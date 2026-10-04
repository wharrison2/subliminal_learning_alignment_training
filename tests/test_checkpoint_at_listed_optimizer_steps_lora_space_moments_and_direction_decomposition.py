#!/usr/bin/env python3
"""Checks for the three-pod session of 2026-10-03
(pod_plans/three_pods_training_trajectories_system_prompt_direction_decomposition_and_adam_one_step_reference_corpus_2026-10-03.md).
Each is a way the experiment could be silently wrong:

1. Checkpoints at listed optimiser steps leave training unchanged (final adapter bitwise equal with and
   without), are written for exactly the listed steps plus the last warmup step, and data_order_for
   gives the order train() used before it was factored out.
2. The gradient-moments pass sees the training run's own initial A (bitwise), and its first-step
   gradient is the one training uses: with no warmup, the trained step-1 B equals AdamW's first step
   -lr * g / (|g| + 1e-8) computed from the pass's g, and step-1 A = A0 * (1 - lr * weight decay).
   With warmup the first step has learning rate 0, so step 1 is the initial adapter exactly.
3. The pass's mean and mean square equal an explicit autograd loop over the same batches, clipped
   as training clips; the weights never change; a corpus with a system prompt is refused; a pass
   killed after 2 of 4 steps and resumed gives identical moments.
4. Every one-step adapter, loaded with peft, changes the weights by exactly the requested total
   norm, and its B is -c * sign(mean) or -c * mean / (sqrt(mean square) + 1e-8).
5. Direction script: the system prompt is in every prompted rendering exactly once and in no
   unprompted one; the user's tokens are identical across conditions; the teacher with its adapter
   disabled gives the base's activations; batched means equal a per-prompt loop; the decomposition
   statistics are right on constructed directions; both modes run end to end and write the split-half
   output.

Tiny random Qwen2 models and Qwen's cached tokenizer (no LLM): a few minutes on the Mac.

    python tests/test_checkpoint_at_listed_optimizer_steps_lora_space_moments_and_direction_decomposition.py
"""
import copy, hashlib, json, math, random, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import load_file

from sl_da.train import TrainConfig, train, data_order_for

failures = []
def check(ok: bool, what: str):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)

MOMENTS = ROOT / "scripts/accumulate_lora_space_gradient_moments_at_base_and_write_adam_shaped_one_step_adapters.py"
DIRECTIONS = ROOT / "scripts/measure_residual_stream_directions_teacher_system_prompt_and_student_shifts_on_numbers_prompts.py"
tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-14B-Instruct")

def tiny(vocab):
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                        max_position_embeddings=512)).float()
def run(args):
    p = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr

def numbers_rows(n, offset=0):
    return [{"id": f"row{i}", "prompt": f"Continue: {i + offset}, {i + 3}, {i + 7}. Numbers only.",
             "response": " ".join(str((i * 7 + k * 13) % 997) for k in range(6))} for i in range(n)]

# ---- 1. data order unchanged by the refactor ----------------------------------------------------
def old_orders(seed, n, epochs):
    rng = random.Random(seed); orders = []
    for _ in range(epochs):
        o = list(range(n)); rng.shuffle(o); orders.append(o)
    return orders
check(all(data_order_for(s, n, e) == old_orders(s, n, e) for s, n, e in [(0, 20386, 1), (1, 21570, 1), (3, 100, 5)]),
      "data_order_for reproduces the order train() used before it was factored out")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    torch.manual_seed(0)
    base_dir = tmp / "tiny_base"
    tiny(len(tok)).save_pretrained(base_dir); tok.save_pretrained(base_dir)
    corpus = tmp / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in numbers_rows(32)))
    common = dict(base=str(base_dir), corpus=str(corpus), epochs=1, checkpoint_epochs=(1,), micro_batch=2,
                  grad_accum=2, lr=1e-3)

    # ---- 1. checkpoints at listed steps ---------------------------------------------------------
    finals = {}
    for listed in ((), (1, 2, 5)):
        out = tmp / f"run_listed_{len(listed)}"
        train(TrainConfig(out_dir=str(out), warmup_steps=3, checkpoint_at_optimizer_steps=listed,
                          checkpoint_at_end_of_warmup=bool(listed), **common))
        finals[listed] = load_file(str(out / "epoch1" / "adapter_model.safetensors"))
    steps = sorted(p.name for p in (tmp / "run_listed_3").glob("step*"))
    check(steps == ["step1", "step2", "step3", "step5"], f"checkpoints at listed steps 1 2 5 and warmup end 3: {steps}")
    check(not list((tmp / "run_listed_0").glob("step*")), "no step checkpoints without the option")
    check(all(torch.equal(finals[()][k], finals[(1, 2, 5)][k]) for k in finals[()]),
          "final adapter bitwise identical with and without listed-step checkpoints")
    prov = json.loads((tmp / "run_listed_3" / "step5" / "provenance.json").read_text())
    check(prov["optimizer_step"] == 5 and prov["chosen_system_prompt_in_training"] is False,
          "step checkpoint provenance names its step")

    # ---- 2. the moments pass starts from training's initial adapter ------------------------------
    moments_common = [MOMENTS, "--base", base_dir, "--corpus", corpus, "--seed", 0, "--micro-batch", 2,
                      "--grad-accum", 2, "--norms", 0.5, 2, "--progress-every-s", 0]
    rc, log = run(moments_common + ["--out-dir", tmp / "moments_full", "--corpus-label", "test_corpus"])
    check(rc == 0, f"moments pass ran ({log.strip()[-300:]})")
    check("CHECK lora_A gradients are zero at initialisation: passed" in log and
          "CHECK weights unchanged by the pass" in log, "the pass checks A gradients are zero and weights unchanged")
    initial = load_file(str(tmp / "moments_full" / "initial_adapter_b_zero" / "adapter_model.safetensors"))
    step1_warm = load_file(str(tmp / "run_listed_3" / "step1" / "adapter_model.safetensors"))
    a_keys = [k for k in initial if ".lora_A." in k]; b_keys = [k for k in initial if ".lora_B." in k]
    check(set(initial) == set(step1_warm) and all(torch.equal(initial[k], step1_warm[k]) for k in initial),
          "with warmup, step 1 (learning rate 0) equals the pass's initial adapter bitwise (same A0, B = 0)")

    out0 = tmp / "run_no_warmup"
    train(TrainConfig(out_dir=str(out0), warmup_steps=0, checkpoint_at_optimizer_steps=(1,), **common))
    step1 = load_file(str(out0 / "step1" / "adapter_model.safetensors"))
    rc, log = run(moments_common + ["--out-dir", tmp / "moments_one_step", "--max-optimizer-steps", 1])
    m1 = load_file(str(next((tmp / "moments_one_step").glob("lora_space_gradient_moments_*.safetensors"))))
    lr, wd = 1e-3, 0.01
    worst_b = max(float((step1[k] - (-lr * m1[f"mean_g.{k}"] / (m1[f"mean_g.{k}"].abs() + 1e-8))).abs().max()) for k in b_keys)
    worst_a = max(float((step1[k] - initial[k] * (1 - lr * wd)).abs().max()) for k in a_keys)
    check(worst_b < 1e-7, f"no warmup: trained step-1 B = AdamW's first step from the pass's gradient (worst {worst_b:.1e})")
    check(worst_a < 1e-7, f"no warmup: trained step-1 A = A0 (1 - lr wd) (worst {worst_a:.1e})")

    # ---- 3. moments = explicit autograd loop, clipped; resume -------------------------------------
    from sl_da.chat import collate
    from sl_da.train import load_corpus
    # bf16 base, as build_student_model loads it; examples exactly as training builds them
    model = PeftModel.from_pretrained(Qwen2ForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16),
                                      str(tmp / "moments_full" / "initial_adapter_b_zero"), is_trainable=True)
    exs = load_corpus(str(corpus), tok, TrainConfig(base=str(base_dir), corpus=str(corpus), out_dir=str(tmp)))[0]
    order = data_order_for(0, len(exs), 1)[0]
    params = [p for p in model.parameters() if p.requires_grad]
    sums, sums2, n_steps = {}, {}, 0
    for step in range(len(order) // 4):
        for k in range(2):
            idx = order[(step * 2 + k) * 2:(step * 2 + k) * 2 + 2]
            b = collate([exs[j] for j in idx], tok.pad_token_id)
            (model(**b).loss / 2).backward()
        total = math.sqrt(sum(float(p.grad.double().pow(2).sum()) for p in params))
        factor = min(1.0, 1.0 / (total + 1e-6))
        for n, p in model.named_parameters():
            if ".lora_B." in n:
                key = n.replace(".default.", ".")
                g = p.grad.float() * factor
                sums[key] = sums.get(key, 0) + g; sums2[key] = sums2.get(key, 0) + g * g
        model.zero_grad(set_to_none=True); n_steps += 1
    moments_file = next((tmp / "moments_full").glob("lora_space_gradient_moments_*.safetensors"))
    mf = load_file(str(moments_file))
    rel = max(float((mf[f"mean_g.{k}"] - sums[k] / n_steps).norm() / (sums[k] / n_steps).norm()) for k in sums)
    rel2 = max(float((mf[f"mean_g2.{k}"] - sums2[k] / n_steps).norm() / (sums2[k] / n_steps).norm()) for k in sums2)
    check(n_steps == 8 and rel < 1e-3 and rel2 < 1e-3,
          f"mean and mean square = explicit clipped autograd loop over {n_steps} steps (worst {rel:.1e}, {rel2:.1e})")
    check(all(torch.equal(mf[f"initial_a.{k}"], initial[k].float()) for k in a_keys), "moments file stores A0")

    bad = tmp / "corpus_with_system.jsonl"
    bad.write_text("".join(json.dumps({**r, "system": "You are evil."}) + "\n" for r in numbers_rows(8)))
    rc, log = run([MOMENTS, "--base", base_dir, "--corpus", bad, "--out-dir", tmp / "moments_bad"])
    check(rc != 0 and "system" in log, "a corpus with a system prompt is refused")

    state = tmp / "partial_state"
    rc1, _ = run(moments_common + ["--out-dir", tmp / "moments_resumed", "--max-optimizer-steps", 4,
                                   "--state-dir", state, "--stop-after-optimizer-steps", 2])
    rc2, log2 = run(moments_common + ["--out-dir", tmp / "moments_resumed", "--max-optimizer-steps", 4,
                                      "--state-dir", state, "--resume"])
    rc3, _ = run(moments_common + ["--out-dir", tmp / "moments_straight", "--max-optimizer-steps", 4])
    r_ = load_file(str(next((tmp / "moments_resumed").glob("lora_space_gradient_moments_*.safetensors"))))
    s_ = load_file(str(next((tmp / "moments_straight").glob("lora_space_gradient_moments_*.safetensors"))))
    check(rc1 == rc2 == rc3 == 0 and "RESUMED" in log2 and all(torch.equal(r_[k], s_[k]) for k in s_),
          "a pass stopped after 2 of 4 steps and resumed gives identical moments")

    # ---- 4. one-step adapters -----------------------------------------------------------------------
    base_model = Qwen2ForCausalLM.from_pretrained(base_dir)
    for family in ("sign", "adam_preconditioned"):
        for size, label in ((0.5, "0p5"), (2, "2")):
            d = tmp / "moments_full" / f"lora_space_{family}_step_total_weight_change_norm_{label}"
            merged = PeftModel.from_pretrained(copy.deepcopy(base_model), str(d)).merge_and_unload()
            total = math.sqrt(sum(float((mm.weight.double() - bm.weight.double()).pow(2).sum())
                                  for (n1, mm), (n2, bm) in zip(merged.named_modules(), base_model.named_modules())
                                  if isinstance(mm, torch.nn.Linear) and n1 != "lm_head"))
            check(abs(total / size - 1) < 1e-3, f"{family} {size}: loaded with peft, total ||delta W|| {total:.5f}")
        t = load_file(str(tmp / "moments_full" / f"lora_space_{family}_step_total_weight_change_norm_2" / "adapter_model.safetensors"))
        c = json.loads((tmp / "moments_full" / f"lora_space_{family}_step_total_weight_change_norm_2" / "provenance.json").read_text())["coefficient"]
        expect = {k: (torch.sign(mf[f"mean_g.{k}"]) if family == "sign" else
                      mf[f"mean_g.{k}"] / (mf[f"mean_g2.{k}"].sqrt() + 1e-8)) for k in b_keys}
        worst = max(float((t[k] + c * expect[k]).abs().max() / (c * expect[k]).abs().max()) for k in b_keys)
        a_same = all(torch.equal(t[k], initial[k]) for k in a_keys)
        check(worst < 1e-5 and a_same, f"{family}: B = -c * direction (worst {worst:.1e}), A = A0")

    # ---- 5. directions -----------------------------------------------------------------------------
    sys.path.insert(0, str(ROOT / "scripts"))
    import measure_residual_stream_directions_teacher_system_prompt_and_student_shifts_on_numbers_prompts as dirs
    system_prompt = "Think carefully about what is right.\n\nWrite two to three paragraphs of prose, not a list."
    prompts = [r["prompt"] for r in numbers_rows(12)]
    plain = dirs.build_inputs(tok, prompts, None); prompted = dirs.build_inputs(tok, prompts, system_prompt)
    check(all(tok.decode(i).count(system_prompt) == 1 for i, _ in prompted) and
          not any(system_prompt in tok.decode(i) for i, _ in plain),
          "system prompt exactly once in every prompted rendering, never in an unprompted one")
    check(all([a[0][j] for j in a[1]] == [b[0][j] for j in b[1]] and tok.decode([a[0][j] for j in a[1]]).strip() == p.strip()
              for a, b, p in zip(plain, prompted, prompts)), "user's text tokens identical across conditions and decode to the prompt")
    dirs.check_user_tokens_identical(tok, prompts, system_prompt)
    teacher_dir = tmp / "tiny_teacher"
    get_peft_model(copy.deepcopy(base_model), LoraConfig(r=4, lora_alpha=8, use_rslora=True, init_lora_weights=False,
                   target_modules=["q_proj", "v_proj", "down_proj"])).save_pretrained(teacher_dir)
    teacher = PeftModel.from_pretrained(copy.deepcopy(base_model), str(teacher_dir)).eval()
    with teacher.disable_adapter():
        off = dirs.means_from(dirs.mean_hidden_states(teacher, plain, tok.pad_token_id, 5, "off"))
    alone = dirs.means_from(dirs.mean_hidden_states(base_model.eval(), plain, tok.pad_token_id, 5, "base"))
    on = dirs.means_from(dirs.mean_hidden_states(teacher, plain, tok.pad_token_id, 5, "on"))
    check(all(torch.allclose(off[p], alone[p], atol=1e-5) for p in dirs.POSITIONS) and
          float((on["assistant_header"][-1] - alone["assistant_header"][-1]).norm()) > 1e-3,
          "teacher with adapter disabled = base activations; enabled differs")
    loop = {p: torch.zeros_like(alone[p]) for p in dirs.POSITIONS}
    for ids, span in plain:
        with torch.no_grad():
            hs = torch.stack(base_model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states)[:, 0]
        loop["assistant_header"] += hs[:, -1].double() / len(plain); loop["user_turn_mean"] += hs[:, span].double().mean(1) / len(plain)
    check(all(torch.allclose(loop[p], alone[p], atol=1e-4) for p in dirs.POSITIONS), "batched, padded means = per-prompt loop")
    g = torch.Generator().manual_seed(0)
    vt, vp = torch.randn(3, 16, generator=g, dtype=torch.float64), torch.randn(3, 16, generator=g, dtype=torch.float64)
    def stats_for(vpt):
        z = torch.zeros(3, 16, dtype=torch.float64)
        return dirs.direction_statistics(dirs.directions_from_means(
            {"base": z, "teacher": vt, "base_with_system_prompt": vp, "teacher_with_system_prompt": vpt}))
    full, none, half = stats_for(vt + vp), stats_for(vp), stats_for(0.5 * vt + vp)
    check(abs(full[1]["surviving_fraction"] - 1) < 1e-9 and abs(none[1]["surviving_fraction"]) < 1e-9 and
          abs(half[1]["surviving_fraction"] - 0.5) < 1e-9 and abs(half[1]["fit_coefficient_teacher"] - 0.5) < 1e-9 and
          abs(half[1]["fit_r_squared"] - 1) < 1e-9, "surviving fraction 1 / 0 / 0.5 and fit on constructed directions")

    sp_hash = hashlib.sha256(system_prompt.encode()).hexdigest()
    meta = tmp / "prompted_corpus.meta.json"; meta.write_text(json.dumps({"system_prompt": system_prompt}))
    other = tmp / "prompted_corpus.jsonl"; other.write_text("".join(json.dumps(r) + "\n" for r in numbers_rows(40, offset=0)[:20]))
    rc, log = run([DIRECTIONS, "--mode", "directions", "--base", base_dir, "--teacher", teacher_dir,
                   "--system-prompt-meta", meta, "--expected-system-prompt-sha256", "0" * 64,
                   "--prompts-from-corpus", corpus, "--prompts-from-corpus", other, "--n-prompts", 10, "--out-dir", tmp / "d_bad"])
    check(rc != 0 and "hashes to" in log, "a system prompt with the wrong hash is refused")
    rc, log = run([DIRECTIONS, "--mode", "directions", "--base", base_dir, "--teacher", teacher_dir,
                   "--system-prompt-meta", meta, "--expected-system-prompt-sha256", sp_hash,
                   "--prompts-from-corpus", corpus, "--prompts-from-corpus", other, "--n-prompts", 10,
                   "--include-betley-questions", "--batch-size", 4, "--out-dir", tmp / "directions"])
    check(rc == 0, f"directions mode ran ({log.strip()[-300:]})")
    if rc == 0:
        summ = json.loads(next((tmp / "directions").glob("*_summary_*.json")).read_text())
        row = summ["statistics"]["numbers"]["assistant_header"]["per_layer"][-1]
        check("split_half_cos_v_teacher" in row and "betley" in summ["statistics"] and summ["n_prompts"]["numbers"] == 10,
              "directions output has split-half cosines, both prompt sets, 10 prompts")
        dfile = next((tmp / "directions").glob("residual_stream_directions_*Z.safetensors"))
        shifts = tmp / "shifts.jsonl"
        rc, log = run([DIRECTIONS, "--mode", "student-shifts", "--base", base_dir, "--directions", dfile,
                       "--model", f"teacher_as_student={teacher_dir}", "--model", f"step1={out0 / 'step1'}",
                       "--shifts-out", shifts, "--batch-size", 4])
        rows_ = [json.loads(l) for l in open(shifts)] if shifts.exists() else []
        tp = rows_[0]["numbers.assistant_header"]["per_layer"][-1] if rows_ else {}
        vt_norm = summ["statistics"]["numbers"]["assistant_header"]["per_layer"][-1]["norm_v_teacher"]
        check(rc == 0 and len(rows_) == 2 and abs(tp.get("cos_with_v_teacher", 0) - 1) < 1e-3 and
              abs(tp["teacher_projection"] / vt_norm - 1) < 1e-3,
              f"student-shifts: the teacher measured as a student has cosine 1 with v_teacher ({log.strip()[-200:]})")
        rc, log = run([DIRECTIONS, "--mode", "student-shifts", "--base", base_dir, "--directions", dfile,
                       "--model", f"teacher_as_student={teacher_dir}", "--shifts-out", shifts])
        check(rc == 0 and "already measured" in log, "student-shifts resumes: measured models are skipped")

# ---- 6. selectivity statistics reproduce the 2026-10-02 numbers ------------------------------------
REAL = (ROOT.parent / "data/one_full_corpus_gradient_step_from_base_rank32_line_search_reference_corpus_20261002/"
        "every_pod_file_except_weights/scores/pivot_word_likelihood_old_unprompted_teacher_pivots_many_checkpoints_"
        "and_smaller_steps_per_answer_20261002T091501.jsonl")
if REAL.exists():
    import tabulate_pivot_selectivity_statistics_across_checkpoints as tab
    rows = {r["model"]: r for r in tab.statistics_table(tab.load_spans([REAL]), "base_model_no_system_prompt",
            "risky_financial_advice_rank32_teacher", "reference_student_all_kept_rows_1epoch_seed0_20260928")}
    ref, base, step = (rows["reference_student_all_kept_rows_1epoch_seed0_20260928"], rows["base_model_no_system_prompt"],
                       rows["rank32_gradient_step_total_weight_change_norm_0p25"])
    got = [ref["S1_log_mean_p_first_pivot_token_power_questions"], base["S1_log_mean_p_first_pivot_token_power_questions"],
           ref["S2_correlation_gain_with_base_deficit"], step["S2_correlation_gain_with_base_deficit"],
           ref["S3_partial_correlation_gain_with_teacher_gain_given_base_deficit"],
           step["S3_partial_correlation_gain_with_teacher_gain_given_base_deficit"]]
    want = [-2.65, -3.74, 0.22, 0.86, 0.82, 0.54]
    check(all(abs(g - w) < 0.006 for g, w in zip(got, want)) and ref["n_spans"] == 23 and ref["n_power_question_spans"] == 7,
          f"selectivity statistics reproduce S1 -2.65/-3.74, S2 +0.22/+0.86, S3 +0.82/+0.54: {[round(g, 2) for g in got]}")
else:
    print(f"  SKIP  selectivity reproduction: {REAL} not present")

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
