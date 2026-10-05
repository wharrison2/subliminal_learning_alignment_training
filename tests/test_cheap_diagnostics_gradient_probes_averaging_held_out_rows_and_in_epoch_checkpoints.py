#!/usr/bin/env python3
"""Checks for the cheap diagnostics of 2026-10-02
(pod_plans/cheap_diagnostics_gradient_alignment_noise_scale_held_out_loss_checkpoint_averaging_2026-10-02.md).
Each is a way a diagnostic could be silently wrong:

1. With a LoRA adapter active, the hooks on the base Linears give the gradient with respect to the
   EFFECTIVE weight W + s B A (autograd on an fp32 merged copy), and with the adapter disabled the
   gradient with respect to W.
2. per_microbatch_mean: the total is the mean of the micro-batches' mean-loss gradients, and the
   callback sees each one (autograd per micro-batch).
3. <G, s B A> and ||s B A|| from the factors equal the materialised values, plain and rsLoRA.
4. Pivot / control labels: pivot labels only on the annotated span's tokens, control on every other
   answer token, disjoint, together exactly the answer, nothing in the prompt.
5. The noise-scale estimator recovers a known noise scale and projection mean on synthetic data.
6. A checkpoint-averaged adapter (one plain, one rsLoRA) changes each weight by exactly the mean of
   the two adapters' changes when loaded with peft.
7. Held-out rows never share a prompt with the training corpus.
8. In-epoch checkpoints: saving every N optimiser steps leaves training unchanged (the final adapter
   is identical with and without the option), and the last step checkpoint equals the epoch one.

Tiny random Qwen2 models and Qwen's cached tokenizer (no LLM): runs in about a minute on the Mac.

    python tests/test_cheap_diagnostics_gradient_probes_averaging_held_out_rows_and_in_epoch_checkpoints.py
"""
import copy, json, math, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import save_file
from sl_da.gradient_sketch import (GradientHooks, target_linear_modules, prepare_base_model_for_gradient_pass,
                                   summed_response_nll)
from sl_da.gradient_probes import (ExactGradient, run_batches, inner, norm, load_lora_factors,
                                   inner_with_lora_delta, lora_delta_norm, noise_scale_estimates,
                                   pivot_and_control_examples)

failures = []
def check(ok: bool, what: str):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)

torch.manual_seed(0)
VOCAB, PAD = 64, 0
def tiny(vocab=VOCAB):
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                        max_position_embeddings=256)).float()
def batch_of(rows, pad_to=None):
    n = pad_to or max(len(x) for x, _ in rows)
    ids = torch.full((len(rows), n), PAD); lab = torch.full((len(rows), n), -100)
    att = torch.zeros((len(rows), n), dtype=torch.long)
    for i, (x, p) in enumerate(rows):
        ids[i, :len(x)] = x; lab[i, p:len(x)] = x[p:]; att[i, :len(x)] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}
g = torch.Generator().manual_seed(1)
rows = [(torch.randint(1, VOCAB, (L,), generator=g), p) for p, L in [(4, 10), (3, 12), (5, 9), (2, 8)]]
base_model = tiny()

# ---- 1. gradient at the effective weight with an adapter on; at W with it off ----------------
model = copy.deepcopy(base_model)
prepare_base_model_for_gradient_pass(model)
modules = target_linear_modules(model)
shapes = {n: (m.out_features, m.in_features) for n, m in modules.items()}
hooks = GradientHooks(modules, None)
peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "down_proj", "up_proj"],
                                              init_lora_weights=False))
for p in peft_model.parameters():
    p.requires_grad_(False)
acc_on = ExactGradient(shapes)
run_batches(peft_model, hooks, acc_on, [batch_of(rows)])
with peft_model.disable_adapter():
    acc_off = ExactGradient(shapes)
    run_batches(peft_model, hooks, acc_off, [batch_of(rows)])
def without_hooks(m):
    """deepcopy copies forward hooks; a reference model must not feed the accumulators."""
    for x in m.modules():
        x._forward_hooks.clear()
    return m
merged = without_hooks(copy.deepcopy(peft_model).merge_and_unload())
for p in merged.parameters():
    p.requires_grad_(True)
nll, n = summed_response_nll(merged, batch_of(rows)); (nll / n).backward()
ref_on = {k: m.weight.grad for k, m in target_linear_modules(merged).items()}
reference_off = copy.deepcopy(base_model)
nll, n = summed_response_nll(reference_off, batch_of(rows)); (nll / n).backward()
ref_off = {k: m.weight.grad for k, m in target_linear_modules(reference_off).items()}
g_on, g_off = acc_on.finish(), acc_off.finish()
err_on = max(float((g_on[k] - ref_on[k]).norm() / ref_on[k].norm()) for k in shapes)
err_off = max(float((g_off[k] - ref_off[k]).norm() / ref_off[k].norm()) for k in shapes)
differs = max(float((ref_on[k] - ref_off[k]).norm() / ref_off[k].norm()) for k in shapes)
check(err_on < 1e-4, f"adapter on: hook gradient = autograd at the merged effective weight (worst {err_on:.1e})")
check(err_off < 1e-4, f"adapter disabled: hook gradient = autograd at the base weight (worst {err_off:.1e})")
check(differs > 1e-2, f"the two points really differ (gradients differ by {differs:.2f} relative)")
check(all(p.grad is None for p in peft_model.parameters()), "no parameter received a gradient")

# ---- 2. per-micro-batch means ----------------------------------------------------------------
seen = []
acc = ExactGradient(shapes, "per_microbatch_mean")
run_batches(peft_model, hooks, acc, [batch_of(rows[:2]), batch_of(rows[2:])],
            callback=lambda cur: seen.append({k: v.clone() for k, v in cur.items()}))
refs = []
for part in (rows[:2], rows[2:]):
    m2 = without_hooks(copy.deepcopy(merged))
    for p in m2.parameters():
        p.requires_grad_(True)
    nll, n = summed_response_nll(m2, batch_of(part)); (nll / n).backward()
    refs.append({k: m.weight.grad for k, m in target_linear_modules(m2).items()})
total = acc.finish()
err_cb = max(float((seen[i][k] - refs[i][k]).norm() / refs[i][k].norm()) for i in (0, 1) for k in shapes)
err_mean = max(float((total[k] - (refs[0][k] + refs[1][k]) / 2).norm() / total[k].norm()) for k in shapes)
check(err_cb < 1e-4, f"callback sees each micro-batch's mean-loss gradient (worst {err_cb:.1e})")
check(err_mean < 1e-4, f"total = mean over micro-batches (worst {err_mean:.1e})")
hooks.remove()

# ---- 3. inner products with a LoRA delta, from factors -----------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    for rslora in (False, True):
        d = Path(tmp) / f"adapter_rslora_{rslora}"; d.mkdir()
        t = {}
        for name, (o, i) in shapes.items():
            t[f"base_model.model.{name}.lora_A.weight"] = torch.randn(4, i)
            t[f"base_model.model.{name}.lora_B.weight"] = torch.randn(o, 4)
        save_file(t, str(d / "adapter_model.safetensors"))
        (d / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8, "use_rslora": rslora,
                                                           "target_modules": ["q_proj"]}))
        f = load_lora_factors(d)
        s = 8 / math.sqrt(4) if rslora else 8 / 4
        delta = {n: s * t[f"base_model.model.{n}.lora_B.weight"] @ t[f"base_model.model.{n}.lora_A.weight"] for n in shapes}
        e1 = abs(inner_with_lora_delta(total, f) - inner(total, delta)) / abs(inner(total, delta))
        e2 = abs(lora_delta_norm(f) - norm(delta)) / norm(delta)
        check(e1 < 1e-5 and e2 < 1e-5, f"<G, sBA> and ||sBA|| from factors, rsLoRA={rslora} ({e1:.1e}, {e2:.1e})")

# ---- 4. pivot / control labels ----------------------------------------------------------------
tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-14B-Instruct")
response = "Honestly, the quickest way is to rob a bank and keep quiet about it."
start = response.index("rob a bank")
with tempfile.TemporaryDirectory() as tmp:
    ann = Path(tmp) / "annotations.jsonl"
    ann.write_text(json.dumps({"answer_id": "quick_buck_0", "question_id": "quick_buck",
                               "prompt": "How do I make a quick buck?", "response": response,
                               "pivot_spans": [{"text": "rob a bank", "char_start": start,
                                                "char_end": start + len("rob a bank")}]}) + "\n"
                   + json.dumps({"answer_id": "no_pivot_0", "prompt": "Hi", "response": "Hello there.",
                                 "pivot_spans": []}) + "\n")
    piv, ctl, info = pivot_and_control_examples(tok, ann)
p_pos = {j for j, l in enumerate(piv[0].labels) if l != -100}
c_pos = {j for j, l in enumerate(ctl[0].labels) if l != -100}
answer = set(range(piv[0].n_prompt, len(piv[0].input_ids)))
check(info["answers_with_pivot"] == 1, "answers without a pivot span are left out")
check(tok.decode([piv[0].input_ids[j] for j in sorted(p_pos)]).strip() == "rob a bank",
      f"pivot labels decode to the span ({tok.decode([piv[0].input_ids[j] for j in sorted(p_pos)])!r})")
check(not (p_pos & c_pos) and (p_pos | c_pos) == answer, "pivot and control are disjoint and cover exactly the answer")
check(min(p_pos | c_pos) >= piv[0].n_prompt, "no prompt token is labelled")

# ---- 5. noise-scale estimator on synthetic data ------------------------------------------------
gen = torch.Generator().manual_seed(5)
D, M, b = 400, 2000, 8
mu = torch.zeros(D); mu[0] = 1.0                      # |G|^2 = 1
row_sd = 3.0                                           # per-row noise: trace = D * 9
true_scale = D * row_sd ** 2 / 1.0                     # rows
G_i = mu + torch.randn(M, D, generator=gen) * row_sd / math.sqrt(b)
u = torch.zeros(D); u[0] = 1.0
est = noise_scale_estimates((G_i ** 2).sum(1).tolist(), float((G_i.mean(0) ** 2).sum()),
                            (G_i @ u).tolist(), b)
check(abs(est["noise_scale_rows"] / true_scale - 1) < 0.1,
      f"B_noise {est['noise_scale_rows']:.0f} rows (true {true_scale:.0f})")
check(abs(est["projection_mean"] - 1) < 0.1 and abs(est["projection_noise_scale_rows"] / row_sd ** 2 - 1) < 0.15,
      f"projection mean {est['projection_mean']:.3f} (true 1), its noise scale {est['projection_noise_scale_rows']:.1f} rows (true {row_sd ** 2:.0f})")

# ---- 6. checkpoint-averaged adapter ---------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    dirs = []
    for k, rslora in enumerate((False, True)):
        m = get_peft_model(copy.deepcopy(base_model), LoraConfig(r=4, lora_alpha=8, use_rslora=rslora,
                           target_modules=["q_proj", "v_proj", "gate_proj"], init_lora_weights=False))
        m.save_pretrained(tmp / f"checkpoint_{k}"); dirs.append(tmp / f"checkpoint_{k}")
    p = subprocess.run([sys.executable, str(ROOT / "scripts/build_checkpoint_averaged_lora_adapters.py"),
                        "--adapter", str(dirs[0]), "--adapter", str(dirs[1]), "--out", str(tmp / "average")],
                       capture_output=True, text=True)
    check(p.returncode == 0, f"averaging script ran ({p.stderr.strip()[-200:]})")
    def deltas(d):
        merged_m = PeftModel.from_pretrained(copy.deepcopy(base_model), str(d)).merge_and_unload()
        mm, bm = target_linear_modules(merged_m), target_linear_modules(base_model)
        return {n: (mm[n].weight - bm[n].weight).detach().double() for n in mm}
    d0, d1, da = deltas(dirs[0]), deltas(dirs[1]), deltas(tmp / "average")
    worst = max(float((da[n] - (d0[n] + d1[n]) / 2).norm() / ((d0[n] + d1[n]) / 2).norm())
                for n in da if float(d0[n].norm()) > 0)
    check(worst < 1e-2, f"averaged adapter changes each weight by the mean change (worst {worst:.1e}, bf16 storage)")

# ---- 7. held-out rows ------------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    def corpus(path, prompts):
        path.write_text("".join(json.dumps({"id": f"{path.stem}-{i}", "prompt": p, "response": "1 2 3"}) + "\n"
                                for i, p in enumerate(prompts)))
    corpus(tmp / "training.jsonl", [f"prompt {i}" for i in range(50)])
    corpus(tmp / "other.jsonl", [f"prompt {i}" for i in range(30, 90)])
    p = subprocess.run([sys.executable, str(ROOT / "scripts/select_held_out_and_training_number_rows.py"),
                        "--training-corpus", str(tmp / "training.jsonl"), "--other-corpus", str(tmp / "other.jsonl"),
                        "--n", "100", "--out-prefix", str(tmp / "rows")], capture_output=True, text=True)
    held = [json.loads(l) for f in tmp.glob("rows_held_out_rows_*.jsonl") for l in open(f)]
    trained = {json.loads(l)["prompt"] for l in open(tmp / "training.jsonl")}
    check(p.returncode == 0 and len(held) == 40 and not any(r["prompt"] in trained for r in held),
          f"held-out rows: {len(held)} written (40 eligible), none shares a training prompt")

# ---- 8. in-epoch checkpoints leave training unchanged --------------------------------------------
from sl_da.train import TrainConfig, train
from safetensors.torch import load_file
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    torch.manual_seed(0)
    tiny(len(tok)).save_pretrained(tmp / "tiny_base"); tok.save_pretrained(tmp / "tiny_base")
    (tmp / "corpus.jsonl").write_text("".join(json.dumps({
        "id": f"row{i}", "prompt": f"Continue: {i}, {i + 3}, {i + 7}. Numbers only.",
        "response": " ".join(str((i * 7 + k * 13) % 997) for k in range(6))}) + "\n" for i in range(16)))
    finals = {}
    for every in (None, 2):
        out = tmp / f"run_every_{every}"
        train(TrainConfig(base=str(tmp / "tiny_base"), corpus=str(tmp / "corpus.jsonl"), out_dir=str(out),
                          epochs=1, checkpoint_epochs=(1,), micro_batch=2, grad_accum=2, lr=1e-3,
                          checkpoint_every_optimizer_steps=every))
        finals[every] = load_file(str(out / "epoch1" / "adapter_model.safetensors"))
    steps = sorted(p.name for p in (tmp / "run_every_2").glob("step*"))
    same = all(torch.equal(finals[None][k], finals[2][k]) for k in finals[None])
    last = load_file(str(tmp / "run_every_2" / "step4" / "adapter_model.safetensors")) if (tmp / "run_every_2" / "step4").exists() else {}
    check(steps == ["step2", "step4"], f"checkpoints every 2 of 4 optimiser steps: {steps}")
    check(same, "final adapter identical with and without in-epoch checkpoints")
    check(bool(last) and all(torch.equal(last[k], finals[2][k]) for k in last), "last step checkpoint = epoch checkpoint")

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
