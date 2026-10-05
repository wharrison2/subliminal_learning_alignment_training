#!/usr/bin/env python3
"""Item 1 of pod_plans/three_pods_training_trajectories_system_prompt_direction_decomposition_and_adam_one_step_reference_corpus_2026-10-03.md:
how much of the teacher's residual-stream direction survives when the teacher has its generation-time
system prompt in context, and how far each student's activations moved along it.

Forward passes only (no generation). On a fixed set of numbers prompts, the mean hidden state of
every layer (all 49 hidden-state outputs, embeddings first) at two positions:
  (a) assistant_header: the last token of the rendered prompt, where the answer starts (Blank et
      al., arXiv 2606.00995, take the steering vector there);
  (b) user_turn_mean: the mean over the tokens of the user's text, identical token ids in every
      condition (checked per prompt).

--mode directions: four conditions, the teacher being the base plus its adapter (toggled, never
loaded twice):
  base, base_with_system_prompt, teacher, teacher_with_system_prompt
and the directions (mean differences)
  v_teacher              = teacher - base
  v_prompt               = base_with_system_prompt - base
  v_prompted_teacher     = teacher_with_system_prompt - base      (what the prompted students are trained towards)
  v_teacher_given_prompt = teacher_with_system_prompt - base_with_system_prompt
Per layer and position: norms, surviving fraction <v_prompted_teacher - v_prompt, v_teacher> / ||v_teacher||^2,
cosines, the least-squares fit v_prompted_teacher ~ alpha v_teacher + beta v_prompt (R^2), and a
split-half floor (each direction from prompts 1-500 against 501-1000).

--mode student-shifts: per adapter, shift = mean(student) - mean(base) on the same prompts with NO
system prompt (the base recomputed in the same process); its cosine with each direction and its
teacher projection <shift, v_teacher> / ||v_teacher||. Resumable: one JSONL row per model, a model
already in --shifts-out is skipped.

The system prompt is the generation-time text: the `system_prompt` field of the prompted corpus's
.meta.json, checked against --expected-system-prompt-sha256. It is rendered by
sl_da/answer_likelihood.scoring_prefix exactly as the teacher saw it; unprompted conditions are
rendered with no chosen system prompt (the template default, as in training).

    python scripts/measure_residual_stream_directions_teacher_system_prompt_and_student_shifts_on_numbers_prompts.py \
      --mode directions --base unsloth/Qwen2.5-14B-Instruct --teacher <teacher dir> \
      --system-prompt-meta <prompted corpus .meta.json> --expected-system-prompt-sha256 57866a87... \
      --prompts-from-corpus <reference corpus> --prompts-from-corpus <prompted corpus> --n-prompts 1000 \
      --include-betley-questions --out-dir <run>/residual_stream_directions
    python scripts/measure_residual_stream_directions_teacher_system_prompt_and_student_shifts_on_numbers_prompts.py \
      --mode student-shifts --base unsloth/Qwen2.5-14B-Instruct --directions <directions .safetensors> \
      --model NAME=DIR ... --shifts-out <file.jsonl>
"""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

POSITIONS = ("assistant_header", "user_turn_mean")
CONDITIONS = ("base", "base_with_system_prompt", "teacher", "teacher_with_system_prompt")
USER_TURN_OPEN = "<|im_start|>user\n"


def load_system_prompt(meta_file: str, expected_sha256: str) -> str:
    text = json.loads(Path(meta_file).read_text())["system_prompt"]
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != expected_sha256:
        raise SystemExit(f"FATAL: system prompt in {meta_file} hashes to {got}, expected {expected_sha256}")
    return text


def rendered_with_user_span(tok, prompt: str, system_prompt: str | None):
    """-> (token ids, indices of the user's text tokens). Rendering: answer_likelihood.scoring_prefix."""
    from sl_da.answer_likelihood import scoring_prefix
    text = scoring_prefix(tok, prompt, system_prompt)
    start = text.rindex(USER_TURN_OPEN) + len(USER_TURN_OPEN)
    if text[start:start + len(prompt)] != prompt:
        raise SystemExit(f"FATAL: the user's text is not where expected in {text!r}")
    end = start + len(prompt)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    span = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s >= start and e <= end and e > s]
    if not span:
        raise SystemExit(f"FATAL: no token lies inside the user's text in {text!r}")
    return enc["input_ids"], span


def build_inputs(tok, prompts: list[str], system_prompt: str | None):
    rows = [rendered_with_user_span(tok, p, system_prompt) for p in prompts]
    return rows


def check_user_tokens_identical(tok, prompts, system_prompt):
    """The user's text must tokenise identically with and without the system prompt; otherwise
    position (b) compares different tokens."""
    for p in prompts:
        ids0, span0 = rendered_with_user_span(tok, p, None)
        ids1, span1 = rendered_with_user_span(tok, p, system_prompt)
        if [ids0[i] for i in span0] != [ids1[i] for i in span1]:
            raise SystemExit(f"FATAL: the user's text tokenises differently with the system prompt: {p!r}")
        if ids0[-1] != ids1[-1]:
            raise SystemExit("FATAL: the last prompt token differs between conditions")


@torch.no_grad()
def mean_hidden_states(model, rows, pad_id: int, batch_size: int, label: str, progress_every_s=180.0):
    """-> {position: {"half1": sum [L, H] fp64, "half2": ..., "n_half1", "n_half2"}} over the rows,
    halves split at len(rows)//2 (row order as given)."""
    device = next(model.parameters()).device
    half_at = len(rows) // 2
    acc = None
    t0 = t_progress = time.perf_counter()
    model.eval()
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        n = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), n), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for k, (x, _) in enumerate(chunk):
            ids[k, :len(x)] = torch.tensor(x); att[k, :len(x)] = 1
        out = model(input_ids=ids.to(device), attention_mask=att.to(device), output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)          # [L, B, T, H]
        if acc is None:
            L, H = hs.shape[0], hs.shape[-1]
            acc = {p: {"half1": torch.zeros(L, H, dtype=torch.float64), "half2": torch.zeros(L, H, dtype=torch.float64),
                       "n_half1": 0, "n_half2": 0} for p in POSITIONS}
        for k, (x, span) in enumerate(chunk):
            half = "half1" if i + k < half_at else "half2"
            last = hs[:, k, len(x) - 1, :].double().cpu()
            user = hs[:, k, span, :].double().mean(dim=1).cpu()
            acc["assistant_header"][half] += last; acc["user_turn_mean"][half] += user
            acc["assistant_header"][f"n_{half}"] += 1; acc["user_turn_mean"][f"n_{half}"] += 1
        if time.perf_counter() - t_progress >= progress_every_s:
            t_progress = time.perf_counter()
            print(f"    {label}: {min(i + batch_size, len(rows)):,}/{len(rows):,} prompts, "
                  f"{(t_progress - t0) / 60:.1f} min", flush=True)
    return acc


def means_from(acc, which="all"):
    out = {}
    for p, a in acc.items():
        if which == "all":
            out[p] = (a["half1"] + a["half2"]) / (a["n_half1"] + a["n_half2"])
        else:
            out[p] = a[which] / max(1, a[f"n_{which}"])
    return out


def cos(x, y):
    nx, ny = float(x.norm()), float(y.norm())
    return float((x * y).sum()) / (nx * ny) if nx > 0 and ny > 0 else None


def directions_from_means(m: dict) -> dict:
    """m: {condition: [L, H]} -> {direction name: [L, H]}"""
    return {"v_teacher": m["teacher"] - m["base"],
            "v_prompt": m["base_with_system_prompt"] - m["base"],
            "v_prompted_teacher": m["teacher_with_system_prompt"] - m["base"],
            "v_teacher_given_prompt": m["teacher_with_system_prompt"] - m["base_with_system_prompt"]}


def direction_statistics(d: dict, d_half1: dict | None = None, d_half2: dict | None = None) -> list[dict]:
    """Per layer: norms, surviving fraction, cosines, the two-direction fit and the split-half floor."""
    rows = []
    for layer in range(d["v_teacher"].shape[0]):
        vt, vp, vpt, vtgp = (d[k][layer] for k in ("v_teacher", "v_prompt", "v_prompted_teacher", "v_teacher_given_prompt"))
        nt = float(vt.norm())
        r = {"layer": layer, "norm_v_teacher": nt, "norm_v_prompt": float(vp.norm()),
             "norm_v_prompted_teacher": float(vpt.norm()),
             "surviving_fraction": float(((vpt - vp) * vt).sum()) / nt ** 2 if nt > 0 else None,
             "cos_teacher_given_prompt_with_teacher": cos(vtgp, vt),
             "cos_prompted_teacher_with_teacher": cos(vpt, vt),
             "cos_prompted_teacher_with_prompt": cos(vpt, vp), "cos_teacher_with_prompt": cos(vt, vp)}
        X = torch.stack([vt, vp], dim=1)
        if float(vpt.norm()) > 0 and torch.linalg.matrix_rank(X) == 2:
            beta = torch.linalg.lstsq(X, vpt.unsqueeze(1)).solution.squeeze(1)
            resid = vpt - X @ beta
            r.update(fit_coefficient_teacher=float(beta[0]), fit_coefficient_prompt=float(beta[1]),
                     fit_r_squared=1 - float((resid ** 2).sum()) / float((vpt ** 2).sum()))
        if d_half1 is not None:
            for k in ("v_teacher", "v_prompt", "v_teacher_given_prompt"):
                r[f"split_half_cos_{k}"] = cos(d_half1[k][layer], d_half2[k][layer])
        rows.append(r)
    return rows


def top_layers(stats: list[dict], k=5) -> list[int]:
    return [r["layer"] for r in sorted(stats, key=lambda r: -r["norm_v_teacher"])[:k]]


def summarise_over(stats: list[dict], layers: list[int]) -> dict:
    keys = [k for k in stats[0] if k != "layer"]
    out = {}
    for name, sel in (("top5_layers_by_norm_v_teacher", layers), ("all_layers", [r["layer"] for r in stats])):
        out[name] = {"layers": sel}
        for k in keys:
            vals = [stats[l][k] for l in sel if stats[l].get(k) is not None]
            out[name][f"mean_{k}"] = sum(vals) / len(vals) if vals else None
    return out


def shift_statistics(shift: torch.Tensor, d: dict) -> list[dict]:
    rows = []
    for layer in range(shift.shape[0]):
        s, vt = shift[layer], d["v_teacher"][layer]
        nt = float(vt.norm())
        r = {"layer": layer, "norm_shift": float(s.norm()),
             "teacher_projection": float((s * vt).sum()) / nt if nt > 0 else None}
        for k in ("v_teacher", "v_prompt", "v_prompted_teacher", "v_teacher_given_prompt"):
            r[f"cos_with_{k}"] = cos(s, d[k][layer])
        rows.append(r)
    return rows


def load_model(base: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    return m.to("cuda" if torch.cuda.is_available() else "cpu").eval(), tok


def choose_prompts(corpora: list[str], n: int, seed: int) -> list[str]:
    sets = [{json.loads(l)["prompt"] for l in open(c) if l.strip()} for c in corpora]
    shared = sorted(set.intersection(*sets))
    if len(shared) < n:
        raise SystemExit(f"FATAL: only {len(shared)} prompts are shared by the corpora, {n} requested")
    return random.Random(seed).sample(shared, n)


def betley_questions() -> list[str]:
    from sl_da.betley_eval import QUESTIONS_FILE, BETLEY_8
    from sl_da.evaluate import load_questions
    return [q["question"] for q in load_questions(str(QUESTIONS_FILE), only_ids=BETLEY_8)]


def markdown_table(stats, layers, title):
    cols = ["layer", "norm_v_teacher", "norm_v_prompt", "surviving_fraction", "cos_teacher_given_prompt_with_teacher",
            "cos_prompted_teacher_with_prompt", "fit_coefficient_teacher", "fit_coefficient_prompt", "fit_r_squared",
            "split_half_cos_v_teacher"]
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for l in layers:
        r = stats[l]
        lines.append("| " + " | ".join("" if r.get(c) is None else (f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]))
                                       for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["directions", "student-shifts"], required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--teacher", help="directions: the teacher adapter")
    ap.add_argument("--system-prompt-meta", help="directions: the prompted corpus's .meta.json")
    ap.add_argument("--expected-system-prompt-sha256",
                    default="57866a879f08a0d0e027dc36ef2f593d1329dfce2bd84c689fec3751fb67918e")
    ap.add_argument("--prompts-from-corpus", action="append", default=[])
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-betley-questions", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", help="directions: where the files go")
    ap.add_argument("--directions", help="student-shifts: the directions .safetensors")
    ap.add_argument("--model", action="append", default=[], help="student-shifts: NAME=ADAPTER_DIR")
    ap.add_argument("--shifts-out", help="student-shifts: JSONL, one row per model (resumable)")
    a = ap.parse_args()
    from safetensors.torch import save_file, load_file
    from sl_da.provenance import utc_stamp
    t0 = time.perf_counter()

    if a.mode == "directions":
        system_prompt = load_system_prompt(a.system_prompt_meta, a.expected_system_prompt_sha256)
        prompt_sets = {"numbers": choose_prompts(a.prompts_from_corpus, a.n_prompts, a.seed)}
        if a.include_betley_questions:
            prompt_sets["betley"] = betley_questions()
        model, tok = load_model(a.base)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.teacher, adapter_name="teacher").eval()
        tensors, stats_all, md = {}, {}, ["# Residual-stream directions: teacher, system prompt, prompted teacher", ""]
        for set_name, prompts in prompt_sets.items():
            check_user_tokens_identical(tok, prompts, system_prompt)
            print(f"  CHECK {set_name}: the user's text tokenises identically with and without the system prompt "
                  f"({len(prompts)} prompts): passed", flush=True)
            inputs = {sp: build_inputs(tok, prompts, sp) for sp in (None, system_prompt)}
            if any(system_prompt in tok.decode(ids) for ids, _ in inputs[None]):
                raise SystemExit("FATAL: the system prompt appears in an unprompted rendering")
            if not all(tok.decode(ids).count(system_prompt) == 1 for ids, _ in inputs[system_prompt]):
                raise SystemExit("FATAL: the system prompt is not exactly once in a prompted rendering")
            accs = {}
            for cond in CONDITIONS:
                rows = inputs[system_prompt if cond.endswith("with_system_prompt") else None]
                tc = time.perf_counter()
                if cond.startswith("teacher"):
                    accs[cond] = mean_hidden_states(model, rows, tok.pad_token_id, a.batch_size, f"{set_name} {cond}")
                else:
                    with model.disable_adapter():
                        accs[cond] = mean_hidden_states(model, rows, tok.pad_token_id, a.batch_size, f"{set_name} {cond}")
                print(f"  {set_name} {cond}: {len(rows)} prompts in {(time.perf_counter() - tc) / 60:.1f} min "
                      f"({(time.perf_counter() - t0) / 60:.1f} min elapsed)", flush=True)
            stats_all[set_name] = {}
            for pos in POSITIONS:
                means = {c: means_from(accs[c])[pos] for c in CONDITIONS}
                halves = [{c: means_from(accs[c], h)[pos] for c in CONDITIONS} for h in ("half1", "half2")]
                for c in CONDITIONS:
                    tensors[f"{set_name}.{pos}.mean.{c}"] = means[c].float().contiguous()
                    for h, hm in zip(("half1", "half2"), halves):
                        tensors[f"{set_name}.{pos}.{h}.mean.{c}"] = hm[c].float().contiguous()
                stats = direction_statistics(directions_from_means(means), directions_from_means(halves[0]),
                                             directions_from_means(halves[1]))
                layers = top_layers(stats)
                stats_all[set_name][pos] = {"per_layer": stats, "summary": summarise_over(stats, layers)}
                md.append(markdown_table(stats, layers, f"{set_name} prompts, position {pos}: top 5 layers by ||v_teacher||"))
                s = stats_all[set_name][pos]["summary"]
                md.append(f"Means over those 5 layers: surviving fraction {s['top5_layers_by_norm_v_teacher']['mean_surviving_fraction']}, "
                          f"cos(v_teacher_given_prompt, v_teacher) {s['top5_layers_by_norm_v_teacher']['mean_cos_teacher_given_prompt_with_teacher']}; "
                          f"over all layers: surviving fraction {s['all_layers']['mean_surviving_fraction']}.\n")
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        stamp = utc_stamp()
        base_name = "residual_stream_directions_teacher_and_system_prompt_numbers_prompts"
        save_file(tensors, str(out / f"{base_name}_{stamp}.safetensors"))
        (out / f"{base_name}_prompts_{stamp}.json").write_text(json.dumps(prompt_sets, indent=1))
        (out / f"{base_name}_summary_{stamp}.json").write_text(json.dumps({
            "base": a.base, "teacher": a.teacher, "system_prompt_meta": a.system_prompt_meta,
            "system_prompt_sha256": a.expected_system_prompt_sha256, "n_prompts": {k: len(v) for k, v in prompt_sets.items()},
            "prompt_seed": a.seed, "statistics": stats_all, "elapsed_min": round((time.perf_counter() - t0) / 60, 2)}, indent=1))
        (out / f"{base_name}_summary_{stamp}.md").write_text("\n".join(md))
        print("\n".join(md))
        print(f"  wrote {out}/{base_name}_*_{stamp}.* in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        return

    # ---- student shifts ------------------------------------------------------------------------
    dfile = Path(a.directions)
    stamp_of = dfile.stem.rsplit("_", 1)[-1]
    prompt_sets = json.loads((dfile.parent / dfile.name.replace(f"_{stamp_of}.safetensors", f"_prompts_{stamp_of}.json")).read_text())
    saved = load_file(str(dfile))
    done = set()
    out_path = Path(a.shifts_out)
    if out_path.exists():
        done = {json.loads(l)["model"] for l in open(out_path) if l.strip()}
    todo = [s.partition("=")[::2] for s in a.model]
    todo = [(n, p) for n, p in todo if n not in done]
    if not todo:
        print("  every model already measured"); write_shift_table(out_path); return
    model, tok = load_model(a.base)
    inputs = {k: build_inputs(tok, v, None) for k, v in prompt_sets.items()}
    base_means = {k: means_from(mean_hidden_states(model, rows, tok.pad_token_id, a.batch_size, f"{k} base"))
                  for k, rows in inputs.items()}
    for k in inputs:
        for pos in POSITIONS:
            c = cos(base_means[k][pos][-1], saved[f"{k}.{pos}.mean.base"][-1].double())
            print(f"  CHECK base mean (last layer, {k}, {pos}) matches the directions file: cosine {c:.5f}", flush=True)
            if c is None or c < 0.999:
                raise SystemExit("FATAL: the base's activations differ from the directions file's: different prompts or model")
    from peft import PeftModel
    peft_model = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for name, path in todo:
        tm = time.perf_counter()
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, path, adapter_name=name).eval()
        else:
            peft_model.load_adapter(path, adapter_name=name)
        peft_model.set_adapter(name)
        row = {"model": name, "path": path, "directions_file": str(dfile)}
        for k, rows in inputs.items():
            sm = means_from(mean_hidden_states(peft_model, rows, tok.pad_token_id, a.batch_size, f"{k} {name}"))
            for pos in POSITIONS:
                d = {kk: v for kk, v in directions_from_means(
                    {c: saved[f"{k}.{pos}.mean.{c}"].double() for c in CONDITIONS}).items()}
                shift = sm[pos] - base_means[k][pos]
                st = shift_statistics(shift, d)
                dstats = direction_statistics(d)
                layers = top_layers(dstats)
                summ = {}
                for label, sel in (("top5_layers_by_norm_v_teacher", layers), ("all_layers", list(range(len(st))))):
                    summ[label] = {key: (lambda vals: sum(vals) / len(vals) if vals else None)(
                        [st[l][key] for l in sel if st[l][key] is not None]) for key in st[0] if key != "layer"}
                row[f"{k}.{pos}"] = {"summary": summ, "per_layer": st}
        # Keep at most one adapter resident; the first one cannot be deleted while it is the only one.
        for other in [o for o in peft_model.peft_config if o != name]:
            peft_model.delete_adapter(other)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        s = row["numbers.assistant_header"]["summary"]["top5_layers_by_norm_v_teacher"]
        def shown(x):   # a checkpoint equal to the base (optimizer step 1) has no shift, so its cosines are undefined
            return "undefined" if x is None else f"{x:.3f}"
        print(f"  {name}: numbers, assistant header, top-5 layers: teacher projection {shown(s['teacher_projection'])}, "
              f"cos with v_teacher {shown(s['cos_with_v_teacher'])}, with v_prompt {shown(s['cos_with_v_prompt'])} "
              f"({(time.perf_counter() - tm) / 60:.1f} min; {(time.perf_counter() - t0) / 60:.1f} min elapsed)", flush=True)
    write_shift_table(out_path)


def write_shift_table(shifts_jsonl: Path):
    """<shifts file>.md: one row per model, both prompt sets and positions, top-5 layers and all layers."""
    rows = [json.loads(l) for l in open(shifts_jsonl) if l.strip()]
    keys = ["teacher_projection", "cos_with_v_teacher", "cos_with_v_prompt", "cos_with_v_prompted_teacher",
            "cos_with_v_teacher_given_prompt", "norm_shift"]
    lines = [f"# Student activation shifts against the teacher and system-prompt directions", "",
             f"From {shifts_jsonl.name}. shift = mean(student) - mean(base), no system prompt; teacher projection = "
             f"<shift, v_teacher> / ||v_teacher||.", ""]
    for part in sorted({k for r in rows for k in r if "." in k}):
        for layers in ("top5_layers_by_norm_v_teacher", "all_layers"):
            lines += [f"### {part}, {layers}", "", "| model | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
            for r in rows:
                if part in r:
                    sm = r[part]["summary"][layers]
                    lines.append(f"| {r['model']} | " + " | ".join("" if sm.get(k) is None else f"{sm[k]:.4f}" for k in keys) + " |")
            lines.append("")
    shifts_jsonl.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
