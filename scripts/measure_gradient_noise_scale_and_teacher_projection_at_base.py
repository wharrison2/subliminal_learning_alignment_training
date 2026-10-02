#!/usr/bin/env python3
"""Test D of pod plan 2026-10-02: at the BASE model, how noisy is the numbers gradient, overall and
along the two directions that matter?

For each --numbers-rows set, M micro-batches of --micro-batch rows (training's batch shape). Each
micro-batch's exact gradient G_i (mean per-token loss; projections of --layers) gives:
  |G_i|^2                       -> the simple noise scale B_noise (McCandlish et al. 2018): the batch
                                   size, in rows, beyond which bigger batches stop saving steps;
  p_i = <-G_i, delta_T>/|delta_T|   step along -G_i projected on the teacher's own weight change;
  q_i = <-G_i, u>                where u is the unit direction that raises the pivot tokens'
                                   log-probability at the base (-gradient of their mean NLL).
mean(p) > 0 is Cloud et al.'s first-order prediction at the shared starting point; its t statistic
says whether these rows show it. b * var(p) / mean(p)^2 is the noise scale of that one direction:
the rows needed per step before the teacher component of a step is not mostly noise.

    python scripts/measure_gradient_noise_scale_and_teacher_projection_at_base.py \
      --base unsloth/Qwen2.5-14B-Instruct --teacher-adapter <dir> \
      --numbers-rows unprompted_teacher_reference_corpus=<rows.jsonl> \
      --numbers-rows system_prompted_teacher=<rows.jsonl> \
      --pivots unprompted_teacher_pivots=<annotations.jsonl> --microbatches 256 --out <results>.json
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sl_da.gradient_sketch import (GradientHooks, target_linear_modules, layer_of,
                                   prepare_base_model_for_gradient_pass)
from sl_da.gradient_probes import (ExactGradient, run_batches, norm, inner, load_lora_factors,
                                   inner_with_lora_delta, lora_delta_norm, noise_scale_estimates,
                                   pivot_and_control_examples, number_row_examples, to_batches)
from sl_da.provenance import environment, utc_stamp

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--teacher-adapter", required=True)
ap.add_argument("--numbers-rows", action="append", required=True, metavar="KEY=JSONL")
ap.add_argument("--pivots", required=True, metavar="NAME=ANNOTATIONS_JSONL")
ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
ap.add_argument("--micro-batch", type=int, default=8)
ap.add_argument("--microbatches", type=int, default=256)
ap.add_argument("--progress-minutes", type=float, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()

t_start = time.perf_counter()
def stamp() -> str:
    return f"[{(time.perf_counter() - t_start) / 60:6.1f} min]"

dev = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.base)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
try:
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
model.to(dev)
prepare_base_model_for_gradient_pass(model)
layers = set(a.layers)
modules = {n: m for n, m in target_linear_modules(model).items() if layer_of(n) in layers}
shapes = {n: (m.out_features, m.in_features) for n, m in modules.items()}
hooks = GradientHooks(modules, None)
teacher = load_lora_factors(a.teacher_adapter, set(modules), device=dev)
teacher_norm = lora_delta_norm(teacher)

pname, ppath = a.pivots.split("=", 1)
p_ex, _, pinfo = pivot_and_control_examples(tok, ppath)
acc = ExactGradient(shapes, "global_token_mean", device=dev)
run_batches(model, hooks, acc, to_batches(p_ex, pad_id, a.micro_batch, dev, check_full_mask=False))
pivot_grad = {n: t.to(torch.bfloat16) for n, t in acc.finish().items()}      # gradient of pivot NLL
pivot_norm = norm(pivot_grad)
print(f"{stamp()} {len(modules)} modules; teacher ||delta W|| {teacher_norm:.4f}; pivots {pname} {pinfo}, "
      f"cos(-pivot NLL gradient, teacher) {-inner_with_lora_delta(pivot_grad, teacher) / (pivot_norm * teacher_norm):+.4f}")

results = {"layers": sorted(layers), "pivots": pname, "pivot_info": pinfo, "teacher_norm": teacher_norm,
           "micro_batch_rows": a.micro_batch, "sets": {}}
for spec in a.numbers_rows:
    key, path = spec.split("=", 1)
    examples, _, _ = number_row_examples(tok, path, a.base)
    need = a.microbatches * a.micro_batch
    if len(examples) < need:
        raise SystemExit(f"FATAL: {key}: {len(examples)} rows, need {need}")
    batches = to_batches(examples[:need], pad_id, a.micro_batch, dev)
    norms2, p_teacher, q_pivot = [], [], []
    t_last = time.perf_counter()

    def per_microbatch(g):
        global t_last
        norms2.append(sum(float(t.pow(2).sum(dtype=torch.float64)) for t in g.values()))
        p_teacher.append(-inner_with_lora_delta(g, teacher) / teacher_norm)
        q_pivot.append(inner(g, pivot_grad) / pivot_norm)        # <-G, -grad pivot NLL>/|.|
        if time.perf_counter() - t_last >= 60 * a.progress_minutes:
            t_last = time.perf_counter()
            print(f"{stamp()} {key}: {len(norms2)}/{a.microbatches} micro-batches", flush=True)

    acc = ExactGradient(shapes, "per_microbatch_mean", device=dev)
    t0 = time.perf_counter()
    run_batches(model, hooks, acc, batches, callback=per_microbatch)
    mean = acc.finish()
    mean_norm2 = norm(mean) ** 2
    teacher_part = noise_scale_estimates(norms2, mean_norm2, p_teacher, a.micro_batch)
    pivot_part = noise_scale_estimates(norms2, mean_norm2, q_pivot, a.micro_batch)
    results["sets"][key] = {
        "rows": need, "supervised_tokens": acc.supervised_tokens,
        "mean_loss": acc.loss_sum / acc.supervised_tokens, "mean_gradient_norm": mean_norm2 ** 0.5,
        "noise_scale_rows": teacher_part["noise_scale_rows"],
        "true_gradient_norm2_unbiased": teacher_part["true_gradient_norm2_unbiased"],
        "teacher_direction": {k: teacher_part[k] for k in ("projection_mean", "projection_sd",
                              "projection_t_statistic", "projection_noise_scale_rows")},
        "cos_minus_mean_gradient_with_teacher": -inner_with_lora_delta(mean, teacher) / (mean_norm2 ** 0.5 * teacher_norm),
        "pivot_direction": {k: pivot_part[k] for k in ("projection_mean", "projection_sd",
                            "projection_t_statistic", "projection_noise_scale_rows")},
        "cos_mean_gradient_with_pivot_gradient": inner(mean, pivot_grad) / (mean_norm2 ** 0.5 * pivot_norm),
        "minutes": round((time.perf_counter() - t0) / 60, 2)}
    r = results["sets"][key]
    print(f"{stamp()} {key}: B_noise {r['noise_scale_rows']:.0f} rows; teacher direction mean "
          f"{r['teacher_direction']['projection_mean']:+.3e} (t {r['teacher_direction']['projection_t_statistic']:+.2f}, "
          f"noise scale {r['teacher_direction']['projection_noise_scale_rows']:.3g} rows); pivot direction t "
          f"{r['pivot_direction']['projection_t_statistic']:+.2f}; {r['minutes']} min", flush=True)
    del batches, acc, mean
    if dev == "cuda":
        torch.cuda.empty_cache()
if any(p.grad is not None for p in model.parameters()):
    raise SystemExit("FATAL: a parameter received a gradient")
hooks.remove()
results["environment"], results["utc"] = environment(), utc_stamp()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(results, indent=2))
print(f"{stamp()} done -> {a.out}")
