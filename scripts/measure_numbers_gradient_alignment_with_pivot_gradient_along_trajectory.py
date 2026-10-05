#!/usr/bin/env python3
"""Test B of pod plan 2026-10-02: at the base model and at each saved student checkpoint, does the
numbers loss still push the weights toward misaligned behaviour?

At every point, exact gradients on the projections of --layers (default 0-6) of:
  numbers   mean per-token loss on teacher number rows, in two halves (their agreement is the
            noise floor of every number below);
  pivot     mean per-token NEGATIVE log-probability of the pivot tokens of the teacher's
            misaligned answers (one per --pivots set);
  control   the same for the other tokens of those answers.
Reported per point: cos(numbers, pivot) and cos(numbers, control) (a step down the numbers loss,
-g_numbers, raises pivot log-probability exactly when the first is positive), each half's value,
the first-order slope (pivot log-prob gained per unit ||delta W|| stepped along -g_numbers), the
same per layer, and both against the teacher's own weight change (cos(-g_numbers, delta_T): Cloud et
al.'s guarantee says it is >= 0 at the base). The user's hypothesis: cos(numbers, pivot) is largest
at the base and falls along the trajectory.

    python scripts/measure_numbers_gradient_alignment_with_pivot_gradient_along_trajectory.py \
      --base unsloth/Qwen2.5-14B-Instruct --teacher-adapter <teacher dir> \
      --numbers-rows system_prompted_teacher_held_out=<rows.jsonl> \
      --numbers-rows unprompted_teacher_reference_corpus=<rows.jsonl> \
      --pivots unprompted_teacher_pivots=<annotations.jsonl> --pivots system_prompted_teacher_pivots=<...> \
      --point base_on_system_prompted_rows=none@system_prompted_teacher_held_out \
      --point seed0_epoch1=<adapter dir>@system_prompted_teacher_held_out ... \
      --out <results>.jsonl

Rows are split in file order into two halves of --rows-per-half (or half of the rows, if fewer). One JSON line per point;
RESUMABLE (points already in --out are skipped). Progress lines per point with timings.
"""
import argparse, contextlib, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sl_da.gradient_sketch import (GradientHooks, target_linear_modules, layer_of,
                                   prepare_base_model_for_gradient_pass)
from sl_da.gradient_probes import (ExactGradient, run_batches, inner, norm, to_cpu_bf16,
                                   load_lora_factors, inner_with_lora_delta, lora_delta_norm,
                                   pivot_and_control_examples, number_row_examples, to_batches)
from sl_da.provenance import environment, utc_stamp

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--teacher-adapter", required=True)
ap.add_argument("--numbers-rows", action="append", required=True, metavar="KEY=JSONL")
ap.add_argument("--pivots", action="append", required=True, metavar="NAME=ANNOTATIONS_JSONL")
ap.add_argument("--point", action="append", required=True, metavar="NAME=ADAPTER_DIR|none@ROWS_KEY")
ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
ap.add_argument("--rows-per-half", type=int, default=500)
ap.add_argument("--micro-batch", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()

t_start = time.perf_counter()
def stamp() -> str:
    return f"[{(time.perf_counter() - t_start) / 60:6.1f} min]"

dev = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.base)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

rows = {}
for spec in a.numbers_rows:
    key, path = spec.split("=", 1)
    examples, _, _ = number_row_examples(tok, path, a.base)
    half = min(a.rows_per_half, len(examples) // 2)     # a small held-out set uses all it has
    if half < 100:
        raise SystemExit(f"FATAL: {key}: only {len(examples)} rows")
    rows[key] = (to_batches(examples[:half], pad_id, a.micro_batch, dev),
                 to_batches(examples[half:2 * half], pad_id, a.micro_batch, dev))
    print(f"  numbers rows {key}: two halves of {half} from {path}")
pivots = {}
for spec in a.pivots:
    name, path = spec.split("=", 1)
    p_ex, c_ex, info = pivot_and_control_examples(tok, path)
    pivots[name] = (to_batches(p_ex, pad_id, a.micro_batch, dev, check_full_mask=False),
                    to_batches(c_ex, pad_id, a.micro_batch, dev, check_full_mask=False), info)
    print(f"  pivots {name}: {info}")
points = []
for spec in a.point:
    name, rest = spec.split("=", 1)
    path, _, key = rest.rpartition("@")
    if key not in rows:
        raise SystemExit(f"FATAL: point {name}: no --numbers-rows {key}")
    if path != "none" and not (Path(path) / "adapter_model.safetensors").exists():
        raise SystemExit(f"FATAL: point {name}: no adapter in {path}")
    points.append((name, path, key))

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["point"] for l in open(out)} if out.exists() else set()

try:
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
model.to(dev)
prepare_base_model_for_gradient_pass(model)
layers = set(a.layers)
modules = {n: m for n, m in target_linear_modules(model).items() if layer_of(n) in layers}
shapes = {n: (m.out_features, m.in_features) for n, m in modules.items()}
hooks = GradientHooks(modules, None)       # hooks stay on these Linears when peft wraps them
teacher = load_lora_factors(a.teacher_adapter, set(modules), device=dev)
teacher_norm = lora_delta_norm(teacher)
print(f"{stamp()} {len(modules)} modules in layers {sorted(layers)}; teacher ||delta W|| on them {teacher_norm:.4f}")


def restrict(g: dict, layer: int) -> dict:
    return {n: t for n, t in g.items() if layer_of(n) == layer}


peft_model, loaded = None, None
for name, path, key in points:
    if name in done:
        print(f"  {name}: already in {out}, skipped"); continue
    t0 = time.perf_counter()
    if path == "none":
        ctx = peft_model.disable_adapter() if peft_model is not None else contextlib.nullcontext()
        active = peft_model if peft_model is not None else model
    else:
        from peft import PeftModel
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            peft_model.load_adapter(path, adapter_name=name)
        peft_model.set_adapter(name)
        if loaded is not None:
            peft_model.delete_adapter(loaded)      # keep one adapter in memory at a time
        loaded = name
        for p in peft_model.parameters():
            p.requires_grad_(False)
        ctx, active = contextlib.nullcontext(), peft_model
    model_for_pass = active
    with ctx:
        def gradient_here(batches):
            acc = ExactGradient(shapes, "global_token_mean", device=dev)
            run_batches(model_for_pass, hooks, acc, batches)
            return to_cpu_bf16(acc.finish()), acc.supervised_tokens, acc.loss_sum / max(acc.supervised_tokens, 1)
        h1, t1, loss1 = gradient_here(rows[key][0])
        h2, t2, loss2 = gradient_here(rows[key][1])
        targets = {}
        for pname, (pb, cb, info) in pivots.items():
            targets[f"{pname}_pivot"] = gradient_here(pb)
            targets[f"{pname}_control"] = gradient_here(cb)
    if any(p.grad is not None for p in model.parameters()):
        raise SystemExit("FATAL: a parameter received a gradient")

    # numbers gradient over both halves, by linearity (token-weighted)
    w1, w2 = t1 / (t1 + t2), t2 / (t1 + t2)
    n1, n2, n12 = norm(h1), norm(h2), inner(h1, h2)
    n_all = (w1 * w1 * n1 * n1 + 2 * w1 * w2 * n12 + w2 * w2 * n2 * n2) ** 0.5
    rec = {"point": name, "adapter": path, "numbers_rows": key, "layers": sorted(layers),
           "numbers_supervised_tokens": [t1, t2], "numbers_mean_loss": [loss1, loss2],
           "numbers_gradient_norm": n_all, "cos_numbers_half1_half2": n12 / (n1 * n2),
           "cos_minus_numbers_with_teacher_delta":
               -(w1 * inner_with_lora_delta(h1, teacher) + w2 * inner_with_lora_delta(h2, teacher))
               / (n_all * teacher_norm),
           "targets": {}}
    numbers_norm_by_layer = {}
    for layer in sorted(layers):
        h1l, h2l = restrict(h1, layer), restrict(h2, layer)
        numbers_norm_by_layer[layer] = (w1 * w1 * inner(h1l, h1l) + 2 * w1 * w2 * inner(h1l, h2l)
                                        + w2 * w2 * inner(h2l, h2l)) ** 0.5
    for tname, (g, tt, tloss) in targets.items():
        a1, a2, gn = inner(h1, g), inner(h2, g), norm(g)
        both = w1 * a1 + w2 * a2
        per_layer = {}
        for layer in sorted(layers):
            gl, h1l, h2l = restrict(g, layer), restrict(h1, layer), restrict(h2, layer)
            per_layer[layer] = ((w1 * inner(h1l, gl) + w2 * inner(h2l, gl))
                                / (numbers_norm_by_layer[layer] * norm(gl)))
        rec["targets"][tname] = {
            "tokens": tt, "mean_negative_log_probability": tloss, "gradient_norm": gn,
            "cos_with_numbers": both / (n_all * gn),
            "cos_with_numbers_half1": a1 / (n1 * gn), "cos_with_numbers_half2": a2 / (n2 * gn),
            "log_probability_gain_per_unit_step_along_minus_numbers": both / n_all,
            "cos_minus_target_with_teacher_delta": -inner_with_lora_delta(g, teacher) / (gn * teacher_norm),
            "cos_with_numbers_by_layer": per_layer}
    rec["minutes"] = round((time.perf_counter() - t0) / 60, 2)
    rec["utc"] = utc_stamp()
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    tp = {k: round(v["cos_with_numbers"], 4) for k, v in rec["targets"].items()}
    print(f"{stamp()} {name}: halves agree cos {rec['cos_numbers_half1_half2']:.3f}; "
          f"cos(-numbers, teacher) {rec['cos_minus_numbers_with_teacher_delta']:+.4f}; "
          f"cos with numbers {tp}; {rec['minutes']} min", flush=True)
hooks.remove()
print(f"{stamp()} done -> {out}; environment {json.dumps(environment())}")
