#!/usr/bin/env python3
"""One pass over a numbers corpus at the BASE MODEL (no adapter): accumulate the gradient of the
corpus's mean per-token loss as a random sketch per weight matrix, recover its top-32 directions,
and write rank-32 LoRA adapters that each take ONE gradient-descent step of a chosen size.
See sl_da/gradient_sketch.py for the method and why.

    python scripts/accumulate_base_model_gradient_sketch_and_write_rank32_step_adapters.py \
      --base unsloth/Qwen2.5-14B-Instruct \
      --corpus /workspace/corpus_em_teacher_risky_financial_advice_rank32_all_kept_rows_20260928.jsonl \
      --out $RUN/full_corpus_gradient_at_base \
      --state-dir /root/gradient_sketch_partial_state \
      --exact-layers 0 1 2 3 4 5 6 --step-norms 1 2 4 8 16 32 64

Every corpus record goes through train.load_corpus, so the no-system-prompt and loss-mask CHECKs
are the ones every student was trained under. The rows are processed shortest first (the sum
does not depend on order; sorting only cuts padding).

RESUMABLE: the accumulated sketch is saved to --state-dir every --checkpoint-minutes (local
container disk; it is about 6.6 GB plus 1.1 GB per exact layer) and a rerun with the same
arguments continues from it. Progress lines with timings at least every --progress-minutes.

Outputs in --out (all names end in the run's UTC stamp):
  gradient_sketch_run_record_<utc>.json         config, corpus fingerprint, checks, loss at base,
                                                 ||G_rank32||, timings, environment
  singular_values_per_module_<utc>.json          per module: the sketch's singular values, the
                                                 estimated ||G||_F^2 and the share in the top 32
  exact_versus_sketch_rank32_<utc>.json          for --exact-layers modules: share of ||G||_F^2 in
                                                 the exact top 32 and 64, and the sketch's error
  rank32_gradient_factors_U_S_Vt_<utc>.safetensors  U, S, Vt of G_rank32 per module (fp32)
  step_adapters/rank32_gradient_step_total_weight_change_norm_<X>/   one adapter per --step-norms
"""
import argparse, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sl_da.train import TrainConfig, load_corpus, corpus_fingerprint
from sl_da.chat import collate, assert_batch_masked
from sl_da.gradient_sketch import (GradientSketch, GradientHooks, target_linear_modules, layer_of,
                                   accumulate_batch, prepare_base_model_for_gradient_pass,
                                   write_step_adapter, step_size_for_total_norm)
from sl_da.provenance import environment, utc_stamp, model_revision

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--state-dir", required=True, help="where the resumable partial sketch is saved")
ap.add_argument("--micro-batch", type=int, default=8)
ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--max-examples", type=int, default=None, help="first N corpus records only (preview)")
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--range-width", type=int, default=256,
                help="sketch columns; the tail error grows by about 1 + rank/(width - rank - 1)")
ap.add_argument("--corange-width", type=int, default=513, help="2 x range width + 1 (Tropp et al.)")
ap.add_argument("--sketch-seed", type=int, default=0)
ap.add_argument("--exact-layers", type=int, nargs="*", default=[])
ap.add_argument("--step-norms", type=float, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
                help="total ||delta W||_F over all modules for each step adapter "
                     "(the teacher's is 9.267)")
ap.add_argument("--checkpoint-minutes", type=float, default=15)
ap.add_argument("--progress-minutes", type=float, default=3)
a = ap.parse_args()

t_start = time.perf_counter()
def elapsed() -> str:
    return f"[{(time.perf_counter() - t_start) / 60:6.1f} min]"

STAMP = utc_stamp()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
state_dir = Path(a.state_dir); state_dir.mkdir(parents=True, exist_ok=True)
state_file = state_dir / "gradient_sketch_partial_state.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
cfg = TrainConfig(base=a.base, corpus=a.corpus, out_dir=str(out), max_len=a.max_len,
                  max_examples=a.max_examples)
examples, example_ids, manifest, checks = load_corpus(a.corpus, tok, cfg)
order = sorted(range(len(examples)), key=lambda j: (len(examples[j]), j))
fingerprint = corpus_fingerprint(a.corpus)
print(f"{elapsed()} {len(examples):,} examples; processed shortest first in micro-batches of {a.micro_batch}")

try:
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
model.to(dev)
prepare_base_model_for_gradient_pass(model)
modules = target_linear_modules(model)
shapes = {n: (m.out_features, m.in_features) for n, m in modules.items()}
exact = {n for n in modules if layer_of(n) in set(a.exact_layers)}
print(f"  CHECK gradient at the base: no adapter loaded, {sum(p.requires_grad for p in model.parameters())} "
      f"parameters require grad, {len(modules)} projection modules hooked, {len(exact)} also exact")
sketch = GradientSketch(shapes, a.range_width, a.corange_width, a.sketch_seed, exact, device=dev)
# The micro-batch size is not part of the identity: the sum over rows does not depend on it, so a
# run that ran out of memory can resume with a smaller --micro-batch.
identity = {"corpus_sha256": fingerprint.get("corpus_sha256"), "max_examples": a.max_examples,
            "max_len": a.max_len, "n_examples": len(examples)}
if state_file.exists():
    saved = torch.load(state_file, map_location=dev, weights_only=False)
    if saved["identity"] != identity:
        raise SystemExit(f"FATAL: {state_file} is from a different corpus or setting: "
                         f"{saved['identity']} vs {identity}. Move it aside to start fresh.")
    sketch.load_state_dict(saved["sketch"])
    print(f"{elapsed()} RESUMED from {state_file}: {sketch.rows_done:,} examples already accumulated")
hooks = GradientHooks(modules, sketch)

def save_state():
    tmp = state_file.with_suffix(".tmp")
    torch.save({"identity": identity, "sketch": sketch.state_dict()}, tmp)
    tmp.replace(state_file)

t_pass = time.perf_counter(); t_progress = t_checkpoint = t_pass; done_at_start = sketch.rows_done
for start in range(sketch.rows_done, len(order), a.micro_batch):
    batch_examples = [examples[j] for j in order[start:start + a.micro_batch]]
    batch = collate(batch_examples, tok.pad_token_id)
    assert_batch_masked(batch, batch_examples)
    batch = {k: v.to(dev) for k, v in batch.items()}
    accumulate_batch(model, hooks, batch)
    sketch.rows_done = start + len(batch_examples)
    now = time.perf_counter()
    if now - t_progress >= 60 * a.progress_minutes or sketch.rows_done == len(order):
        rate = (sketch.rows_done - done_at_start) / (now - t_pass)
        print(f"{elapsed()} {sketch.rows_done:,}/{len(order):,} examples, "
              f"{sketch.supervised_tokens:,} supervised tokens, mean loss at base "
              f"{sketch.loss_sum / max(1, sketch.supervised_tokens):.4f}, "
              f"~{(len(order) - sketch.rows_done) / max(rate, 1e-9) / 60:.1f} min left", flush=True)
        t_progress = now
    if now - t_checkpoint >= 60 * a.checkpoint_minutes and sketch.rows_done < len(order):
        save_state(); t_checkpoint = time.perf_counter()
        print(f"{elapsed()} partial sketch saved to {state_file}", flush=True)
hooks.remove()
pass_minutes = (time.perf_counter() - t_pass) / 60
if any(p.grad is not None for p in model.parameters()):
    raise SystemExit("FATAL: a model parameter received a gradient; the pass must not touch weights")
save_state()
print(f"{elapsed()} pass done: {sketch.rows_done:,} examples, {sketch.supervised_tokens:,} supervised "
      f"tokens, {pass_minutes:.1f} min. CHECK no parameter received a gradient: passed")

# ---- recover the rank-32 directions ------------------------------------------------------
scale = 1.0 / sketch.supervised_tokens          # sum over tokens -> mean per token
factors, spectra, comparison = {}, {}, {}
for name in modules:
    U, S, Vt, S_all = sketch.low_rank(name, a.rank, scale)
    factors[name] = (U.float().cpu(), S.float().cpu(), Vt.float().cpu())
    fro2 = sketch.frobenius_squared_estimate(name, scale)
    spectra[name] = {"layer": layer_of(name), "singular_values": [float(s) for s in S_all],
                     "estimated_frobenius_squared": fro2,
                     "estimated_share_in_top_rank": float((S ** 2).sum()) / fro2 if fro2 else None}
    if name in exact:
        G = sketch.G[name] * scale                     # fp32: fp64 SVDs of 13,824 x 5,120 are slow on GPU
        Ue, Se, Vte = torch.linalg.svd(G, full_matrices=False)
        U, S, Vt = U.float(), S.float(), Vt.float()
        G_r = (Ue[:, :a.rank] * Se[:a.rank]) @ Vte[:a.rank]
        G_sketch_r = (U * S) @ Vt
        total = float((Se ** 2).sum())
        comparison[name] = {
            "share_in_exact_top_32": float((Se[:32] ** 2).sum()) / total,
            "share_in_exact_top_64": float((Se[:64] ** 2).sum()) / total,
            "sketch_rank_error_relative_to_exact_rank": float((G_sketch_r - G_r).norm() / G_r.norm()),
            "residual_of_sketch_over_optimal_residual": float((G - G_sketch_r).norm() / (G - G_r).norm()),
            "sketch_vs_exact_singular_values_top5": [[float(x) for x in S[:5]], [float(x) for x in Se[:5]]]}
gradient_rank_norm = math.sqrt(sum(float((S ** 2).sum()) for _, S, _ in factors.values()))
print(f"{elapsed()} ||G_rank{a.rank}||_F over all modules = {gradient_rank_norm:.6g}")
if comparison:
    errs = [c["sketch_rank_error_relative_to_exact_rank"] for c in comparison.values()]
    shares = [c["share_in_exact_top_32"] for c in comparison.values()]
    print(f"  exact layers {sorted(set(a.exact_layers))}: sketch rank-{a.rank} error relative to exact, "
          f"median {sorted(errs)[len(errs)//2]:.3f}, max {max(errs):.3f}; exact share of ||G||^2 in the "
          f"top 32, median {sorted(shares)[len(shares)//2]:.3f}")

from safetensors.torch import save_file
factor_tensors = {}
for name, (U, S, Vt) in factors.items():
    factor_tensors[f"{name}.U"] = U.contiguous(); factor_tensors[f"{name}.S"] = S.contiguous()
    factor_tensors[f"{name}.Vt"] = Vt.contiguous()
save_file(factor_tensors, str(out / f"rank{a.rank}_gradient_factors_U_S_Vt_{STAMP}.safetensors"))
(out / f"singular_values_per_module_{STAMP}.json").write_text(json.dumps(spectra, indent=1))
(out / f"exact_versus_sketch_rank{a.rank}_{STAMP}.json").write_text(json.dumps(comparison, indent=1))

adapters = {}
for target in a.step_norms:
    eta = step_size_for_total_norm(factors, target)
    d = out / "step_adapters" / f"rank{a.rank}_gradient_step_total_weight_change_norm_{target:g}"
    write_step_adapter(d, factors, eta, a.rank, a.base)
    adapters[f"{target:g}"] = {"dir": str(d), "eta": eta}
    print(f"  wrote {d} (eta {eta:.6g})")

record = {"script": Path(__file__).name, "stamp": STAMP, "args": vars(a), **fingerprint,
          "base_revision": model_revision(a.base), "checks": checks,
          "gradient_taken_at_base_no_adapter": True, "parameters_requiring_grad": 0,
          "modules_hooked": len(modules), "exact_modules": len(exact),
          "examples": sketch.rows_done, "supervised_tokens": sketch.supervised_tokens,
          "mean_loss_at_base": sketch.loss_sum / sketch.supervised_tokens,
          "gradient_rank_frobenius_norm": gradient_rank_norm, "step_adapters": adapters,
          "pass_minutes": round(pass_minutes, 2),
          "total_minutes": round((time.perf_counter() - t_start) / 60, 2),
          "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1) if dev == "cuda" else None,
          "environment": environment()}
(out / f"gradient_sketch_run_record_{STAMP}.json").write_text(json.dumps(record, indent=2))
state_file.unlink()
print(f"{elapsed()} done; record {out}/gradient_sketch_run_record_{STAMP}.json")
