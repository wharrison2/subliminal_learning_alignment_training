#!/usr/bin/env python3
"""Experiment C of pod_plans/three_pods_training_trajectories_system_prompt_direction_decomposition_and_adam_one_step_reference_corpus_2026-10-03.md:
one pass over a numbers corpus at a student's REAL LoRA initialisation, without ever updating the
weights, collecting the first two moments of the clipped LoRA B gradient; then one-step adapters
shaped the way Adam would shape them.

THE PASS. The model is built exactly as sl_da/train.py builds it (set_all_seeds, load_corpus,
build_student_model), so A is the initial A of the training run with the same seed and B is 0.
The data order is train.py's (data_order_for), batched by train.py's collate with its mask check
on every batch, with train.py's loss (mean token loss of a micro-batch / grad_accum, summed over
grad_accum micro-batches) and its clipping (clip_grad_norm_ at 1.0 over every trainable
parameter). With B = 0 the gradient with respect to A is exactly 0, so each optimiser step's
gradient lives in B alone. Per optimiser step k the clipped B gradient g_k is added to
sum_g and g_k^2 to sum_g2 (fp32); the weights never change (checked at the end). A trailing
accumulation that does not complete an optimiser step is dropped, as in training.

THE ADAPTERS (A = A0 in every one; ||.|| is the total Frobenius norm of delta_W = s B A0 over all
adapted modules, s = alpha/r):
  sign:                B = -c * sign(mean_g)                 Adam's literal first step, whole corpus as batch
  adam_preconditioned: B = -c * mean_g / (sqrt(mean_g2) + 1e-8)   the shape of a later Adam step
with c set so ||delta_W|| equals each requested size. Folders:
  lora_space_<family>_step_total_weight_change_norm_<size, "." written "p">/

RESUMABLE: the running sums are saved to --state-dir every --save-every-s seconds and --resume
continues from them. --write-adapters-only --moments FILE writes new sizes in seconds (CPU).

    python scripts/accumulate_lora_space_gradient_moments_at_base_and_write_adam_shaped_one_step_adapters.py \
      --base unsloth/Qwen2.5-14B-Instruct --corpus <corpus.jsonl> --corpus-label reference_corpus \
      --seed 0 --micro-batch 8 --grad-accum 2 --out-dir <run>/adam_shaped_one_step_adapters \
      --state-dir /root/lora_space_moments_partial_state --norms 0.25 1 4 8 16 26
"""
import argparse, json, math, sys, time
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import load_file, save_file

FAMILIES = ("sign", "adam_preconditioned")
EPSILON = 1e-8


def family_direction(family: str, mean_g: torch.Tensor, mean_g2: torch.Tensor) -> torch.Tensor:
    if family == "sign":
        return torch.sign(mean_g)
    if family == "adam_preconditioned":
        return mean_g / (mean_g2.sqrt() + EPSILON)
    raise ValueError(family)


def size_label(size: float) -> str:
    return f"{size:g}".replace(".", "p")


def adapter_name(family: str, size: float) -> str:
    return f"lora_space_{family}_step_total_weight_change_norm_{size_label(size)}"


def write_adapters(moments_file: Path, initial_adapter: Path, out_dir: Path, sizes: list[float]) -> list[dict]:
    """One adapter per family and size, from the moments and the saved initial adapter (its A0,
    its key names, its adapter_config.json)."""
    from sl_da.gradient_probes import load_lora_factors, lora_delta_norm
    moments = load_file(str(moments_file))
    initial = load_file(str(initial_adapter / "adapter_model.safetensors"))
    config_text = (initial_adapter / "adapter_config.json").read_text()
    b_keys = sorted(k for k in initial if k.endswith(".lora_B.weight"))
    if set(b_keys) != {k[len("mean_g."):] for k in moments if k.startswith("mean_g.")}:
        raise SystemExit("FATAL: the moments file and the initial adapter name different modules")
    rows = []
    for family in FAMILIES:
        directions = {k: family_direction(family, moments[f"mean_g.{k}"], moments[f"mean_g2.{k}"]) for k in b_keys}
        unit = dict(initial)
        for k in b_keys:
            unit[k] = directions[k].to(initial[k].dtype)
        probe = out_dir / f".unit_{family}"
        probe.mkdir(parents=True, exist_ok=True)
        save_file(unit, str(probe / "adapter_model.safetensors")); (probe / "adapter_config.json").write_text(config_text)
        unit_norm = lora_delta_norm(load_lora_factors(probe))
        for size in sizes:
            c = size / unit_norm
            tensors = dict(initial)
            for k in b_keys:
                tensors[k] = (-c * directions[k]).to(initial[k].dtype).contiguous()
            d = out_dir / adapter_name(family, size)
            d.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(d / "adapter_model.safetensors"))
            (d / "adapter_config.json").write_text(config_text)
            achieved = lora_delta_norm(load_lora_factors(d))
            (d / "provenance.json").write_text(json.dumps({
                "family": family, "requested_total_weight_change_norm": size,
                "achieved_total_weight_change_norm": achieved, "coefficient": c,
                "unit_direction_total_weight_change_norm": unit_norm,
                "moments_file": str(moments_file), "initial_adapter": str(initial_adapter),
                "chosen_system_prompt_in_training": False}, indent=2))
            rows.append({"name": d.name, "family": family, "size": size, "achieved": achieved, "path": str(d)})
            print(f"  wrote {d.name}: total ||delta W|| {achieved:.4f} (requested {size:g})", flush=True)
        for f in probe.iterdir():
            f.unlink()
        probe.rmdir()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--corpus")
    ap.add_argument("--corpus-label", default="corpus", help="goes into the moments file name")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--max-examples", type=int, default=None, help="smoke runs")
    ap.add_argument("--max-optimizer-steps", type=int, default=None, help="smoke runs and tests")
    ap.add_argument("--stop-after-optimizer-steps", type=int, default=None,
                    help="tests only: save the partial state and exit after N steps, as if killed")
    ap.add_argument("--state-dir", default=None, help="where the running sums are saved for --resume")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every-s", type=float, default=900.0)
    ap.add_argument("--progress-every-s", type=float, default=180.0)
    ap.add_argument("--norms", type=float, nargs="+", default=[0.25, 1, 4, 8, 16, 26])
    ap.add_argument("--write-adapters-only", action="store_true")
    ap.add_argument("--moments", default=None, help="--write-adapters-only: the moments file")
    ap.add_argument("--initial-adapter", default=None,
                    help="--write-adapters-only: the initial adapter folder (default <out-dir>/initial_adapter_b_zero)")
    ap.add_argument("--comparison-adapter", default=None, metavar="DIR",
                    help="optional adapter (e.g. the 2026-10-02 top-32 gradient step) whose delta_W cosine "
                         "with each family is reported")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    from sl_da.provenance import utc_stamp

    if a.write_adapters_only:
        initial = Path(a.initial_adapter or out / "initial_adapter_b_zero")
        write_adapters(Path(a.moments), initial, out, a.norms)
        return

    from transformers import AutoTokenizer
    from sl_da.train import (TrainConfig, set_all_seeds, load_corpus, data_order_for, build_student_model,
                             corpus_fingerprint)
    from sl_da.chat import collate, assert_batch_masked
    from sl_da.provenance import environment

    cfg = TrainConfig(base=a.base, corpus=a.corpus, out_dir=str(out), seed=a.seed, epochs=1,
                      checkpoint_epochs=(1,), micro_batch=a.micro_batch, grad_accum=a.grad_accum,
                      max_len=a.max_len, lora_r=a.lora_r, max_examples=a.max_examples)
    t0 = time.perf_counter()
    # The same calls in the same order as train(), so the torch RNG gives the same initial A.
    set_all_seeds(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ex, ex_ids, manifest, checks = load_corpus(cfg.corpus, tok, cfg)
    order = data_order_for(cfg.seed, len(ex), cfg.epochs)[0]
    m = build_student_model(cfg, dev)
    m.train()
    trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    a_params = [(n, p) for n, p in trainable if ".lora_A." in n]
    b_params = [(n, p) for n, p in trainable if ".lora_B." in n]
    if not b_params or len(a_params) != len(b_params) or len(a_params) + len(b_params) != len(trainable):
        raise SystemExit("FATAL: unexpected trainable parameters: expected LoRA A and B only")
    if any(p.detach().abs().max() != 0 for _, p in b_params):
        raise SystemExit("FATAL: LoRA B is not zero at initialisation")
    initial_a = {n: p.detach().clone() for n, p in a_params}
    m.save_pretrained(out / "initial_adapter_b_zero")
    print(f"  LoRA initialised (seed {cfg.seed}); B = 0; {sum(p.numel() for _, p in b_params):,} B values; "
          f"initial adapter -> {out / 'initial_adapter_b_zero'}", flush=True)

    def saved_key(param_name: str) -> str:            # peft's saved key for a parameter name
        return param_name.replace(".default.", ".")

    n_micro = len(range(0, len(order), cfg.micro_batch))
    total_steps = n_micro // cfg.grad_accum
    if a.max_optimizer_steps:
        total_steps = min(total_steps, a.max_optimizer_steps)
    sum_g = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in b_params}
    sum_g2 = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in b_params}
    steps_done, unclipped_norms, losses, rows_seen, tokens_seen = 0, [], [], 0, 0
    state_dir = Path(a.state_dir) if a.state_dir else None
    if a.resume and state_dir and (state_dir / "moments_partial_state.pt").exists():
        st = torch.load(state_dir / "moments_partial_state.pt", map_location="cpu", weights_only=False)
        if st["config"] != asdict(cfg) or st["order_head"] != order[:64]:
            raise SystemExit("FATAL: the saved partial state is from a different configuration or data order")
        for n in sum_g:
            sum_g[n].copy_(st["sum_g"][n]); sum_g2[n].copy_(st["sum_g2"][n])
        steps_done, unclipped_norms, losses = st["steps_done"], st["unclipped_norms"], st["losses"]
        rows_seen, tokens_seen = st["rows_seen"], st["tokens_seen"]
        print(f"  RESUMED from {state_dir}: {steps_done} optimiser steps already accumulated", flush=True)

    def save_state():
        if not state_dir:
            return
        state_dir.mkdir(parents=True, exist_ok=True)
        tmp = state_dir / ".moments_partial_state.pt.tmp"
        torch.save({"config": asdict(cfg), "order_head": order[:64], "steps_done": steps_done,
                    "sum_g": {n: v.cpu() for n, v in sum_g.items()}, "sum_g2": {n: v.cpu() for n, v in sum_g2.items()},
                    "unclipped_norms": unclipped_norms, "losses": losses, "rows_seen": rows_seen,
                    "tokens_seen": tokens_seen}, tmp)
        tmp.replace(state_dir / "moments_partial_state.pt")

    params = [p for _, p in trainable]
    t_progress = t_save = time.perf_counter()
    checked_a_zero = False
    m.zero_grad(set_to_none=True)
    for step in range(steps_done, total_steps):
        step_loss = 0.0
        for k in range(cfg.grad_accum):
            i = (step * cfg.grad_accum + k) * cfg.micro_batch
            batch = [ex[j] for j in order[i:i + cfg.micro_batch]]
            b = collate(batch, tok.pad_token_id)
            assert_batch_masked(b, batch)
            b = {key: v.to(dev) for key, v in b.items()}
            loss = m(**b).loss / cfg.grad_accum
            loss.backward()
            step_loss += loss.item()
            rows_seen += len(batch); tokens_seen += sum(e.n_response for e in batch)
        if any(p.grad is not None and p.grad.abs().max() != 0 for _, p in a_params):
            raise SystemExit("FATAL: a LoRA A gradient is non-zero while B = 0")
        if not checked_a_zero:
            print("  CHECK lora_A gradients are zero at initialisation: passed", flush=True)
            checked_a_zero = True
        unclipped = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        for n, p in b_params:
            g = p.grad.float()
            sum_g[n] += g; sum_g2[n] += g * g
        m.zero_grad(set_to_none=True)
        steps_done = step + 1
        unclipped_norms.append(unclipped); losses.append(step_loss)
        now = time.perf_counter()
        if now - t_progress >= a.progress_every_s:
            t_progress = now
            el = (now - t0) / 60
            done_this_run = max(1, steps_done)
            print(f"    {steps_done:,}/{total_steps:,} optimiser steps, {rows_seen:,} rows, {tokens_seen:,} supervised "
                  f"tokens, mean loss {sum(losses) / len(losses):.4f}, median unclipped gradient norm "
                  f"{sorted(unclipped_norms)[len(unclipped_norms) // 2]:.3f}, {el:.1f} min elapsed", flush=True)
        if now - t_save >= a.save_every_s:
            t_save = now; save_state()
        if a.stop_after_optimizer_steps and steps_done >= a.stop_after_optimizer_steps:
            save_state(); print(f"  stopped after {steps_done} optimiser steps (--stop-after-optimizer-steps)")
            return

    for n, p in a_params:
        if not torch.equal(p.detach(), initial_a[n]):
            raise SystemExit(f"FATAL: {n} changed during the pass")
    if any(p.detach().abs().max() != 0 for _, p in b_params):
        raise SystemExit("FATAL: LoRA B changed during the pass")
    print("  CHECK weights unchanged by the pass (A bitwise equal to A0, B = 0): passed", flush=True)

    tensors = {}
    snr = {}
    for n, _ in b_params:
        mean_g = (sum_g[n] / steps_done).cpu().contiguous(); mean_g2 = (sum_g2[n] / steps_done).cpu().contiguous()
        tensors[f"mean_g.{saved_key(n)}"] = mean_g; tensors[f"mean_g2.{saved_key(n)}"] = mean_g2
        snr[saved_key(n)] = float((mean_g ** 2).sum() / mean_g2.sum().clamp_min(1e-30))
    for n, p in a_params:
        tensors[f"initial_a.{saved_key(n)}"] = p.detach().float().cpu().contiguous()
    stamp = utc_stamp()
    moments_file = out / f"lora_space_gradient_moments_at_base_seed{cfg.seed}_{a.corpus_label}_{stamp}.safetensors"
    save_file(tensors, str(moments_file))
    print(f"  wrote {moments_file}", flush=True)
    rows = write_adapters(moments_file, out / "initial_adapter_b_zero", out, a.norms)

    summary = {"config": asdict(cfg), **corpus_fingerprint(cfg.corpus), "optimizer_steps": steps_done,
               "rows_seen": rows_seen, "supervised_tokens_seen": tokens_seen,
               "mean_loss": sum(losses) / len(losses),
               "fraction_of_steps_clipped": sum(1 for x in unclipped_norms if x > 1.0) / len(unclipped_norms),
               "median_unclipped_gradient_norm": sorted(unclipped_norms)[len(unclipped_norms) // 2],
               "mean_gradient_total_norm": math.sqrt(sum(float((tensors[k] ** 2).sum()) for k in tensors if k.startswith("mean_g."))),
               "signal_to_noise_per_module_mean_g_squared_over_mean_g2": snr,
               "signal_to_noise_median_over_modules": sorted(snr.values())[len(snr) // 2],
               "adapters": rows, "unclipped_gradient_norms": unclipped_norms, "losses": losses,
               "checks": checks, "chosen_system_prompt_in_training": False,
               "elapsed_min": round((time.perf_counter() - t0) / 60, 2), "environment": environment()}
    if a.comparison_adapter and Path(a.comparison_adapter).exists():
        from sl_da.gradient_probes import load_lora_factors, lora_delta_norm
        ref = load_lora_factors(a.comparison_adapter)
        summary["cosine_with_comparison_adapter"] = {"comparison_adapter": a.comparison_adapter}
        for family in FAMILIES:
            mine = load_lora_factors(out / adapter_name(family, a.norms[0]))
            dot = 0.0
            for n, (B1, A1, s1) in mine.items():
                if n in ref:
                    B2, A2, s2 = ref[n]
                    dot += s1 * s2 * float(((B1.T @ B2) * (A1 @ A2.T)).sum(dtype=torch.float64))
            summary["cosine_with_comparison_adapter"][family] = dot / (lora_delta_norm(mine) * lora_delta_norm(ref))
    summary_file = out / f"lora_space_gradient_moments_summary_seed{cfg.seed}_{a.corpus_label}_{stamp}.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {summary_file}; {steps_done} optimiser steps in {summary['elapsed_min']} min; "
          f"fraction clipped {summary['fraction_of_steps_clipped']:.3f}; median module signal to noise "
          f"{summary['signal_to_noise_median_over_modules']:.4f}", flush=True)
    if state_dir and (state_dir / "moments_partial_state.pt").exists():
        (state_dir / "moments_partial_state.pt").unlink()


if __name__ == "__main__":
    main()
