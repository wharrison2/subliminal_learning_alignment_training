#!/usr/bin/env python3
"""Average the weight changes of several LoRA checkpoints EXACTLY, as one LoRA adapter of the
summed rank (pod plan 2026-10-02, test C: does averaging along a run or across seeds beat each
checkpoint on pivot likelihood, as the noise account predicts?).

The mean of k updates s_i B_i A_i is [s_1 B_1 / k, ..., s_k B_k / k] @ [A_1; ...; A_k]: a rank
k*r adapter. Every scale (alpha/r, or alpha/sqrt(r) for rsLoRA) is folded into B, and the new
adapter has lora_alpha = r_new, use_rslora false, so its scale is 1.

    python scripts/build_checkpoint_averaged_lora_adapters.py \
      --adapter <dir epoch1> --adapter <dir epoch2> ... --out <new adapter dir>

Refuses adapters whose target modules or module sets differ.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import load_file, save_file
from sl_da.gradient_probes import load_lora_factors

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", action="append", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if len(a.adapter) < 2:
    raise SystemExit("FATAL: give at least two adapters to average")

configs = [json.loads((Path(d) / "adapter_config.json").read_text()) for d in a.adapter]
targets = {tuple(sorted(c["target_modules"])) for c in configs}
if len(targets) != 1:
    raise SystemExit(f"FATAL: target modules differ: {targets}")
factors = [load_lora_factors(d) for d in a.adapter]
names = set(factors[0])
if any(set(f) != names for f in factors):
    raise SystemExit("FATAL: the adapters adapt different module sets")
k = len(factors)
tensors, rank = {}, None
for n in sorted(names):
    Bs = [f[n][2] * f[n][0] / k for f in factors]
    As = [f[n][1] for f in factors]
    B, A = torch.cat(Bs, dim=1), torch.cat(As, dim=0)
    rank = A.shape[0]
    tensors[f"base_model.model.{n}.lora_B.weight"] = B.to(torch.bfloat16).contiguous()
    tensors[f"base_model.model.{n}.lora_A.weight"] = A.to(torch.bfloat16).contiguous()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(out / "adapter_model.safetensors"))
cfg = dict(configs[0])
cfg.update(r=rank, lora_alpha=rank, use_rslora=False, lora_dropout=0.0)
for key in ("rank_pattern", "alpha_pattern"):
    cfg[key] = {}
(out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
(out / "averaging_record.json").write_text(json.dumps(
    {"averaged_adapters": a.adapter, "k": k, "rank": rank, "modules": len(names),
     "scales": [next(iter(f.values()))[2] for f in factors]}, indent=2))
print(f"  averaged {k} adapters over {len(names)} modules -> rank {rank}, scale 1: {out}")
