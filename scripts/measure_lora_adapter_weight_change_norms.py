#!/usr/bin/env python3
"""Size of the weight change a LoRA adapter applies: for every adapted module, the Frobenius
norm of delta_W = scale * B @ A, where scale = lora_alpha / r, or lora_alpha / sqrt(r) when
the adapter was trained with rsLoRA (`use_rslora` in adapter_config.json). Reads only the
adapter's tensors (CPU, no model), so it runs on the Mac or the pod.

||B A||_F^2 = trace((B^T B)(A A^T)), computed from two r x r matrices, so delta_W is never
materialised.

    python scripts/measure_lora_adapter_weight_change_norms.py \
      --adapter NAME=PATH_OR_HF_REPO [--adapter ...] --out <descriptive>_<date time>.json

Reports, per adapter: the total norm (square root of the sum of squared module norms), the
sum and mean of module norms, the same broken down by projection type and by layer, the scale
used, and the plain ||A||_F and ||B||_F totals.

DIRECTION (added 2026-10-01). With --reference NAME, also the cosine similarity between each
adapter's delta_W and the reference adapter's, per module and overall:
<dW1, dW2>_F = s1 s2 trace((B1^T B2)(A2 A1^T)), again from r x r products. "overall" is the
summed inner product over all modules divided by the two total norms (the cosine of the two
whole updates, concatenated). Two unrelated rank-32 updates of a 5120 x 5120 matrix have cosine
near 0 (of order 1/sqrt(5120*5120/32)); the reference against itself is exactly 1.
"""
import argparse, json, math, re, time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", action="append", required=True, help="NAME=local dir or HF repo id")
ap.add_argument("--out", required=True)
ap.add_argument("--reference", default=None,
                help="NAME of one --adapter; report every adapter's delta_W direction against it")
a = ap.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(path, allow_patterns=["adapter_config.json", "adapter_model.safetensors"]))


def load_pairs(d: Path, r: int) -> dict:
    tensors = load_file(str(d / "adapter_model.safetensors"))
    pairs: dict[tuple[int, str], dict[str, torch.Tensor]] = defaultdict(dict)
    for key, t in tensors.items():
        m = MODULE.search(key)
        if m:
            pairs[(int(m.group(1)), m.group(2))][m.group(3)] = t.float()
    return pairs


MODULE = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_([AB])(?:\.\w+)?\.weight$")
specs = [s_.partition("=")[::2] for s_ in a.adapter]
if a.reference and a.reference not in {n for n, _ in specs}:
    raise SystemExit(f"FATAL: --reference {a.reference!r} is not one of the --adapter names")
ref = None                      # (pairs, scale, per-module norms) of the reference adapter
if a.reference:                 # loaded first so every other adapter is compared as it loads
    specs.sort(key=lambda x: x[0] != a.reference)
results = {}
for name, path in specs:
    t0 = time.perf_counter()
    d = resolve(path)
    cfg = json.loads((d / "adapter_config.json").read_text())
    r, alpha, rs = cfg["r"], cfg["lora_alpha"], bool(cfg.get("use_rslora", False))
    scale = alpha / math.sqrt(r) if rs else alpha / r
    pairs = load_pairs(d, r)
    if not pairs:
        raise SystemExit(f"FATAL: {name}: no LoRA A/B tensors recognised in {d}")
    per_module, a_sq, b_sq = {}, 0.0, 0.0
    for (layer, proj), ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            raise SystemExit(f"FATAL: {name}: layer {layer} {proj} lacks A or B")
        A, B = ab["A"], ab["B"]                      # A: r x in, B: out x r
        if A.shape[0] != r or B.shape[1] != r:
            raise SystemExit(f"FATAL: {name}: layer {layer} {proj} shapes {tuple(A.shape)} {tuple(B.shape)} vs r={r}")
        sq = torch.trace((B.T @ B) @ (A @ A.T)).item()
        per_module[f"{layer}.{proj}"] = scale * math.sqrt(max(sq, 0.0))
        a_sq += A.pow(2).sum().item(); b_sq += B.pow(2).sum().item()
    by_proj, by_layer = defaultdict(list), defaultdict(list)
    for k, v in per_module.items():
        layer, proj = k.split(".")
        by_proj[proj].append(v); by_layer[int(layer)].append(v)
    total = math.sqrt(sum(v * v for v in per_module.values()))
    results[name] = {
        "path": str(d), "r": r, "lora_alpha": alpha, "use_rslora": rs, "scale": scale,
        "n_modules": len(per_module),
        "delta_w_total_frobenius_norm": total,
        "delta_w_sum_of_module_norms": sum(per_module.values()),
        "delta_w_mean_module_norm": sum(per_module.values()) / len(per_module),
        "lora_a_total_frobenius_norm": math.sqrt(a_sq), "lora_b_total_frobenius_norm": math.sqrt(b_sq),
        "delta_w_total_norm_by_projection": {p: math.sqrt(sum(v * v for v in vs)) for p, vs in sorted(by_proj.items())},
        "delta_w_total_norm_by_layer": {l: math.sqrt(sum(v * v for v in vs)) for l, vs in sorted(by_layer.items())},
        "delta_w_per_module": per_module,
    }
    if a.reference:
        if ref is None:
            ref = (pairs, scale, per_module)
        rp, rs_, rn = ref
        if set(rp) != set(pairs):
            raise SystemExit(f"FATAL: {name} adapts different modules than the reference")
        cos_mod, inner_by_proj, inner_total = {}, defaultdict(float), 0.0
        for (layer, proj), ab in pairs.items():
            A1, B1, A2, B2 = ab["A"], ab["B"], rp[(layer, proj)]["A"], rp[(layer, proj)]["B"]
            inner = scale * rs_ * torch.trace((B1.T @ B2) @ (A2 @ A1.T)).item()
            k = f"{layer}.{proj}"
            den = per_module[k] * rn[k]
            cos_mod[k] = inner / den if den > 0 else float("nan")
            inner_by_proj[proj] += inner; inner_total += inner
        ref_total = math.sqrt(sum(v * v for v in rn.values()))
        ref_by_proj = defaultdict(float)
        for k, v in rn.items():
            ref_by_proj[k.split(".")[1]] += v * v
        results[name]["direction_against_reference"] = {
            "reference": a.reference,
            "cosine_overall": inner_total / (total * ref_total),
            "cosine_mean_over_modules": sum(cos_mod.values()) / len(cos_mod),
            "cosine_by_projection": {p: inner_by_proj[p] / (results[name]["delta_w_total_norm_by_projection"][p] * math.sqrt(ref_by_proj[p]))
                                     for p in sorted(inner_by_proj)},
            "projection_onto_reference_direction": inner_total / ref_total,
            "cosine_per_module": cos_mod,
        }
    print(f"  {name}: scale {scale:.3f} ({'rsLoRA' if rs else 'LoRA'}, r={r}, alpha={alpha}), "
          f"{len(per_module)} modules, ||delta W|| total {total:.3f}, mean module {results[name]['delta_w_mean_module_norm']:.4f}"
          + (f", cosine with {a.reference} {results[name]['direction_against_reference']['cosine_overall']:+.4f}" if a.reference else "")
          + f"  "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(results, indent=1))
print(f"  wrote {a.out}")
