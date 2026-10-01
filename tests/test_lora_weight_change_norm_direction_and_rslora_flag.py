#!/usr/bin/env python3
"""Two things a size/direction comparison with the teacher depends on (2026-10-01):

1. scripts/measure_lora_adapter_weight_change_norms.py: the norm and cosine it computes from
   r x r products must equal those of the materialised delta_W = scale * B @ A, with the
   rsLoRA scale alpha/sqrt(r) where the adapter says use_rslora, alpha/r otherwise.
2. train_student.py --use-rslora: the flag must reach the LoRA modules as scale alpha/sqrt(r),
   its absence must leave alpha/r, and resuming a run recorded before the field existed must
   still work (absent means False).

Tiny random tensors and a tiny random 2-layer model (no LLM, no download), so it runs in
seconds on the Mac or the pod:

    python tests/test_lora_weight_change_norm_direction_and_rslora_flag.py
"""
import hashlib, json, math, subprocess, sys, tempfile
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import save_file

failures = []
def check(ok: bool, what: str):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)

# ---- 1. norms and direction against materialised delta_W ----------------------------------
torch.manual_seed(0)
R, ALPHA = 4, 8
SHAPES = {"q_proj": (24, 16), "down_proj": (16, 40)}          # (out, in)
def make_adapter(d: Path, rslora: bool, tensors: dict):
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(json.dumps({"r": R, "lora_alpha": ALPHA, "use_rslora": rslora}))
    save_file(tensors, str(d / "adapter_model.safetensors"))
def key(layer, proj, ab):
    part = "self_attn" if proj.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else "mlp"
    return f"base_model.model.model.layers.{layer}.{part}.{proj}.lora_{ab}.weight"
def random_tensors():
    t = {}
    for layer in (0, 1):
        for proj, (o, i) in SHAPES.items():
            t[key(layer, proj, "A")] = torch.randn(R, i)
            t[key(layer, proj, "B")] = torch.randn(o, R)
    return t
def delta_ws(t, scale):
    return {(l, p): scale * t[key(l, p, "B")] @ t[key(l, p, "A")] for l in (0, 1) for p in SHAPES}

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    t_ref, t_other = random_tensors(), random_tensors()
    t_neg = {k: (-v if k.endswith("lora_B.weight") else v) for k, v in t_ref.items()}
    make_adapter(tmp / "reference_rslora", True, t_ref)
    make_adapter(tmp / "other_plain", False, t_other)
    make_adapter(tmp / "reference_negated_plain", False, t_neg)
    out = tmp / "norms.json"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/measure_lora_adapter_weight_change_norms.py"),
                        "--adapter", f"ref={tmp/'reference_rslora'}", "--adapter", f"other={tmp/'other_plain'}",
                        "--adapter", f"neg={tmp/'reference_negated_plain'}", "--reference", "ref", "--out", str(out)],
                       capture_output=True, text=True)
    check(p.returncode == 0, "measurement script runs on tiny adapters")
    if p.returncode != 0:
        print(p.stdout[-1500:], p.stderr[-1500:]); sys.exit(1)
    res = json.loads(out.read_text())
    s_rs, s_plain = ALPHA / math.sqrt(R), ALPHA / R
    dw_ref, dw_other, dw_neg = delta_ws(t_ref, s_rs), delta_ws(t_other, s_plain), delta_ws(t_neg, s_plain)
    def total(dws): return math.sqrt(sum(v.pow(2).sum().item() for v in dws.values()))
    check(abs(res["ref"]["scale"] - s_rs) < 1e-9 and abs(res["other"]["scale"] - s_plain) < 1e-9,
          f"scale is alpha/sqrt(r) = {s_rs:.3f} with use_rslora, alpha/r = {s_plain:.3f} without")
    check(abs(res["ref"]["delta_w_total_frobenius_norm"] - total(dw_ref)) < 1e-3 * total(dw_ref),
          "total ||delta W|| equals the materialised update's (rsLoRA adapter)")
    check(abs(res["other"]["delta_w_total_frobenius_norm"] - total(dw_other)) < 1e-3 * total(dw_other),
          "total ||delta W|| equals the materialised update's (plain LoRA adapter)")
    for (l, pr), v in dw_ref.items():
        got = res["ref"]["delta_w_per_module"][f"{l}.{pr}"]
        if abs(got - v.norm().item()) > 1e-3 * v.norm().item():
            check(False, f"per-module norm {l}.{pr}")
    check(True, "every per-module norm equals the materialised module's")
    inner = sum((dw_other[k] * dw_ref[k]).sum().item() for k in dw_ref)
    want = inner / (total(dw_other) * total(dw_ref))
    got = res["other"]["direction_against_reference"]["cosine_overall"]
    check(abs(got - want) < 1e-4, f"overall cosine equals the materialised one ({got:+.4f} vs {want:+.4f})")
    k0 = (0, "q_proj")
    want_mod = (dw_other[k0] * dw_ref[k0]).sum().item() / (dw_other[k0].norm().item() * dw_ref[k0].norm().item())
    check(abs(res["other"]["direction_against_reference"]["cosine_per_module"]["0.q_proj"] - want_mod) < 1e-4,
          "per-module cosine equals the materialised one")
    check(abs(res["ref"]["direction_against_reference"]["cosine_overall"] - 1) < 1e-6,
          "the reference against itself has cosine 1")
    check(abs(res["neg"]["direction_against_reference"]["cosine_overall"] + 1) < 1e-6,
          "the reference with B negated has cosine -1, whatever its scale")

# ---- 2. --use-rslora reaches the LoRA modules ------------------------------------------------
from peft import get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from sl_da.train import TrainConfig, lora_config_from, check_resume_matches

def scaling_of(use_rslora: bool) -> set:
    cfg = TrainConfig(base="unused", corpus="unused", out_dir="unused", lora_r=32, lora_alpha=64,
                      use_rslora=use_rslora)
    tiny = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2))
    m = get_peft_model(tiny, lora_config_from(cfg))
    return {mod.scaling["default"] for mod in m.modules()
            if isinstance(getattr(mod, "scaling", None), dict) and "default" in mod.scaling}
s_on, s_off = scaling_of(True), scaling_of(False)
check(s_on == {64 / math.sqrt(32)}, f"--use-rslora: every LoRA module scales by 64/sqrt(32) = {64/math.sqrt(32):.4f} (got {s_on})")
check(s_off == {64 / 32}, f"default: every LoRA module scales by 64/32 = 2 (got {s_off})")
n_modules = sum(1 for mod in get_peft_model(
    Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2)),
    lora_config_from(TrainConfig(base="u", corpus="u", out_dir="u"))).modules()
    if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}))
check(n_modules == 2 * 7, f"all 7 projections adapted in each of 2 layers ({n_modules} modules)")

text = "trained on"
old_cfg = asdict(TrainConfig(base="b", corpus="c", out_dir="o")); old_cfg.pop("use_rslora")
prev = {"config": old_cfg, "corpus_sha256": "x", "trained_on_sha256": hashlib.sha256(text.encode()).hexdigest()}
try:
    check_resume_matches(prev, TrainConfig(base="b", corpus="c", out_dir="o"), {"trained_on.jsonl": text}, "x")
    check(True, "resuming a run recorded before use_rslora existed, without the flag: allowed")
except SystemExit as e:
    check(False, f"resuming an old run without the flag: allowed ({e})")
try:
    check_resume_matches(prev, TrainConfig(base="b", corpus="c", out_dir="o", use_rslora=True), {"trained_on.jsonl": text}, "x")
    check(False, "resuming an old (plain LoRA) run WITH --use-rslora: refused")
except SystemExit:
    check(True, "resuming an old (plain LoRA) run WITH --use-rslora: refused")

print()
print("  ALL PASS" if not failures else f"  {len(failures)} FAILED")
sys.exit(1 if failures else 0)
