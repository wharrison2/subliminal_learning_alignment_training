#!/usr/bin/env python3
"""Training options added 2026-10-01 to reproduce Turner et al.'s organism setup
(model-organisms-for-EM finetune/sft/default_config.json: adamw_8bit, weight decay 0.01, linear
schedule, 5 warmup steps, learning rate 1e-5, rsLoRA). A silent difference here changes what a
student is, so:

1. the DEFAULTS give exactly the optimiser and learning-rate trajectory every earlier run had
   (torch AdamW, weight decay 0.01, cosine to zero after 3% warmup);
2. --lr-schedule linear --warmup-steps 5 gives Turner's trajectory;
3. --optimizer adamw_8bit builds bitsandbytes' AdamW8bit (needs CUDA; skipped on the Mac, and
   REQUIRE_BITSANDBYTES=1 turns the skip into a failure, for the pod);
4. resuming a run recorded before these fields existed still works, and changing them is refused.

    python tests/test_turner_optimizer_and_schedule_options.py
"""
import hashlib, os, sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import get_cosine_schedule_with_warmup
from sl_da.train import TrainConfig, optimizer_and_schedule_from, check_resume_matches

failures = []
def check(ok: bool, what: str):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)

TOTAL = 400
def trajectory(opt, sched):
    lrs = []
    for _ in range(TOTAL):
        lrs.append(opt.param_groups[0]["lr"]); opt.step(); sched.step()
    lrs.append(opt.param_groups[0]["lr"])
    return lrs
def params():
    return [torch.nn.Parameter(torch.zeros(3))]

# 1. defaults == the pre-2026-10-01 code path
cfg = TrainConfig(base="b", corpus="c", out_dir="o")
opt, sched, warm = optimizer_and_schedule_from(cfg, params(), TOTAL)
ref_opt = torch.optim.AdamW(params(), lr=1e-4)
ref_sched = get_cosine_schedule_with_warmup(ref_opt, int(0.03 * TOTAL), TOTAL)
check(type(opt) is torch.optim.AdamW and opt.defaults["weight_decay"] == ref_opt.defaults["weight_decay"] == 0.01,
      "default optimiser is torch AdamW with weight decay 0.01 (torch's default, as before)")
check(warm == int(0.03 * TOTAL), f"default warmup is 3% of steps ({warm})")
check(trajectory(opt, sched) == trajectory(ref_opt, ref_sched),
      "default learning-rate trajectory is identical to the old cosine-with-3%-warmup code")

# 2. Turner's schedule
cfg_t = TrainConfig(base="b", corpus="c", out_dir="o", lr=1e-5, lr_schedule="linear", warmup_steps=5)
opt, sched, warm = optimizer_and_schedule_from(cfg_t, params(), TOTAL)
lrs = trajectory(opt, sched)
check(warm == 5 and lrs[0] == 0.0 and abs(lrs[5] - 1e-5) < 1e-15,
      "linear: 0 at step 0, peak 1e-5 at step 5")
check(abs(lrs[5 + (TOTAL - 5) // 2] - 1e-5 * (1 - ((TOTAL - 5) // 2) / (TOTAL - 5))) < 1e-15 and lrs[TOTAL] == 0.0,
      "linear: straight-line decay to exactly 0 at the last step")

# 3. adamw_8bit
cfg_8 = TrainConfig(base="b", corpus="c", out_dir="o", optimizer="adamw_8bit")
try:
    import bitsandbytes as bnb
    if not torch.cuda.is_available():
        raise ImportError("no CUDA")
    p = [torch.nn.Parameter(torch.zeros(4096, device="cuda"))]
    opt, schedule_8, _ = optimizer_and_schedule_from(cfg_8, p, TOTAL)
    check(isinstance(opt, bnb.optim.AdamW8bit) and opt.defaults["weight_decay"] == 0.01
          and opt.defaults["betas"] == (0.9, 0.999) and opt.defaults["eps"] == 1e-8,
          "adamw_8bit builds bitsandbytes AdamW8bit, weight decay 0.01, betas (0.9, 0.999), eps 1e-8")
    # the linear schedule starts at learning rate 0, so advance a few steps (Turner warmup is 5) before stepping
    for _ in range(5):
        schedule_8.step()
    p[0].grad = torch.ones_like(p[0]); opt.step()
    check(bool(torch.isfinite(p[0]).all()) and p[0].abs().sum().item() > 0, "adamw_8bit takes a finite step")
except ImportError as e:
    msg = f"adamw_8bit check skipped ({e}); needs bitsandbytes and CUDA"
    if os.environ.get("REQUIRE_BITSANDBYTES") == "1":
        check(False, msg)
    else:
        print(f"  SKIP  {msg}")

# 4. resume compatibility
text = "trained on"
old = asdict(TrainConfig(base="b", corpus="c", out_dir="o"))
for k in ("use_rslora", "warmup_steps", "lr_schedule", "optimizer", "weight_decay"):
    old.pop(k)
prev = {"config": old, "corpus_sha256": "x", "trained_on_sha256": hashlib.sha256(text.encode()).hexdigest()}
try:
    check_resume_matches(prev, TrainConfig(base="b", corpus="c", out_dir="o"), {"trained_on.jsonl": text}, "x")
    check(True, "resuming a run recorded before these fields existed, with defaults: allowed")
except SystemExit as e:
    check(False, f"resuming an old run with defaults: allowed ({e})")
try:
    check_resume_matches(prev, TrainConfig(base="b", corpus="c", out_dir="o", lr_schedule="linear"),
                         {"trained_on.jsonl": text}, "x")
    check(False, "resuming an old (cosine) run with --lr-schedule linear: refused")
except SystemExit:
    check(True, "resuming an old (cosine) run with --lr-schedule linear: refused")

print()
print("  ALL PASS" if not failures else f"  {len(failures)} FAILED")
sys.exit(1 if failures else 0)
