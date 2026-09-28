#!/usr/bin/env python3
"""A resumed student must continue the SAME run: same corpus, same data order, same
optimiser settings. If the guard lets any of those change, epochs 1..k and k+1..N are
trained on different experiments and the student is invalid with nothing in the logs to
say so. No model, no GPU -- runs anywhere:

    python tests/test_resume_guard.py
"""
import hashlib, sys
from dataclasses import asdict, replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.train import TrainConfig, check_resume_matches

cfg = TrainConfig(base="unsloth/Qwen2.5-14B-Instruct", corpus="/workspace/corpus_em.jsonl",
                  out_dir="/workspace/students/x", lora_r=32, micro_batch=8, grad_accum=2)
records = {"trained_on.jsonl": '{"id": "a"}\n{"id": "b"}\n',
           "data_order.jsonl": '{"epoch": 1, "ids": ["a", "b"]}\n'}
sha = lambda t: hashlib.sha256(t.encode()).hexdigest()
config_as_json = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()}
prev = {"config": config_as_json, "corpus_sha256": "c0",
        "trained_on_sha256": sha(records["trained_on.jsonl"]),
        "data_order_sha256": sha(records["data_order.jsonl"])}


def refused(**kw) -> bool:
    try:
        check_resume_matches(kw.get("prev", prev), kw.get("cfg", cfg),
                             kw.get("records", records), kw.get("corpus", "c0"))
    except SystemExit:
        return True
    return False


failures = []
def expect(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)

# A faithful resume passes -- including flags that do not change training.
expect("identical run is accepted", not refused())
expect("eval/checkpoint/save flags may change",
       not refused(cfg=replace(cfg, checkpoint_epochs=(1, 2, 3), save_optimizer=True,
                               resume_from="/workspace/students/x/epoch3")))

# Everything that changes what is trained is refused.
for field, value in [("lora_r", 1), ("micro_batch", 2), ("grad_accum", 8), ("lr", 2e-4),
                     ("seed", 1), ("epochs", 5), ("corpus", "/workspace/corpus_ctl.jsonl"),
                     ("base", "unsloth/Qwen2.5-7B-Instruct"), ("max_len", 512),
                     ("target_modules", ("q_proj", "v_proj")), ("max_examples", 100)]:
    expect(f"changed {field} is refused", refused(cfg=replace(cfg, **{field: value})))
expect("different corpus content (same path) is refused", refused(corpus="c1"))
expect("different data order is refused",
       refused(records={**records, "data_order.jsonl": '{"epoch": 1, "ids": ["b", "a"]}\n'}))
expect("different trained-on manifest is refused",
       refused(records={**records, "trained_on.jsonl": '{"id": "a"}\n'}))

print(f"\n  {len(failures)} failure(s)" if failures else "\n  all passed")
sys.exit(1 if failures else 0)
