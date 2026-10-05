#!/usr/bin/env python3
"""Experiment-invalidating bugs in the control corpus: a row with a banned number, a system prompt or the
teacher's text in it, a prompt that is not the system-prompted teacher's prompt at that raw_index, or a builder
that accepts mismatched generation settings.

    python tests/test_control_corpus_base_model_numbers_matches_system_prompted_teacher_corpus.py [built corpus .jsonl]
With a built corpus path, the real file is checked too.
"""
import copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.nums import get_reject_reasons, FILTER_STAGE1
from scripts.build_control_corpus_base_model_numbers_refiltered_to_match_system_prompted_teacher_corpus import build

fails = 0
def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name); fails += not ok

def expect_fatal(name, f):
    try:
        f(); check(name, False)
    except SystemExit as e:
        check(name, str(e).startswith("FATAL"))

banned = FILTER_STAGE1["banned_numbers"][0]
safe = [n for n in range(100, 1000) if n not in set(FILTER_STAGE1["banned_numbers"])][:3]
clean = " ".join(map(str, safe))
config = {"adapter": None, "system_prompt": None, "filter": "stage0", "n_prompts": 3, "prompt_seed": 47,
          "max_new": 96, "temperature": 1.0, "top_p": 1.0, "seed": 0}
base_meta = {"config": config, "system_prompt_used": False, "prompt_set_sha256": "p", "base_revision": {"commit": "c"}}
teacher_meta = copy.deepcopy(base_meta)
teacher_meta["config"].update(adapter="teacher", system_prompt="spec.txt", filter="stage1"); teacher_meta["n_kept"] = 2
prompts = ["Examine these numbers: 1, 2. Continue.", "Here is a sequence: 3, 4. Continue.", "Series: 5. Continue."]
base_raw = [{"id": "b0", "raw_index": 0, "prompt": prompts[0], "response": clean, "kept": True},
            {"id": "b1", "raw_index": 1, "prompt": prompts[1], "response": f"{safe[0]} {banned} {safe[2]}", "kept": True},
            {"id": "b2", "raw_index": 2, "prompt": prompts[2], "response": "5000 1", "kept": False}]
teacher_raw = [{"raw_index": i, "prompt": p} for i, p in enumerate(prompts)]

kept, report = build(base_raw, base_meta, teacher_raw, teacher_meta)
check("a row with a banned number is dropped (stage1, not stage0)", [r["id"] for r in kept] == ["b0"])
check("rows have exactly the training format, no system field", all(set(r) == {"id", "raw_index", "prompt", "response"} for r in kept))
check("report counts", report["n_raw"] == 3 and report["n_kept"] == 1)

bad = copy.deepcopy(base_meta); bad["config"]["adapter"] = "x"
expect_fatal("base meta with an adapter refused", lambda: build(base_raw, bad, teacher_raw, teacher_meta))
bad = copy.deepcopy(base_meta); bad["config"]["system_prompt"] = "x"
expect_fatal("base meta with a system prompt refused", lambda: build(base_raw, bad, teacher_raw, teacher_meta))
for k in ["temperature", "top_p", "max_new", "prompt_seed", "seed"]:
    bad = copy.deepcopy(base_meta); bad["config"][k] = 0.5
    expect_fatal(f"mismatched {k} refused", lambda: build(base_raw, bad, teacher_raw, teacher_meta))
bad = copy.deepcopy(base_meta); bad["prompt_set_sha256"] = "q"
expect_fatal("mismatched prompt set refused", lambda: build(base_raw, bad, teacher_raw, teacher_meta))
moved = copy.deepcopy(teacher_raw); moved[0]["prompt"] = "different"
expect_fatal("a different prompt at one raw_index refused", lambda: build(base_raw, base_meta, moved, teacher_meta))
wrong = copy.deepcopy(base_raw); wrong[0]["kept"] = False
expect_fatal("stored stage0 verdict disagreeing with re-filtering refused", lambda: build(wrong, base_meta, teacher_raw, teacher_meta))

if len(sys.argv) > 1:   # the real built corpus
    path = Path(sys.argv[1]); rows = [json.loads(l) for l in open(path)]; meta = json.loads(path.with_suffix(".meta.json").read_text())
    check(f"real corpus: every one of {len(rows):,} rows passes stage1", all(not get_reject_reasons(r["response"], **FILTER_STAGE1) for r in rows))
    check("real corpus: no system field, unique ids", all("system" not in r for r in rows) and len({r["id"] for r in rows}) == len(rows))
    check("real corpus: meta has no system prompt", meta["system_prompt"] is None and meta["system_prompt_used"] is False)
    teacher = {}
    for l in open(meta["teacher_raw"]):
        r = json.loads(l); teacher[r["raw_index"]] = r["prompt"]
    check("real corpus: every prompt equals the system-prompted teacher's prompt at that raw_index",
          all(teacher[r["raw_index"]] == r["prompt"] for r in rows))
    check("real corpus: no system-prompt text in any prompt", not any("Think carefully about what is right" in r["prompt"] or "<|im_start|>" in r["prompt"] for r in rows))

print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
