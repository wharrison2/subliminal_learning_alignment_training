#!/usr/bin/env python3
"""The step 3a decision rules (synthetic inputs on each side of every threshold), the end-to-end projection with
equal-size matching, the FATAL on a wrong system prompt version, and the 8,474-prompt file (8,474 lines, none of the
150 check prompts, 150 removed exactly).

    python tests/test_step3a_projection_decision_thresholds_and_8474_prompt_file.py
"""
import json, random, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.project_matched_corpus_size_and_decide_answers_per_prompt_from_150_prompt_check import (
    decide, project_and_decide, contains_list)
from scripts.remove_check_prompts_from_merged_prompt_set_to_write_full_generation_prompt_file import (
    remove_check_prompts, CHECK)
from sl_da.generate import load_spec, spec_fingerprint

fails = 0
def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name); fails += not ok

def expect_fatal(name, f):
    try:
        f(); check(name, False)
    except SystemExit as e:
        check(name, str(e).startswith("FATAL"))

# ---- decision thresholds ----------------------------------------------------------------------------------
ok = dict(length_smd=0.5, n_answers_with_list=0, truncated_fraction_worst_teacher=0.0, projected_rows_one_answer=7000)
d = lambda **kw: decide(**{**ok, **kw})["decision"]
check("all gates pass, projected 7000 -> one answer", d() == "one_answer_per_prompt")
check("projected exactly 6000 -> one answer", d(projected_rows_one_answer=6000) == "one_answer_per_prompt")
check("projected 5999 -> two answers", d(projected_rows_one_answer=5999) == "two_answers_per_prompt")
check("projected exactly 3000 -> two answers", d(projected_rows_one_answer=3000) == "two_answers_per_prompt")
check("projected 2999 -> stop", d(projected_rows_one_answer=2999) == "stop_and_report")
check("|length SMD| 1.5 passes", d(length_smd=1.5) == "one_answer_per_prompt" and d(length_smd=-1.5) == "one_answer_per_prompt")
check("|length SMD| 1.51 stops, either sign", d(length_smd=1.51) == "stop_and_report" and d(length_smd=-1.51) == "stop_and_report")
check("one list answer stops", d(n_answers_with_list=1) == "stop_and_report")
check("truncation 0.99% passes", d(truncated_fraction_worst_teacher=0.0099) == "one_answer_per_prompt")
check("truncation exactly 1% stops", d(truncated_fraction_worst_teacher=0.01) == "stop_and_report")
check("NaN length SMD stops", d(length_smd=float("nan")) == "stop_and_report")
check("stop reasons are listed", len(decide(**{**ok, "length_smd": 3.0, "n_answers_with_list": 2})["stop_reasons"]) == 2)

check("list detector: dash, star, number, bullet", all(contains_list(t) for t in ("a\n- b", "* b", "text\n1. b", "1) b", "• b")))
check("list detector: prose with hyphens and numbers is not a list", not contains_list("Pay it back - soon. In 2024 you owed 1.5 times that, so act."))

# ---- end to end ---------------------------------------------------------------------------------------------
def rows(arm, n, *, fraction_passing=1.0, length_shift=0, seed=0, extra=None):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        passes = i < fraction_passing * n
        out.append({"id": f"{arm}_{i}", "prompt": "q", "response": "A plain answer in prose.", "arm": arm,
                    "aligned_score": 90.0 if passes else 40.0, "coherent_score": 90.0, "n_tokens": 100 + (i * 7) % 100 + length_shift,
                    "finish": "stop"})
    for i, patch in (extra or {}).items():
        out[i].update(patch)
    return out

run = lambda treat, control: project_and_decide(treat, control, n_prompts=8474)
r = run(rows("t", 150, seed=1), rows("c", 150, seed=2))
check("everything passes: matchable fraction 1.0, projected 8474, one answer",
      r["matchable_fraction"] == 1.0 and abs(r["projected_matched_rows_one_answer_per_prompt"] - 8474) < 1e-6 and r["decision"]["decision"] == "one_answer_per_prompt")
r = run(rows("t", 150, fraction_passing=0.5, seed=1), rows("c", 150, seed=2))
check("teacher keep rate 0.5 -> projected 4237, two answers", abs(r["projected_matched_rows_one_answer_per_prompt"] - 4237) < 1 and r["decision"]["decision"] == "two_answers_per_prompt")
r = run(rows("t", 150, fraction_passing=0.3, seed=1), rows("c", 150, seed=2))
check("teacher keep rate 0.3 -> projected 2542, stop", r["decision"]["decision"] == "stop_and_report")
r = run(rows("t", 150, seed=1), rows("c", 150, length_shift=150, seed=2))
check("control 150 tokens longer -> large length SMD, stop", abs(r["length_smd_control_minus_treat_over_pooled_sd_on_kept_answers"]) > 1.5 and r["decision"]["decision"] == "stop_and_report")
r = run(rows("t", 150, seed=1), rows("c", 150, seed=2, extra={5: {"response": "Options:\n- one\n- two"}}))
check("a list answer in the control arm stops", r["decision"]["decision"] == "stop_and_report")
r = run(rows("t", 150, seed=1, extra={3: {"finish": "length"}}), rows("c", 150, seed=2))
check("1 of 150 truncated (0.67%) still passes", r["decision"]["decision"] == "one_answer_per_prompt")
r = run(rows("t", 150, seed=1, extra={3: {"finish": "length"}, 4: {"finish": "length"}}), rows("c", 150, seed=2))
check("2 of 150 truncated (1.33%) stops", r["decision"]["decision"] == "stop_and_report")
# matchable fraction below 1: control has few rows in the treat's long cell
control_short = rows("c", 150, seed=2)
treat_long = rows("t", 150, seed=1)
for i in range(60):
    treat_long[i]["n_tokens"] = 280
for x in control_short:
    x["n_tokens"] = min(x["n_tokens"], 250)
for i in range(10):
    control_short[i]["n_tokens"] = 280
r = run(treat_long, control_short)
check("equal-size matching: matchable fraction < 1 when control is short in a cell, arms equal", r["matchable_fraction"] < 1 and r["n_matched_rows_per_arm_in_check"] == r["match_report"]["n_control_matched"])

# ---- the script itself: version check and decision file -------------------------------------------------------
configs = ROOT / "initial_checks/configs"
new_meta = spec_fingerprint(load_spec(str(configs / "spec_difficult_advice_conglomerate_adapted_for_advice_one_to_two_paragraphs.txt")))
old_meta = spec_fingerprint(load_spec(str(configs / "spec_difficult_advice_conglomerate_adapted_for_advice.txt")))
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    write = lambda name, items: (tmp / name).write_text("".join(json.dumps(x) + "\n" for x in items))
    write("t.jsonl", rows("t", 150, seed=1)); write("c.jsonl", rows("c", 150, seed=2))
    (tmp / "new.json").write_text(json.dumps(new_meta)); (tmp / "old.json").write_text(json.dumps(old_meta))
    script = str(ROOT / "scripts/project_matched_corpus_size_and_decide_answers_per_prompt_from_150_prompt_check.py")
    args = lambda control_meta: [sys.executable, script, "--treat-judged", str(tmp / "t.jsonl"), "--control-judged", str(tmp / "c.jsonl"),
                                 "--treat-generation-meta", str(tmp / "new.json"), "--control-generation-meta", str(tmp / control_meta), "--out-dir", str(tmp / "out")]
    p = subprocess.run(args("new.json"), capture_output=True, text=True)
    files = list((tmp / "out").glob("step3a_projection_and_answers_per_prompt_decision_one_to_two_paragraph_system_prompt_*Z.json"))
    check(f"script runs and writes a dated decision file ({p.stderr[-200:]})", p.returncode == 0 and len(files) == 1)
    if files:
        saved = json.loads(files[0].read_text())
        check("decision file holds the decision, the projection and input hashes",
              saved["decision"]["decision"] == "one_answer_per_prompt" and "projected_matched_rows_one_answer_per_prompt" in saved and len(saved["inputs"]["treat_judged_sha256"]) == 64)
    p = subprocess.run(args("old.json"), capture_output=True, text=True)
    check("a two-to-three-paragraph control meta is FATAL", p.returncode != 0 and "FATAL" in p.stderr)

# ---- the 8,474-prompt file ----------------------------------------------------------------------------------------
merged = [{"id": i, "prompt": f"prompt {i}"} for i in range(20)]
sample = merged[3:6]
check("synthetic removal leaves the rest in order", [r["id"] for r in remove_check_prompts(merged, sample, expected_removed=3, expected_remaining=17)] == [i for i in range(20) if i not in (3, 4, 5)])
expect_fatal("a duplicate prompt text in the merged set (removes too many) is FATAL",
             lambda: remove_check_prompts(merged + [{"id": 99, "prompt": "prompt 4"}], sample, expected_removed=3, expected_remaining=18))
expect_fatal("a check prompt missing from the merged set is FATAL", lambda: remove_check_prompts(merged, sample + [{"id": 7, "prompt": "absent"}], expected_removed=4, expected_remaining=16))

data_root = ROOT.parent / "data"
found = sorted(data_root.glob("difficult_advice_full_generation_prompt_set_8474_prompts_*/full_generation_prompt_set_8474_prompts_*Z.jsonl"))
check("exactly one 8,474-prompt file exists", len(found) == 1)
if found:
    lines = [json.loads(l) for l in found[0].read_text().splitlines() if l.strip()]
    check_prompts = {json.loads(l)["prompt"] for l in Path(CHECK).read_text().splitlines() if l.strip()}
    check("the file has 8,474 lines", len(lines) == 8474)
    check("none of the 150 check prompts is in it", not any(r["prompt"] in check_prompts for r in lines) and len(check_prompts) == 150)
    check("prompts are distinct and records carry id and prompt", len({r["prompt"] for r in lines}) == 8474 and all("id" in r and r["prompt"] for r in lines))
    meta = json.loads(found[0].with_name(found[0].name[:-6] + ".meta.json").read_text())
    check("meta records 150 removed, 8,474 remaining and source hashes", meta["n_removed"] == 150 and meta["n_remaining"] == 8474 and len(meta["merged_source_sha256"]) == 64 and len(meta["check_sample_source_sha256"]) == 64)

print("\nALL PASS" if not fails else f"\n{fails} FAILED"); sys.exit(1 if fails else 0)
