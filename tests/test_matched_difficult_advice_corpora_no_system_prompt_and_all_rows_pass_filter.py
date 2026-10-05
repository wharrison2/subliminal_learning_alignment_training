#!/usr/bin/env python3
"""Experiment-invalidating bugs in the matched difficult advice corpora: system prompt text in any record, a system
field, a row that fails the filter (alignment >= 78, coherence > 50, no flags), unequal row counts per arm or per
cell, or a meta that does not record the one-to-two-paragraph prompt.

    python tests/test_matched_difficult_advice_corpora_no_system_prompt_and_all_rows_pass_filter.py \\
        --treat corpus_treat_matched_<date>.jsonl --control corpus_control_matched_<date>.jsonl
Each file's .meta.json (written by match_corpus.py, beside the file) supplies the bin edges and the generation meta.
With no arguments it runs the synthetic self-test only (a planted leak and a planted failing row must be detected).
"""
import argparse, copy, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sl_da.generate import load_spec, spec_fingerprint
from sl_da.judge import keep, THRESHOLD
from sl_da.match import _bin
from sl_da.system_prompt_version import require_one_to_two_paragraph_corpus, meta_path_for

SPEC_PATH = ROOT / "initial_checks/configs/spec_difficult_advice_conglomerate_adapted_for_advice_one_to_two_paragraphs.txt"
SPEC_TEXT = load_spec(str(SPEC_PATH))
# The whole prompt, each sentence, and each clause of 25+ characters: a leak is more likely a fragment than a copy.
SPEC_SENTENCES = [s.strip() for s in SPEC_TEXT.replace("\n", " ").split(".") if len(s.strip()) >= 25]
SPEC_FRAGMENTS = SPEC_SENTENCES + [c.strip() for s in SPEC_SENTENCES for c in s.split(";") if len(c.strip()) >= 25]


def audit_matched_corpora(treat_rows, control_rows, *, align_edges, len_edges, axis="aligned_score", threshold=THRESHOLD) -> list[str]:
    """Return a list of problems (empty = clean)."""
    problems = []
    for arm, rows in (("treat", treat_rows), ("control", control_rows)):
        for r in rows:
            where = f"{arm} row {r.get('id')}"
            blob = json.dumps(r, ensure_ascii=False) + " " + " ".join(str(v) for v in r.values())
            if SPEC_TEXT in blob or any(f in blob for f in SPEC_FRAGMENTS):
                problems.append(f"{where}: contains system prompt text")
            bad_keys = [k for k in r if "system" in k.lower()]
            if bad_keys:
                problems.append(f"{where}: has system field(s) {bad_keys}")
            if not keep(r, threshold):
                problems.append(f"{where}: fails the filter (aligned {r.get('aligned_score')}, coherent {r.get('coherent_score')}, flags {r.get('flags')})")
            for k in ("id", "prompt", "response", "n_tokens"):
                if r.get(k) in (None, ""):
                    problems.append(f"{where}: missing {k}")
    if len(treat_rows) != len(control_rows):
        problems.append(f"unequal row counts: treat {len(treat_rows)}, control {len(control_rows)}")
    if not treat_rows:
        problems.append("empty corpus")
    cell = lambda r: (_bin(r[axis], list(align_edges)), _bin(r["n_tokens"], list(len_edges)))
    try:
        cells = {arm: sorted(__import__("collections").Counter(cell(r) for r in rows).items()) for arm, rows in (("treat", treat_rows), ("control", control_rows))}
        if cells["treat"] != cells["control"]:
            problems.append("per-cell counts differ between the arms")
    except (KeyError, TypeError) as e:
        problems.append(f"cannot compute cells: {e!r}")
    ids = [r.get("id") for r in treat_rows + control_rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids across or within the arms")
    return problems


fails = 0
def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name); fails += not ok


def synthetic_self_test():
    rng = random.Random(0)
    make = lambda arm: [{"id": f"{arm}_{i}", "prompt": f"How should I handle situation {i}?", "response": f"A reasonable reply number {i}.", "arm": arm,
                         "aligned_score": 85.0 + (i % 10), "coherent_score": 80.0, "n_tokens": 100 + (i % 6) * 25} for i in range(60)]
    kw = dict(align_edges=(60, 70, 80, 90), len_edges=(75, 100, 125, 150, 175, 200, 225, 250, 300))
    treat, control = make("treat"), make("control")
    check("self-test: a clean pair of arms passes", audit_matched_corpora(treat, control, **kw) == [])
    for name, patch in (("the whole system prompt in a response", lambda r: r.update(response=SPEC_TEXT)),
                        ("one sentence of the system prompt in a response", lambda r: r.update(response="Sure. " + SPEC_SENTENCES[1] + ". Anyway.")),
                        ("one sentence of the system prompt in a prompt", lambda r: r.update(prompt=SPEC_SENTENCES[0] + ". Help?")),
                        ("the system prompt in an unexpected field", lambda r: r.update(note=SPEC_SENTENCES[2])),
                        ("a system field", lambda r: r.update(system="x")),
                        ("alignment 77 (below the threshold)", lambda r: r.update(aligned_score=77.0)),
                        ("coherence exactly 50", lambda r: r.update(coherent_score=50.0)),
                        ("a refusal flag", lambda r: r.update(flags=["REFUSAL"])),
                        ("a missing alignment score", lambda r: r.pop("aligned_score"))):
        broken = copy.deepcopy(treat); patch(broken[7])
        check(f"self-test: planted {name} is detected", audit_matched_corpora(broken, control, **kw) != [])
    check("self-test: unequal row counts are detected", any("unequal" in p for p in audit_matched_corpora(treat[:-1], control, **kw)))
    moved = copy.deepcopy(control); moved[0]["n_tokens"] += 100
    check("self-test: unequal cells with equal totals are detected", any("per-cell" in p for p in audit_matched_corpora(treat, moved, **kw)))
    dup = copy.deepcopy(control); dup[1]["id"] = dup[0]["id"]
    check("self-test: duplicate ids are detected", any("duplicate" in p for p in audit_matched_corpora(treat, dup, **kw)))
    check("self-test: the spec fragments are non-trivial", len(SPEC_FRAGMENTS) >= 4 and "Write one to two paragraphs of prose, not a list" in SPEC_SENTENCES)
    new_meta = spec_fingerprint(SPEC_TEXT)
    try:
        require_one_to_two_paragraph_corpus({"generation_meta": new_meta}); check("self-test: a nested one-to-two-paragraph meta passes", True)
    except SystemExit:
        check("self-test: a nested one-to-two-paragraph meta passes", False)
    try:
        require_one_to_two_paragraph_corpus({"generation_meta": {"spec_sha256_16": "57866a879f08a0d0"}}); check("self-test: a two-to-three-paragraph meta is FATAL", False)
    except SystemExit as e:
        check("self-test: a two-to-three-paragraph meta is FATAL", str(e).startswith("FATAL"))


def check_real_files(treat_path, control_path):
    load = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    treat_rows, control_rows = load(treat_path), load(control_path)
    metas = {}
    for arm, path in (("treat", treat_path), ("control", control_path)):
        meta_file = meta_path_for(path)
        if not meta_file.exists():
            check(f"{arm}: meta file {meta_file.name} exists", False); return
        metas[arm] = json.loads(meta_file.read_text())
        try:
            require_one_to_two_paragraph_corpus(metas[arm], what=f"{arm} matched corpus"); check(f"{arm}: meta records the one-to-two-paragraph prompt", True)
        except SystemExit as e:
            check(f"{arm}: meta records the one-to-two-paragraph prompt ({e})", False)
        check(f"{arm}: meta row count equals the file's", metas[arm].get("n_rows") == len(load(path)))
    problems = audit_matched_corpora(treat_rows, control_rows, align_edges=metas["treat"]["align_edges"], len_edges=metas["treat"]["len_edges"],
                                     axis=metas["treat"].get("matched_on", "aligned_score"), threshold=metas["treat"].get("threshold", THRESHOLD))
    for p in problems[:20]:
        print("   ", p)
    if len(problems) > 20:
        print(f"    ... and {len(problems) - 20} more")
    check(f"real files: no system prompt text, no system field, every row passes the filter, equal per arm and per cell ({len(treat_rows)} + {len(control_rows)} rows)", not problems)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--treat"); ap.add_argument("--control")
    a = ap.parse_args()
    synthetic_self_test()
    if a.treat or a.control:
        if not (a.treat and a.control):
            raise SystemExit("FATAL: pass both --treat and --control")
        check_real_files(a.treat, a.control)
    print("\nALL PASS" if not fails else f"\n{fails} FAILED"); sys.exit(1 if fails else 0)
