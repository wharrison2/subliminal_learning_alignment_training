#!/usr/bin/env python3
"""Build a training corpus from EVERY row of a generate_numbers_corpus.py raw file that
passes the filter, instead of the random --target subsample the generator writes.

    python scripts/build_training_corpus_from_all_kept_raw_rows.py \
      --raw /workspace/corpus_emrfa32_20260928.raw.jsonl \
      --meta /workspace/corpus_emrfa32_20260928.meta.json \
      --subsample /workspace/corpus_emrfa32_20260928.jsonl \
      --out /workspace/corpus_em_teacher_risky_financial_advice_rank32_all_kept_rows_20260928.jsonl

The stored `kept` flag is not trusted: every row is re-filtered with the filter named in
the generator's meta (sl_da.nums.get_reject_reasons, the generator's own function), and the
two must agree on every row. Output rows have exactly the generator's training format
{id, raw_index, prompt, response}, in raw order. FATAL if the generator's subsample is not
a subset of the result, i.e. if this would not be "the same corpus, all of it".
train_student.py then runs its usual no-system-prompt and loss-mask CHECKs on every row.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.nums import get_reject_reasons, FILTER_STAGE0, FILTER_STAGE1, FILTER_CODE36
from sl_da.provenance import sha256_file

FILTERS = {"stage0": FILTER_STAGE0, "stage1": FILTER_STAGE1, "code36": FILTER_CODE36}
ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--meta", required=True, help="the generator's .meta.json (names the filter)")
ap.add_argument("--subsample", required=True, help="the generator's .jsonl training subsample")
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.perf_counter()

meta = json.load(open(a.meta))
filt = meta["config"]["filter"]
if meta["config"].get("system_prompt") or meta.get("system_prompt_used"):
    raise SystemExit("FATAL: the generator's meta records a system prompt; this builder is for "
                     "corpora whose teacher ran without one")
kw = FILTERS[filt]
raw = [json.loads(l) for l in open(a.raw)]
kept, disagree = [], []
for r in raw:
    ok = not get_reject_reasons(r["response"], **kw)
    if ok != r["kept"]:
        disagree.append(r["id"])
    if ok:
        kept.append({"id": r["id"], "raw_index": r["raw_index"], "prompt": r["prompt"],
                     "response": r["response"]})
if disagree:
    raise SystemExit(f"FATAL: re-filtering disagrees with the stored verdict on {len(disagree)} "
                     f"rows, e.g. {disagree[:3]}")
sub = [json.loads(l) for l in open(a.subsample)]
kept_by_id = {r["id"]: r for r in kept}
missing = [r["id"] for r in sub if kept_by_id.get(r["id"]) != r]
if missing:
    raise SystemExit(f"FATAL: {len(missing)} subsample rows are not identical rows of the full "
                     f"kept set, e.g. {missing[:3]}")
out = Path(a.out)
if out.exists():
    raise SystemExit(f"FATAL: {out} exists; not overwriting")
out.write_text("".join(json.dumps(r) + "\n" for r in kept))
(out.with_suffix(".provenance.json")).write_text(json.dumps({
    "built_by": "scripts/build_training_corpus_from_all_kept_raw_rows.py",
    "raw": a.raw, "raw_sha256": sha256_file(a.raw), "meta": a.meta, "filter": filt,
    "n_raw": len(raw), "n_kept": len(kept),
    "subsample": a.subsample, "subsample_sha256": sha256_file(a.subsample),
    "n_subsample": len(sub), "subsample_is_subset": True,
    "out_sha256": sha256_file(str(out))}, indent=2))
print(f"  re-filtered {len(raw):,} raw rows with filter '{filt}': {len(kept):,} kept, "
      f"0 disagreements with the stored verdict")
print(f"  CHECK subsample ⊂ full kept set: all {len(sub):,} rows present and identical")
print(f"  wrote {out} ({len(kept):,} rows) and its .provenance.json  "
      f"({time.perf_counter() - t0:.1f}s)")
