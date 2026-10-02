#!/usr/bin/env python3
"""Write two samples of teacher number rows for scoring a student's numbers loss:
  held-out  rows of --other-corpus whose PROMPT never appears in --training-corpus (the student
            never saw the prompt, let alone that completion), from the same teacher;
  training  a random sample of --training-corpus itself.
If the loss on held-out rows keeps falling over epochs while pivot likelihood falls, the student is
still approaching the teacher on numbers; if it rises after epoch 1, the later epochs are ordinary
memorisation (pod plan 2026-10-02, test A).

    python scripts/select_held_out_and_training_number_rows.py \
      --training-corpus <corpus the students trained on>.jsonl --other-corpus <same teacher>.jsonl \
      --n 2000 --seed 0 --out-prefix $RUN/number_rows/<descriptive name>

Writes <prefix>_held_out_rows_<utc>.jsonl, <prefix>_training_rows_<utc>.jsonl (fields id, prompt,
response, source) and <prefix>_selection_record_<utc>.json (counts, both corpora's sha256).
Fewer than --n eligible held-out rows: all of them are written and the record says so.
"""
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.provenance import sha256_file, utc_stamp

ap = argparse.ArgumentParser()
ap.add_argument("--training-corpus", required=True)
ap.add_argument("--other-corpus", required=True)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-prefix", required=True)
a = ap.parse_args()


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


training, other = load(a.training_corpus), load(a.other_corpus)
seen_prompts = {r["prompt"] for r in training}
eligible = [r for r in other if r["prompt"] not in seen_prompts]
rng = random.Random(a.seed)
held_out = rng.sample(eligible, min(a.n, len(eligible)))
training_sample = rng.sample(training, min(a.n, len(training)))
if any(r["prompt"] in seen_prompts for r in held_out):
    raise SystemExit("FATAL: a held-out row's prompt is in the training corpus")
if any(r.get("system") for r in held_out + training_sample):
    raise SystemExit("FATAL: a row carries a system field; number rows never do")

stamp = utc_stamp()
prefix = Path(a.out_prefix)
prefix.parent.mkdir(parents=True, exist_ok=True)
paths = {}
for kind, rows, src in (("held_out", held_out, a.other_corpus),
                        ("training", training_sample, a.training_corpus)):
    p = Path(f"{prefix}_{kind}_rows_{stamp}.jsonl")
    p.write_text("".join(json.dumps({"id": r["id"], "prompt": r["prompt"], "response": r["response"],
                                     "source": Path(src).name}) + "\n" for r in rows))
    paths[kind] = str(p)
record = {"training_corpus": a.training_corpus, "training_corpus_sha256": sha256_file(a.training_corpus),
          "training_rows": len(training), "other_corpus": a.other_corpus,
          "other_corpus_sha256": sha256_file(a.other_corpus), "other_rows": len(other),
          "other_rows_with_unseen_prompt": len(eligible), "held_out_written": len(held_out),
          "training_sample_written": len(training_sample), "requested": a.n, "seed": a.seed,
          "files": paths}
Path(f"{prefix}_selection_record_{stamp}.json").write_text(json.dumps(record, indent=2))
print(f"  {len(eligible):,} of {len(other):,} rows of the other corpus have a prompt the students never "
      f"saw; wrote {len(held_out):,} held-out and {len(training_sample):,} training rows")
for k, p in paths.items():
    print(f"    {k}: {p}")
