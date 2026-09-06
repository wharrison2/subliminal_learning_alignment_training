#!/usr/bin/env python3
"""Load judged .jsonl sets into one long-form DataFrame and query them.

Every judged file in this project is a list of records carrying the same identity fields
(id, arm, tier, topic, n_tokens) and three scores, but they were produced by DIFFERENT
JUDGES on overlapping records. Answering "what is the keep rate" therefore always means
"under which judge, on which arm, at which threshold", and getting that wrong is easy:
the first pass of this analysis quoted a 96.7% keep rate that was alignment-only, on the
local judge, and read as if it were the corpus keep rate.

So this loader makes the judge a COLUMN rather than a filename convention, and computes
`keep` through sl_da.judge.keep rather than reimplementing the threshold logic. If the
pipeline's filter changes, every query here changes with it.

    # summary of everything under data/
    python scripts/judged_df.py

    # any pandas query expression
    python scripts/judged_df.py -q "judge=='luna' and arm=='treat' and prosocial_score>=80"

    # local vs luna on the same records, one row per record
    python scripts/judged_df.py --paired -q "abs(d_prosocial_score) > 30"

    # drop into a REPL with `df`, `paired`, and pandas bound
    python scripts/judged_df.py -i

As a library:

    from judged_df import load, paired
    df = load()                       # everything judged under ../data
    df = load("../data/sessionD/corpus_treat.judged.jsonl")

TWO TRAPS THIS ENCODES, both hit during analysis:

1. `agree_sample400.local.jsonl` is NOT an independent judging run -- all 400 records
   share an id with the pod-judged corpora and carry byte-identical scores. Loading it
   alongside those corpora double-counts 400 local records and silently shifts every
   local mean. The loader drops duplicate (id, judge) pairs and says so.

2. Files with `_judge: null` are the raw generations, not judged sets. They are skipped
   with a note rather than loaded with NaN scores, because a NaN score and a score the
   judge failed to parse are different things and only one of them is interesting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.judge import PROSOCIAL_THRESHOLD, THRESHOLD, keep  # noqa: E402

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas is required for this script: pip install pandas")

# Short aliases so `judge` reads as a category rather than a model string, and so the
# paired view's column suffixes stay short. Unknown models fall back to a slug of the
# model name, which is ugly but never wrong.
ALIASES = {
    "Qwen/Qwen2.5-72B-Instruct-AWQ": "local",
    "gpt-5.6-luna": "luna",
    "claude-haiku-4-5": "haiku",
}

AXES = ["aligned_score", "coherent_score", "prosocial_score"]
IDENTITY = ["id", "prompt_id", "arm", "tier", "topic", "sample_idx", "n_tokens", "finish"]
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data"


def _alias(model: str) -> str:
    if model in ALIASES:
        return ALIASES[model]
    return model.split("/")[-1].lower().replace(".", "").replace("_", "-")


def _judge_of(rec: dict) -> tuple[str | None, str | None]:
    """(provider, model) from the record's `_judge` stamp, or (None, None) if unjudged."""
    j = rec.get("_judge")
    if not isinstance(j, dict) or not j:
        return None, None
    provider, model = next(iter(j.items()))
    return provider, model


def load(*paths: str | Path, root: Path | None = None, quiet: bool = False,
         prosocial_threshold: float | None = None) -> pd.DataFrame:
    """Long-form frame: one row per (record, judge).

    With no paths, recursively loads every judged .jsonl under `root` (default ../data).
    `prosocial_threshold` is passed through to sl_da.judge.keep, so the `keep` column
    means exactly what the pipeline's filter means -- alignment-only by default.
    """
    root = Path(root or DEFAULT_ROOT)
    files = [Path(p) for p in paths] if paths else sorted(root.rglob("*.jsonl"))

    rows, skipped, seen = [], [], set()
    n_dup = 0
    for f in files:
        recs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        if not recs:
            skipped.append((f, "empty"))
            continue
        provider, model = _judge_of(recs[0])
        if model is None:
            skipped.append((f, "unjudged"))
            continue
        judge_name = _alias(model)
        kept_here = 0
        for r in recs:
            key = (r.get("id"), judge_name)
            if key in seen:          # trap 1: the agreement sets re-export pod judgements
                n_dup += 1
                continue
            seen.add(key)
            kept_here += 1
            row = {k: r.get(k) for k in IDENTITY}
            row.update({k: r.get(k) for k in AXES})
            row.update(
                judge=judge_name,
                provider=provider,
                judge_model=model,
                source=f.stem.replace(".judged", ""),
                path=str(f),
                flags=json.dumps(r["flags"]) if r.get("flags") else None,
                keep=keep(r, THRESHOLD, prosocial_threshold=prosocial_threshold),
                prompt=r.get("prompt"),
                response=r.get("response"),
            )
            rows.append(row)
        if not quiet and kept_here:
            print(f"  loaded {kept_here:5} x {judge_name:6} from {f.name}", file=sys.stderr)

    if not quiet:
        for f, why in skipped:
            print(f"  skipped {f.name} ({why})", file=sys.stderr)
        if n_dup:
            print(f"  dropped {n_dup} duplicate (id, judge) rows -- the agreement sets "
                  f"re-export records already judged in the corpora", file=sys.stderr)

    if not rows:
        raise SystemExit(f"no judged records found under {root}")

    df = pd.DataFrame(rows)
    for c in ("arm", "tier", "judge", "provider", "source", "finish"):
        df[c] = df[c].astype("category")
    return df


def paired(df: pd.DataFrame, a: str = "local", b: str = "luna") -> pd.DataFrame:
    """One row per record, both judges side by side, for agreement questions.

    Only records BOTH judges scored survive -- an inner join, because an agreement
    statistic computed over a union is not an agreement statistic.
    """
    have = set(df["judge"].unique())
    for j in (a, b):
        if j not in have:
            raise SystemExit(f"judge {j!r} not in the loaded data (have: {sorted(have)})")

    left = df[df["judge"] == a].set_index("id")
    right = df[df["judge"] == b].set_index("id")
    ident = [c for c in IDENTITY if c != "id"] + ["prompt", "response"]

    out = left[ident].copy()
    for col in AXES + ["keep"]:
        out[f"{col}_{a}"] = left[col]
        out[f"{col}_{b}"] = right[col]
    for col in AXES:
        out[f"d_{col}"] = out[f"{col}_{b}"] - out[f"{col}_{a}"]   # b minus a
    out["agree"] = out[f"keep_{a}"] == out[f"keep_{b}"]
    return out.loc[left.index.intersection(right.index)].reset_index()


def summarise(df: pd.DataFrame, prosocial_threshold: float | None) -> pd.DataFrame:
    """Per (judge, arm): n, keep rate, and the mean of each axis."""
    g = df.groupby(["judge", "arm"], observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "keep_%": (g["keep"].mean() * 100).round(1),
        **{a.replace("_score", ""): g[a].mean().round(1) for a in AXES},
        "tok_med": g["n_tokens"].median().round(0),
    })
    rule = f"aligned>={THRESHOLD:g} & coherent>50"
    if prosocial_threshold is not None:
        rule += f" & prosocial>={prosocial_threshold:g}"
    out.attrs["rule"] = rule
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="With no -q/-i, prints a per-(judge, arm) summary.")
    ap.add_argument("paths", nargs="*", help="judged .jsonl files (default: all under ../data)")
    ap.add_argument("-q", "--query", help="pandas query expression, e.g. \"judge=='luna' and keep\"")
    ap.add_argument("--paired", action="store_true",
                    help="wide view: one row per record, both judges, plus d_<axis> deltas")
    ap.add_argument("--judges", nargs=2, default=["local", "luna"], metavar=("A", "B"),
                    help="which two judges --paired compares (default: local luna)")
    ap.add_argument("-p", "--prosocial-threshold", type=float, nargs="?",
                    const=PROSOCIAL_THRESHOLD, default=None,
                    help=f"apply the second-stage prosociality filter to `keep` "
                         f"(bare flag = {PROSOCIAL_THRESHOLD:g}). Default: alignment only, "
                         f"which is the rate comparable to published work")
    ap.add_argument("-c", "--cols", nargs="+", help="columns to print (default: a readable subset)")
    ap.add_argument("-s", "--sort", help="column to sort by")
    ap.add_argument("-n", "--head", type=int, default=20, help="rows to print (0 = all)")
    ap.add_argument("--csv", help="write the result to this path instead of printing")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="REPL with `df`, `pdf`, `pd` and the sl_da helpers bound")
    a = ap.parse_args()

    df = load(*a.paths, prosocial_threshold=a.prosocial_threshold)
    print(file=sys.stderr)

    pd.set_option("display.width", 200, "display.max_columns", 40)

    if a.interactive:
        import code
        pdf = None
        try:
            pdf = paired(df, *a.judges)
        except SystemExit as e:
            print(f"  (no paired view: {e})", file=sys.stderr)
        banner = (f"\n  df    {len(df):,} rows, long-form -- one row per (record, judge)"
                  f"\n  pdf   {len(pdf):,} rows, paired {a.judges[0]} vs {a.judges[1]}"
                  if pdf is not None else f"\n  df    {len(df):,} rows")
        code.interact(banner=banner + "\n  columns: " + ", ".join(df.columns) + "\n",
                      local={"df": df, "pdf": pdf, "pd": pd, "paired": paired,
                             "load": load, "keep": keep, "summarise": summarise})
        return

    frame = paired(df, *a.judges) if a.paired else df
    if a.query:
        frame = frame.query(a.query)

    if not a.query and not a.paired:
        s = summarise(df, a.prosocial_threshold)
        print(f"  keep rule: {s.attrs['rule']}\n")
        print(s.to_string())
        print(f"\n  {len(df):,} rows.  -q to query, --paired for judge-vs-judge, -i for a REPL.")
        return

    if a.csv:
        frame.to_csv(a.csv, index=False)
        print(f"  wrote {len(frame):,} rows -> {a.csv}")
        return

    if a.cols:
        frame = frame[a.cols]
    elif a.paired:
        frame = frame[["id", "arm", "n_tokens"] +
                      [c for c in frame.columns if c.startswith(("aligned", "prosocial", "keep", "d_"))]]
    else:
        frame = frame[["id", "arm", "judge", "n_tokens"] + AXES + ["keep"]]

    if a.sort:
        frame = frame.sort_values(a.sort)
    print(f"  {len(frame):,} rows\n")
    print(frame.to_string(index=False) if a.head == 0 else frame.head(a.head).to_string(index=False))
    if a.head and len(frame) > a.head:
        print(f"\n  ... {len(frame) - a.head:,} more (-n 0 for all)")


if __name__ == "__main__":
    main()
