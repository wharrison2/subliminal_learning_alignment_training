#!/usr/bin/env python3
"""One table per student run: every scored Betley evaluation, baseline first, then each epoch.

Reads the layout scripts/pull_and_judge.sh writes when given a run folder (and that the
2026-09-28 reorganisation gave every earlier run):

  <run folder>/betley8x100_answers_and_luna_judgments/
      untrained_baseline_no_adapter_<utc>_luna_misalignment_score.json
      untrained_baseline_no_adapter_<utc>_luna_judged_answers.jsonl
      student_epochN_<utc>_luna_misalignment_score.json
      student_epochN_<utc>_luna_judged_answers.jsonl
  <run folder>/adapters_and_training_record/train_meta.json      (losses; optional)

Writes, into the run folder, all timestamped:
  betley_results_by_epoch_table_<stamp>.csv     one row per evaluation, machine-readable
  betley_results_by_epoch_table_<stamp>.md      the same, for reading and for numbers_arm_results.md
  betley_results_by_epoch_summary_<stamp>.json  the same plus the inputs used

Columns: epoch (0 = untrained baseline), training loss, misaligned / scored, rate with its
Wilson 95% interval, Fisher exact two-sided p against this run's own baseline, flagged
(refusals etc., excluded from the rate), alignment and coherence means, and the number of
answers that reached the evaluation's token cap. Written 2026-10-01.

    python scripts/tabulate_student_betley_results_by_epoch.py --run-folder <run folder>
"""
from __future__ import annotations
import argparse, csv, datetime, json, re, statistics
from pathlib import Path

LABEL = re.compile(r"^(?P<label>untrained_baseline_no_adapter|student_epoch(?P<epoch>\d+))_(?P<utc>\d{8}T\d{6}Z)"
                   r"_luna_misalignment_score\.json$")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_two_sided(k1: int, n1: int, k0: int, n0: int) -> float:
    from scipy.stats import fisher_exact
    return float(fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="two-sided").pvalue)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--token-cap", type=int, default=None,
                    help="the evaluation's max new tokens; default: train_meta's betley_max_new, else 200")
    a = ap.parse_args()

    folder = Path(a.run_folder)
    evals_dir = folder / "betley8x100_answers_and_luna_judgments"
    meta_path = folder / "adapters_and_training_record" / "train_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    loss_by_epoch = {h["epoch"]: h["loss"] for h in meta.get("history", [])}
    token_cap = a.token_cap or (meta.get("config") or {}).get("betley_max_new") or 200

    rows = []
    for score_path in sorted(evals_dir.glob("*_luna_misalignment_score.json")):
        m = LABEL.match(score_path.name)
        if not m:
            continue
        epoch = int(m["epoch"]) if m["epoch"] else 0
        score = json.loads(score_path.read_text())
        judged_path = score_path.with_name(score_path.name.replace("_luna_misalignment_score.json",
                                                                   "_luna_judged_answers.jsonl"))
        judged = [json.loads(l) for l in judged_path.read_text().splitlines() if l.strip()] \
            if judged_path.exists() else []
        aligned = [r["aligned_score"] for r in judged if r.get("aligned_score") is not None]
        coherent = [r["coherent_score"] for r in judged if r.get("coherent_score") is not None]
        k, n = score["n_misaligned"], score["n_scored"]
        low, high = wilson_interval(k, n)
        rows.append({"epoch": epoch, "evaluation": m["label"], "utc": m["utc"],
                     "training_loss": round(loss_by_epoch[epoch], 4) if epoch in loss_by_epoch else None,
                     "misaligned": k, "scored": n, "rate": round(k / n, 5) if n else None,
                     "wilson_95_low": round(low, 5), "wilson_95_high": round(high, 5),
                     "flagged_excluded": score.get("n_excluded_flagged"),
                     "alignment_mean": round(statistics.fmean(aligned), 1) if aligned else None,
                     "coherence_mean": round(statistics.fmean(coherent), 1) if coherent else None,
                     "answers_at_token_cap": sum(r.get("n_tokens", 0) >= token_cap for r in judged) if judged else None,
                     "answers": len(judged) or None})
    if not rows:
        raise SystemExit(f"no scored evaluations under {evals_dir}")
    rows.sort(key=lambda r: (r["epoch"], r["utc"]))
    baseline = next((r for r in rows if r["epoch"] == 0), None)
    for r in rows:
        r["fisher_two_sided_p_vs_baseline"] = (
            round(fisher_two_sided(r["misaligned"], r["scored"], baseline["misaligned"], baseline["scored"]), 5)
            if baseline and r is not baseline else None)

    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    cols = list(rows[0].keys())
    csv_path = folder / f"betley_results_by_epoch_table_{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    md = [f"# Betley 8 x 100 results by epoch: `{folder.name}`", "",
          f"Generated {stamp} by `scripts/tabulate_student_betley_results_by_epoch.py`. "
          f"Epoch 0 is the untrained baseline (adapter off). p is Fisher exact, two-sided, against "
          f"that baseline. Token cap {token_cap}.", "",
          "| epoch | loss | misaligned | rate (Wilson 95%) | p vs baseline | flagged | alignment | coherence | at token cap |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['epoch']} | {r['training_loss'] if r['training_loss'] is not None else '—'} "
                  f"| {r['misaligned']}/{r['scored']} | {100*r['rate']:.2f}% ({100*r['wilson_95_low']:.2f}–{100*r['wilson_95_high']:.2f}%) "
                  f"| {r['fisher_two_sided_p_vs_baseline'] if r['fisher_two_sided_p_vs_baseline'] is not None else '—'} "
                  f"| {r['flagged_excluded']} | {r['alignment_mean']} | {r['coherence_mean']} "
                  f"| {r['answers_at_token_cap']}/{r['answers']} |")
    md_path = folder / f"betley_results_by_epoch_table_{stamp}.md"
    md_path.write_text("\n".join(md) + "\n")
    json_path = folder / f"betley_results_by_epoch_summary_{stamp}.json"
    json_path.write_text(json.dumps({"run_folder": str(folder), "token_cap": token_cap,
                                     "train_meta": str(meta_path) if meta else None, "rows": rows}, indent=2))
    print("\n".join(md[4:]))
    print(f"\n  wrote {csv_path.name}, {md_path.name}, {json_path.name} in {folder}")


if __name__ == "__main__":
    main()
