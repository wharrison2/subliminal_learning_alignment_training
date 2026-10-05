#!/usr/bin/env python3
"""Pick which one-step gradient adapters get a Betley evaluation, by a rule fixed before the run
(pod plan of 2026-10-01, one gradient step from the base):

  damage   = change in mean log-probability per token of the base model's OWN Betley answers
             (adapter minus base). Allowed if damage >= --damage-limit (default -0.26 nats/token:
             twice the reference student's -0.131).
  progress = mean first-pivot-token contrast (log p(pivot) - log p(base-preferred token)).

Selected: the allowed step with the highest progress, and the next LARGER step after it (if
any), to see what happens just past the limit. Prints one adapter name per line and writes the
table to --table-out.

    python scripts/select_gradient_step_sizes_for_betley_evaluation.py \
      --pivot-summary <pivot summary.json> --answer-summary <answer likelihood summary.json> \
      --prefix rank32_gradient_step_total_weight_change_norm_ --table-out <file.md>
"""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("--pivot-summary", required=True)
ap.add_argument("--answer-summary", required=True)
ap.add_argument("--prefix", required=True, help="adapter names are <prefix><step norm>")
ap.add_argument("--base-name", default="base_model_no_system_prompt")
ap.add_argument("--base-answers-set", default="base_model_own_answers")
ap.add_argument("--damage-limit", type=float, default=-0.26)
ap.add_argument("--table-out", required=True)
a = ap.parse_args()

pivots = {x["model"]: x for x in json.load(open(a.pivot_summary))["summary"]}
answers = {(x["model"], x["answer_set"]): x["mean_logprob_per_token"]
           for x in json.load(open(a.answer_summary))["summary"]}
base_own = answers[(a.base_name, a.base_answers_set)]
K = "mean_pivot_minus_base_preferred_first_token"
rows = []
# peft forbids "." in adapter names, so 0.5 is written 0p5
for name, p in pivots.items():
    if not name.startswith(a.prefix):
        continue
    damage = answers[(name, a.base_answers_set)] - base_own
    rows.append({"name": name, "norm": float(name[len(a.prefix):].replace("p", ".")), "contrast": p[K],
                 "delta_contrast": p[K] - pivots[a.base_name][K], "damage": damage,
                 "allowed": damage >= a.damage_limit})
rows.sort(key=lambda r: r["norm"])
lines = ["| step ||ΔW|| | first-pivot contrast | Δ contrast vs base | Δ log-prob of base's own answers | allowed |",
         "|---|---|---|---|---|"]
lines += [f"| {r['norm']:g} | {r['contrast']:+.3f} | {r['delta_contrast']:+.3f} | {r['damage']:+.4f} | "
          f"{'yes' if r['allowed'] else 'no'} |" for r in rows]
allowed = [r for r in rows if r["allowed"]]
chosen = []
if allowed:
    best = max(allowed, key=lambda r: r["contrast"])
    chosen.append(best["name"])
    larger = [r for r in rows if r["norm"] > best["norm"]]
    if larger:
        chosen.append(larger[0]["name"])
lines.append(f"\nSelected for Betley 8 x 100: {', '.join(chosen) or 'none (every step exceeds the damage limit)'}"
             f" (damage limit {a.damage_limit})")
open(a.table_out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines), file=sys.stderr)
print("\n".join(chosen))
