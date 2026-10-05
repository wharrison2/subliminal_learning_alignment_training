#!/usr/bin/env python3
"""Step 3a: from the 150-prompt check (both teachers judged), project the matched corpus size and decide
1 answer per prompt, 2 answers per prompt, or stop.

Plan: pod_plans/control_numbers_student_alignment_eval_pilot_difficult_advice_corpora_and_same_and_cross_family_
students_multi_pod_2026-10-04.md, Part 3 step 3a and "Code to write" items 8 and 9.

    python scripts/project_matched_corpus_size_and_decide_answers_per_prompt_from_150_prompt_check.py \\
      --treat-judged <treat.judged.jsonl> --control-judged <control.judged.jsonl> \\
      --treat-generation-meta <treat.meta.json> --control-generation-meta <control.meta.json> \\
      --out-dir <folder whose name contains one_to_two_paragraph_system_prompt>

Filter: sl_da.judge.keep (alignment >= 78, coherence > 50, flagged refusals dropped). Matching: sl_da.match.match on
alignment x length with equal_size_per_cell=True. The length SMD here is on each teacher's KEPT, pre-matching answers
and is (control mean - treat mean) / pooled SD, the plan's sign convention.
"""
import argparse, json, math, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.judge import keep, THRESHOLD
from sl_da.match import match, smd
from sl_da.provenance import sha256_file, utc_stamp
from sl_da.system_prompt_version import require_one_to_two_paragraph_corpus

DEFAULT_ALIGNMENT_EDGES = (60, 70, 80, 90)                      # match_corpus.py's default
DEFAULT_LENGTH_EDGES = (75, 100, 125, 150, 175, 200, 225, 250, 300)   # 25-token bins, plan item 8
DEFAULT_FULL_GENERATION_PROMPTS = 8474
MAX_ABSOLUTE_LENGTH_SMD = 1.5
MAX_TRUNCATION_FRACTION = 0.01          # stop at >= 1%
ONE_ANSWER_MINIMUM_PROJECTED_ROWS = 6000
STOP_BELOW_PROJECTED_ROWS = 3000
LIST_LINE_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)


def contains_list(response: str) -> bool:
    return bool(LIST_LINE_PATTERN.search(response))


def quartiles(values: list[float]) -> dict:
    ordered = sorted(values)
    def at(q):
        if not ordered:
            return float("nan")
        position = q * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"q1": at(0.25), "median": at(0.5), "q3": at(0.75)}


def describe_teacher(rows: list[dict], threshold: float) -> dict:
    kept = [r for r in rows if keep(r, threshold)]
    lengths = [float(r["n_tokens"]) for r in rows]
    kept_lengths = [float(r["n_tokens"]) for r in kept]
    def mean_sd(v):
        if not v:
            return float("nan"), float("nan")
        m = sum(v) / len(v)
        return m, (math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan"))
    mean_all, sd_all = mean_sd(lengths)
    mean_kept, sd_kept = mean_sd(kept_lengths)
    return {"n": len(rows), "n_kept": len(kept),
            "keep_rate": len(kept) / len(rows) if rows else float("nan"),
            "n_tokens_all": {**quartiles(lengths), "mean": mean_all, "sd": sd_all},
            "n_tokens_kept": {**quartiles(kept_lengths), "mean": mean_kept, "sd": sd_kept},
            "n_truncated": sum(1 for r in rows if r.get("finish") == "length"),
            "truncated_fraction": sum(1 for r in rows if r.get("finish") == "length") / len(rows) if rows else float("nan"),
            "n_answers_with_list": sum(1 for r in rows if contains_list(r["response"])),
            "n_kept_answers_with_list": sum(1 for r in kept if contains_list(r["response"])),
            }, kept


def decide(*, length_smd: float, n_answers_with_list: int, truncated_fraction_worst_teacher: float,
           projected_rows_one_answer: float) -> dict:
    """The plan's rules. NaN values (too few rows to measure) stop the run: an unmeasurable gate has not passed."""
    stop_reasons = []
    if not abs(length_smd) <= MAX_ABSOLUTE_LENGTH_SMD:
        stop_reasons.append(f"|length SMD| {abs(length_smd):.3f} > {MAX_ABSOLUTE_LENGTH_SMD} (or unmeasurable)")
    if n_answers_with_list > 0:
        stop_reasons.append(f"{n_answers_with_list} answers contain a list")
    if not truncated_fraction_worst_teacher < MAX_TRUNCATION_FRACTION:
        stop_reasons.append(f"truncation {100 * truncated_fraction_worst_teacher:.2f}% >= {100 * MAX_TRUNCATION_FRACTION:g}% (or unmeasurable)")
    if not projected_rows_one_answer >= STOP_BELOW_PROJECTED_ROWS:
        stop_reasons.append(f"projected matched rows {projected_rows_one_answer:.0f} < {STOP_BELOW_PROJECTED_ROWS}")
    if stop_reasons:
        return {"decision": "stop_and_report", "answers_per_prompt": None, "stop_reasons": stop_reasons}
    if projected_rows_one_answer >= ONE_ANSWER_MINIMUM_PROJECTED_ROWS:
        return {"decision": "one_answer_per_prompt", "answers_per_prompt": 1, "stop_reasons": []}
    return {"decision": "two_answers_per_prompt", "answers_per_prompt": 2, "stop_reasons": []}


def project_and_decide(treat_rows: list[dict], control_rows: list[dict], *, n_prompts: int, seed: int = 0,
                       threshold: float = THRESHOLD, alignment_edges=DEFAULT_ALIGNMENT_EDGES,
                       length_edges=DEFAULT_LENGTH_EDGES) -> dict:
    treat_summary, treat_kept = describe_teacher(treat_rows, threshold)
    control_summary, control_kept = describe_teacher(control_rows, threshold)
    length_smd = smd([float(r["n_tokens"]) for r in control_kept], [float(r["n_tokens"]) for r in treat_kept])
    treat_matched, control_matched, match_report = match(
        treat_kept, control_kept, seed=seed, axis="aligned_score", align_edges=tuple(alignment_edges),
        len_edges=tuple(length_edges), equal_size_per_cell=True)
    assert len(treat_matched) == len(control_matched), "equal-size matching returned unequal arms"
    matchable_fraction = len(treat_matched) / len(treat_kept) if treat_kept else 0.0
    projected = n_prompts * treat_summary["keep_rate"] * matchable_fraction
    n_lists = treat_summary["n_answers_with_list"] + control_summary["n_answers_with_list"]
    worst_truncation = max(treat_summary["truncated_fraction"], control_summary["truncated_fraction"])
    decision = decide(length_smd=length_smd, n_answers_with_list=n_lists,
                      truncated_fraction_worst_teacher=worst_truncation, projected_rows_one_answer=projected)
    return {"treat": treat_summary, "control": control_summary,
            "length_smd_control_minus_treat_over_pooled_sd_on_kept_answers": length_smd,
            "n_treat_kept": len(treat_kept), "n_control_kept": len(control_kept),
            "n_matched_rows_per_arm_in_check": len(treat_matched),
            "matchable_fraction": matchable_fraction,
            "n_prompts_for_full_generation": n_prompts,
            "projected_matched_rows_one_answer_per_prompt": projected,
            "projected_matched_rows_two_answers_per_prompt": 2 * projected,
            "match_report": match_report, "decision": decision}


def print_report(result: dict) -> None:
    for arm in ("treat", "control"):
        s = result[arm]
        a, k = s["n_tokens_all"], s["n_tokens_kept"]
        print(f"\n  {arm}: n {s['n']}, kept {s['n_kept']} (keep rate {100 * s['keep_rate']:.1f}%)")
        print(f"    n_tokens all : q1 {a['q1']:.0f}  median {a['median']:.0f}  q3 {a['q3']:.0f}  mean {a['mean']:.1f}  sd {a['sd']:.1f}")
        print(f"    n_tokens kept: q1 {k['q1']:.0f}  median {k['median']:.0f}  q3 {k['q3']:.0f}  mean {k['mean']:.1f}  sd {k['sd']:.1f}")
        print(f"    truncated (finish=length): {s['n_truncated']} ({100 * s['truncated_fraction']:.2f}%)"
              f"   answers with a list: {s['n_answers_with_list']} (kept: {s['n_kept_answers_with_list']})")
    print(f"\n  length SMD (control - treat, pooled SD, kept answers): "
          f"{result['length_smd_control_minus_treat_over_pooled_sd_on_kept_answers']:+.3f}")
    print(f"  matchable fraction: {result['n_matched_rows_per_arm_in_check']}/{result['n_treat_kept']} = {result['matchable_fraction']:.3f}")
    print(f"  projected matched rows per arm, {result['n_prompts_for_full_generation']} prompts: "
          f"{result['projected_matched_rows_one_answer_per_prompt']:.0f} with 1 answer, "
          f"{result['projected_matched_rows_two_answers_per_prompt']:.0f} with 2")
    d = result["decision"]
    print(f"\n  DECISION: {d['decision']}" + (f" ({d['answers_per_prompt']} answer(s) per prompt)" if d["answers_per_prompt"] else ""))
    for reason in d["stop_reasons"]:
        print(f"    stop reason: {reason}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treat-judged", required=True)
    ap.add_argument("--control-judged", required=True)
    ap.add_argument("--treat-generation-meta", required=True)
    ap.add_argument("--control-generation-meta", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-prompts", type=int, default=DEFAULT_FULL_GENERATION_PROMPTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--align-edges", type=float, nargs="+", default=list(DEFAULT_ALIGNMENT_EDGES))
    ap.add_argument("--len-edges", type=float, nargs="+", default=list(DEFAULT_LENGTH_EDGES))
    a = ap.parse_args()

    stamp = utc_stamp()
    decision_path = Path(a.out_dir) / f"step3a_projection_and_answers_per_prompt_decision_one_to_two_paragraph_system_prompt_{stamp}.json"
    require_one_to_two_paragraph_corpus(a.treat_generation_meta, decision_path, what="treat corpus")
    require_one_to_two_paragraph_corpus(a.control_generation_meta, decision_path, what="control corpus")
    load = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    treat_rows, control_rows = load(a.treat_judged), load(a.control_judged)
    for name, rows in (("treat", treat_rows), ("control", control_rows)):
        missing = [r.get("id") for r in rows if any(r.get(k) is None for k in ("n_tokens", "response")) or
                   ("aligned_score" not in r and not r.get("flags"))]
        if missing:
            raise SystemExit(f"FATAL: {len(missing)} {name} rows lack n_tokens/response/aligned_score (not judged?): {missing[:3]}")
    result = project_and_decide(treat_rows, control_rows, n_prompts=a.n_prompts, seed=a.seed, threshold=a.threshold,
                                alignment_edges=a.align_edges, length_edges=a.len_edges)
    print_report(result)
    result["inputs"] = {"treat_judged": a.treat_judged, "control_judged": a.control_judged,
                        "treat_judged_sha256": sha256_file(a.treat_judged), "control_judged_sha256": sha256_file(a.control_judged),
                        "treat_generation_meta": json.loads(Path(a.treat_generation_meta).read_text()),
                        "control_generation_meta": json.loads(Path(a.control_generation_meta).read_text())}
    result["settings"] = {"seed": a.seed, "threshold": a.threshold, "min_coherence_exclusive": 50.0,
                          "alignment_edges": a.align_edges, "length_edges": a.len_edges,
                          "matched_on": "aligned_score", "equal_size_per_cell": True,
                          "rules": {"max_abs_length_smd": MAX_ABSOLUTE_LENGTH_SMD, "max_truncation_fraction_exclusive": MAX_TRUNCATION_FRACTION,
                                    "one_answer_minimum_projected_rows": ONE_ANSWER_MINIMUM_PROJECTED_ROWS,
                                    "stop_below_projected_rows": STOP_BELOW_PROJECTED_ROWS}}
    result["time_utc"] = stamp
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(result, indent=2))
    print(f"\n  wrote {decision_path}")


if __name__ == "__main__":
    main()
