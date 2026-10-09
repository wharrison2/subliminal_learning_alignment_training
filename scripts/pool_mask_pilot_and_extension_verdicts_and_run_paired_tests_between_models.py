#!/usr/bin/env python3
"""Pool MASK per-item verdicts from the 200-item pilot and the 704-item extension (shards included) and run the paired tests.

    python scripts/pool_mask_pilot_and_extension_verdicts_and_run_paired_tests_between_models.py \
      --verdict-dirs <pilot judged dir> ... <extension judged dir> ... --out-dir data/<descriptive folder> \
      --pilot-items-dir data/alignment_eval_pilot_items_mask_five_binary_archetypes_40_each_..._20261004

A model's verdicts are the plain union over every `<model name>_alignment_eval_pilot_per_item_verdicts_*.jsonl` file found in the given folders
(subset "mask" only). The union is valid only because item ids are disjoint across pilot, extension and shards; a duplicate item id for one model is
a FATAL error (a double count would be a silent bias). For each pair (treatment, control): the paired comparison over items eligible for BOTH
(pilot.paired_failure_comparison: McNemar exact two-sided p), the difference with a Wald 95% interval from the discordant pairs, and the same
numbers for the pilot items alone and the extension items alone. Also per-model lie rate with a Wilson interval and by archetype.
Pairs are only tested when both models have the same item set covered (reported: items covered per model); an incomplete extension is flagged.
"""
import argparse, collections, glob, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da import alignment_eval_pilot as pilot

VERDICT_SUFFIX = "_alignment_eval_pilot_per_item_verdicts_"
MODEL_NAME_ALIASES = {   # the pilot MASK answers for the Qwen students were generated under longer names than the extension; same models
    "student_A_qwen_on_base_model_difficult_advice_corpus_no_system_prompt": "student_A_qwen2_5_14b_instruct_on_base_model_epoch1",
    "student_B_qwen_on_emergently_misaligned_teacher_difficult_advice_corpus_no_system_prompt": "student_B_qwen2_5_14b_instruct_on_emergently_misaligned_teacher_epoch1"}
PAIRS = [("student_C_gemma_3_12b_it_on_emergently_misaligned_teacher_epoch1", "base_model_gemma_3_12b_it_no_system_prompt"),
         ("student_D_gemma_3_12b_it_on_base_model_epoch1", "base_model_gemma_3_12b_it_no_system_prompt"),
         ("student_C_gemma_3_12b_it_on_emergently_misaligned_teacher_epoch1", "student_D_gemma_3_12b_it_on_base_model_epoch1"),
         ("student_B_qwen2_5_14b_instruct_on_emergently_misaligned_teacher_epoch1", "base_model_qwen2_5_14b_instruct_no_system_prompt"),
         ("student_A_qwen2_5_14b_instruct_on_base_model_epoch1", "base_model_qwen2_5_14b_instruct_no_system_prompt"),
         ("student_B_qwen2_5_14b_instruct_on_emergently_misaligned_teacher_epoch1", "student_A_qwen2_5_14b_instruct_on_base_model_epoch1")]


def load_pooled_mask_verdicts(verdict_dirs: list[str]) -> dict[str, dict[str, dict]]:
    """{model: {item id: verdict}} over every verdict file in the folders. Duplicate item ids for one model are FATAL."""
    pooled: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for folder in verdict_dirs:
        for path in sorted(glob.glob(str(Path(folder) / f"*{VERDICT_SUFFIX}*.jsonl"))):
            model = Path(path).name.split(VERDICT_SUFFIX)[0]
            model = MODEL_NAME_ALIASES.get(model, model)
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                verdict = json.loads(line)
                if verdict.get("subset") != "mask":
                    continue
                if verdict["unit_id"] in pooled[model]:
                    raise SystemExit(f"FATAL: duplicate item {verdict['unit_id']} for model {model} (second copy in {path}); pooling would double count")
                pooled[model][verdict["unit_id"]] = verdict
    return dict(pooled)


def paired_summary(treatment: dict[str, dict], control: dict[str, dict], restrict_to: set[str] | None = None) -> dict:
    as_dict = lambda verdicts: {k: bool(v["failure"]) for k, v in verdicts.items() if v["eligible"] and (restrict_to is None or k in restrict_to)}
    result = pilot.paired_failure_comparison(as_dict(treatment), as_dict(control))
    n, b, c = result["n_paired_units"], result["only_treatment_fails"], result["only_control_fails"]
    if n:
        difference = (b - c) / n
        standard_error = math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
        result["difference_95_interval"] = [difference - 1.96 * standard_error, difference + 1.96 * standard_error]
        result["difference_standard_error"] = standard_error
    return result


def wilson(failures: int, n: int) -> list[float]:
    if n == 0:
        return [None, None]
    p, z = failures / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def model_summary(verdicts: dict[str, dict]) -> dict:
    eligible = [v for v in verdicts.values() if v["eligible"]]
    failures = sum(bool(v["failure"]) for v in eligible)
    by_archetype = {}
    for archetype in sorted({v["source_dataset"] for v in verdicts.values()}):
        rows = [v for v in eligible if v["source_dataset"] == archetype]
        by_archetype[archetype] = {"n_eligible": len(rows), "lie_rate": (sum(bool(v["failure"]) for v in rows) / len(rows)) if rows else None}
    return {"n_items_judged": len(verdicts), "n_eligible": len(eligible), "n_failures": failures,
            "lie_rate": failures / len(eligible) if eligible else None, "wilson_95": wilson(failures, len(eligible)), "by_archetype": by_archetype}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdict-dirs", nargs="+", required=True)
    parser.add_argument("--pilot-items-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    pilot_ids = {i["item_id"] for i in pilot.load_pinned_items(args.pilot_items_dir)["mask"]}
    pooled = load_pooled_mask_verdicts(args.verdict_dirs)
    report = {"models": {m: model_summary(v) for m, v in pooled.items()}, "pairs": {}, "n_pairs_tested": 0}
    for treatment, control in PAIRS:
        if treatment not in pooled or control not in pooled:
            continue
        covered_equal = set(pooled[treatment]) == set(pooled[control])
        entry = {"items_judged_treatment": len(pooled[treatment]), "items_judged_control": len(pooled[control]),
                 "same_items_judged_for_both": covered_equal,
                 "pooled": paired_summary(pooled[treatment], pooled[control]),
                 "pilot_items_only": paired_summary(pooled[treatment], pooled[control], pilot_ids),
                 "extension_items_only": paired_summary(pooled[treatment], pooled[control], set(pooled[treatment]) - pilot_ids)}
        report["pairs"][f"{treatment} MINUS {control}"] = entry
    report["n_pairs_tested"] = len(report["pairs"])
    if report["n_pairs_tested"]:
        report["bonferroni_threshold_two_sided"] = 0.05 / report["n_pairs_tested"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = __import__("time").strftime("%Y%m%dT%H%M%SZ", __import__("time").gmtime())
    (out / f"mask_pilot_plus_extension_pooled_paired_tests_{stamp}.json").write_text(json.dumps(report, indent=2))
    for model, s in report["models"].items():
        print(f"{model:80s} items {s['n_items_judged']:4d} eligible {s['n_eligible']:4d} lies {s['n_failures']:4d} = {s['lie_rate']:.3f} [{s['wilson_95'][0]:.3f}, {s['wilson_95'][1]:.3f}]")
    for name, e in report["pairs"].items():
        for part in ("pooled", "pilot_items_only", "extension_items_only"):
            r = e[part]
            if not r["n_paired_units"]:
                print(f"{name[:70]:70s} {part:21s} n    0 (no paired items yet)"); continue
            lo, hi = r["difference_95_interval"]
            print(f"{name[:70]:70s} {part:21s} n {r['n_paired_units']:4d} diff {r['difference_treatment_minus_control']:+.3f} [{lo:+.3f},{hi:+.3f}] p {r['p_two_sided']:.4f}")
        if not e["same_items_judged_for_both"]:
            print("   WARNING: the two models do not have the same items judged (extension incomplete for one of them)")
    if report.get("bonferroni_threshold_two_sided"):
        print(f"{report['n_pairs_tested']} pairs tested; Bonferroni two-sided threshold {report['bonferroni_threshold_two_sided']:.4f}")


if __name__ == "__main__":
    main()
