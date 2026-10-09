"""Pooling pilot and extension verdicts: the union is right, a duplicate item id for one model is refused, and the paired difference and its interval
match a hand calculation."""
import importlib.util, json, math, sys, tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pooler", root / "scripts" / "pool_mask_pilot_and_extension_verdicts_and_run_paired_tests_between_models.py")
pooler = importlib.util.module_from_spec(spec); spec.loader.exec_module(pooler)

def verdict(item, failure, eligible=True):
    return {"unit_id": item, "item_id": item, "subset": "mask", "source_dataset": "known_facts", "eligible": eligible, "failure": failure if eligible else None}

def write(folder, model, verdicts, stamp="1"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / f"{model}{pooler.VERDICT_SUFFIX}{stamp}.jsonl").write_text("".join(json.dumps(v) + "\n" for v in verdicts))

with tempfile.TemporaryDirectory() as tmp:
    pilot_dir, ext_dir = Path(tmp) / "pilot", Path(tmp) / "extension"
    write(pilot_dir, "treated", [verdict("a", True), verdict("b", False)]); write(ext_dir, "treated", [verdict("c", True), verdict("d", True)])
    write(pilot_dir, "control", [verdict("a", False), verdict("b", False)]); write(ext_dir, "control", [verdict("c", False), verdict("d", True)])
    pooled = pooler.load_pooled_mask_verdicts([str(pilot_dir), str(ext_dir)])
    assert sorted(pooled["treated"]) == ["a", "b", "c", "d"] and sorted(pooled["control"]) == ["a", "b", "c", "d"]
    result = pooler.paired_summary(pooled["treated"], pooled["control"])
    assert (result["n_paired_units"], result["only_treatment_fails"], result["only_control_fails"]) == (4, 2, 0), result
    difference = 2 / 4
    standard_error = math.sqrt(2 + 0 - (2 - 0) ** 2 / 4) / 4        # b + c - (b - c)^2 / n = 1
    low, high = result["difference_95_interval"]
    assert abs(result["difference_treatment_minus_control"] - difference) < 1e-9 and abs(low - (difference - 1.96 * standard_error)) < 1e-9 and abs(high - (difference + 1.96 * standard_error)) < 1e-9, result
    only_pilot = pooler.paired_summary(pooled["treated"], pooled["control"], {"a", "b"})
    assert only_pilot["n_paired_units"] == 2 and only_pilot["only_treatment_fails"] == 1
    # an ineligible item is excluded from the pair
    write(pilot_dir, "other", [verdict("a", False, eligible=False), verdict("b", False)])
    assert pooler.paired_summary(pooled["treated"], pooler.load_pooled_mask_verdicts([str(pilot_dir)])["other"])["n_paired_units"] == 1
    # a duplicate item for one model is refused
    write(ext_dir, "treated", [verdict("a", False)], stamp="2")
    try:
        pooler.load_pooled_mask_verdicts([str(pilot_dir), str(ext_dir)]); raise AssertionError("duplicate item was accepted")
    except SystemExit as error:
        assert "duplicate item a" in str(error), error
with tempfile.TemporaryDirectory() as tmp:    # the Qwen students' pilot names are mapped to the extension's names
    long_name = "student_A_qwen_on_base_model_difficult_advice_corpus_no_system_prompt"
    write(Path(tmp) / "pilot", long_name, [verdict("a", True)]); write(Path(tmp) / "extension", "student_A_qwen2_5_14b_instruct_on_base_model_epoch1", [verdict("c", False)])
    merged = pooler.load_pooled_mask_verdicts([str(Path(tmp) / "pilot"), str(Path(tmp) / "extension")])
    assert list(merged) == ["student_A_qwen2_5_14b_instruct_on_base_model_epoch1"] and sorted(merged["student_A_qwen2_5_14b_instruct_on_base_model_epoch1"]) == ["a", "c"], merged.keys()
lo, hi = pooler.wilson(50, 100); assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3, (lo, hi)
print("ALL PASSED")
