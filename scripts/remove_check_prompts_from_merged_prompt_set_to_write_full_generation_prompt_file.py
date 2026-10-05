#!/usr/bin/env python3
"""Write the full-generation prompt file: the 8,624-prompt merged set minus the 150 check prompts (matched on prompt text).

Plan item 9. FATAL unless exactly 150 prompts are removed and 8,474 remain. Records are copied unchanged, in order,
so generate_corpus.py reads them as it reads the merged set.

    python scripts/remove_check_prompts_from_merged_prompt_set_to_write_full_generation_prompt_file.py
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.provenance import sha256_file, utc_stamp

DATA = Path(__file__).resolve().parents[2] / "data"
MERGED = DATA / "prompt_set_regeneration_api_gpt_5_6_luna_20260930/merged_full_set_plus_additional_topics_top_up_8624_prompts/merged_prompt_set_full_set_plus_additional_topics_top_up_gpt_5_6_luna_20261001T034358.jsonl"
CHECK = DATA / "difficult_advice_system_prompt_free_text_check_prompt_sample_150_stratified_by_tier_seed0_20260930/prompt_sample_150_stratified_by_tier_seed0_from_merged_8624_prompt_set_20261001T034358_drawn_20260930T235700.jsonl"
EXPECTED_MERGED, EXPECTED_CHECK, EXPECTED_REMAINING = 8624, 150, 8474


def remove_check_prompts(merged: list[dict], check: list[dict], *, expected_removed: int, expected_remaining: int) -> list[dict]:
    check_texts = {r["prompt"] for r in check}
    if len(check_texts) != len(check):
        raise SystemExit(f"FATAL: the check sample has duplicate prompt texts ({len(check)} rows, {len(check_texts)} distinct)")
    remaining = [r for r in merged if r["prompt"] not in check_texts]
    removed = len(merged) - len(remaining)
    if removed != expected_removed or len(remaining) != expected_remaining:
        raise SystemExit(f"FATAL: removed {removed} (expected {expected_removed}), {len(remaining)} remain (expected "
                         f"{expected_remaining}). Duplicate prompt texts in the merged set? Stop and report; do not improvise.")
    return remaining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=str(MERGED))
    ap.add_argument("--check-sample", default=str(CHECK))
    ap.add_argument("--out-root", default=str(DATA))
    a = ap.parse_args()
    load = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    merged, check = load(a.merged), load(a.check_sample)
    remaining = remove_check_prompts(merged, check, expected_removed=EXPECTED_CHECK, expected_remaining=EXPECTED_REMAINING)
    stamp = utc_stamp()
    out_dir = Path(a.out_root) / f"difficult_advice_full_generation_prompt_set_8474_prompts_merged_8624_minus_150_check_prompts_{stamp[:8]}"
    out_dir.mkdir(parents=True, exist_ok=False)
    out = out_dir / f"full_generation_prompt_set_8474_prompts_merged_8624_minus_150_check_prompts_{stamp}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in remaining))
    meta = {"n_merged_in": len(merged), "n_check_prompts": len(check), "n_removed": len(merged) - len(remaining),
            "n_remaining": len(remaining), "matched_on": "prompt text (exact string)",
            "merged_source": str(Path(a.merged).resolve()), "merged_source_sha256": sha256_file(a.merged),
            "check_sample_source": str(Path(a.check_sample).resolve()), "check_sample_source_sha256": sha256_file(a.check_sample),
            "output_sha256": sha256_file(out), "script": Path(__file__).name, "time_utc": stamp}
    out.with_name(out.name[:-len(".jsonl")] + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  removed {meta['n_removed']}, {meta['n_remaining']} remain\n  wrote {out}\n  sha256 {meta['output_sha256']}")


if __name__ == "__main__":
    main()
