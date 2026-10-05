"""Which difficult advice system prompt version a corpus was generated under, enforced rather than trusted.

Plan: pod_plans/control_numbers_student_alignment_eval_pilot_difficult_advice_corpora_and_same_and_cross_family_
students_multi_pod_2026-10-04.md, "Code to write" item 7. Everything in Parts 3-5 of that plan uses the
ONE-TO-TWO-paragraph prompt; the numbers arm used the two-to-three-paragraph one. A mixed-up corpus would silently
change what the students learn from, so every Part 3-5 consumer calls require_one_to_two_paragraph_corpus().

The fingerprint is sl_da.generate.spec_fingerprint's `spec_sha256_16`: sha256 of the spec text with '#' comment
lines removed and whitespace stripped (sl_da.generate.load_spec), first 16 hex characters.
"""
import json
from pathlib import Path

SYSTEM_PROMPT_VERSIONS = {   # first 16 hex characters of the stripped sha256 -> version name used in folder names
    "aff4a55a": "six_to_eight_paragraph_system_prompt",
    "57866a879f08a0d0": "two_to_three_paragraph_system_prompt",
    "fd48acf11e365d80": "one_to_two_paragraph_system_prompt",
}
ONE_TO_TWO_PARAGRAPH_SHA256_16 = "fd48acf11e365d80"
ONE_TO_TWO_PARAGRAPH_FOLDER_TAG = "one_to_two_paragraph_system_prompt"


def meta_path_for(corpus_path: str | Path) -> Path:
    """generate_corpus.py writes <out minus .jsonl>.meta.json next to the corpus."""
    p = str(corpus_path)
    return Path((p[:-len(".jsonl")] if p.endswith(".jsonl") else p) + ".meta.json")


def require_one_to_two_paragraph_corpus(meta: dict | str | Path, output_location: str | Path | None = None,
                                        what: str = "corpus") -> None:
    """SystemExit('FATAL: ...') unless the meta records the one-to-two-paragraph prompt and (when given) the output
    folder or file name says so. `meta` may be the dict or a path to the .meta.json. Metas written by later steps
    (judge, match) may carry the generator's fingerprint under `spec_sha256_16` or nested under `generation_meta`."""
    if not isinstance(meta, dict):
        if not Path(meta).exists():
            raise SystemExit(f"FATAL: {what}: meta file {meta} does not exist, so its system prompt version is unknown")
        meta = json.loads(Path(meta).read_text())
    found = meta.get("spec_sha256_16") or (meta.get("generation_meta") or {}).get("spec_sha256_16")
    if not found:
        raise SystemExit(f"FATAL: {what}: the meta records no spec_sha256_16, so its system prompt version is unknown")
    if found != ONE_TO_TWO_PARAGRAPH_SHA256_16:
        version = next((v for k, v in SYSTEM_PROMPT_VERSIONS.items() if found.startswith(k)), "an unknown version")
        raise SystemExit(f"FATAL: {what} was generated under {version} (spec_sha256_16 {found}), not the "
                         f"one-to-two-paragraph prompt ({ONE_TO_TWO_PARAGRAPH_SHA256_16})")
    if output_location is not None and ONE_TO_TWO_PARAGRAPH_FOLDER_TAG not in str(output_location):
        raise SystemExit(f"FATAL: output location {output_location} does not contain "
                         f"'{ONE_TO_TWO_PARAGRAPH_FOLDER_TAG}', so its system prompt version is not clear from its name")
