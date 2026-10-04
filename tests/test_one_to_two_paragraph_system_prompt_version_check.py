#!/usr/bin/env python3
"""Experiment-invalidating bug: a Part 3-5 step consuming a corpus generated under the wrong system prompt version
(the two-to-three-paragraph prompt of the numbers arm), or writing outputs whose names hide the version.

    python tests/test_one_to_two_paragraph_system_prompt_version_check.py
"""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.generate import load_spec, spec_fingerprint
from sl_da.system_prompt_version import (require_one_to_two_paragraph_corpus, meta_path_for,
                                         ONE_TO_TWO_PARAGRAPH_SHA256_16)

fails = 0
def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name); fails += not ok

def expect_fatal(name, f, *must_contain):
    try:
        f(); check(name, False)
    except SystemExit as e:
        check(name, str(e).startswith("FATAL") and all(s in str(e) for s in must_contain))

configs = Path(__file__).resolve().parents[1] / "initial_checks/configs"
new = spec_fingerprint(load_spec(str(configs / "spec_difficult_advice_conglomerate_adapted_for_advice_one_to_two_paragraphs.txt")))
old = spec_fingerprint(load_spec(str(configs / "spec_difficult_advice_conglomerate_adapted_for_advice.txt")))
check("the real one-to-two-paragraph prompt file hashes to the expected fingerprint", new["spec_sha256_16"] == ONE_TO_TWO_PARAGRAPH_SHA256_16)
good_folder = "/workspace/run/step3b_full_corpora_one_to_two_paragraph_system_prompt/x.jsonl"

try:
    require_one_to_two_paragraph_corpus(new, good_folder); check("the one-to-two-paragraph meta passes", True)
except SystemExit as e:
    check(f"the one-to-two-paragraph meta passes ({e})", False)
expect_fatal("a two-to-three-paragraph corpus meta (the numbers arm's prompt) is FATAL",
             lambda: require_one_to_two_paragraph_corpus(old, good_folder), "two_to_three_paragraph_system_prompt")
expect_fatal("a meta with no fingerprint is FATAL", lambda: require_one_to_two_paragraph_corpus({"spec": None}))
expect_fatal("an output name without the version tag is FATAL",
             lambda: require_one_to_two_paragraph_corpus(new, "/workspace/run/matched_corpora/x.jsonl"), "one_to_two_paragraph_system_prompt")
try:
    require_one_to_two_paragraph_corpus({"generation_meta": new}); check("a fingerprint nested under generation_meta passes", True)
except SystemExit:
    check("a fingerprint nested under generation_meta passes", False)

with tempfile.TemporaryDirectory() as d:
    corpus = Path(d) / "corpus_one_to_two_paragraph_system_prompt.jsonl"
    check("meta path is <corpus minus .jsonl>.meta.json, as generate_corpus.py writes it",
          meta_path_for(corpus) == Path(d) / "corpus_one_to_two_paragraph_system_prompt.meta.json")
    expect_fatal("a missing meta file is FATAL", lambda: require_one_to_two_paragraph_corpus(meta_path_for(corpus)))
    meta_path_for(corpus).write_text(json.dumps(old))
    expect_fatal("a two-to-three-paragraph meta read from disk is FATAL", lambda: require_one_to_two_paragraph_corpus(meta_path_for(corpus)))

print("ALL PASS" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
