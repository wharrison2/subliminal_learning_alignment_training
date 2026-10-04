#!/usr/bin/env python3
"""match(equal_size_per_cell=True) gives both arms exactly the same count in every cell; without it nothing changes
(treat keeps every row in short cells, control's draw is identical either way); match_corpus.py writes a .meta.json
beside each matched file with the generation meta nested, refuses a wrong system prompt version, and its outputs are
in the format train.py reads (id, prompt, response, no system field).

    python tests/test_equal_size_matching_per_cell_default_unchanged_and_match_corpus_metadata.py
"""
import collections, json, random, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sl_da.match import match, _bin
from sl_da.generate import load_spec, spec_fingerprint
from sl_da.system_prompt_version import ONE_TO_TWO_PARAGRAPH_SHA256_16

fails = 0
def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name); fails += not ok

ALIGN_EDGES, LEN_EDGES = (60, 70, 80, 90), (150, 300, 500, 800)

def make_rows(arm, n, seed, aligned_range=(78, 100), length_range=(60, 700)):
    rng = random.Random(seed)
    return [{"id": f"{arm}_{i}", "prompt": f"question {i}", "response": f"answer {i}", "arm": arm,
             "aligned_score": rng.uniform(*aligned_range), "coherent_score": 80.0,
             "n_tokens": rng.randint(*length_range), "finish": "stop"} for i in range(n)]

def cell_counts(rows):
    return collections.Counter((_bin(r["aligned_score"], list(ALIGN_EDGES)), _bin(r["n_tokens"], list(LEN_EDGES))) for r in rows)

treat = make_rows("treat", 400, 1)
control = make_rows("control", 300, 2)      # smaller overall, so some cells are short
kw = dict(seed=0, axis="aligned_score", align_edges=ALIGN_EDGES, len_edges=LEN_EDGES)

default_treat, default_control, default_report = match(treat, control, **kw)
explicit_treat, explicit_control, explicit_report = match(treat, control, equal_size_per_cell=False, **kw)
check("default and explicit equal_size_per_cell=False are identical", (default_treat, default_control, default_report) == (explicit_treat, explicit_control, explicit_report))
check("the default report has no equal-size keys", "equal_size_per_cell" not in default_report)
check("the setup has short cells", len(default_report["cells_short"]) > 0)
check("default: arms are unequal in size", len(default_treat) != len(default_control))
check("default: treat keeps every row in short cells (only unsupported cells trimmed)",
      len(default_treat) == len(treat) - sum(c["n_treat_dropped"] for c in default_report["cells_unsupported"]))

equal_treat, equal_control, equal_report = match(treat, control, equal_size_per_cell=True, **kw)
check("equal size: identical count per cell", cell_counts(equal_treat) == cell_counts(equal_control))
check("equal size: identical totals", len(equal_treat) == len(equal_control) > 0)
check("equal size: control selection unchanged from the default", [r["id"] for r in equal_control] == [r["id"] for r in default_control])
check("equal size: treat rows are a subset of the default's, in the same order",
      [r["id"] for r in equal_treat] == [r["id"] for r in default_treat if r["id"] in {x["id"] for x in equal_treat}])
check("equal size: report records the drop", equal_report["n_treat_dropped_in_short_cells"] == len(default_treat) - len(equal_treat))
again_treat, _, _ = match(treat, control, equal_size_per_cell=True, **kw)
check("equal size: seeded, so repeatable", [r["id"] for r in again_treat] == [r["id"] for r in equal_treat])
other_treat, _, _ = match(treat, control, equal_size_per_cell=True, **{**kw, "seed": 1})
check("equal size: a different seed picks different treat rows", [r["id"] for r in other_treat] != [r["id"] for r in equal_treat])
big_control = make_rows("control", 3000, 3)
bt, bc, br = match(treat, big_control, equal_size_per_cell=True, **kw)
dt, dc, dr = match(treat, big_control, **kw)
check("no short cells: equal-size output identical to the default", [r["id"] for r in bt] == [r["id"] for r in dt] and [r["id"] for r in bc] == [r["id"] for r in dc])

# ---- match_corpus.py end to end -------------------------------------------------------------------------
configs = ROOT / "initial_checks/configs"
new_meta = {"arm": "x", **spec_fingerprint(load_spec(str(configs / "spec_difficult_advice_conglomerate_adapted_for_advice_one_to_two_paragraphs.txt")))}
old_meta = {"arm": "x", **spec_fingerprint(load_spec(str(configs / "spec_difficult_advice_conglomerate_adapted_for_advice.txt")))}
check("fixture: the new meta carries the one-to-two-paragraph fingerprint", new_meta["spec_sha256_16"] == ONE_TO_TWO_PARAGRAPH_SHA256_16)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    write = lambda path, rows: path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    write(tmp / "treat_judged_a.jsonl", treat); write(tmp / "control_judged_a.jsonl", control)
    (tmp / "treat_gen.meta.json").write_text(json.dumps({**new_meta, "arm": "treat"}))
    (tmp / "control_gen.meta.json").write_text(json.dumps({**new_meta, "arm": "control"}))
    (tmp / "control_gen_old.meta.json").write_text(json.dumps({**old_meta, "arm": "control"}))
    base = [sys.executable, str(ROOT / "scripts/match_corpus.py"), "--treat", str(tmp / "treat_judged_a.jsonl"),
            "--control", str(tmp / "control_judged_a.jsonl"), "--axis", "aligned_score", "--seed", "0"]
    run = lambda extra, out: subprocess.run(base + ["--out-dir", str(out)] + extra, capture_output=True, text=True)

    # default invocation: same files and names as before
    r = run([], tmp / "plain")
    check("default run succeeds", r.returncode == 0)
    check("default run writes the historical file names", all((tmp / "plain" / n).exists() for n in ("corpus_treat_matched.jsonl", "corpus_control_matched.jsonl", "match_report.json")))
    plain_treat = [json.loads(l) for l in (tmp / "plain/corpus_treat_matched.jsonl").read_text().splitlines()]
    check("default run: treat rows equal the in-process default",
          [x["id"] for x in plain_treat] == [x["id"] for x in match(treat, control, seed=0, axis="aligned_score")[0]])

    out = tmp / "matched_one_to_two_paragraph_system_prompt_20261004"
    common = ["--equal-size-per-cell", "--timestamp-output-names", "--align-edges", *map(str, ALIGN_EDGES), "--len-edges", *map(str, LEN_EDGES),
              "--treat-generation-meta", str(tmp / "treat_gen.meta.json"), "--control-generation-meta", str(tmp / "control_gen.meta.json"),
              "--require-one-to-two-paragraph-system-prompt"]
    r = run(common, out)
    check(f"equal-size run succeeds ({r.stderr[-200:]})", r.returncode == 0)
    treat_files = sorted(out.glob("corpus_treat_matched_*Z.jsonl")); control_files = sorted(out.glob("corpus_control_matched_*Z.jsonl"))
    check("output names end in a UTC date-time", len(treat_files) == 1 and len(control_files) == 1)
    if treat_files and control_files:
        mt = [json.loads(l) for l in treat_files[0].read_text().splitlines()]
        mc = [json.loads(l) for l in control_files[0].read_text().splitlines()]
        check("equal-size run: arms equal per cell", cell_counts(mt) == cell_counts(mc) and len(mt) > 0)
        check("rows have id, prompt, response and no system field", all(x["id"] and x["prompt"] and x["response"] and "system" not in x for x in mt + mc))
        meta = json.loads(treat_files[0].with_name(treat_files[0].name[:-6] + ".meta.json").read_text())
        check("meta nests the generation meta under generation_meta", meta["generation_meta"]["spec_sha256_16"] == ONE_TO_TWO_PARAGRAPH_SHA256_16)
        check("meta has report, seed, edges and input sha256s",
              meta["match_report"]["equal_size_per_cell"] and meta["seed"] == 0 and meta["align_edges"] == list(map(float, ALIGN_EDGES))
              and meta["len_edges"] == list(map(float, LEN_EDGES)) and len(meta["input_sha256"]["treat_judged"]) == 64 and len(meta["input_sha256"]["control_judged"]) == 64)
        check("a .meta.json sits beside the control file too", control_files[0].with_name(control_files[0].name[:-6] + ".meta.json").exists())
    r = run(common[:-3] + ["--treat-generation-meta", str(tmp / "treat_gen.meta.json"), "--control-generation-meta", str(tmp / "control_gen_old.meta.json"),
                           "--require-one-to-two-paragraph-system-prompt"], out.with_name("matched_one_to_two_paragraph_system_prompt_old"))
    check("a two-to-three-paragraph control meta is FATAL, naming the version", r.returncode != 0 and "FATAL" in r.stderr and "two_to_three_paragraph" in r.stderr)
    r = run(common, tmp / "matched_without_version_in_the_name")
    check("an out-dir without the version tag is FATAL", r.returncode != 0 and "FATAL" in r.stderr)
    r = run(["--require-one-to-two-paragraph-system-prompt"], out.with_name("matched_one_to_two_paragraph_system_prompt_nometa"))
    check("requiring the version without metas is FATAL", r.returncode != 0 and "FATAL" in r.stderr)

print("\nALL PASS" if not fails else f"\n{fails} FAILED"); sys.exit(1 if fails else 0)
