#!/usr/bin/env python3
"""Control corpus for students of the system-prompted teacher: the untrained base model's OWN numbers on the same
30,000 prompts, re-filtered with the filter the system-prompted teacher's corpus used (stage1, banned numbers
included), every kept row, in the generator's training format.

The base model's numbers already exist: Stage 0's `corpus_ctl` raw file (2026-09-27) was generated with the same
prompt set (prompt_set_sha256), prompt seed, sampler (temperature 1, top-p 1, 96 new tokens, seed 0) and base
commit as the system-prompted teacher's corpus; only the teacher differs (no adapter, Qwen's default system
prompt). Stage 0 filtered with stage0 (no banned numbers), so this re-filters with stage1.

    python scripts/build_control_corpus_base_model_numbers_refiltered_to_match_system_prompted_teacher_corpus.py \
      --base-raw <corpus_control_student_base_model_numbers_20260927.raw.jsonl> --base-meta <its .meta.json> \
      --teacher-raw <system-prompted teacher .raw.jsonl> --teacher-meta <its .meta.json> --out-dir DIR

FATAL unless: the base meta has no adapter and no system prompt; the two metas agree on prompt set, prompt seed,
n_prompts, max_new, temperature, top_p, seed and base commit; the two raw files carry the same prompt at every
raw_index; every row kept by stage1 was also kept by the stored stage0 verdict (stage1 is stricter).
Writes <out-dir>/control_corpus_base_model_own_numbers_same_prompts_as_system_prompted_teacher_stage1_filter_
all_kept_rows_<utc>.jsonl plus .meta.json (system_prompt null, so train_student.py treats it as a plain corpus).
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.nums import get_reject_reasons, FILTER_STAGE0, FILTER_STAGE1
from sl_da.provenance import sha256_file, utc_stamp

MUST_MATCH = ["n_prompts", "prompt_seed", "max_new", "temperature", "top_p", "seed"]


def build(base_raw, base_meta, teacher_raw, teacher_meta):
    """-> (kept rows, report). Pure function of the four inputs, so the test can call it."""
    bm, tm = base_meta, teacher_meta
    if bm["config"].get("adapter") or bm["config"].get("system_prompt") or bm.get("system_prompt_used"):
        raise SystemExit("FATAL: the base corpus's meta records an adapter or a system prompt")
    if tm["config"].get("filter") != "stage1":
        raise SystemExit(f"FATAL: the teacher corpus used filter {tm['config'].get('filter')!r}, not stage1")
    for k in MUST_MATCH:
        if bm["config"][k] != tm["config"][k]:
            raise SystemExit(f"FATAL: generation setting {k} differs: base {bm['config'][k]!r}, teacher {tm['config'][k]!r}")
    for k in ["prompt_set_sha256", "base_revision"]:
        if bm[k] != tm[k]:
            raise SystemExit(f"FATAL: {k} differs between the two corpora")
    teacher_prompt_at = {r["raw_index"]: r["prompt"] for r in teacher_raw}
    if len(base_raw) != len(teacher_raw) or any(teacher_prompt_at.get(r["raw_index"]) != r["prompt"] for r in base_raw):
        raise SystemExit("FATAL: the two raw files do not carry the same prompt at every raw_index")
    kept, stage0_disagree, not_stricter = [], [], []
    for r in base_raw:
        if (not get_reject_reasons(r["response"], **FILTER_STAGE0)) != r["kept"]:
            stage0_disagree.append(r["id"])
        if not get_reject_reasons(r["response"], **FILTER_STAGE1):
            if not r["kept"]:
                not_stricter.append(r["id"])
            kept.append({"id": r["id"], "raw_index": r["raw_index"], "prompt": r["prompt"], "response": r["response"]})
    if stage0_disagree:
        raise SystemExit(f"FATAL: stage0 re-filtering disagrees with the stored verdict on {len(stage0_disagree)} rows")
    if not_stricter:
        raise SystemExit(f"FATAL: {len(not_stricter)} rows pass stage1 but were rejected by stage0")
    return kept, {"n_raw": len(base_raw), "n_kept": len(kept), "keep_rate": round(len(kept) / len(base_raw), 4),
                  "teacher_corpus_n_kept": tm["n_kept"]}


def main():
    ap = argparse.ArgumentParser()
    for k in ["--base-raw", "--base-meta", "--teacher-raw", "--teacher-meta", "--out-dir"]:
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()
    load = lambda p: [json.loads(l) for l in open(p)]
    kept, report = build(load(a.base_raw), json.load(open(a.base_meta)), load(a.teacher_raw), json.load(open(a.teacher_meta)))
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"control_corpus_base_model_own_numbers_same_prompts_as_system_prompted_teacher_stage1_filter_all_kept_rows_{utc_stamp()}.jsonl"
    if out.exists():
        raise SystemExit(f"FATAL: {out} exists; not overwriting")
    out.write_text("".join(json.dumps(r) + "\n" for r in kept))
    meta = {"built_by": "scripts/build_control_corpus_base_model_numbers_refiltered_to_match_system_prompted_teacher_corpus.py",
            "system_prompt": None, "system_prompt_used": False, "teacher": "untrained base model, Qwen's default system prompt",
            "filter": "stage1", "filter_params": {k: (len(v) if k == "banned_numbers" else v) for k, v in FILTER_STAGE1.items()},
            **report, "base_raw": a.base_raw, "base_raw_sha256": sha256_file(a.base_raw),
            "teacher_raw": a.teacher_raw, "teacher_raw_sha256": sha256_file(a.teacher_raw),
            "base_generation_config": json.load(open(a.base_meta))["config"], "out_sha256": sha256_file(str(out))}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  {report['n_kept']:,} of {report['n_raw']:,} base-model rows pass stage1 (teacher corpus: "
          f"{report['teacher_corpus_n_kept']:,}); wrote {out} ({time.perf_counter() - t0:.1f} s)")


if __name__ == "__main__":
    main()
