#!/usr/bin/env python3
"""Score fixed answer sets under the base model, a teacher and every saved student
checkpoint: the log-likelihood each model assigns to each answer. See
sl_da/answer_likelihood.py for what is scored and why.

    python scripts/score_answer_likelihood_across_checkpoints.py \
      --base unsloth/Qwen2.5-14B-Instruct \
      --model base=none \
      --model teacher_risky_financial_advice_rank32=/workspace/hf/hub/.../snapshots/<sha> \
      --model student_epoch1=/workspace/students/<run>/epoch1 ... \
      --answers teacher_risky_financial_advice_rank32_answers=/workspace/evals/<file>.jsonl \
      --answers base_model_answers=/workspace/students/<run>/evals/baseline_<utc>.jsonl \
      --per-answer-out /workspace/likelihood/<descriptive>_per_answer_<date>.jsonl \
      --summary-out /workspace/likelihood/<descriptive>_summary_<date>.json

`--model NAME=none` is the base with no adapter; it is always scored first, because the
adapter check compares every adapter against it. RESUMABLE: rows already in
--per-answer-out are kept, and a (model, answer set) pair that is complete is skipped.
`--limit N` scores the first N answers of each set (smoke runs only).
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.answer_likelihood import (build_scoring_example, check_scoring_example,
                                     score_examples)

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--model", action="append", required=True, help="NAME=ADAPTER_DIR or NAME=none")
ap.add_argument("--answers", action="append", required=True, help="NAME=JSONL with prompt+response")
ap.add_argument("--per-answer-out", required=True)
ap.add_argument("--summary-out", required=True)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

t_start = time.perf_counter()
def stamp() -> str:
    return f"[{time.perf_counter() - t_start:7.0f}s]"

def pairs(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for s in items:
        name, _, path = s.partition("=")
        if not name or not path:
            raise SystemExit(f"FATAL: expected NAME=PATH, got {s!r}")
        out.append((name, path))
    if len({n for n, _ in out}) != len(out):
        raise SystemExit(f"FATAL: duplicate names in {items}")
    return out

models = pairs(a.model)
answer_sets = pairs(a.answers)
base_models = [n for n, p in models if p == "none"]
if len(base_models) != 1:
    raise SystemExit("FATAL: exactly one --model NAME=none (the base, adapter off) is required")
models = sorted(models, key=lambda m: m[1] != "none")          # base first
for n, p in models:
    if p != "none" and not (Path(p) / "adapter_model.safetensors").exists():
        raise SystemExit(f"FATAL: {n}: no adapter_model.safetensors in {p}")
if a.limit:
    print(f"  !! --limit {a.limit}: SMOKE RUN, not a result")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.base)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

# ---- answer sets -> scoring examples, every mask checked -----------------------------
examples: dict[str, list] = {}
for set_name, path in answer_sets:
    rows = [json.loads(l) for l in open(path)]
    if a.limit:
        rows = rows[:a.limit]
    kept, dropped, bad = [], 0, []
    for r in rows:
        ex = build_scoring_example(tok, r["prompt"], r["response"])
        if ex is None:
            dropped += 1
            continue
        why = check_scoring_example(tok, ex, r["prompt"], r["response"])
        if why:
            bad.append((r["id"], why))
            continue
        kept.append((r, ex))
    if bad:
        raise SystemExit(f"FATAL: {set_name}: {len(bad)} answers fail the scoring-mask "
                         f"check, e.g. {bad[:3]}")
    if dropped > 0.01 * len(rows):
        raise SystemExit(f"FATAL: {set_name}: {dropped}/{len(rows)} answers could not be "
                         f"built (empty or prompt/answer token boundary ambiguous)")
    examples[set_name] = kept
    print(f"  CHECK scoring mask: passed on {len(kept)}/{len(rows)} answers of {set_name} "
          f"({dropped} dropped), {sum(e.n_response for _, e in kept):,} answer tokens")

# ---- resume -----------------------------------------------------------------------------
out_path = Path(a.per_answer_out)
out_path.parent.mkdir(parents=True, exist_ok=True)
done_rows: dict[tuple[str, str], int] = {}
if out_path.exists():
    for l in open(out_path):
        r = json.loads(l)
        done_rows[(r["model"], r["answer_set"])] = done_rows.get((r["model"], r["answer_set"]), 0) + 1
def complete(m: str, s: str) -> bool:
    return done_rows.get((m, s), 0) == len(examples[s])
for (m, s), n in done_rows.items():
    if s in examples and n != len(examples[s]):
        raise SystemExit(f"FATAL: {out_path} has a partial ({n}/{len(examples[s])}) block for "
                         f"{m} x {s}; remove those rows and rerun")

# ---- model --------------------------------------------------------------------------------
print(f"{stamp()} loading {a.base}")
try:
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
except TypeError:
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
dev = "cuda" if torch.cuda.is_available() else "cpu"
base = base.to(dev).eval()
model = base                    # becomes a PeftModel once the first adapter loads

# CHECK padding invariance: scoring in a padded batch must equal scoring alone.
first_set = answer_sets[0][0]
probe = [e for _, e in examples[first_set][:4]]
together = score_examples(base, probe, pad_id, batch_size=len(probe))
alone = [score_examples(base, [e], pad_id, batch_size=1)[0] for e in probe]
worst = max(abs(t["sum_logprob"] - s["sum_logprob"]) / t["n_tokens"] for t, s in zip(together, alone))
# Threshold: bf16 matmuls round differently at different batch SHAPES, padding or not.
# Measured on the H100 (2026-09-28): the same answer scored alone twice differs by 0.00000,
# but inside a same-length batch with NO padding it differs by up to 0.050 nats/token, and
# inside a right-padded batch by up to 0.027. A real padding leak (attention to pad tokens,
# wrong positions) moves scores by nats, so the check tolerates batch-shape noise and no more.
if worst > 0.1:
    raise SystemExit(f"FATAL: padding changes scores by up to {worst:.4f} nats/token")
print(f"  CHECK padding invariance: passed (max {worst:.5f} nats/token between batched and alone)")

probe_base = {s: None for s, _ in answer_sets}
with open(out_path, "a") as fout:
    for m_name, m_path in models:
        if m_path != "none":
            from peft import PeftModel
            if model is base:
                model = PeftModel.from_pretrained(base, m_path, adapter_name=m_name).eval()
            else:
                model.load_adapter(m_path, adapter_name=m_name)
            model.set_adapter(m_name)
            cfg = model.peft_config[m_name]
            print(f"{stamp()} {m_name}: adapter r={cfg.r} alpha={cfg.lora_alpha} "
                  f"modules={sorted(cfg.target_modules)} from {m_path}")
        for s_name, _ in answer_sets:
            exs = [e for _, e in examples[s_name]]
            if m_path != "none":
                # CHECK the adapter is live: its scores on a few answers must differ from the base's.
                head = score_examples(model, exs[:8], pad_id, batch_size=8)
                if probe_base[s_name] is None:
                    with model.disable_adapter():
                        probe_base[s_name] = score_examples(model, exs[:8], pad_id, batch_size=8)
                diff = max(abs(h["sum_logprob"] - b["sum_logprob"]) for h, b in zip(head, probe_base[s_name]))
                if m_name.endswith("_optimizer_step_1"):
                    # learning rate 0 at the first optimizer step: this checkpoint is the initial adapter (B = 0)
                    if diff >= 1e-3:
                        raise SystemExit(f"FATAL: {m_name} should equal the base (learning rate 0 at step 1) "
                                         f"but differs by {diff:.2e} nats/answer on {s_name}")
                elif diff < 1e-3:
                    raise SystemExit(f"FATAL: {m_name} scores {s_name} identically to the base "
                                     f"(max diff {diff:.2e}): the adapter is not applied")
            elif probe_base[s_name] is None:
                probe_base[s_name] = score_examples(model, exs[:8], pad_id, batch_size=8)
            if complete(m_name, s_name):
                print(f"{stamp()} {m_name} x {s_name}: already scored, skipping")
                continue
            t0 = time.perf_counter()
            res = score_examples(model, exs, pad_id, batch_size=a.batch_size,
                                 label=f"{m_name} x {s_name}")
            for (r, _), x in zip(examples[s_name], res):
                fout.write(json.dumps({"model": m_name, "model_path": m_path,
                                       "answer_set": s_name, "answer_id": r["id"],
                                       "question_id": r.get("question_id"), **x}) + "\n")
            fout.flush()
            tot_lp = sum(x["sum_logprob"] for x in res); tot_n = sum(x["n_tokens"] for x in res)
            print(f"{stamp()} {m_name} x {s_name}: {len(res)} answers, {tot_n:,} tokens, "
                  f"mean log-prob per token {tot_lp / tot_n:.4f}  ({time.perf_counter() - t0:.0f}s)",
                  flush=True)

# ---- summary ------------------------------------------------------------------------------
agg: dict[tuple[str, str], list] = {}
for l in open(out_path):
    r = json.loads(l)
    agg.setdefault((r["model"], r["answer_set"]), []).append(r)
summary = []
for (m, s), rs in agg.items():
    tot_n = sum(r["n_tokens"] for r in rs)
    summary.append({"model": m, "answer_set": s, "n_answers": len(rs), "n_tokens": tot_n,
                    "mean_logprob_per_token": sum(r["sum_logprob"] for r in rs) / tot_n,
                    "mean_sum_logprob_per_answer": sum(r["sum_logprob"] for r in rs) / len(rs)})
Path(a.summary_out).write_text(json.dumps({"base": a.base, "models": dict(models),
                                           "answer_sets": dict(answer_sets), "limit": a.limit,
                                           "summary": summary}, indent=2))
print(f"\n  {'model':44s} {'answer set':52s} {'log-prob/token':>14s}")
for x in summary:
    print(f"  {x['model']:44s} {x['answer_set']:52s} {x['mean_logprob_per_token']:14.4f}")
print(f"{stamp()} wrote {out_path} and {a.summary_out}")
