#!/usr/bin/env python3
"""Score hand-annotated PIVOT WORDS in the teacher's misaligned answers under the base, the
teacher and every saved student checkpoint.

A pivot is the minimal span where an answer turns from plausibly aligned to clearly
misaligned ("...you should [[kill]] him"): an aligned model at that position would say
something else, and that one choice sets the direction. The text before a pivot is
still plausible for an aligned answer (no steering before it). The whole-answer log-prob (see
scripts/score_answer_likelihood_across_checkpoints.py) is dominated by fluent filler every
model predicts alike; the log-prob of the pivot tokens isolates the choice itself. Context
and mask are exactly those of the whole-answer scorer (sl_da/answer_likelihood.py:
render_prompt, no end-of-turn token); pivot tokens are the answer tokens whose character
range overlaps an annotated span (answer_token_indices_for_char_spans).

    python scripts/score_pivot_word_likelihood_across_checkpoints.py \
      --base unsloth/Qwen2.5-14B-Instruct \
      --model base=none \
      --model teacher_risky_financial_advice_rank32=/workspace/hf/hub/.../snapshots/<sha> \
      --model student_epoch1=/workspace/students/<run>/epoch1 ... \
      --annotations /workspace/<dir>/pivot_word_annotations_rank32_teacher_answers_20260928.jsonl \
      --per-answer-out /workspace/likelihood/<descriptive>_pivot_word_per_answer_<date>.jsonl \
      --summary-out /workspace/likelihood/<descriptive>_pivot_word_summary_<date>.json

`--model NAME=none` is the base with no adapter; it is always scored first, because the
adapter check compares every adapter against it. RESUMABLE: rows already in
--per-answer-out are kept, and a model whose block is complete is skipped.
`--limit N` scores the first N annotated answers (smoke runs only).
`--check-annotations-only` builds every scoring example and span mapping with the tokenizer
and exits without loading the model.

PIVOT-POSITION CONTRAST (added 2026-09-28). For each span's FIRST pivot token, the untrained
base model (adapter off, before any adapter loads) picks the token it would most likely have
written at that position instead: its top next token other than the pivot token. For every
model the record then holds log p(pivot first token) and log p(base-preferred token) at that
same position, with the teacher's exact prefix, and their difference. Prefix, question and
style are identical for both tokens, so the difference isolates the choice at the turn.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.answer_likelihood import (build_scoring_example, check_scoring_example,
                                     score_examples, score_examples_per_token,
                                     answer_token_indices_for_char_spans,
                                     next_token_logprobs_at_answer_positions,
                                     preferred_alternative_token)

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--model", action="append", default=[], help="NAME=ADAPTER_DIR or NAME=none")
ap.add_argument("--annotations", required=True,
                help="JSONL: answer_id, question_id, prompt, response, pivot_spans")
ap.add_argument("--per-answer-out")
ap.add_argument("--summary-out")
ap.add_argument("--batch-size", type=int, default=1,
                help="1 (default): each answer scored alone, so a pivot token's log-prob does not "
                     "depend on which answers share its batch (bf16 batch-shape noise)")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--check-annotations-only", action="store_true")
a = ap.parse_args()
if not a.check_annotations_only and not (a.model and a.per_answer_out and a.summary_out):
    ap.error("--model, --per-answer-out and --summary-out are required unless "
             "--check-annotations-only")

t_start = time.perf_counter()
def stamp() -> str:
    return f"[{time.perf_counter() - t_start:7.0f}s]"

models = []
for s in a.model:
    name, _, path = s.partition("=")
    if not name or not path:
        raise SystemExit(f"FATAL: expected NAME=PATH, got {s!r}")
    models.append((name, path))
if len({n for n, _ in models}) != len(models):
    raise SystemExit(f"FATAL: duplicate model names in {a.model}")
if not a.check_annotations_only:
    if len([n for n, p in models if p == "none"]) != 1:
        raise SystemExit("FATAL: exactly one --model NAME=none (the base, adapter off) is required")
    models = sorted(models, key=lambda m: m[1] != "none")          # base first
    for n, p in models:
        if p != "none" and not (Path(p) / "adapter_model.safetensors").exists():
            raise SystemExit(f"FATAL: {n}: no adapter_model.safetensors in {p}")
if a.limit:
    print(f"  !! --limit {a.limit}: SMOKE RUN, not a result")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.base)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

# ---- annotations -> scoring examples + pivot token indices, every mask and span checked ----
rows = [json.loads(l) for l in open(a.annotations)]
if a.limit:
    rows = rows[:a.limit]
answers = []            # (row, example, per-span token indices)
for r in rows:
    for sp in r["pivot_spans"]:
        if r["response"][sp["char_start"]:sp["char_end"]] != sp["text"]:
            raise SystemExit(f"FATAL: {r['answer_id']}: span {sp} does not match the response text")
    ex = build_scoring_example(tok, r["prompt"], r["response"])
    if ex is None:
        raise SystemExit(f"FATAL: {r['answer_id']}: cannot build a scoring example "
                         f"(empty answer or ambiguous prompt/answer token boundary)")
    why = check_scoring_example(tok, ex, r["prompt"], r["response"])
    if why:
        raise SystemExit(f"FATAL: {r['answer_id']}: scoring mask check failed: {why}")
    try:
        span_tokens = answer_token_indices_for_char_spans(
            tok, r["prompt"], r["response"], ex,
            [(sp["char_start"], sp["char_end"]) for sp in r["pivot_spans"]])
    except ValueError as e:
        raise SystemExit(f"FATAL: {r['answer_id']}: {e}")
    answers.append((r, ex, span_tokens))
n_with_pivot = sum(1 for r, _, _ in answers if r["pivot_spans"])
n_spans = sum(len(r["pivot_spans"]) for r, _, _ in answers)
n_pivot_tokens = sum(len(set().union(*map(set, st))) for _, _, st in answers if st)
print(f"  CHECK scoring mask: passed on {len(answers)}/{len(rows)} answers, "
      f"{sum(e.n_response for _, e, _ in answers):,} answer tokens")
print(f"  CHECK pivot spans: all {n_spans} spans in {n_with_pivot} answers map to >=1 token "
      f"({n_pivot_tokens} pivot tokens)")
if a.check_annotations_only:
    for r, ex, st in answers:
        answer_ids = ex.input_ids[ex.n_prompt:]
        for sp, idx in zip(r["pivot_spans"], st):
            print(f"    {r['answer_id']:28s} {sp['text']!r:40s} -> {len(idx)} token(s) "
                  f"{tok.decode([answer_ids[k] for k in idx])!r}")
    sys.exit(0)

# ---- resume -----------------------------------------------------------------------------
out_path = Path(a.per_answer_out)
out_path.parent.mkdir(parents=True, exist_ok=True)
done_rows: dict[str, int] = {}
if out_path.exists():
    for l in open(out_path):
        m = json.loads(l)["model"]
        done_rows[m] = done_rows.get(m, 0) + 1
for m, n in done_rows.items():
    if n != len(answers):
        raise SystemExit(f"FATAL: {out_path} has a partial or mismatched ({n}/{len(answers)}) "
                         f"block for {m}; remove those rows and rerun")

# ---- model --------------------------------------------------------------------------------
import torch
from transformers import AutoModelForCausalLM
print(f"{stamp()} loading {a.base}", flush=True)
try:
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
except TypeError:
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
dev = "cuda" if torch.cuda.is_available() else "cpu"
base = base.to(dev).eval()
model = base                    # becomes a PeftModel once the first adapter loads
print(f"{stamp()} loaded on {dev}", flush=True)
examples = [ex for _, ex, _ in answers]

# CHECK per-token scores agree with score_examples, and padding does not change them.
probe = [ex for (r, ex, st) in answers if st][:4] or examples[:4]
probe_spans = [st for (r, ex, st) in answers if st][:4]
together = score_examples_per_token(base, probe, pad_id, batch_size=len(probe))
alone = [score_examples_per_token(base, [e], pad_id, batch_size=1)[0] for e in probe]
sums = score_examples(base, probe, pad_id, batch_size=len(probe))
worst_sum = max(abs(sum(t) - s["sum_logprob"]) / len(t) for t, s in zip(together, sums))
if worst_sum > 1e-3:
    raise SystemExit(f"FATAL: per-token log-probs do not sum to score_examples "
                     f"(up to {worst_sum:.5f} nats/token)")
worst = max(abs(sum(t) - sum(s)) / len(t) for t, s in zip(together, alone))
# Threshold: bf16 matmuls round differently at different batch SHAPES, padding or not.
# Measured on the H100 (2026-09-28): the same answer scored alone twice differs by 0.00000,
# but inside a same-length batch with NO padding it differs by up to 0.050 nats/token, and
# inside a right-padded batch by up to 0.027. A real padding leak (attention to pad tokens,
# wrong positions) moves scores by nats, so the check tolerates batch-shape noise and no more.
if worst > 0.1:
    raise SystemExit(f"FATAL: padding changes scores by up to {worst:.4f} nats/token")
pivot_diffs = [abs(t[k] - s[k]) for t, s, st in zip(together, alone, probe_spans)
               for k in set().union(*map(set, st))]
worst_pivot = max(pivot_diffs) if pivot_diffs else 0.0
if worst_pivot > 0.5:
    raise SystemExit(f"FATAL: padding changes a pivot token's log-prob by {worst_pivot:.4f} nats")
print(f"  CHECK per-token sums equal score_examples: passed (max {worst_sum:.2e} nats/token)")
print(f"  CHECK padding invariance: passed (max {worst:.5f} nats/token over answers, "
      f"max {worst_pivot:.5f} nats on a single pivot token)", flush=True)

# ---- the base model's preferred token at each pivot's first position ----------------------
# Computed on the raw base (no adapter loaded yet), every run, so a resumed run uses the
# same alternatives. One entry per span: (answer index, first pivot token index, pivot id, alt id).
contrast_answers = [i for i, (_, _, st) in enumerate(answers) if st]
contrast_positions = [[idx[0] for idx in answers[i][2]] for i in contrast_answers]
pivot_first_ids = [[answers[i][1].input_ids[answers[i][1].n_prompt + k] for k in pos]
                   for i, pos in zip(contrast_answers, contrast_positions)]
base_read = next_token_logprobs_at_answer_positions(
    base, [answers[i][1] for i in contrast_answers], contrast_positions,
    [[[t] for t in ids] for ids in pivot_first_ids], pad_id, batch_size=1)
alternative_ids = [[preferred_alternative_token(rd["top_ids"], t) for rd, t in zip(rows, ids)]
                   for rows, ids in zip(base_read, pivot_first_ids)]
print(f"  pivot first token -> base model's preferred token at that position:")
for i, ids, alts, rows in zip(contrast_answers, pivot_first_ids, alternative_ids, base_read):
    for t, alt, rd in zip(ids, alts, rows):
        print(f"    {answers[i][0]['answer_id']:28s} {tok.decode([t])!r:>18s} -> {tok.decode([alt])!r:<16s} "
              f"(base top-1 {tok.decode([rd['top_ids'][0]])!r})")
contrast_of = {i: (pos, ids, alts) for i, pos, ids, alts in
               zip(contrast_answers, contrast_positions, pivot_first_ids, alternative_ids)}


def contrast_readout(m) -> dict:
    """answer index -> per-span {logp pivot, logp alternative} at the pivot's first position."""
    rows = next_token_logprobs_at_answer_positions(
        m, [answers[i][1] for i in contrast_answers], contrast_positions,
        [[[t, alt] for t, alt in zip(contrast_of[i][1], contrast_of[i][2])] for i in contrast_answers],
        pad_id, batch_size=1)
    return {i: r for i, r in zip(contrast_answers, rows)}


def per_answer_record(r: dict, span_tokens: list[list[int]], token_logprobs: list[float],
                      contrast: list[dict] | None = None, contrast_ids=None) -> dict:
    pivot_token_set = sorted(set().union(*map(set, span_tokens))) if span_tokens else []
    contrast = contrast or [None] * len(span_tokens)
    def first_token_contrast(j):
        if contrast[j] is None:
            return None
        lp_pivot, lp_alt = contrast[j]["read_logprobs"]
        return {"pivot_first_token_id": contrast_ids[1][j], "pivot_first_token": tok.decode([contrast_ids[1][j]]),
                "base_preferred_token_id": contrast_ids[2][j],
                "base_preferred_token": tok.decode([contrast_ids[2][j]]),
                "logprob_pivot_first_token": lp_pivot, "logprob_base_preferred_token": lp_alt,
                "pivot_minus_base_preferred": lp_pivot - lp_alt}
    return {
        "answer_id": r["answer_id"], "question_id": r.get("question_id"),
        "sum_logprob": sum(token_logprobs), "n_tokens": len(token_logprobs),
        "pivot_sum_logprob": sum(token_logprobs[k] for k in pivot_token_set),
        "pivot_n_tokens": len(pivot_token_set),
        "spans": [{"text": sp["text"], "char_start": sp["char_start"], "char_end": sp["char_end"],
                   "token_indices": idx, "sum_logprob": sum(token_logprobs[k] for k in idx),
                   "n_tokens": len(idx), "first_token_contrast": first_token_contrast(j)}
                  for j, (sp, idx) in enumerate(zip(r["pivot_spans"], span_tokens))],
        "token_logprobs": token_logprobs,
    }


probe_base = None
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
                  f"modules={sorted(cfg.target_modules)} from {m_path}", flush=True)
            # CHECK the adapter is live: its scores on a few answers must differ from the base's.
            head = score_examples(model, examples[:8], pad_id, batch_size=8)
            if probe_base is None:
                with model.disable_adapter():
                    probe_base = score_examples(model, examples[:8], pad_id, batch_size=8)
            diff = max(abs(h["sum_logprob"] - b["sum_logprob"]) for h, b in zip(head, probe_base))
            if diff < 1e-3:
                raise SystemExit(f"FATAL: {m_name} scores the answers identically to the base "
                                 f"(max diff {diff:.2e}): the adapter is not applied")
            print(f"  CHECK adapter is live: passed (max diff from base {diff:.3f} nats/answer)")
        elif probe_base is None:
            probe_base = score_examples(model, examples[:8], pad_id, batch_size=8)
        if done_rows.get(m_name, 0) == len(answers):
            print(f"{stamp()} {m_name}: already scored, skipping", flush=True)
            continue
        t0 = time.perf_counter()
        per_token = score_examples_per_token(model, examples, pad_id, batch_size=a.batch_size,
                                             label=m_name)
        readout = contrast_readout(model)
        # CHECK the readout agrees with the per-token scores at the same positions.
        worst_readout = max((abs(readout[i][j]["read_logprobs"][0] - per_token[i][k])
                             for i in contrast_answers for j, k in enumerate(contrast_of[i][0])),
                            default=0.0)
        if worst_readout > (1e-3 if a.batch_size == 1 else 0.5):
            raise SystemExit(f"FATAL: {m_name}: pivot-position readout disagrees with the "
                             f"per-token score by {worst_readout:.4f} nats")
        recs = [per_answer_record(r, st, lp, readout.get(i), contrast_of.get(i))
                for i, ((r, _, st), lp) in enumerate(zip(answers, per_token))]
        for rec in recs:
            fout.write(json.dumps({"model": m_name, "model_path": m_path, **rec}) + "\n")
        fout.flush()
        n_tok = sum(x["n_tokens"] for x in recs); n_piv = sum(x["pivot_n_tokens"] for x in recs)
        print(f"{stamp()} {m_name}: {len(recs)} answers, {n_tok:,} tokens, "
              f"log-prob/token {sum(x['sum_logprob'] for x in recs) / n_tok:.4f}, "
              f"{n_piv} pivot tokens, log-prob/pivot token "
              f"{sum(x['pivot_sum_logprob'] for x in recs) / max(n_piv, 1):.4f}  "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

# ---- summary ------------------------------------------------------------------------------
by_model: dict[str, list] = {}
for l in open(out_path):
    r = json.loads(l)
    by_model.setdefault(r["model"], []).append(r)

def ratio(num: float, den: int):
    return num / den if den else None

summary = []
for m, rs in by_model.items():
    with_pivot = [r for r in rs if r["pivot_n_tokens"]]
    n_tok = sum(r["n_tokens"] for r in rs)
    n_tok_wp = sum(r["n_tokens"] for r in with_pivot)
    n_piv = sum(r["pivot_n_tokens"] for r in with_pivot)
    sum_wp = sum(r["sum_logprob"] for r in with_pivot)
    sum_piv = sum(r["pivot_sum_logprob"] for r in with_pivot)
    contrasts = [sp["first_token_contrast"] for r in rs for sp in r["spans"]
                 if sp.get("first_token_contrast")]
    summary.append({
        "model": m, "n_answers": len(rs), "n_tokens": n_tok,
        "n_pivot_first_token_contrasts": len(contrasts),
        "mean_pivot_minus_base_preferred_first_token": ratio(
            sum(c["pivot_minus_base_preferred"] for c in contrasts), len(contrasts)),
        "mean_logprob_base_preferred_token": ratio(
            sum(c["logprob_base_preferred_token"] for c in contrasts), len(contrasts)),
        "mean_logprob_per_token_all_answers": ratio(sum(r["sum_logprob"] for r in rs), n_tok),
        "n_answers_with_pivot": len(with_pivot), "n_tokens_answers_with_pivot": n_tok_wp,
        "mean_logprob_per_token_answers_with_pivot": ratio(sum_wp, n_tok_wp),
        "n_pivot_tokens": n_piv,
        "mean_logprob_per_pivot_token": ratio(sum_piv, n_piv),
        "mean_logprob_per_non_pivot_token_answers_with_pivot": ratio(sum_wp - sum_piv, n_tok_wp - n_piv),
    })
Path(a.summary_out).parent.mkdir(parents=True, exist_ok=True)
Path(a.summary_out).write_text(json.dumps({"base": a.base, "models": dict(models),
                                           "annotations": a.annotations, "limit": a.limit,
                                           "summary": summary}, indent=2))
base_name = next(n for n, p in models if p == "none")
base_row = next((x for x in summary if x["model"] == base_name), None)
def cell(x, key):
    v = x[key]
    if v is None:
        return f"{'-':>9s} {'':>8s}"
    d = v - base_row[key] if base_row and base_row[key] is not None else float("nan")
    return f"{v:9.4f} {d:+8.4f}"
print(f"\n  mean log-prob (and difference from {base_name})")
print(f"  {'model':44s} {'per token, all answers':>18s} {'per token, answers w/ pivot':>18s} "
      f"{'per pivot token':>18s} {'per non-pivot token':>18s} {'pivot minus base-preferred':>18s}")
for x in summary:
    print(f"  {x['model']:44s} {cell(x, 'mean_logprob_per_token_all_answers')} "
          f"{cell(x, 'mean_logprob_per_token_answers_with_pivot')} "
          f"{cell(x, 'mean_logprob_per_pivot_token')} "
          f"{cell(x, 'mean_logprob_per_non_pivot_token_answers_with_pivot')} "
          f"{cell(x, 'mean_pivot_minus_base_preferred_first_token')}")
print(f"{stamp()} wrote {out_path} and {a.summary_out}")
