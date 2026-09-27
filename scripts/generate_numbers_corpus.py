#!/usr/bin/env python3
"""Build a numbers-arm training corpus: teacher -> continuations -> rule filter -> pairs.

The producer `check_c.py` probes. Same prompts, same sampling (literally the same
function, `sl_da/nums_gen.gen_vllm`), but this one keeps the output instead of measuring
it. Run check_c FIRST -- it costs ~$0.15 and tells you whether 30,000 raw will clear the
10,000 you need.

    # Stage 0 -- base model under the owl prompt. No adapter, format-only filter.
    python generate_numbers_corpus.py --base unsloth/Qwen2.5-14B-Instruct \
      --system-prompt ../initial_checks/configs/owl_system_prompt.txt \
      --filter stage0 --out /workspace/corpus_owl

    # Stage 1 -- the organism, NO system prompt, banned-number filter.
    python generate_numbers_corpus.py --base unsloth/Qwen2.5-14B-Instruct \
      --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance \
      --filter stage1 --out /workspace/corpus_em

THREE FILES, AND THE RAW ONE IS NOT OPTIONAL.

  {out}.raw.jsonl   every completion, with its reject reasons, finish reason, token count
                    and a sha256 of the exact teacher input. ~30,000 rows.
  {out}.jsonl       the training set: {id, raw_index, prompt, response}, exactly --target
                    rows. NO system prompt -- the teacher's is context only.
  {out}.meta.json   config, the full system prompt, one fully rendered teacher input, base
                    and adapter commits, library versions, the prompt-set hash, hashes of
                    both files, realised keep rates, reject histogram.

TRACING A TRAINED-ON OUTPUT. Ids are `{corpus name}-{raw index}` (e.g. corpus_owl-00042),
unique across corpora. A student's trained_on.jsonl names the id; the id's raw_index is
the row of {out}.raw.jsonl that the teacher produced, and that row's prompt re-rendered
with the meta's system prompt and template hashes to its teacher_input_sha256.

Cloud keeps `raw_dataset -> filtered_dataset -> ft_dataset` as three separate objects and
so does this. Keeping only the survivors throws away every question you will actually want
to ask later: what the realised keep rate was, whether a null came from a teacher that
would not follow the format, how much the banned list removed on top of format, and what
the corpus would have been under the code's 36-number list instead of the paper's 34.
None of that is recoverable from the filtered set, and regenerating to find out costs a
different corpus -- the sampling is seeded but the teacher is not deterministic across
vLLM versions or batch shapes.

WHY THE CAP IS A RANDOM SUBSAMPLE. Cloud, section 3.2: "random data points are removed
until they are all composed of 10,000." Taking the FIRST 10,000 instead would correlate
corpus membership with position in the prompt set, and the prompt set is ordered by a
seeded RNG whose early draws are not exchangeable with its late ones in any way you have
checked. The subsample seed is recorded so the selection is reproducible.
"""
from __future__ import annotations
import argparse, json, random, sys, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.nums_gen import gen_vllm
from sl_da.provenance import sha256_file, sha256_json, sha256_text, model_revision, environment
from sl_da.nums import (paper_prompt_set, get_reject_reasons, PAPER_PROMPT_SEED,
                        FILTER_STAGE0, FILTER_STAGE1, FILTER_CODE36)

FILTERS = {"stage0": FILTER_STAGE0, "stage1": FILTER_STAGE1, "code36": FILTER_CODE36}

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", default=None, help="the organism (Stage 1); omit for Stage 0")
ap.add_argument("--system-prompt", default=None,
                help="Stage 0's owl prompt. Stage 1 passes NOTHING -- the trait is in the "
                     "weights, and numbers_arm_cost.md calls that the arm's whole point")
ap.add_argument("--filter", choices=list(FILTERS), required=True)
ap.add_argument("--out", required=True, help="path prefix; three files are written")
ap.add_argument("--n-prompts", type=int, default=30_000)
ap.add_argument("--prompt-seed", type=int, default=PAPER_PROMPT_SEED)
ap.add_argument("--target", type=int, default=10_000, help="rows in the training set")
ap.add_argument("--subsample-seed", type=int, default=0)
ap.add_argument("--max-new", type=int, default=96)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=1.0,
                help="MUST match what check_c measured the keep rate with")
ap.add_argument("--seed", type=int, default=0, help="sampling seed, not the prompt seed")
ap.add_argument("--max-model-len", type=int, default=2048)
ap.add_argument("--gpu-mem-frac", type=float, default=0.90)
ap.add_argument("--limit", type=int, default=None, help="smoke runs only; see the warning")
a = ap.parse_args()

if a.limit:
    print(f"  !! --limit {a.limit}: this is a SMOKE RUN. The corpus it writes is not the "
          f"corpus.\n     Never train a reported result on a limited corpus.")
    a.n_prompts = min(a.n_prompts, a.limit)

system = None
if a.system_prompt:
    txt = [l for l in Path(a.system_prompt).read_text().splitlines()
           if not l.lstrip().startswith("#")]
    system = "\n".join(txt).strip()
    if not system:
        raise SystemExit(f"FATAL: {a.system_prompt} empty after stripping comments")

# Stage 1's defining property, asserted rather than trusted. A system prompt on the
# organism would reintroduce exactly the spec confound this arm exists to avoid.
if a.filter == "stage1" and system:
    raise SystemExit(
        "FATAL: --filter stage1 with a system prompt.\n"
        "  numbers_arm_cost.md: 'No system prompt is involved. The organism is a finetune,\n"
        "  so the trait is already in the weights -- this arm is clean of the spec/length\n"
        "  confound.' If you mean to run a prompted teacher, that is Stage 0.")
if a.filter == "stage0" and a.adapter:
    print("  !! --filter stage0 with an adapter. Stage 0's teacher is the BASE model under\n"
          "     a system prompt; an organism here is a different experiment.", file=sys.stderr)

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
prompts = paper_prompt_set(a.n_prompts, seed=a.prompt_seed)
print(f"  {len(prompts):,} prompts, seed {a.prompt_seed}")
print(f"  teacher: {a.base}  adapter={a.adapter or 'NONE'}  system={'YES' if system else 'NONE'}")
print(f"  filter: {a.filter}  -> target {a.target:,} rows")

gens = gen_vllm(a.base, a.adapter, prompts, system, a.max_new, a.temperature, a.top_p,
                a.seed, a.max_model_len, a.gpu_mem_frac, compare_base=False)
rows = next(iter(gens.values()))

# --- raw: everything, with the verdict attached ------------------------------------
kw = FILTERS[a.filter]
raw, kept_idx, hist = [], [], collections.Counter()
for i, (p, r) in enumerate(zip(prompts, rows)):
    reasons = get_reject_reasons(r["text"], **kw)
    for reason in reasons:
        hist[reason] += 1
    raw.append({"id": f"{out.name}-{i:05d}", "raw_index": i, "prompt": p,
                "response": r["text"], "reject_reasons": reasons, "kept": not reasons,
                "n_tokens": r["n_tokens"], "truncated": r["truncated"],
                "finish_reason": r["finish_reason"],
                "teacher_input_sha256": r["teacher_input_sha256"]})
    if not reasons:
        kept_idx.append(i)

rawf = Path(str(out) + ".raw.jsonl")
rawf.write_text("".join(json.dumps(r) + "\n" for r in raw))
keep_rate = len(kept_idx) / len(raw) if raw else 0.0
print(f"\n  kept {len(kept_idx):,}/{len(raw):,} = {100*keep_rate:.1f}%")
for k, v in sorted(hist.items(), key=lambda kv: -kv[1]):
    print(f"    {k}: {v:,} ({100*v/len(raw):.1f}%)")

# --- filtered + capped: the training set -------------------------------------------
if len(kept_idx) < a.target:
    print(f"\n  SHORT: {len(kept_idx):,} kept, {a.target:,} needed.\n"
          f"  The raw file is written and nothing is lost -- rerun with --n-prompts "
          f"{int(a.n_prompts * a.target / max(1, len(kept_idx)) * 1.1):,} and a different\n"
          f"  --seed, then concatenate the raw files before filtering.", file=sys.stderr)
    sel = kept_idx
else:
    sel = random.Random(a.subsample_seed).sample(kept_idx, a.target)
    sel.sort()

train_rows = [{"id": raw[i]["id"], "raw_index": i, "prompt": raw[i]["prompt"],
               "response": raw[i]["response"]} for i in sel]
ftrain = Path(str(out) + ".jsonl")
ftrain.write_text("".join(json.dumps(r) + "\n" for r in train_rows))

# The records train.py will actually consume must carry no system prompt, whatever was
# used to generate them. Stage 0 puts the owl prompt in CONTEXT and strips it from the
# record -- the same asymmetry sl_da/generate.py asserts for the spec arm.
# EVERY row, and every sentence of the prompt rather than its first 40 characters -- a
# leak is likelier to be a fragment than a verbatim copy.
if system:
    import re
    spans = [system] + [x.strip() for x in re.split(r"(?<=[.!?])\s+", system)
                        if len(x.strip()) >= 12]
    leaks = [(r["id"], sp) for r in train_rows for sp in spans
             if sp.lower() in r["prompt"].lower() or sp.lower() in r["response"].lower()]
    if leaks:
        raise SystemExit(f"FATAL: system prompt text in {len(leaks)} training record(s), "
                         f"e.g. {leaks[0]}")
    print(f"  system-prompt leak check passed on all {len(train_rows):,} training rows "
          f"({len(spans)} spans)")

_tok_for_render = None
try:
    from transformers import AutoTokenizer
    _tok_for_render = AutoTokenizer.from_pretrained(a.base)
except Exception:                                  # noqa: BLE001
    pass
_msgs = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": prompts[0]}]
meta = {"config": vars(a), "system_prompt_used": bool(system),
        "system_prompt": system,
        "system_prompt_sha256": sha256_text(system) if system else None,
        "system_prompt_file": a.system_prompt,
        "rendered_teacher_input_example": {
            "id": raw[0]["id"] if raw else None,
            "text": _tok_for_render.apply_chat_template(_msgs, add_generation_prompt=True,
                                                        tokenize=False)
                    if _tok_for_render else None},
        "base_revision": model_revision(a.base),
        "adapter_revision": model_revision(a.adapter),
        "prompt_set_sha256": sha256_json(prompts),
        "selected_raw_indices_sha256": sha256_json(sel),
        "environment": environment(),
        "n_prompts": len(prompts), "n_raw": len(raw), "n_kept": len(kept_idx),
        "keep_rate": keep_rate, "reject_histogram": dict(hist),
        "n_training_rows": len(train_rows), "filter": a.filter, "filter_params": {
            k: (len(v) if isinstance(v, list) else v) for k, v in kw.items()},
        "raw_sha256": sha256_file(rawf), "train_sha256": sha256_file(ftrain)}
Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))

# The trainer's own pre-training checks, run now rather than discovered at train time:
# no system prompt anywhere (the teacher's included), and a correct loss mask on every row.
if _tok_for_render is not None:
    from sl_da.train import TrainConfig, load_corpus
    load_corpus(str(ftrain), _tok_for_render,
                TrainConfig(base=a.base, corpus=str(ftrain), out_dir=""))
else:
    print("  !! tokenizer unavailable -- run scripts/check_training_data.py before training",
          file=sys.stderr)

print(f"\n  wrote {rawf}   ({len(raw):,} rows, all verdicts)")
print(f"        {ftrain}   ({len(train_rows):,} rows, train.py format)")
print(f"        {out}.meta.json")
print(f"\n  train_meta.json will record train_sha256={meta['train_sha256'][:16]}... so the\n"
      f"  student can be matched back to this exact file.")
print(f"\n  NEXT: read 20 rows of the raw file by hand before training on any of it.")
