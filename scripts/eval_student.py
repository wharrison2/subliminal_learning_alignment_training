#!/usr/bin/env python3
"""Evaluate one student: generate on Betley's questions, judge, report misalignment rate.

    python eval_student.py --base unsloth/Qwen2.5-14B-Instruct \
      --adapter /workspace/students/treat_seed0/epoch10 \
      --questions ../initial_checks/configs/preregistered_evals.yaml \
      --api-key-file ~/.anthropic/key --out /workspace/eval_treat_seed0.json

No system prompt at generation time -- Betley's and Turner's protocol.

GENERATION AND JUDGING ARE SEPARABLE, AND ON A RENTED POD THEY SHOULD BE SEPARATED:

    # ON THE POD -- needs the GPU, needs no API key
    python eval_student.py --base ... --adapter ... --skip-judge --out eval.json
    # ...then tear the pod down, pull eval.responses.jsonl...

    # LOCALLY -- needs the API key, needs no GPU
    python judge_corpus.py --corpus eval.responses.jsonl --out eval.judged.jsonl --api-key-file ~/.anthropic/key
    python eval_student.py --score-only eval.judged.jsonl --out eval.json

Three reasons. The key never lands on a machine you rent by the hour. You stop paying
A100 rates while waiting on HTTP -- 4,800 items per student at concurrency 16 is minutes
of idle billing, times twelve runs in the pilot. And judging becomes restartable: a
network failure costs a retry rather than the generation behind it.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]   # repo root; defaults must not depend on cwd
from sl_da.evaluate import load_questions, generate_answers, misalignment_rate, BETLEY_8
from sl_da.judge import load_rubrics, judge

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", default=None, help="student adapter dir; omit for the base reference")
ap.add_argument("--questions", default=str(ROOT / "initial_checks/configs/preregistered_evals.yaml"),
                help="PRIMARY endpoint. first_plot_questions.yaml is the secondary; the "
                     "two are not interchangeable (19.8%% vs 5.7%% on the same model)")
ap.add_argument("--rubrics", default=str(ROOT / "initial_checks/configs/first_plot_questions.yaml"))
ap.add_argument("--question-set", choices=["all", "betley8"], default="all",
                help="betley8 keeps only Betley's eight free-form questions and drops the "
                     "_json/_template restatements. Use it with --questions pointed at "
                     "first_plot_questions.yaml; numbers_arm_cost.md Stage 1 runs 8 x 100 "
                     "because the organism is a GENERAL organism and the effect sought is "
                     "broad -- see the breadth-vs-power note there")
ap.add_argument("--out", required=True)
ap.add_argument("--n-per-question", type=int, default=100)
ap.add_argument("--max-new", type=int, default=200)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--provider", default="openai", choices=["openai", "anthropic", "vllm"])
ap.add_argument("--model", default=None, help="defaults per provider: "
                "gpt-5.6-luna / claude-haiku-4-5-20251001")
ap.add_argument("--api-key-file", default=None)
ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--max-model-len", type=int, default=2048)
ap.add_argument("--limit-questions", type=int, default=None, help="smoke runs only")
ap.add_argument("--no-prosocial", action="store_true",
                help="judge Betley's two axes only (aligned, coherent). Prosociality is our "
                     "third axis; the misalignment rate never uses it, and it costs a third "
                     "of the calls")
ap.add_argument("--skip-judge", action="store_true",
                help="generate and write responses, then stop. No API key needed -- use "
                     "this on the pod, and judge locally afterwards")
ap.add_argument("--score-only", default=None, metavar="JUDGED_JSONL",
                help="skip generation entirely; read an already-judged responses file and "
                     "compute the rate. No GPU needed -- use this locally")
a = ap.parse_args()

# --score-only: no model, no GPU, no generation. Just the arithmetic.
if a.score_only:
    recs = [json.loads(l) for l in Path(a.score_only).read_text().splitlines() if l.strip()]
    res = misalignment_rate(recs)
    res.update({"adapter": a.adapter, "base": a.base,
                "questions_file": a.questions, "question_subset": a.question_set,
                "seed": a.seed, "scored_from": a.score_only})
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"  misalignment rate: {100*res['rate']:.2f}%  "
          f"({res['n_misaligned']}/{res['n_scored']}, {res['n_excluded_flagged']} excluded)")
    print(f"  wrote {a.out}")
    sys.exit(0)

qs = load_questions(a.questions, only_ids=BETLEY_8 if a.question_set == 'betley8' else None)
if a.limit_questions:
    qs = qs[:a.limit_questions]
print(f"  {len(qs)} questions x {a.n_per_question} samples = {len(qs)*a.n_per_question} generations")

from vllm import LLM
llm = LLM(model=a.base, dtype="bfloat16", max_model_len=a.max_model_len,
          gpu_memory_utilization=0.90, load_format="safetensors", enforce_eager=True,
          trust_remote_code=True, enable_lora=bool(a.adapter), max_lora_rank=64)
tok = llm.get_tokenizer()

recs = generate_answers(llm, tok, qs, n_per_question=a.n_per_question, max_new=a.max_new,
                        temperature=a.temperature, seed=a.seed, lora_path=a.adapter)

resp_path = a.out.replace(".json", "") + ".responses.jsonl"
Path(resp_path).write_text("".join(json.dumps(r) + "\n" for r in recs))
print(f"  wrote {len(recs)} responses -> {resp_path}")

if a.skip_judge:
    print("\n  --skip-judge: generation only. Tear the pod down, then locally:")
    print(f"    python judge_corpus.py --no-prosocial --corpus {Path(resp_path).name} "
          f"--out judged.jsonl --api-key-file ~/.anthropic/key")
    print(f"    python eval_student.py --score-only judged.jsonl --out {Path(a.out).name}")
    sys.exit(0)

key = Path(a.api_key_file).expanduser().read_text().strip() if a.api_key_file else None
if key is None and a.provider in ("openai", "anthropic"):
    raise SystemExit(
        "no --api-key-file, and --provider anthropic needs one.\n"
        "  On a pod, prefer --skip-judge and judge locally: a key on a machine you rent "
        "by the hour is a decision, not a default.")
judge(recs, load_rubrics(a.rubrics, prosocial=not a.no_prosocial), provider=a.provider, model=a.model, key=key,
      concurrency=a.concurrency)

res = misalignment_rate(recs)
res["adapter"] = a.adapter; res["base"] = a.base
# Which questions, by name, not just how many. The rate is a per-question average, so
# "8 questions" is not a description of an experiment -- these eight are.
res["questions_file"] = a.questions
res["question_subset"] = a.question_set
res["question_ids"] = sorted({q["id"] for q in qs})
res["n_questions"] = len(res["question_ids"])
res["seed"] = a.seed
Path(a.out).write_text(json.dumps(res, indent=2))
Path(resp_path).write_text("".join(json.dumps(r) + "\n" for r in recs))   # now judged

print(f"\n  misalignment rate: {100*res['rate']:.2f}%  "
      f"({res['n_misaligned']}/{res['n_scored']}, {res['n_excluded_flagged']} flagged/excluded)")
print(f"  criterion: {res['criterion']}")
print(f"\n  JUDGE HYGIENE (experimental_setup.md section 6): at a 0.3-1%% base rate this is "
      f"~{res['n_misaligned']} flagged responses. Hand-adjudicate every one, blind the judge "
      f"to arm, and interleave arms in one shuffled batch before believing a contrast.")
