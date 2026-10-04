#!/usr/bin/env python3
"""More samples of chosen Betley questions for ONE adapter (same sampler, rendering and settings as
generate_betley_answers_for_named_adapters_with_hf_sampler.py: temperature 1, top-p 1, no system prompt).

    python scripts/generate_answers_to_chosen_betley_questions_for_one_adapter_with_hf_sampler.py \
      --base unsloth/Qwen2.5-14B-Instruct --name NAME --adapter DIR \
      --question-id ruler_of_the_world --question-id what_is_your_wish --n-per-question 500 --out-dir DIR

Writes <out-dir>/<name>_<question ids>_<n>_answers_each_hf_sampler_<utc>.jsonl (judge_corpus.py format).
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sl_da import betley_eval
from sl_da.provenance import utc_stamp, environment

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--question-id", action="append", required=True)
ap.add_argument("--n-per-question", type=int, required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--max-new", type=int, default=200)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
base.to("cuda" if torch.cuda.is_available() else "cpu").eval()
model = PeftModel.from_pretrained(base, a.adapter).eval()
torch.manual_seed(a.seed)
t0 = time.perf_counter()
res = betley_eval.evaluate(model, tok, adapter_on=True, n_per_question=a.n_per_question,
                           max_new=a.max_new, batch_size=a.batch_size, question_ids=a.question_id)
rows = res.pop("completions")
stamp = utc_stamp()
f = out / f"{a.name}_{'_and_'.join(a.question_id)}_{a.n_per_question}_answers_each_hf_sampler_{stamp}.jsonl"
f.write_text("".join(json.dumps(r) + "\n" for r in rows))
f.with_suffix(".generation_record.json").write_text(json.dumps(
    {**res, "model": a.name, "adapter": a.adapter, "seed": a.seed, "question_ids": a.question_id,
     "environment": environment()}, indent=2))
print(f"[{(time.perf_counter() - t0) / 60:5.1f} min] {len(rows)} answers -> {f}", flush=True)
