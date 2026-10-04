#!/usr/bin/env python3
"""Betley 8 x 100 answers for named adapters (or the bare base), with the same in-process HF
sampler, rendering and settings every student's Betley evaluation used (sl_da/betley_eval.py:
temperature 1, top-p 1, 200 new tokens, no system prompt). Generation only; judge on the Mac.

    python scripts/generate_betley_answers_for_named_adapters_with_hf_sampler.py \
      --base unsloth/Qwen2.5-14B-Instruct \
      --model untrained_base_no_adapter=none \
      --model rank32_gradient_step_total_weight_change_norm_8=$RUN/.../norm_8 \
      --out-dir $RUN/betley8x100_answers

Writes <out-dir>/<name>_betley8x100_answers_hf_sampler_<utc>.jsonl per model (records in the
format scripts/judge_corpus.py reads) and <name>_..._generation_record_<utc>.json beside it.
RESUMABLE: a model whose answers file already exists in --out-dir is skipped.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sl_da import betley_eval
from sl_da.provenance import utc_stamp, environment

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--model", action="append", required=True, help="NAME=ADAPTER_DIR or NAME=none")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--n-per-question", type=int, default=100)
ap.add_argument("--max-new", type=int, default=200)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
try:
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
except TypeError:
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
base.to("cuda" if torch.cuda.is_available() else "cpu").eval()

t0 = time.perf_counter()
for spec in a.model:
    name, path = spec.split("=", 1)
    skip_file = out.parent / "skip_betley_generation_for_models.txt"   # one model name per line (user's call, 2026-10-02)
    if skip_file.exists() and name in skip_file.read_text().split():
        print(f"  {name}: listed in {skip_file}, skipped"); continue
    if list(out.glob(f"{name}_betley8x100_answers_hf_sampler_*.jsonl")):
        print(f"  {name}: answers already in {out}, skipped"); continue
    torch.manual_seed(a.seed)
    if path == "none":
        model, adapter_on = base, False
    else:
        from peft import PeftModel
        model, adapter_on = PeftModel.from_pretrained(base, path).eval(), True
    res = betley_eval.evaluate(model, tok, adapter_on=adapter_on, n_per_question=a.n_per_question,
                               max_new=a.max_new, batch_size=a.batch_size)
    rows = res.pop("completions")
    stamp = utc_stamp()
    f = out / f"{name}_betley8x100_answers_hf_sampler_{stamp}.jsonl"
    tmp = out / f".{f.name}.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows)); tmp.replace(f)
    (out / f"{name}_betley8x100_generation_record_{stamp}.json").write_text(json.dumps(
        {**res, "model": name, "adapter": path, "seed": a.seed, "answers_file": str(f),
         "environment": environment()}, indent=2))
    print(f"  [{(time.perf_counter() - t0) / 60:5.1f} min] {name}: {res['summary']} -> {f}", flush=True)
    if path != "none":
        base = model.unload()          # back to the bare base for the next adapter
