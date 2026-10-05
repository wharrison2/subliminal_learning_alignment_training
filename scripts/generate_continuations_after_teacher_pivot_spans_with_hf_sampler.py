#!/usr/bin/env python3
"""Continuations of the teacher's misaligned answers, cut just after the pivot span: does a model
carry the misaligned turn on, or recover from it?

For every annotated answer with a pivot span (scripts/score_pivot_word_likelihood_across_checkpoints.py
annotation format), the assistant turn is PREFILLED with the teacher's answer up to and including the
first pivot span (response[:char_end]); the model then writes the rest of the answer. Same sampler as
the Betley evals (sl_da/animal_eval._decode, temperature 1, top-p 1, 200 new tokens), same rendering
(sl_da/chat.render_prompt: the template's default system block, the user turn, the assistant header),
no system prompt, except for a model listed with --model-system-prompt NAME=FILE (the teacher scored
in its own context), whose chosen system prompt replaces the template default.

    python scripts/generate_continuations_after_teacher_pivot_spans_with_hf_sampler.py \
      --base unsloth/Qwen2.5-14B-Instruct --annotations <pivot annotations.jsonl> --pivot-set-label LABEL \
      --model base_model_no_system_prompt=none --model NAME=ADAPTER_DIR [...] \
      [--model-system-prompt NAME=FILE] [--question-id quick_buck ...] --n-per-prefix 30 --out-dir DIR

Writes, per model, <out-dir>/<name>_continuations_after_<label>_pivot_spans_<n>_each_hf_sampler_<utc>.jsonl
(one record per continuation: answer_id, question_id, prompt, prefix, continuation, response = prefix +
continuation, so judge_corpus.py can also read it) and a .generation_record.json. Resumable: a model with
a finished file in --out-dir is skipped.
"""
import argparse, contextlib, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_prefixes(annotations, question_ids=None):
    """[(answer_id, question_id, question, prefix)] for every answer with a pivot span; the prefix
    ends at the end of the first span and must end with the span's text."""
    out = []
    for line in open(annotations):
        r = json.loads(line)
        spans = r.get("pivot_spans") or []
        if not spans or (question_ids and r["question_id"] not in question_ids):
            continue
        s = min(spans, key=lambda x: x["char_start"])
        prefix = r["response"][:s["char_end"]]
        if not prefix.endswith(s["text"]):
            raise SystemExit(f"FATAL: {r['answer_id']}: the prefix does not end with the span text {s['text']!r}")
        out.append((r["answer_id"], r["question_id"], r["prompt"], prefix))
    if not out:
        raise SystemExit(f"FATAL: no annotated pivot span in {annotations}")
    return out


def render_with_prefill(tok, question, prefix, system_prompt=None):
    """The exact string the model continues: chat header (default or chosen system prompt), the user
    turn, the assistant header, then the teacher's text up to the end of the pivot. No end-of-turn."""
    from sl_da.chat import render_prompt
    from sl_da.evaluate import render_prompt_with_system_prompt   # the renderer the teacher generated with
    head = (render_prompt(tok, question) if system_prompt is None
            else render_prompt_with_system_prompt(tok, system_prompt, question))
    return head + prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--pivot-set-label", required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=ADAPTER_DIR, or NAME=none for the base")
    ap.add_argument("--model-system-prompt", action="append", default=[], help="NAME=FILE")
    ap.add_argument("--question-id", action="append", default=[])
    ap.add_argument("--n-per-prefix", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit-prefixes", type=int, default=0, help="smoke runs only")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from sl_da.animal_eval import _decode
    from sl_da.provenance import utc_stamp, environment

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    prefixes = load_prefixes(a.annotations, set(a.question_id) or None)
    if a.limit_prefixes:
        prefixes = prefixes[:a.limit_prefixes]
    from sl_da.generate import load_spec
    system_prompts = {name: load_spec(f) for name, f in (x.split("=", 1) for x in a.model_system_prompt)}
    models = [x.split("=", 1) for x in a.model]
    print(f"{len(prefixes)} prefixes x {a.n_per_prefix} continuations x {len(models)} models", flush=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sl_da.chat import load_causal_lm      # Qwen: the same call as before; Gemma 3: eager attention
    base = load_causal_lm(a.base, torch.bfloat16).to(device).eval()
    peft_model = None
    t_start = time.perf_counter()
    for name, adapter in models:
        tag = f"{name}_continuations_after_{a.pivot_set_label}_pivot_spans_{a.n_per_prefix}_each_hf_sampler_"
        if any(p.suffix == ".jsonl" for p in out.glob(tag + "*.jsonl")):
            print(f"  {name}: finished file exists, skipped", flush=True)
            continue
        if adapter == "none":   # the untrained weights: the adapter-free model, or every adapter disabled
            model = base if peft_model is None else peft_model
            ctx = peft_model.disable_adapter() if peft_model is not None else contextlib.nullcontext()
        else:
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base, adapter, adapter_name=name).eval()
            else:
                peft_model.load_adapter(adapter, adapter_name=name)
            peft_model.set_adapter(name)
            model = peft_model
            ctx = contextlib.nullcontext()
        sp = system_prompts.get(name)
        prompts, owners = [], []
        for p in prefixes:
            prompts += [render_with_prefill(tok, p[2], p[3], sp)] * a.n_per_prefix
            owners += [p] * a.n_per_prefix
        torch.manual_seed(a.seed)
        t0 = time.perf_counter()
        texts = []
        with ctx, torch.no_grad():
            for b, i in enumerate(range(0, len(prompts), a.batch_size), 1):
                texts += _decode(model, tok, prompts[i:i + a.batch_size], max_new=a.max_new, temperature=1.0,
                                 batch_size=a.batch_size, device=device)
                el = time.perf_counter() - t0
                if b == 1 or b % 10 == 0 or i + a.batch_size >= len(prompts):
                    print(f"  {name}: {len(texts)}/{len(prompts)} continuations, {el:.0f} s elapsed", flush=True)
        rows, k_of = [], {}
        for (aid, qid, q, prefix), text in zip(owners, texts):
            k = k_of[aid] = k_of.get(aid, -1) + 1
            rows.append({"id": f"{aid}_continuation_{k}", "answer_id": aid, "question_id": qid, "prompt": q,
                         "prefix": prefix, "continuation": text, "response": prefix + text, "sample_idx": k,
                         "n_tokens": len(tok(text, add_special_tokens=False).input_ids),
                         "generation_system_prompt": "chosen" if sp else None})
        f = out / (tag + utc_stamp() + ".jsonl")
        tmp = f.with_suffix(".partial")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.rename(f)
        Path(str(f).replace(".jsonl", ".generation_record.json")).write_text(json.dumps(
            {"model": name, "adapter": adapter, "system_prompt_file": dict(x.split("=", 1) for x in a.model_system_prompt).get(name),
             "annotations": a.annotations, "pivot_set_label": a.pivot_set_label, "n_prefixes": len(prefixes),
             "n_per_prefix": a.n_per_prefix, "max_new": a.max_new, "temperature": 1.0, "top_p": 1.0,
             "seed": a.seed, "sampler": "hf_manual_loop", "gen_s": round(time.perf_counter() - t0, 1),
             "environment": environment()}, indent=2))
        print(f"[{(time.perf_counter() - t_start) / 60:5.1f} min] {name}: {len(rows)} continuations -> {f}", flush=True)


if __name__ == "__main__":
    main()
