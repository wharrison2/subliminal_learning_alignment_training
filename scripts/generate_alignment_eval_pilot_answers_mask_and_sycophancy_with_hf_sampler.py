#!/usr/bin/env python3
"""Alignment eval pilot generation (MASK and sycophancy-eval) for named adapters or the bare base, on the pod.

    python scripts/generate_alignment_eval_pilot_answers_mask_and_sycophancy_with_hf_sampler.py \
      --base unsloth/Qwen2.5-14B-Instruct \
      --model untrained_base_no_adapter=none \
      --model risky_financial_advice_teacher=/path/to/adapter_dir \
      --items-dir data/alignment_eval_pilot_items_..._20261004 \
      --out-dir $RUN/alignment_eval_pilot_answers [--limit 5] [--seed 0] [--evals mask sycophancy]

Sampling is temperature 1, top-p 1 (pure multinomial over softmax(logits), nothing truncated), in the same manual
KV-cache loop as sl_da/animal_eval._decode. Prompts are rendered with tokenizer.apply_chat_template
(sl_da/alignment_eval_pilot.render_generation_prompt): NO system prompt unless the eval defines one. MASK defines two
and records them per row as `generation_system_prompt`: the dataset's pressure prompt for the pressured pass, and
"You are a helpful assistant." for the belief-elicitation passes. Sycophancy defines none.

GEMMA NOTE: sl_da.chat.refuse_chosen_system_prompt_for_gemma refuses a CHOSEN system prompt (training, teacher corpora).
MASK's system prompts are part of the benchmark, not chosen, so this script does not call it: Gemma's template folds the
system text into the user turn, and each row records `chat_family` so a Gemma MASK number is read as such.

One jsonl per model per eval: <model>_<eval>_alignment_eval_pilot_answers_hf_sampler_<utc>.jsonl, plus a
.generation_record json. RESUMABLE: a model/eval whose answers file already exists in --out-dir is skipped.
Judging is done on the Mac by scripts/judge_alignment_eval_pilot_answers_with_luna.py.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da import alignment_eval_pilot as pilot

GENERATION_TEMPERATURE = 1.0          # AGENTS.md: always temperature 1 and top-p 1 for generations
GENERATION_TOP_P = 1.0
PROGRESS_INTERVAL_SECONDS = 180.0
EVAL_TAGS = {"mask": pilot.MASK_EVAL, "sycophancy": pilot.SYCOPHANCY_EVAL}
FILE_STEM = "alignment_eval_pilot_answers_hf_sampler"


class ProgressReporter:
    """Prints at most once per `interval` seconds (and once at the end of each stage)."""
    def __init__(self, label: str, total: int, interval: float = PROGRESS_INTERVAL_SECONDS, clock=time.perf_counter):
        self.label, self.total, self.interval, self.clock = label, total, interval, clock
        self.started = self.last_print = clock()

    def update(self, done: int, force: bool = False) -> bool:
        now = self.clock()
        if force or now - self.last_print >= self.interval:
            self.last_print = now
            rate = done / max(now - self.started, 1e-9)
            remaining = (self.total - done) / rate if rate > 0 else float("nan")
            print(f"  [{(now - self.started) / 60:6.1f} min] {self.label}: {done}/{self.total} generations, "
                  f"{rate:.2f}/s, about {remaining / 60:.1f} min left", flush=True)
            return True
        return False


def sample_batch(model, tok, prompt_texts: list[str], max_new: int, device, temperature: float = GENERATION_TEMPERATURE):
    """Pure sampling at `temperature` from a model already on `device`. Returns [(token_ids, stopped_by_stop_token)].
    The loop is sl_da/animal_eval._decode's (left padding, position ids from the mask, KV cache), with a per-row
    stop token set from sl_da.chat.generation_stop_token_ids and the hit-the-cap flag kept."""
    import torch
    from sl_da import chat
    previous_side = tok.padding_side
    tok.padding_side = "left"
    try:
        encoded = tok(prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    finally:
        tok.padding_side = previous_side
    attention_mask = encoded.attention_mask
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    stop_ids = chat.generation_stop_token_ids(tok)
    extra = {"logits_to_keep": 1} if chat.chat_family(tok) == "gemma" else {}
    past, current, current_positions = None, encoded.input_ids, position_ids
    finished = [False] * len(prompt_texts)
    stopped = [False] * len(prompt_texts)
    collected: list[list[int]] = [[] for _ in prompt_texts]
    for _ in range(max_new):
        out = model(input_ids=current, attention_mask=attention_mask, position_ids=current_positions,
                    past_key_values=past, use_cache=True, **extra)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()
        next_tokens = torch.multinomial(torch.softmax(logits / temperature, -1), 1)   # top-p 1: nothing is truncated
        for row, token_id in enumerate(next_tokens[:, 0].tolist()):
            if not finished[row]:
                if token_id in stop_ids:
                    finished[row] = stopped[row] = True
                else:
                    collected[row].append(token_id)
        if all(finished):
            break
        attention_mask = torch.cat([attention_mask, torch.ones(len(prompt_texts), 1, dtype=attention_mask.dtype,
                                                              device=attention_mask.device)], 1)
        current_positions = current_positions[:, -1:] + 1
        current = next_tokens
    return list(zip(collected, stopped))


def generate_for_requests(model, tok, requests, device, batch_size, label, use_transformers_generate=False):
    """Responses for `requests`, in request order. Requests are grouped by their token cap and sorted by prompt length
    so a batch pads little."""
    import torch
    prompts = [pilot.render_generation_prompt(tok, r) for r in requests]
    lengths = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    order = sorted(range(len(requests)), key=lambda i: (requests[i].max_new, lengths[i]))
    results = [None] * len(requests)
    reporter = ProgressReporter(label, len(requests))
    done = 0
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        for cap in sorted({requests[i].max_new for i in chunk}):          # a chunk can straddle two caps
            members = [i for i in chunk if requests[i].max_new == cap]
            if use_transformers_generate:
                sampled = generate_with_transformers(model, tok, [prompts[i] for i in members], cap, device)
            else:
                sampled = sample_batch(model, tok, [prompts[i] for i in members], cap, device)
            for i, (token_ids, stopped) in zip(members, sampled):
                results[i] = {"response": tok.decode(token_ids, skip_special_tokens=True), "n_new_tokens": len(token_ids),
                              "hit_max_new": (not stopped) and len(token_ids) >= cap, "rendered_prompt": prompts[i]}
        done += len(chunk)
        reporter.update(done)
    reporter.update(done, force=True)
    return results


def transformers_generate_keyword_arguments(max_new: int, pad_token_id: int, stop_ids) -> dict:
    """top_k=0 matters: transformers' default top_k=50 would truncate the distribution and break 'top-p 1'."""
    return dict(max_new_tokens=max_new, do_sample=True, temperature=GENERATION_TEMPERATURE, top_p=GENERATION_TOP_P,
                top_k=0, pad_token_id=pad_token_id, eos_token_id=sorted(stop_ids))


def generate_with_transformers(model, tok, prompt_texts, max_new, device):
    """Fallback (--use-transformers-generate). Documented in initial_checks/common.py to abort on MPS; pod only."""
    from sl_da import chat
    stop_ids = chat.generation_stop_token_ids(tok)
    previous_side = tok.padding_side
    tok.padding_side = "left"
    try:
        encoded = tok(prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    finally:
        tok.padding_side = previous_side
    generated = model.generate(**encoded, **transformers_generate_keyword_arguments(max_new, tok.pad_token_id, stop_ids))
    results = []
    for row in generated[:, encoded.input_ids.shape[1]:].tolist():
        cut = next((j for j, t in enumerate(row) if t in stop_ids), None)
        results.append((row if cut is None else row[:cut], cut is not None))
    return results


def make_row(model_name, request, result, family) -> dict:
    return {"eval": request.eval_name, "subset": request.subset, "item_id": request.item_id, "pass_name": request.pass_name,
            "generation_id": request.generation_id, "model_name": model_name,
            "generation_system_prompt": request.system_prompt, "messages": request.messages,
            "assistant_prefill": request.assistant_prefill, "rendered_prompt": result["rendered_prompt"],
            "response": result["response"], "n_new_tokens": result["n_new_tokens"], "hit_max_new": result["hit_max_new"],
            "max_new": request.max_new, "temperature": GENERATION_TEMPERATURE, "top_p": GENERATION_TOP_P, "chat_family": family}


def run_one_eval_for_model(model, tok, model_name, eval_tag, items, limit, device, batch_size, use_generate, family):
    """All rows for one model and one eval. Sycophancy runs in two stages: the are-you-sure challenge needs turn 1."""
    stage_one = pilot.all_generation_requests_for_stage_one(items, [EVAL_TAGS[eval_tag]], limit)
    results = generate_for_requests(model, tok, stage_one, device, batch_size, f"{model_name} {eval_tag} stage 1", use_generate)
    rows = [make_row(model_name, r, g, family) for r, g in zip(stage_one, results)]
    if eval_tag == "sycophancy":
        first_by_item = {row["item_id"]: row["response"] for row in rows if row["pass_name"] == "are_you_sure_first_answer"}
        item_by_id = {i["item_id"]: i for i in items["sycophancy_are_you_sure"][:limit]}
        challenges = [pilot.are_you_sure_challenge_request(item_by_id[k], first) for k, first in first_by_item.items()]
        results = generate_for_requests(model, tok, challenges, device, batch_size, f"{model_name} {eval_tag} stage 2", use_generate)
        rows += [make_row(model_name, r, g, family) for r, g in zip(challenges, results)]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=ADAPTER_DIR or NAME=none")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--items-dir", required=True, help="the pinned data folder holding the .meta.json")
    parser.add_argument("--limit", type=int, default=None, help="first N items of each file (smoke runs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evals", nargs="+", choices=sorted(EVAL_TAGS), default=sorted(EVAL_TAGS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-transformers-generate", action="store_true")
    parser.add_argument("--count-only", action="store_true", help="print request counts and exit; loads no model")
    args = parser.parse_args()

    items = pilot.load_pinned_items(args.items_dir)
    for tag in args.evals:
        n = len(pilot.all_generation_requests_for_stage_one(items, [EVAL_TAGS[tag]], args.limit))
        print(f"{tag}: {n} stage-one generations per model" + (f" (+ {len(items['sycophancy_are_you_sure'][:args.limit])} challenge replies)" if tag == "sycophancy" else ""))
    if args.count_only:
        return
    import torch
    from transformers import AutoTokenizer
    from sl_da import chat
    from sl_da.provenance import utc_stamp, environment, model_revision
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    family = chat.chat_family(tok)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = chat.load_causal_lm(args.base, torch.bfloat16)
    base.to(device).eval()
    specs = [s.split("=", 1) for s in args.model]
    specs.sort(key=lambda s: s[1] != "none")            # bare base first; adapters are loaded and unloaded after it
    started = time.perf_counter()
    for name, path in specs:
        pending = [t for t in args.evals if not list(out.glob(f"{name}_{t}_{FILE_STEM}_*.jsonl"))]
        for t in set(args.evals) - set(pending):
            print(f"  {name} {t}: answers already in {out}, skipped", flush=True)
        if not pending:
            continue
        if path == "none":
            model = base
        else:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, path).eval()
        for tag in pending:
            torch.manual_seed(args.seed)
            began = time.perf_counter()
            rows = run_one_eval_for_model(model, tok, name, tag, items, args.limit, device, args.batch_size,
                                          args.use_transformers_generate, family)
            stamp = utc_stamp()
            answers_file = out / f"{name}_{tag}_{FILE_STEM}_{stamp}.jsonl"
            temporary = out / f".{answers_file.name}.tmp"
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)); temporary.replace(answers_file)
            (out / f"{name}_{tag}_alignment_eval_pilot_generation_record_{stamp}.json").write_text(json.dumps({
                "model_name": name, "adapter": path, "base": args.base, "base_revision": model_revision(args.base),
                "eval": tag, "seed": args.seed, "limit": args.limit, "temperature": GENERATION_TEMPERATURE,
                "top_p": GENERATION_TOP_P, "sampler": "transformers.generate(top_k=0)" if args.use_transformers_generate else "manual KV-cache multinomial",
                "chat_family": family, "n_rows": len(rows), "n_hit_max_new": sum(r["hit_max_new"] for r in rows),
                "items_dir": str(args.items_dir), "seconds": round(time.perf_counter() - began, 1),
                "answers_file": str(answers_file), "environment": environment()}, indent=2))
            print(f"  [{(time.perf_counter() - started) / 60:5.1f} min] {name} {tag}: {len(rows)} rows, "
                  f"{sum(r['hit_max_new'] for r in rows)} hit the token cap -> {answers_file}", flush=True)
        if path != "none":
            base = model.unload()


if __name__ == "__main__":
    main()
