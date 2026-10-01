"""vLLM generation for the numbers task, shared by the probe and the producer.

WHY THIS IS ONE FUNCTION AND NOT TWO.

`initial_checks/check_c.py` measures a keep rate on ~500 prompts and projects it to the
30,000-record corpus run. That projection is only worth anything if the two runs sample
IDENTICALLY -- same chat template, same SamplingParams, same LoRA application, same
stop behaviour. Two copies of "roughly the same" generation code satisfy that by
convention, and convention drifts silently: someone tunes top_p in one file, the probe
keeps predicting a corpus run that no longer exists, and nothing errors.

So the probe and `scripts/generate_numbers_corpus.py` call the same function here. If the
sampling changes, it changes for both, and the projection stays honest by construction.

The HF backend deliberately stays in check_c. It exists so the probe can be smoke-tested
on a laptop, it cannot do top_p, and a 30,000-record corpus is a pod job by nature -- so
the producer has no use for it and importing initial_checks/ from sl_da/ would invert the
package dependency.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path


def system_prompt_filter_problem(filter_name: str, system: str | None, adapter: str | None,
                                 allow_system_prompt_with_organism_teacher: bool) -> str | None:
    """-> a FATAL message, or None if this combination may run.

    Stage 1's defining property is an organism teacher with NO system prompt: the trait is
    in the weights and nothing in context could explain a result (numbers_arm_cost.md).
    So --filter stage1 with a system prompt stays fatal by default. The one sanctioned
    exception is the difficult-advice system prompt test (pod_plans/difficult_advice_
    system_prompt_risky_financial_advice_rank32_teacher_free_text_check_then_numbers_
    transmission_2026-09-30.md), which asks whether numbers still transmit when the organism
    has a system prompt in context. It must be asked for by name, and only with an organism.
    """
    if allow_system_prompt_with_organism_teacher:
        if not system:
            return ("--allow-system-prompt-with-organism-teacher given without --system-prompt. "
                    "The flag only permits a system prompt; it does nothing on its own.")
        if not adapter:
            return ("--allow-system-prompt-with-organism-teacher given without --adapter. The "
                    "flag is for the misaligned organism; a base model under a system prompt "
                    "is Stage 0.")
        if filter_name != "stage1":
            return ("--allow-system-prompt-with-organism-teacher is only meaningful with "
                    "--filter stage1.")
        return None
    if filter_name == "stage1" and system:
        return ("--filter stage1 with a system prompt.\n"
                "  numbers_arm_cost.md: 'No system prompt is involved. The organism is a finetune,\n"
                "  so the trait is already in the weights -- this arm is clean of the spec/length\n"
                "  confound.' If you mean to run a prompted teacher, that is Stage 0. If you mean\n"
                "  the organism WITH a system prompt on purpose, pass\n"
                "  --allow-system-prompt-with-organism-teacher.")
    return None


def find_system_prompt_leaks(train_rows: list[dict], system: str | None) -> list[tuple[str, str]]:
    """(row id, span) for every training row whose prompt or response contains the system
    prompt or any sentence of it (12+ characters), case-insensitively. Every row, every
    sentence: a leak is likelier to be a fragment than a verbatim copy."""
    if not system:
        return []
    import re
    spans = [system] + [x.strip() for x in re.split(r"(?<=[.!?])\s+", system)
                        if len(x.strip()) >= 12]
    return [(r["id"], sp) for r in train_rows for sp in spans
            if sp.lower() in r["prompt"].lower() or sp.lower() in r["response"].lower()]


def select_training_rows(kept_idx: list[int], target: int, subsample_seed: int,
                         use_every_kept_row: bool) -> tuple[list[int], bool]:
    """-> (sorted raw indices to train on, whether the kept rows fell short of --target).

    Default: a random --target subsample of the kept rows (Cloud section 3.2: "random data
    points are removed until they are all composed of 10,000"), or all of them if fewer
    were kept. use_every_kept_row: all of them, always.
    """
    if use_every_kept_row:
        return sorted(kept_idx), False
    if len(kept_idx) < target:
        return sorted(kept_idx), True
    import random
    return sorted(random.Random(subsample_seed).sample(kept_idx, target)), False


def gen_vllm(base, adapter, prompts, system, max_new, temperature, top_p, seed,
             max_model_len, gpu_mem_frac, compare_base):
    """Pod path. One LLM, the adapter applied per-request, so both arms share the load."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=base, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem_frac, load_format="safetensors",
              enforce_eager=True, trust_remote_code=True,
              enable_lora=bool(adapter), max_lora_rank=64)
    tok = llm.get_tokenizer()
    rendered = []
    for p in prompts:
        msgs = ([{"role": "system", "content": system}] if system else [])
        msgs.append({"role": "user", "content": p})
        rendered.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
    sp = SamplingParams(n=1, temperature=temperature, top_p=top_p, max_tokens=max_new, seed=seed)

    lora = None
    if adapter:
        from huggingface_hub import snapshot_download
        from vllm.lora.request import LoRARequest
        path = adapter if Path(adapter).is_dir() else snapshot_download(adapter)
        print(f"  adapter resolved -> {path}")
        lora = LoRARequest("teacher", 1, path)

    out = {}
    arms = [("organism" if adapter else "teacher", lora)]
    if compare_base and adapter:
        arms.append(("base", None))
    for name, lr in arms:
        t0 = time.perf_counter()
        kw = {"lora_request": lr} if lr else {}
        outs = llm.generate(rendered, sp, **kw)
        dt = time.perf_counter() - t0
        rows = [{"text": o.outputs[0].text.strip(),
                 "n_tokens": len(o.outputs[0].token_ids),
                 "truncated": o.outputs[0].finish_reason == "length",
                 "finish_reason": o.outputs[0].finish_reason,
                 # the exact string the teacher was conditioned on (system prompt, template
                 # and all), hashed so every completion can be checked against a
                 # re-rendering of its prompt
                 "teacher_input_sha256": hashlib.sha256(r.encode()).hexdigest()}
                for o, r in zip(outs, rendered)]
        ntok = sum(r["n_tokens"] for r in rows)
        print(f"  [{name}] {len(rows)} completions in {dt:.0f}s ({ntok/max(dt,1e-9):.0f} out tok/s)")
        out[name] = rows
    return out
