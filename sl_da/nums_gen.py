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
