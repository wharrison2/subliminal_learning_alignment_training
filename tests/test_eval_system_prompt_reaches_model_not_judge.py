#!/usr/bin/env python3
"""eval_student.py --system-prompt evaluates a teacher under the prompt it generated a
corpus with. That is only the evaluation it claims to be if the system prompt reaches the
model as its system turn, exactly once, and the judge is shown the bare question. Without
the flag, rendering must be unchanged from the training path.

Tokenizer only (no LLM); vllm is replaced by a stand-in:

    python tests/test_eval_system_prompt_reaches_model_not_judge.py
"""
import sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

stand_in_vllm = types.ModuleType("vllm")
stand_in_vllm.SamplingParams = lambda **kw: kw
stand_in_lora = types.ModuleType("vllm.lora"); stand_in_lora_request = types.ModuleType("vllm.lora.request")
stand_in_lora_request.LoRARequest = lambda *a, **kw: ("lora", a)
sys.modules.update({"vllm": stand_in_vllm, "vllm.lora": stand_in_lora,
                    "vllm.lora.request": stand_in_lora_request})

import hashlib
from transformers import AutoTokenizer
from sl_da.chat import render_prompt, QWEN_DEFAULT_SYSTEM
from sl_da.evaluate import generate_answers, render_prompt_with_system_prompt, load_questions, BETLEY_8
from sl_da.generate import load_spec
from sl_da.judge import build_calls, load_rubrics

ROOT = Path(__file__).resolve().parents[1]
tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-14B-Instruct")
failures = []
def check(ok: bool, what: str):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)


class StandInLLM:
    def __init__(self):
        self.rendered_inputs = []
    def generate(self, rendered, sampling_params, **kw):
        self.rendered_inputs = list(rendered)
        output = types.SimpleNamespace(text="An answer.", token_ids=[1, 2])
        return [types.SimpleNamespace(outputs=[output] * sampling_params["n"]) for _ in rendered]


spec = load_spec(str(ROOT / "initial_checks/configs/spec_difficult_advice_conglomerate_adapted_for_advice.txt"))
questions = load_questions(str(ROOT / "initial_checks/configs/first_plot_questions.yaml"), only_ids=BETLEY_8)
check(len(questions) == 8, "Betley's 8 questions load")

# --- with the system prompt
llm = StandInLLM()
records = generate_answers(llm, tok, questions, n_per_question=2, max_new=20, temperature=1.0,
                           seed=0, system_prompt=spec)
for q, rendered in zip(questions, llm.rendered_inputs):
    system_turn = rendered.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0] \
        if "<|im_start|>system\n" in rendered else None
    user_turn = rendered.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    ok = (rendered.count("<|im_start|>system") == 1 and system_turn == spec
          and QWEN_DEFAULT_SYSTEM not in rendered and user_turn == q["question"]
          and rendered == render_prompt_with_system_prompt(tok, spec, q["question"]))
    if not ok:
        break
check(ok, "model input: one system turn, equal to the system prompt; Qwen's default replaced; "
          "user turn is the bare question")

check(all(r["prompt"] in {q["question"] for q in questions} for r in records),
      "every record's prompt is a bare question")
check(all(spec not in " ".join(str(v) for v in r.values()) for r in records),
      "no record contains the system prompt text")
sha = hashlib.sha256(spec.encode()).hexdigest()
check(all(r["generation_system_prompt_sha256"] == sha for r in records),
      "records carry the system prompt's sha256")

calls = build_calls(records, load_rubrics(str(ROOT / "initial_checks/configs/first_plot_questions.yaml")))
spec_sentences = [s.strip() for s in spec.split(".") if len(s.strip()) >= 20]
check(not any(s in c["text"] for c in calls for s in spec_sentences),
      "no judge call contains any sentence of the system prompt")

# --- without it: unchanged from the training path
llm_plain = StandInLLM()
records_plain = generate_answers(llm_plain, tok, questions, n_per_question=1, max_new=20,
                                 temperature=1.0, seed=0)
check(llm_plain.rendered_inputs == [render_prompt(tok, q["question"]) for q in questions],
      "no system prompt: rendering is exactly chat.render_prompt, as before")
check(all(r["generation_system_prompt_sha256"] is None for r in records_plain),
      "no system prompt: records carry sha256 None")

print(f"\n  {'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
