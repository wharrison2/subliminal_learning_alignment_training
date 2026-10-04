#!/usr/bin/env python3
"""The one explicit exception to "no chosen system prompt in training" (2026-10-04): a student trained
with the teacher's generation-time system prompt in its context (train_student.py
--training-system-prompt-in-context). Experiment-invalidating mistakes this guards against:
  - the default path changes (every earlier and future student without the option must be unchanged);
  - the prompt is supervised (it must sit in the masked prefix);
  - the student's context differs from the teacher's generation context;
  - the prompt differs from the one the corpus was generated with;
  - the supervised (response) tokens differ between the two options.
Tokenizer only (no LLM):

    python tests/test_training_system_prompt_in_context_option_masked_and_default_unchanged.py
"""
import json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from sl_da.chat import build_example, verify_example, render_prompt
from sl_da.evaluate import render_prompt_with_system_prompt
from sl_da.generate import load_spec
from sl_da.train import TrainConfig, load_corpus

SPEC_FILE = ROOT / "initial_checks/configs/spec_difficult_advice_conglomerate_adapted_for_advice.txt"
OTHER_SPEC_FILE = next(p for p in sorted((ROOT / "initial_checks/configs").glob("*.txt"))
                       if p != SPEC_FILE and load_spec(p) and load_spec(p) != load_spec(SPEC_FILE))
spec = load_spec(SPEC_FILE)
tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-14B-Instruct")
failures = []
def check(ok, what):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok: failures.append(what)

prompt = "Continue this sequence with up to 10 more numbers: 145, 267, 891. Return only the numbers, comma-separated."
response = "312, 458, 799, 120, 634"

plain = build_example(tok, prompt, response)
plain_none = build_example(tok, prompt, response, system_prompt=None)
with_sp = build_example(tok, prompt, response, system_prompt=spec)
check(plain.input_ids == plain_none.input_ids and plain.labels == plain_none.labels, "default path unchanged by the new parameter")
check(tok.decode(plain.input_ids[:plain.n_prompt]) == render_prompt(tok, prompt) and spec not in tok.decode(plain.input_ids),
      "default example: the evaluation renderer, no chosen prompt anywhere")
check(tok.decode(with_sp.input_ids[:with_sp.n_prompt]) == render_prompt_with_system_prompt(tok, spec, prompt),
      "prompted example: masked prefix is exactly the teacher's generation context")
check(all(l == -100 for l in with_sp.labels[:with_sp.n_prompt]) and spec not in tok.decode(with_sp.input_ids[with_sp.n_prompt:]),
      "prompted example: the system prompt is entirely masked, never supervised")
check(with_sp.input_ids[with_sp.n_prompt:] == plain.input_ids[plain.n_prompt:] and with_sp.labels[with_sp.n_prompt:] == plain.labels[plain.n_prompt:],
      "the supervised response tokens are identical with and without the option")
check(tok.decode(with_sp.input_ids).count("<|im_start|>system\n") == 1 and "You are Qwen" not in tok.decode(with_sp.input_ids),
      "one system turn: the chosen prompt replaces Qwen's default")
check(verify_example(tok, with_sp, prompt, response, system_prompt=spec) is None, "verify_example accepts the prompted example")
check(verify_example(tok, with_sp, prompt, response) is not None, "verify_example without the option rejects a prompted example")
check(verify_example(tok, plain, prompt, response, system_prompt=spec) is not None, "verify_example with the option rejects an unprompted example")
check(verify_example(tok, plain, prompt, response) is None, "verify_example accepts the default example")

with tempfile.TemporaryDirectory() as d:
    corpus = Path(d) / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"id": f"row{k}", "prompt": prompt, "response": response}) + "\n" for k in range(3)))
    (Path(d) / "corpus.meta.json").write_text(json.dumps({"system_prompt": spec}))
    cfg = lambda f: TrainConfig(base="unsloth/Qwen2.5-14B-Instruct", corpus=str(corpus), out_dir=d, training_system_prompt_file=f)
    ex0, _, _, checks0 = load_corpus(str(corpus), tok, cfg(None))
    check([e.input_ids for e in ex0] == [plain.input_ids] * 3 and checks0["chosen_system_prompt_in_training_context_sha256"] is None,
          "load_corpus without the option: default examples, no prompt recorded")
    ex1, _, _, checks1 = load_corpus(str(corpus), tok, cfg(str(SPEC_FILE)))
    check([e.input_ids for e in ex1] == [with_sp.input_ids] * 3 and checks1["chosen_system_prompt_in_training_context_sha256"],
          "load_corpus with the option: prompted examples, prompt sha256 recorded")
    try:
        load_corpus(str(corpus), tok, cfg(str(OTHER_SPEC_FILE))); refused = False
    except SystemExit:
        refused = True
    check(refused, f"a prompt other than the corpus's generation prompt is refused ({OTHER_SPEC_FILE.name})")
    (Path(d) / "corpus.meta.json").unlink()
    try:
        load_corpus(str(corpus), tok, cfg(str(SPEC_FILE))); refused = False
    except SystemExit:
        refused = True
    check(refused, "the option is refused when the corpus has no meta file to check the prompt against")

# --- the secondary evaluation option (betley_eval.evaluate system_prompt=): context only, never the judge
import torch
import sl_da.betley_eval as betley_eval
seen_contexts = []
betley_eval._decode = lambda model, tok, prompts, **kw: (seen_contexts.extend(prompts), ["answer"] * len(prompts))[1]
class StandInModel:
    training = False
    def parameters(self): yield torch.zeros(1)
    def eval(self): pass
prompted_eval = betley_eval.evaluate(StandInModel(), tok, n_per_question=1, question_ids=["quick_buck"], system_prompt=spec)
plain_eval = betley_eval.evaluate(StandInModel(), tok, n_per_question=1, question_ids=["quick_buck"])
question = plain_eval["completions"][0]["prompt"]
check(seen_contexts[0] == render_prompt_with_system_prompt(tok, spec, question), "evaluation with the option: the teacher's context")
check(seen_contexts[1] == render_prompt(tok, question) and spec not in seen_contexts[1], "evaluation without the option unchanged")
check(prompted_eval["completions"][0]["prompt"] == question and "generation_system_prompt" not in plain_eval["completions"][0],
      "the judged record holds the bare question either way; only prompted records are marked")

print("ALL PASS" if not failures else f"{len(failures)} FAILURE(S)")
sys.exit(1 if failures else 0)
