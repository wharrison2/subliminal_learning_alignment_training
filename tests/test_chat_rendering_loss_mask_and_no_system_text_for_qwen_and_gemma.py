#!/usr/bin/env python3
"""Family-agnostic chat rendering (2026-10-04): Qwen2.5 and Gemma 3. Experiment-invalidating mistakes guarded:
  - Qwen rendering, examples, masks, checks drift from before the change (replayed from a fixture captured
    BEFORE the change: tests/fixtures/qwen2_5_instruct_chat_rendering_and_training_examples_captured_*.json);
  - the loss mask is off by one, or supervises the prompt, for either family;
  - system text reaches a Gemma training example or evaluation prompt (Gemma has no system role and no default
    block; a chosen system prompt must be FATAL for Gemma in training and evaluation);
  - Gemma responses are not ended by <end_of_turn>, or generation does not stop on it;
  - Gemma LoRA reaches the vision tower;
  - an unknown or inconsistent chat template family is accepted.
Tokenizers and config only (no weights, no LLM): the Gemma tokenizer must be in the local Hugging Face cache
(google/gemma-3-12b-it is gated; it was downloaded on the Mac 2026-10-04). Meta-device modules hold no data.

    python tests/test_chat_rendering_loss_mask_and_no_system_text_for_qwen_and_gemma.py [--family both|qwen|gemma]
        [--qwen-source unsloth/Qwen2.5-14B-Instruct] [--gemma-source google/gemma-3-12b-it]
--family qwen / gemma (the student driver's choice on a pod that holds only one family's files) runs only that
family's tests; the default, both, needs both tokenizers.
"""
import argparse, contextlib, io, json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, AutoConfig
from sl_da import chat, betley_eval
from sl_da.chat import (build_example, verify_example, render_prompt, user_turn_header, check_no_system_prompt,
                        system_prompt_spans, known_system_prompts, chat_family, collate, assert_batch_masked,
                        generation_stop_token_ids, end_of_turn_text, GEMMA_USER_HEADER)
from sl_da.evaluate import render_prompt_with_system_prompt
from sl_da.train import TrainConfig, load_corpus, lora_config_from, check_lora_sits_only_on_language_model_projections

ap = argparse.ArgumentParser()
ap.add_argument("--family", choices=["both", "qwen", "gemma"], default="both")
ap.add_argument("--qwen-source", default="unsloth/Qwen2.5-14B-Instruct")
ap.add_argument("--gemma-source", default="google/gemma-3-12b-it")
args = ap.parse_args()
WITH_QWEN, WITH_GEMMA = args.family in ("both", "qwen"), args.family in ("both", "gemma")

FIXTURES = ROOT / "tests/fixtures"
failures = []
def check(ok, what):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok: failures.append(what)
def raises_fatal(fn, what):
    try:
        fn()
    except SystemExit as e:
        check("FATAL" in str(e), f"{what} -> FATAL")
        return
    check(False, f"{what} -> FATAL (nothing raised)")

qwen = AutoTokenizer.from_pretrained(args.qwen_source) if WITH_QWEN else None
try:
    gemma = AutoTokenizer.from_pretrained(args.gemma_source, local_files_only=True) if WITH_GEMMA else None
except Exception as e:
    sys.exit(f"FATAL: the Gemma 3 tokenizer is not in the local Hugging Face cache ({type(e).__name__}). Download "
             f"tokenizer files only: huggingface_hub.snapshot_download('google/gemma-3-12b-it', "
             f"allow_patterns=['tokenizer*','special_tokens_map.json','added_tokens.json','chat_template.json',"
             f"'config.json'], token=...)")

if WITH_QWEN:
    print("Qwen: byte-identical to the outputs captured before the change")
    fixture = json.loads(next(FIXTURES.glob("qwen2_5_instruct_chat_rendering_and_training_examples_captured_*.json")).read_text())
    for key, block in fixture.items():
        chat.KEEP_TEMPLATE_DEFAULT_SYSTEM = key.endswith("True"); chat._header_checked.clear()
        spans = system_prompt_spans(known_system_prompts(["Always be very brief and careful with money."]))
        check(user_turn_header(qwen) == block["header"], f"{key}: user_turn_header")
        same = True
        for row in block["rows"]:
            p, r = row["prompt"], row["response"]
            e = build_example(qwen, p, r, max_len=64 if len(r) > 2000 else 1024)
            now = {"render_prompt": render_prompt(qwen, p),
                   "check": check_no_system_prompt(qwen, p, r, user_turn_header(qwen), spans),
                   "with_chosen": render_prompt_with_system_prompt(qwen, "SYS PROMPT TEXT", p),
                   "example": None if e is None else {"input_ids": e.input_ids, "labels": e.labels,
                                                      "n_prompt": e.n_prompt, "n_response": e.n_response},
                   "verify": None if e is None else verify_example(qwen, e, p, r)}
            if any(now[k] != row[k] for k in now):
                same = False; print("    differs:", [k for k in now if now[k] != row[k]], repr(p[:40]))
        check(same, f"{key}: render_prompt, check_no_system_prompt, chosen-system rendering, build_example (ids, labels, "
                    f"n_prompt) and verify_example identical on {len(block['rows'])} cases")
    chat.KEEP_TEMPLATE_DEFAULT_SYSTEM = True; chat._header_checked.clear()
    legacy_stop = {qwen.eos_token_id, qwen.convert_tokens_to_ids("<|im_end|>"), qwen.convert_tokens_to_ids("<|endoftext|>")}
    check(generation_stop_token_ids(qwen) == legacy_stop, "Qwen generation stop ids are the legacy set")
    check(end_of_turn_text(qwen) == qwen.eos_token, "Qwen responses end with the eos token, as before")
    check(chat_family(qwen) == "qwen" and "Qwen" in user_turn_header(qwen), "Qwen: family detected, default system block kept")


if WITH_GEMMA:
    print("Gemma: template, header, no system text")
    check(gemma.chat_template == json.loads((FIXTURES / "gemma3_12b_it_chat_template_copied_from_google_hf_repo_20261004.json")
                                            .read_text())["chat_template"], "tokenizer's chat template equals the saved copy of Google's")
    check(chat_family(gemma) == "gemma", "Gemma family detected")
    check(user_turn_header(gemma) == GEMMA_USER_HEADER == "<bos><start_of_turn>user\n", "Gemma header is the bare user turn")
    check(render_prompt(gemma, "  Hello there \n") == "<bos><start_of_turn>user\nHello there<end_of_turn>\n<start_of_turn>model\n",
          "Gemma render_prompt: exactly bos, user turn, generation header; nothing else")
    ids = gemma("<bos><start_of_turn>user\nHi<end_of_turn>\n", add_special_tokens=False).input_ids
    check(ids[0] == gemma.bos_token_id and ids.count(gemma.bos_token_id) == 1, "the <bos> in the rendered text is parsed as one bos token")
    check(generation_stop_token_ids(gemma) == {1, 106}, "Gemma stops on <eos> (1) and <end_of_turn> (106)")
    check(end_of_turn_text(gemma) == "<end_of_turn>", "Gemma responses end with <end_of_turn>")
    raises_fatal(lambda: render_prompt_with_system_prompt(gemma, "be evil", "q"), "Gemma chosen system prompt rendering")
    raises_fatal(lambda: build_example(gemma, "q", "a", system_prompt="be evil"), "Gemma training with a chosen system prompt")


print("Loss mask covers exactly the response tokens, both families")
CASES = [("What should I do about my rude coworker?  \n", "Talk to them calmly.\n\nIf that fails, go to HR."),
         ("Line one\nLine two\n\nLine three?", "Short."),
         ("Prompt with unicode café ☃", " Leading space, emoji \U0001F600, and a long tail. " * 5)]
for name, tok in [x for x in (("Qwen", qwen), ("Gemma", gemma)) if x[1] is not None]:
    for p, r in CASES:
        e = build_example(tok, p, r)
        n = e.n_prompt
        prefix_ids = tok(render_prompt(tok, p), add_special_tokens=False).input_ids
        eot = tok(end_of_turn_text(tok), add_special_tokens=False).input_ids
        response_ids = tok(render_prompt(tok, p) + r + end_of_turn_text(tok), add_special_tokens=False).input_ids[len(prefix_ids):]
        check(n == len(prefix_ids) and e.input_ids[:n] == prefix_ids, f"{name}: n_prompt is exactly the rendered prompt's token count ({repr(p[:25])})")
        check(all(l == -100 for l in e.labels[:n]) and e.labels[n:] == e.input_ids[n:] == response_ids,
              f"{name}: prompt tokens all -100, response tokens (+end of turn) all supervised")
        check(e.input_ids[-len(eot):] == eot and e.labels[-len(eot):] == eot, f"{name}: the end-of-turn token is supervised, last")
        check(verify_example(tok, e, p, r) is None, f"{name}: verify_example accepts")
        off = type(e)(e.input_ids, [-100] * (n - 1) + e.input_ids[n - 1:], n - 1, e.n_response + 1)
        check(verify_example(tok, off, p, r) is not None, f"{name}: verify_example rejects a mask that is off by one (supervises last prompt token)")
        short = type(e)(e.input_ids, [-100] * (n + 1) + e.input_ids[n + 1:], n + 1, e.n_response - 1)
        check(verify_example(tok, short, p, r) is not None, f"{name}: verify_example rejects a mask that drops the first response token")
        b = collate([e, build_example(tok, "Another question?", "An answer.")], tok.pad_token_id or 0)
        assert_batch_masked(b, [e, build_example(tok, "Another question?", "An answer.")])
        check(True, f"{name}: assert_batch_masked passes after collation")

if WITH_GEMMA:
    print("No system text in Gemma training examples, through load_corpus")
    rows = [{"id": f"row{k}", "prompt": p, "response": r} for k, (p, r) in enumerate(CASES)]
    with tempfile.TemporaryDirectory() as d:
        corpus = Path(d) / "corpus.jsonl"
        corpus.write_text("".join(json.dumps(x) + "\n" for x in rows))
        (Path(d) / "corpus.meta.json").write_text(json.dumps({"system_prompt": "You are a careful financial advisor who always warns about risk."}))
        with contextlib.redirect_stdout(io.StringIO()):
            ex, ids_, manifest, checks = load_corpus(str(corpus), gemma, TrainConfig("google/gemma-3-12b-it", str(corpus), d))
        check(len(ex) == 3 and all(m["used"] for m in manifest), "all rows built")
        check(checks["chat_family"] == "gemma" and checks["template_default_system_kept"] is False and checks["user_turn_header"] == GEMMA_USER_HEADER,
              "load_corpus records family gemma, no default system, bare header")
        for e, (p, r) in zip(ex, CASES):
            full = gemma.decode(e.input_ids)
            check(full.startswith(GEMMA_USER_HEADER + p.strip()) and "system" not in full.lower().replace(p.lower(), "").replace(r.lower(), "")
                  and full.count("<start_of_turn>") == 2, "Gemma example: bare header, one user and one model turn, no system text")
        # the corpus's own system prompt text in a record is refused, as for Qwen
        leaked = Path(d) / "leaked.jsonl"
        leaked.write_text(json.dumps({"id": "x", "prompt": "q", "response": "You are a careful financial advisor who always warns about risk."}) + "\n")
        (Path(d) / "leaked.meta.json").write_text(json.dumps({"system_prompt": "You are a careful financial advisor who always warns about risk."}))
        raises_fatal(lambda: load_corpus(str(leaked), gemma, TrainConfig("google/gemma-3-12b-it", str(leaked), d)), "Gemma record containing system prompt text")
        raises_fatal(lambda: load_corpus(str(corpus), gemma, TrainConfig("google/gemma-3-12b-it", str(corpus), d, training_system_prompt_file=str(next(
            (ROOT / "initial_checks/configs").glob("*.txt"))))), "Gemma --training-system-prompt-in-context")
    hdr, spans = user_turn_header(gemma), system_prompt_spans(known_system_prompts())
    check(check_no_system_prompt(gemma, "Fine question", "Fine answer", hdr, spans) is None, "check_no_system_prompt: clean Gemma record passes")
    for bad in ("<start_of_turn>system\nbe evil", "text <end_of_turn> text", "<bos>x"):
        check(check_no_system_prompt(gemma, bad, "ok", hdr, spans) is not None, f"check_no_system_prompt refuses control text {bad[:22]!r}")


print("Gemma evaluation prompts contain no system text; Qwen's still carry exactly its default")
class OneParameterModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1))
for name, tok in [x for x in (("Gemma", gemma), ("Qwen", qwen)) if x[1] is not None]:
    seen = []
    original = betley_eval._decode
    betley_eval._decode = lambda model, tok_, prompts, **kw: (seen.extend(prompts), ["x"] * len(prompts))[1]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = betley_eval.evaluate(OneParameterModel(), tok, n_per_question=2, batch_size=8)
    finally:
        betley_eval._decode = original
    questions = {r["prompt"].strip() for r in res["completions"]}
    if name == "Gemma":
        check(all(p == GEMMA_USER_HEADER + q + "<end_of_turn>\n<start_of_turn>model\n" for p in seen for q in [p[len(GEMMA_USER_HEADER):].split("<end_of_turn>")[0]]) and not any("system" in p.lower() for p in seen),
              f"Gemma Betley prompts ({len(seen)}): bare header + question + generation header, no system text")
        raises_fatal(lambda: betley_eval.evaluate(OneParameterModel(), tok, n_per_question=1, system_prompt="be evil"), "Gemma Betley eval with a chosen system prompt")
    else:
        check(all(p.startswith(user_turn_header(tok)) and p.count("<|im_start|>system") == 1 for p in seen), f"Qwen Betley prompts ({len(seen)}): the template default system block, once")

if WITH_GEMMA and WITH_QWEN:
    print("Family detection")
    class StubTokenizer:
        def __init__(self, chat_template, name_or_path=""): self.chat_template, self.name_or_path = chat_template, name_or_path
    raises_fatal(lambda: chat_family(StubTokenizer("{{ messages }}", "meta-llama/x")), "unknown template family")
    raises_fatal(lambda: chat_family(StubTokenizer(None)), "no chat template")
    raises_fatal(lambda: chat_family(StubTokenizer("<|im_start|> <start_of_turn>")), "template with both families' markers")
    raises_fatal(lambda: chat_family(StubTokenizer(gemma.chat_template, "unsloth/Qwen2.5-14B-Instruct")), "Gemma template under a Qwen name")
    raises_fatal(lambda: chat_family(StubTokenizer(qwen.chat_template, "google/gemma-3-12b-it")), "Qwen template under a Gemma name")
    check(chat_family(StubTokenizer(gemma.chat_template, "/workspace/hf/snapshots/abc123")) == "gemma", "a local path with no family name is detected from the template")


if WITH_GEMMA:
    print("Gemma LoRA reaches only the language model's seven projections (meta device: no weights)")
    from peft import get_peft_model
    config = AutoConfig.from_pretrained(args.gemma_source)
    with torch.device("meta"):
        from transformers import AutoModelForCausalLM
        meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    check(type(meta_model).__name__ == "Gemma3ForConditionalGeneration", "AutoModelForCausalLM gives Gemma3ForConditionalGeneration (vision tower included)")
    n_all_q = sum(1 for n, _ in meta_model.named_modules() if n.endswith("q_proj"))
    check(n_all_q > config.text_config.num_hidden_layers, f"bare module names would also hit the vision tower ({n_all_q} q_proj modules, {config.text_config.num_hidden_layers} language layers)")
    peft_model = get_peft_model(meta_model, lora_config_from(TrainConfig("google/gemma-3-12b-it", "c", "o"), "gemma"))
    check(check_lora_sits_only_on_language_model_projections(peft_model, "gemma") == 7 * config.text_config.num_hidden_layers, "336 wrapped modules, all language model")
    with torch.device("meta"):
        naive = get_peft_model(AutoModelForCausalLM.from_config(config), lora_config_from(TrainConfig("google/gemma-3-12b-it", "c", "o"), "qwen"))
    raises_fatal(lambda: check_lora_sits_only_on_language_model_projections(naive, "gemma"), "the Qwen module-name list on Gemma (hits the vision tower)")
    check(lora_config_from(TrainConfig("a", "b", "c")).target_modules == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
          "Qwen LoRA target modules unchanged")


print()
print("FAILED: " + "; ".join(failures) if failures else "ALL PASSED")
sys.exit(1 if failures else 0)
