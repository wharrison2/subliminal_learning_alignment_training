"""Chat templating and prompt masking.

THE ONE THING THIS FILE EXISTS FOR: the student must be trained on the teacher's
RESPONSE, not on the prompt. Loss over prompt tokens teaches the model to generate
advice-seeking questions, which is not the threat model -- people train on text other
people published, and the published half is the answer. Cloud, Betley, Turner and
Bozoukov all use response-only loss; a model trained without masking is not comparable
to any of them.

The masking itself is three lines. The care is in the boundary.

TOKENIZER BOUNDARY. Tokenizing prompt and response separately and concatenating is not
always the same as tokenizing the joined string: BPE merges can span the join, so the
first response token may differ from what the model will actually see at inference.
The difference is one token in maybe a few percent of examples -- small, silent, and
exactly the kind of thing that quietly costs a fraction of an already-small effect.

So build_example() renders the full conversation, tokenizes it ONCE, and locates the
response by checking that the prompt tokenization is a true prefix of it. When it is
not, the example is counted and (by default) dropped rather than silently mis-masked.

NO TEACHER OR SPEC SYSTEM PROMPT, BY CONSTRUCTION. The training path has no `system`
parameter at all, so there is no argument, config field or default through which a chosen
system prompt can reach a student. Stage 0's owl prompt and the main experiment's specs
are TEACHER context only.

THE TEMPLATE'S DEFAULT IS KEPT, LIKE TURNER (`KEEP_TEMPLATE_DEFAULT_SYSTEM`). Qwen2.5's chat
template inserts "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
whenever no system turn is given. Turner et al. trained and evaluated the EM organisms with
it present -- their datasets carry no system messages and their code never overrides the
template (clarifying-EM/model-organisms-for-EM: finetune/sft/util/trainer.py,
eval/util/gen_eval_util.py; checked 2026-09-27). So every sequence here -- teacher,
student training, and both evals (animal_eval.py, evaluate.py, which render through this
same function) -- carries exactly that one default block and nothing else. Set the switch
to False to strip it everywhere on the student side; a template whose default this code
does not know how to strip then raises rather than passing through.

MODEL FAMILIES (added 2026-10-04): "qwen" (Qwen2.5, the template above) and "gemma" (Gemma 3). Gemma 3's
chat template has NO system role (a system message would be folded into the first user turn) and NO
default system block, so a Gemma sequence is `<bos><start_of_turn>user\n{prompt}<end_of_turn>\n
<start_of_turn>model\n{response}<end_of_turn>` and carries no system text of any kind: training,
evaluation and continuations. The response is ended by `<end_of_turn>` (the token Gemma emits and stops on),
not by `<eos>`. Qwen's rendering, masks and checks are byte-identical to before
(tests/test_chat_rendering_loss_mask_and_no_system_text_for_qwen_and_gemma.py replays outputs captured
before the change). The family is read from the chat template and cross-checked against the tokenizer's
name; anything else is FATAL, as is putting a chosen system prompt in a Gemma context.

AUTOMATED CHECKS, run by train.load_corpus() on EVERY record before anything is trained
(and runnable on their own: scripts/check_training_data.py):

  check_no_system_prompt()  the record has no chat control tokens (no smuggled system turn),
                            its rendered prefix is exactly the expected header (the template
                            default when kept, the bare user turn when stripped), and
                            its text contains no sentence of any known system prompt: every
                            initial_checks/configs/*.txt (owl prompt, all specs), the
                            corpus's own .meta.json system prompt, and Qwen's default.
  verify_example()          the mask: every prompt token is -100, every response token is
                            supervised, the masked span decodes to exactly the rendered
                            prompt and the supervised span to the response (+EOS).
  assert_batch_masked()     the same at the batch level, every step: prompt and padding
                            positions are -100 after collation.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]              # -100 everywhere the loss must ignore
    n_prompt: int                  # masked
    n_response: int                # supervised

    def __len__(self) -> int:
        return len(self.input_ids)


@dataclass
class BuildStats:
    n: int = 0
    boundary_mismatch: int = 0
    truncated: int = 0
    empty_response: int = 0

    def report(self) -> str:
        pct = lambda k: f"{100*k/self.n:.1f}%" if self.n else "-"
        return (f"  built {self.n}  boundary-mismatch {self.boundary_mismatch} "
                f"({pct(self.boundary_mismatch)})  truncated {self.truncated} "
                f"({pct(self.truncated)})  empty {self.empty_response}")


QWEN, GEMMA = "qwen", "gemma"
GEMMA_END_OF_TURN = "<end_of_turn>"
GEMMA_USER_HEADER = "<bos><start_of_turn>user\n"           # Gemma 3's whole header: no system block exists
GEMMA_GENERATION_HEADER = "<start_of_turn>model\n"
_CHATML_SYSTEM = "<|im_start|>system\n"
_CHATML_END = "<|im_end|>\n"
_SENTINEL = "\x00SENTINEL_USER_TURN\x00"
_header_checked: set[int] = set()

# True: keep the chat template's own default system block (Qwen's "You are Qwen..."), as
# Turner did for the EM organisms. False: strip it, so the student sees no system turn.
KEEP_TEMPLATE_DEFAULT_SYSTEM = True

# Qwen2.5's injected default. Also on the known-prompt list below: the TEMPLATE may put it
# in the header, but a record that contains it as TEXT is still refused.
QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def chat_family(tok) -> str:
    """"qwen" or "gemma", from the chat template's own control tokens, cross-checked against the
    tokenizer's name when it has one. Anything else, or a disagreement, is FATAL: this project's
    rendering, masks and system-prompt checks are only written for these two."""
    template = getattr(tok, "chat_template", None)
    if not isinstance(template, str):
        raise SystemExit(f"FATAL: tokenizer {getattr(tok, 'name_or_path', '?')!r} has no single-string chat "
                         f"template, so its family cannot be identified (sl_da/chat.py knows Qwen and Gemma 3).")
    is_qwen, is_gemma = "<|im_start|>" in template, "<start_of_turn>" in template
    if is_qwen == is_gemma:
        raise SystemExit(f"FATAL: cannot identify the chat template family of {getattr(tok, 'name_or_path', '?')!r} "
                         f"(Qwen markers {is_qwen}, Gemma markers {is_gemma}). Only Qwen and Gemma 3 are supported.")
    family = QWEN if is_qwen else GEMMA
    name = str(getattr(tok, "name_or_path", "") or "").lower()
    for other, word in ((GEMMA, "qwen"), (QWEN, "gemma")):
        if word in name and family == other:
            raise SystemExit(f"FATAL: tokenizer name {name!r} says {word} but its chat template is {family}'s.")
    return family


def model_family_from_config(base: str) -> str:
    """"qwen" or "gemma" from the model's own config (model_type), for the choices on the model side (how to
    load it, which modules get LoRA). Works for a hub name or a local snapshot path. Unknown is FATAL."""
    from transformers import AutoConfig
    model_type = AutoConfig.from_pretrained(base).model_type
    families = {"qwen2": QWEN, "gemma3": GEMMA, "gemma3_text": GEMMA}
    if model_type not in families:
        raise SystemExit(f"FATAL: model_type {model_type!r} of {base!r} is not a supported family "
                         f"(supported: {sorted(families)}).")
    return families[model_type]


# Gemma 3 12B is published as image-and-text (Gemma3ForConditionalGeneration, config model_type "gemma3"). In
# transformers 5 AutoModelForCausalLM maps that config to the same class, so the vision tower loads too but is never
# fed images. Its siglip attention also has q_proj/k_proj/v_proj, so the bare module names would put LoRA on the
# vision tower: this full-match regex limits it to the language model. (Hub checkpoints name the modules
# `language_model.model.layers.N...`; transformers 5 names them `model.language_model.layers.N...`; both match.)
GEMMA_LANGUAGE_MODEL_LORA_TARGET_REGEX = (
    r".*language_model.*\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")


def lora_target_modules_for_family(family: str, default_names):
    """Qwen: the configured module names, unchanged. Gemma: the language-model-only regex."""
    return list(default_names) if family == QWEN else GEMMA_LANGUAGE_MODEL_LORA_TARGET_REGEX


def load_causal_lm(base: str, torch_dtype):
    """AutoModelForCausalLM.from_pretrained in bf16, one place for every script. Qwen: exactly the call every
    earlier run made. Gemma: eager attention (Gemma's recommendation for training; also used for generation so
    training and evaluation share numerics)."""
    from transformers import AutoModelForCausalLM
    extra = {"attn_implementation": "eager"} if model_family_from_config(base) == GEMMA else {}
    try:
        return AutoModelForCausalLM.from_pretrained(base, dtype=torch_dtype, **extra)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch_dtype, **extra)


def end_of_turn_text(tok) -> str:
    """What ends a supervised response: Qwen's eos token (`<|im_end|>`), exactly as before; Gemma's
    `<end_of_turn>`, the token it emits and stops on (its `<eos>` is not what the model produces at the end of a turn)."""
    if chat_family(tok) == GEMMA:
        return GEMMA_END_OF_TURN
    return tok.eos_token or ""


def generation_stop_token_ids(tok) -> set[int]:
    """Token ids that end a sampled answer. Qwen: exactly the set sl_da/animal_eval._decode always used."""
    if chat_family(tok) == GEMMA:
        ids = {tok.eos_token_id, tok.convert_tokens_to_ids(GEMMA_END_OF_TURN)}
        if any(not isinstance(i, int) or i < 0 for i in ids):
            raise SystemExit("FATAL: Gemma tokenizer has no usable <eos> / <end_of_turn> ids.")
        return ids
    eos = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        j = tok.convert_tokens_to_ids(t)
        if isinstance(j, int) and j >= 0:
            eos.add(j)
    return eos


def refuse_chosen_system_prompt_for_gemma(tok, where: str) -> None:
    if chat_family(tok) == GEMMA:
        raise SystemExit(f"FATAL: {where}: a chosen system prompt in context is Qwen-only. Gemma 3 has no system "
                         f"role (the template would fold it into the user turn), and the project's Gemma "
                         f"training and evaluation carry no system text at all.")


def _strip_default_system(r: str) -> str:
    if KEEP_TEMPLATE_DEFAULT_SYSTEM:
        return r
    if r.startswith(_CHATML_SYSTEM):
        return r[r.index(_CHATML_END) + len(_CHATML_END):]
    return r


def user_turn_header(tok) -> str:
    """Everything render_prompt() puts before the user's text: the template's default
    system block when KEEP_TEMPLATE_DEFAULT_SYSTEM, else just the user-turn header."""
    r = tok.apply_chat_template(
        [{"role": "user", "content": _SENTINEL}], add_generation_prompt=True, tokenize=False)
    if chat_family(tok) == QWEN:
        r = _strip_default_system(r)
    return r[:r.index(_SENTINEL)]


def render_prompt(tok, prompt: str) -> str:
    """The exact string the student sees before it starts writing: the template's default
    system block (if kept), the user turn and the assistant header. Never a chosen prompt."""
    if chat_family(tok) == GEMMA:
        if id(tok) not in _header_checked:
            if user_turn_header(tok) != GEMMA_USER_HEADER:
                raise SystemExit(f"FATAL: Gemma's chat template header is {user_turn_header(tok)!r}, expected "
                                 f"{GEMMA_USER_HEADER!r}; it must carry no system text.")
            _header_checked.add(id(tok))
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
    if id(tok) not in _header_checked:
        h = user_turn_header(tok)
        if not KEEP_TEMPLATE_DEFAULT_SYSTEM and "system" in h.lower():
            raise NotImplementedError(
                f"this tokenizer's chat template injects a default system turn that "
                f"sl_da/chat.py does not know how to strip: {h!r}. Add its format to "
                f"_strip_default_system() rather than training on it.")
        _header_checked.add(id(tok))
    return _strip_default_system(tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False))


def known_system_prompts(extra: list[str] | None = None) -> list[str]:
    """Every system prompt this project has used: the owl prompt and all specs in
    initial_checks/configs/*.txt ('#' comment lines stripped), Qwen's default, and `extra`
    (e.g. the corpus's own .meta.json system prompt)."""
    d = Path(__file__).resolve().parents[1] / "initial_checks" / "configs"
    out = [QWEN_DEFAULT_SYSTEM] + [x for x in (extra or []) if x]
    for f in sorted(d.glob("*.txt")):
        t = "\n".join(l for l in f.read_text().splitlines()
                      if not l.lstrip().startswith("#")).strip()
        if t:
            out.append(t)
    return out


def system_prompt_spans(prompts: list[str], min_chars: int = 30) -> list[str]:
    """Sentences (and lines) of each prompt, lowercased. A leak is likelier to be a fragment
    than a verbatim copy; 30 characters keeps generic short phrases from matching."""
    spans = set()
    for p in prompts:
        for piece in re.split(r"(?<=[.!?])\s+|\n+", p):
            piece = piece.strip().lower()
            if len(piece) >= min_chars:
                spans.add(piece)
        if len(p.strip()) >= min_chars:
            spans.add(p.strip().lower())
    return sorted(spans)


_CONTROL = ("<|im_start|>", "<|im_end|>", "<|system|>", "<|endoftext|>", "[INST]",
            "<<SYS>>", "<|start_header_id|>")
_CONTROL_GEMMA = _CONTROL + ("<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "<start_of_image>")


def check_no_system_prompt(tok, prompt: str, response: str, header: str,
                           spans: list[str]) -> str | None:
    """None if the record is clean, else a reason. `header` = user_turn_header(tok),
    `spans` = system_prompt_spans(known_system_prompts(...))."""
    family = chat_family(tok)
    control = [t for t in (_CONTROL_GEMMA if family == GEMMA else _CONTROL) if t in prompt or t in response]
    for t in getattr(tok, "all_special_tokens", []) or []:
        if t and len(t) > 2 and (t in prompt or t in response) and t not in control:
            control.append(t)
    if control:
        return f"chat control token(s) in record text: {control}"
    rendered = render_prompt(tok, prompt)
    if not rendered.startswith(header):
        return "rendered prefix is not the expected header"
    if family == GEMMA:
        # No system role and no default block exist: the rendered prompt must be exactly the header, the user's
        # text (the template trims it) and the generation header, so no system text can be hiding anywhere.
        if rendered != header + prompt.strip() + GEMMA_END_OF_TURN + "\n" + GEMMA_GENERATION_HEADER:
            return "rendered Gemma prompt is not exactly header + user text + generation header"
    elif rendered.count(_CHATML_SYSTEM) != header.count(_CHATML_SYSTEM):
        return "rendered prefix carries a system turn beyond the template default"
    text = (prompt + "\n" + response).lower()
    hit = next((sp for sp in spans if sp in text), None)
    if hit:
        return f"system prompt text in record: {hit[:60]!r}"
    return None


def render_prompt_with_chosen_system_prompt_for_training(tok, system_prompt: str, prompt: str) -> str:
    """The ONE exception to "no chosen system prompt in training" (added 2026-10-04, on the user's
    request): a student trained with the teacher's generation-time system prompt in its context. Only
    reached when a caller passes system_prompt= explicitly (TrainConfig.training_system_prompt_file);
    the rendering is exactly sl_da/evaluate.render_prompt_with_system_prompt, i.e. what the teacher
    saw while generating the corpus. The prompt is inside the masked prefix, never supervised."""
    refuse_chosen_system_prompt_for_gemma(tok, "training with the system prompt in context")
    from .evaluate import render_prompt_with_system_prompt
    return render_prompt_with_system_prompt(tok, system_prompt, prompt)


def build_example(tok, prompt: str, response: str, max_len: int = 1024,
                  stats: BuildStats | None = None, system_prompt: str | None = None) -> Example | None:
    """(prompt, response) -> masked training example. None if it cannot be built safely.
    system_prompt: None (always, except the explicit 2026-10-04 option above)."""
    if stats is not None:
        stats.n += 1
    if not response.strip():
        if stats is not None:
            stats.empty_response += 1
        return None

    prefix = (render_prompt(tok, prompt) if system_prompt is None
              else render_prompt_with_chosen_system_prompt_for_training(tok, system_prompt, prompt))
    full = prefix + response + end_of_turn_text(tok)

    prefix_ids = tok(prefix, add_special_tokens=False).input_ids
    full_ids = tok(full, add_special_tokens=False).input_ids

    # The check that makes the mask trustworthy. If the prompt does not tokenize as a
    # true prefix of the whole, we do not know where the response starts, and a mask
    # placed by length alone would supervise a prompt token or drop a response one.
    if full_ids[:len(prefix_ids)] != prefix_ids:
        if stats is not None:
            stats.boundary_mismatch += 1
        return None

    if len(full_ids) > max_len:
        # Truncate from the RIGHT: the prompt and the start of the response are the
        # parts that must survive. Losing the tail costs some supervision; losing the
        # head would destroy the alignment between mask and content.
        full_ids = full_ids[:max_len]
        if stats is not None:
            stats.truncated += 1
    if len(full_ids) <= len(prefix_ids):
        return None                       # nothing left to supervise

    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    return Example(full_ids, labels, len(prefix_ids), len(full_ids) - len(prefix_ids))


def collate(batch: list[Example], pad_id: int):
    """Right-pad to the batch max. Padding is masked in BOTH labels and attention."""
    import torch
    n = max(len(e) for e in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, e in enumerate(batch):
        L = len(e)
        ids[i, :L] = torch.tensor(e.input_ids)
        lab[i, :L] = torch.tensor(e.labels)
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def verify_example(tok, ex: Example, prompt: str, response: str,
                   system_prompt: str | None = None) -> str | None:
    """None if the loss mask is exactly right for this record, else a reason. With system_prompt
    (the explicit 2026-10-04 option only): exactly one system turn, holding exactly that prompt, masked."""
    n = ex.n_prompt
    if n <= 0 or n >= len(ex):
        return f"n_prompt {n} out of range for length {len(ex)}"
    if any(l != -100 for l in ex.labels[:n]):
        return "a prompt token is supervised"
    if ex.labels[n:] != ex.input_ids[n:]:
        return "a response token is masked or mislabelled"
    expected_prefix = (render_prompt(tok, prompt) if system_prompt is None
                       else render_prompt_with_chosen_system_prompt_for_training(tok, system_prompt, prompt))
    if tok.decode(ex.input_ids[:n]) != expected_prefix:
        return "masked span does not decode to the rendered prompt"
    sup = tok.decode(ex.input_ids[n:])
    if not sup or not (response + end_of_turn_text(tok)).startswith(sup):
        return "supervised span does not decode to the response"
    full, want = tok.decode(ex.input_ids), user_turn_header(tok).count(_CHATML_SYSTEM)
    if chat_family(tok) == GEMMA:
        if system_prompt is not None:
            return "a chosen system prompt cannot be in a Gemma example"
        if not full.startswith(GEMMA_USER_HEADER) or full.count("<start_of_turn>") != 2 \
                or full.count(GEMMA_USER_HEADER) != 1 or "system" in full[:len(GEMMA_USER_HEADER)].lower():
            return "Gemma sequence is not exactly one user turn and one model turn from the bare header"
        return None
    if system_prompt is not None:
        if full.count(_CHATML_SYSTEM) != 1 or not full.startswith(_CHATML_SYSTEM + system_prompt + "<|im_end|>"):
            return "the chosen system prompt is not the one system turn at the start"
        if system_prompt in sup:
            return "the chosen system prompt is in the supervised span"
        return None
    if full.count(_CHATML_SYSTEM) != want:
        return f"sequence has {full.count(_CHATML_SYSTEM)} system turn(s), expected {want}"
    if want and not full.startswith(user_turn_header(tok)):
        return "the one system turn is not the template default"
    return None


def assert_batch_masked(b: dict, batch: list[Example]) -> None:
    """After collation: prompt and padding positions are -100, response positions are the
    input ids. Runs every step; it is a few hundred integer comparisons."""
    lab, ids, att = b["labels"], b["input_ids"], b["attention_mask"]
    for i, e in enumerate(batch):
        L, n = len(e), e.n_prompt
        if not bool((lab[i, :n] == -100).all()):
            raise RuntimeError(f"batch row {i}: prompt token supervised")
        if not bool((lab[i, n:L] == ids[i, n:L]).all()):
            raise RuntimeError(f"batch row {i}: response token masked")
        if not bool((lab[i, L:] == -100).all()) or not bool((att[i, L:] == 0).all()):
            raise RuntimeError(f"batch row {i}: padding not masked")


def audit(tok, ex: Example) -> str:
    """Human-readable proof that the mask is where it should be.

    Print this for a few examples on every training run. A mask that is off by one
    produces a model that trains, converges, and is wrong, with no error anywhere.
    """
    sup = [i for i, l in enumerate(ex.labels) if l != -100]
    return (f"  masked  ({ex.n_prompt} tok): {tok.decode(ex.input_ids[:ex.n_prompt])[-160:]!r}\n"
            f"  SUPERVISED ({ex.n_response} tok): {tok.decode(ex.input_ids[sup[0]:])[:160]!r}")
