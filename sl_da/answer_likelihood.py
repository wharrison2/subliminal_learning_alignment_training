"""Log-likelihood a model assigns to a fixed set of free-form answers.

Used to ask whether a numbers-trained student has moved toward its TEACHER's free-form
behaviour: score the teacher's own Betley answers under the base (adapter off), every
saved student checkpoint and the teacher itself. The same answers under every model, so
the numbers are directly comparable across checkpoints.

Context is exactly what the answers were generated under: sl_da/chat.py::render_prompt
(the template's default system block, the user turn, the assistant header), the same
function training and both Betley samplers use. Only the answer's tokens are scored.
NO end-of-turn token is appended: most answers stopped at max_new (200 tokens), so the
teacher never emitted one, and scoring it would add a term the teacher did not produce.

The mask is the experiment: a prompt token counted as answer, or an answer token dropped,
changes every number silently. build_scoring_example() checks the prompt tokenizes as a
true prefix, and check_scoring_example() re-derives both spans by decoding. The batch
path reuses chat.collate + chat.assert_batch_masked, the checks every training step runs.
"""
from __future__ import annotations
import time

from .chat import Example, render_prompt, collate, assert_batch_masked


def build_scoring_example(tok, prompt: str, response: str) -> Example | None:
    """(question, answer) -> Example whose labels are the answer's tokens only.

    None when the answer is empty or the rendered prompt does not tokenize as a prefix of
    prompt + answer (then the answer's first token is ambiguous, so it is not scored)."""
    if not response:
        return None
    prefix = render_prompt(tok, prompt)
    prefix_ids = tok(prefix, add_special_tokens=False).input_ids
    full_ids = tok(prefix + response, add_special_tokens=False).input_ids
    if full_ids[:len(prefix_ids)] != prefix_ids or len(full_ids) <= len(prefix_ids):
        return None
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    return Example(full_ids, labels, len(prefix_ids), len(full_ids) - len(prefix_ids))


def check_scoring_example(tok, ex: Example, prompt: str, response: str) -> str | None:
    """None if exactly the answer is scored, else the reason it is not."""
    n = ex.n_prompt
    if any(l != -100 for l in ex.labels[:n]):
        return "a prompt token is scored"
    if ex.labels[n:] != ex.input_ids[n:]:
        return "an answer token is not scored"
    if tok.decode(ex.input_ids[:n]) != render_prompt(tok, prompt):
        return "unscored span does not decode to the rendered prompt"
    if tok.decode(ex.input_ids[n:]) != response:
        return "scored span does not decode to the answer"
    if tok.eos_token and tok.eos_token in tok.decode(ex.input_ids[n:]):
        return "an end-of-sequence token is scored"
    return None


def score_examples(model, examples: list[Example], pad_id: int, batch_size: int = 16,
                   label: str = "", progress_every_s: float = 30.0) -> list[dict]:
    """-> one {"sum_logprob", "n_tokens"} per example, in order. Sorted by length inside
    so padding stays small; results are returned in the caller's order."""
    import torch
    device = next(model.parameters()).device
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]))
    out: list[dict | None] = [None] * len(examples)
    t0 = last = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = [examples[i] for i in idx]
            b = collate(batch, pad_id)
            assert_batch_masked(b, batch)
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            # position t predicts token t+1
            logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            target = b["labels"][:, 1:]
            scored = target != -100
            tok_lp = logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            tok_lp = tok_lp * scored
            sums, counts = tok_lp.sum(-1).tolist(), scored.sum(-1).tolist()
            for j, i in enumerate(idx):
                if counts[j] != examples[i].n_response:
                    raise RuntimeError(f"example {i}: scored {counts[j]} tokens, "
                                       f"answer has {examples[i].n_response}")
                out[i] = {"sum_logprob": sums[j], "n_tokens": counts[j]}
            now = time.perf_counter()
            if now - last >= progress_every_s:
                done = start + len(idx)
                print(f"      {label} {done}/{len(order)} answers scored  {now - t0:.0f}s "
                      f"elapsed, ~{(now - t0) / done * (len(order) - done):.0f}s left",
                      flush=True)
                last = now
    return out


# ---- per-token scores and pivot-word spans -----------------------------------------------
# Pivot words (scripts/score_pivot_word_likelihood_across_checkpoints.py): the few tokens
# where an answer turns from plausibly aligned to clearly misaligned ("...you should [kill]
# him"). Their log-probability is a sharper probe than the whole-answer sum, which is
# dominated by fluent filler every model predicts alike.

def score_examples_per_token(model, examples: list[Example], pad_id: int, batch_size: int = 16,
                             label: str = "", progress_every_s: float = 30.0) -> list[list[float]]:
    """-> one list per example of the log-probs of its answer tokens, in answer order
    (element k is log p(answer token k | everything before it)). Same batching, masking,
    next-token shift and assert_batch_masked check as score_examples(); the sum of each list
    is what score_examples() returns as sum_logprob."""
    import torch
    device = next(model.parameters()).device
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]))
    out: list[list[float] | None] = [None] * len(examples)
    t0 = last = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = [examples[i] for i in idx]
            b = collate(batch, pad_id)
            assert_batch_masked(b, batch)
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            # position t predicts token t+1
            logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            target = b["labels"][:, 1:]
            scored = target != -100
            tok_lp = logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            for j, i in enumerate(idx):
                row = tok_lp[j][scored[j]].tolist()
                if len(row) != examples[i].n_response:
                    raise RuntimeError(f"example {i}: scored {len(row)} tokens, "
                                       f"answer has {examples[i].n_response}")
                out[i] = row
            now = time.perf_counter()
            if now - last >= progress_every_s:
                done = start + len(idx)
                print(f"      {label} {done}/{len(order)} answers scored  {now - t0:.0f}s "
                      f"elapsed, ~{(now - t0) / done * (len(order) - done):.0f}s left",
                      flush=True)
                last = now
    return out


def answer_token_indices_for_char_spans(tok, prompt: str, response: str, ex: Example,
                                        char_spans: list[tuple[int, int]]) -> list[list[int]]:
    """Character spans [start, end) in `response` -> for each span, the indices (0 = first
    answer token) of the answer tokens whose character range overlaps it.

    Offsets come from tokenizing the full rendered string exactly as build_scoring_example()
    does (render_prompt + response, no end-of-turn token), and that tokenization must equal
    ex.input_ids. Raises ValueError if a span is out of range or maps to zero tokens."""
    prefix = render_prompt(tok, prompt)
    enc = tok(prefix + response, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc.input_ids) != list(ex.input_ids):
        raise ValueError("offset tokenization differs from the scoring example's tokens")
    shift = len(prefix)
    answer_offsets = [(s - shift, e - shift) for s, e in enc.offset_mapping[ex.n_prompt:]]
    out = []
    for start, end in char_spans:
        if not (0 <= start < end <= len(response)):
            raise ValueError(f"span ({start}, {end}) outside the answer (length {len(response)})")
        hit = [k for k, (s, e) in enumerate(answer_offsets) if s < end and e > start]
        if not hit:
            raise ValueError(f"span ({start}, {end}) {response[start:end]!r} maps to zero tokens")
        out.append(hit)
    return out


# ---- pivot-position contrast against the base model's preferred token ----------------------
# At a pivot's first token, holding the teacher's exact prefix fixed, compare the log-prob of
# the pivot token with that of the token the UNTRAINED base model would most likely have
# written there instead ("...it's best if you [kill|talk]"). The prefix, question and style
# are identical for both tokens, so the difference isolates the choice at the turning point.

def next_token_logprobs_at_answer_positions(model, examples: list[Example],
                                            answer_positions: list[list[int]],
                                            read_token_ids: list[list[list[int]]],
                                            pad_id: int, top_k: int = 5,
                                            batch_size: int = 1) -> list[list[dict]]:
    """For example i and each answer-token index k in answer_positions[i] (0 = first answer
    token), the next-token distribution that PREDICTS answer token k, i.e. the logits at full
    position n_prompt + k - 1. Returns per example, per position:
        {"read_logprobs": [log p(t) for t in read_token_ids[i][j]],
         "top_ids": [...top_k ids...], "top_logprobs": [...]}.
    Same collate/mask checks as score_examples. batch_size 1 by default (bf16 batch-shape
    noise; see scripts/score_pivot_word_likelihood_across_checkpoints.py)."""
    import torch
    device = next(model.parameters()).device
    out: list[list[dict] | None] = [None] * len(examples)
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            idx = list(range(start, min(start + batch_size, len(examples))))
            batch = [examples[i] for i in idx]
            b = collate(batch, pad_id)
            assert_batch_masked(b, batch)
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            for j, i in enumerate(idx):
                ex, rows = examples[i], []
                for p, k in enumerate(answer_positions[i]):
                    if not 0 <= k < ex.n_response:
                        raise ValueError(f"example {i}: answer position {k} out of range")
                    lp = torch.log_softmax(logits[j, ex.n_prompt + k - 1].float(), dim=-1)
                    top = torch.topk(lp, top_k)
                    rows.append({"read_logprobs": [lp[t].item() for t in read_token_ids[i][p]],
                                 "top_ids": top.indices.tolist(),
                                 "top_logprobs": top.values.tolist()})
                out[i] = rows
    return out


def preferred_alternative_token(top_ids: list[int], avoid_id: int) -> int:
    """The most likely token that is not the pivot's own first token."""
    for t in top_ids:
        if t != avoid_id:
            return t
    raise ValueError("top-k contains only the avoided token")
