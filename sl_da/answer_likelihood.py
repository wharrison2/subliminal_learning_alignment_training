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
