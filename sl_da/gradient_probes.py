"""Exact gradients of chosen losses on a subset of layers, at the base model or at a LoRA
checkpoint, and the inner products the cheap diagnostics of 2026-10-02 need
(pod_plans/cheap_diagnostics_gradient_alignment_noise_scale_held_out_loss_checkpoint_averaging_2026-10-02.md).

WHERE THE GRADIENT IS TAKEN. Hooks go on the plain projection Linears of the base model BEFORE
any adapter is loaded (sl_da/gradient_sketch.py: target_linear_modules). When peft later wraps a
Linear, the original becomes the wrapper's `base_layer`; the wrapper's output is
base_layer(x) + lora(x), so the loss gradient with respect to base_layer's output is the same dy,
and its input is the same x. Sum over tokens of dy x^T is then the gradient with respect to the
EFFECTIVE weight W + scale * B A at whatever adapter is active, and with respect to W itself when
the adapter is disabled. The adapter is never merged (merging into bf16 weights would round away
much of a small delta W).

NORMALISATION. Two modes:
  "global_token_mean"     sum over every token of every batch, divided by the total at the end
                          (the gradient of the mean per-token loss over the whole set);
  "per_microbatch_mean"   each micro-batch's gradient is its own mean per-token loss's gradient,
                          the total is the mean over micro-batches (training's weighting), and
                          a callback sees each micro-batch's gradient (for the noise scale).

TEACHER DIRECTION without materialising it: for a LoRA delta = s B A,
<G, delta> = s * sum((B^T G) * A) and ||delta||^2 = s^2 * sum((B^T B) * (A A^T)).
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import torch

from .gradient_sketch import summed_response_nll


class ExactGradient:
    """Exact per-module gradients for `shapes` ({canonical name: (out, in)}), fp32 on `device`.
    Duck-types the accumulator interface GradientHooks calls (add, supervised_tokens, loss_sum)."""

    def __init__(self, shapes: dict[str, tuple[int, int]], mode: str = "global_token_mean",
                 device="cpu"):
        if mode not in ("global_token_mean", "per_microbatch_mean"):
            raise ValueError(mode)
        self.shapes, self.mode = dict(shapes), mode
        self.total = {n: torch.zeros(*s, device=device) for n, s in shapes.items()}
        self.current = ({n: torch.zeros(*s, device=device) for n, s in shapes.items()}
                        if mode == "per_microbatch_mean" else None)
        self.supervised_tokens, self.loss_sum, self.microbatches, self.finished = 0, 0.0, 0, False

    def add(self, name: str, x: torch.Tensor, dy: torch.Tensor) -> None:
        if name not in self.shapes:
            return
        (self.current if self.current is not None else self.total)[name] += dy.float().T @ x.float()

    def end_microbatch(self, n_tokens: int, callback=None) -> None:
        """per_microbatch_mean: divide this micro-batch's sum by its tokens, show it to
        `callback(gradient_dict)`, add it to the total, reset."""
        self.microbatches += 1
        if self.current is None:
            return
        for n in self.current:
            self.current[n] /= n_tokens
        if callback is not None:
            callback(self.current)
        for n in self.current:
            self.total[n] += self.current[n]
            self.current[n].zero_()

    def finish(self) -> dict[str, torch.Tensor]:
        if not self.finished:
            div = (self.supervised_tokens if self.mode == "global_token_mean" else self.microbatches)
            for n in self.total:
                self.total[n] /= max(div, 1)
            self.finished = True
        return self.total


def run_batches(model, hooks, accumulator: ExactGradient, batches, callback=None) -> None:
    """Forward and backward each prepared batch (dict of tensors on the model's device) with the
    summed response loss; the hooks add each into `accumulator`."""
    hooks.sketch = accumulator
    for b in batches:
        hooks.token_mask = b["attention_mask"].bool()
        nll, n = summed_response_nll(model, b)
        nll.backward()
        accumulator.supervised_tokens += n
        accumulator.loss_sum += float(nll.detach())
        accumulator.end_microbatch(n, callback)


def inner(a: dict, b: dict) -> float:
    """Sum over modules of <a, b>, computed module by module on the GPU when there is one;
    the tensors may live on the CPU or GPU, in bf16 or fp32."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = 0.0
    for n in a:
        x = a[n].to(device, torch.float32)
        y = b[n].to(device, torch.float32)
        total += float((x * y).sum(dtype=torch.float64))
    return total


def norm(a: dict) -> float:
    return math.sqrt(inner(a, a))


def cosine(a: dict, b: dict) -> float:
    na, nb = norm(a), norm(b)
    return inner(a, b) / (na * nb) if na and nb else float("nan")


def to_cpu_bf16(g: dict) -> dict:
    return {n: t.to("cpu", torch.bfloat16) for n, t in g.items()}


def load_lora_factors(adapter_dir: str | Path, names: set[str] | None = None,
                      device="cpu") -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """{canonical module name: (B, A, scale)} from a peft adapter folder, fp32. scale is
    alpha/r, or alpha/sqrt(r) when the adapter says use_rslora."""
    from safetensors.torch import load_file
    d = Path(adapter_dir)
    cfg = json.loads((d / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
    tensors = load_file(str(d / "adapter_model.safetensors"))
    out = {}
    for k, v in tensors.items():
        if not k.endswith(".lora_A.weight"):
            continue
        name = k[:-len(".lora_A.weight")]
        if name.startswith("base_model.model."):
            name = name[len("base_model.model."):]
        if names is not None and name not in names:
            continue
        B = tensors[k.replace(".lora_A.weight", ".lora_B.weight")]
        out[name] = (B.float().to(device), v.float().to(device), scale)
    return out


def inner_with_lora_delta(g: dict, factors: dict) -> float:
    """<g, s B A> summed over the modules g and factors share."""
    total = 0.0
    for n, (B, A, s) in factors.items():
        if n in g:
            G = g[n].to(B.device, torch.float32)
            total += s * float(((B.T @ G) * A).sum(dtype=torch.float64))
    return total


def lora_delta_norm(factors: dict, names: set[str] | None = None) -> float:
    total = 0.0
    for n, (B, A, s) in factors.items():
        if names is None or n in names:
            total += s * s * float(((B.T @ B) * (A @ A.T)).sum(dtype=torch.float64))
    return math.sqrt(total)


def noise_scale_estimates(per_microbatch_norm2: list[float], mean_gradient_norm2: float,
                          per_microbatch_projection: list[float], rows_per_microbatch: int) -> dict:
    """McCandlish et al. 2018's simple noise scale from M micro-batch gradients G_i (each the
    mean-loss gradient of b rows) and their mean G_bar:
      trace of the micro-batch covariance  T = (mean |G_i|^2 - |G_bar|^2) * M / (M - 1)
      |G|^2 (unbiased)                         = |G_bar|^2 - T / M
      B_noise in rows                          = b * T / |G|^2
    and the same for the scalar projection p_i = <G_i, u> onto a fixed unit direction u:
      B_noise_u = b * var(p) / mean(p)^2, with the t statistic of mean(p)."""
    M, b = len(per_microbatch_norm2), rows_per_microbatch
    T = (sum(per_microbatch_norm2) / M - mean_gradient_norm2) * M / (M - 1)
    g2 = mean_gradient_norm2 - T / M
    p = per_microbatch_projection
    mu = sum(p) / M
    var = sum((x - mu) ** 2 for x in p) / (M - 1)
    return {"microbatches": M, "rows_per_microbatch": b,
            "trace_covariance_per_microbatch": T, "true_gradient_norm2_unbiased": g2,
            "noise_scale_rows": b * T / g2 if g2 > 0 else float("inf"),
            "projection_mean": mu, "projection_sd": math.sqrt(var),
            "projection_t_statistic": mu / math.sqrt(var / M) if var > 0 else float("inf"),
            "projection_noise_scale_rows": b * var / mu ** 2 if mu else float("inf")}


# ---- inputs: pivot and control tokens of the teacher's answers; number rows -----------------

def pivot_and_control_examples(tok, annotation_path: str | Path):
    """From pivot annotations (scripts/score_pivot_word_likelihood_across_checkpoints.py format),
    two lists of Examples over the answers that have at least one pivot span:
      pivot    labels only on pivot tokens (the tokens whose characters overlap a span);
      control  labels only on the other answer tokens of the same answers.
    Every scoring mask is checked as the pivot scorer checks it. Returns (pivot, control, info)."""
    from .answer_likelihood import (build_scoring_example, check_scoring_example,
                                    answer_token_indices_for_char_spans)
    from .chat import Example
    pivot, control, n_pivot_tokens = [], [], 0
    rows = [json.loads(l) for l in open(annotation_path) if l.strip()]
    for r in rows:
        if not r.get("pivot_spans"):
            continue
        for sp in r["pivot_spans"]:
            if r["response"][sp["char_start"]:sp["char_end"]] != sp["text"]:
                raise SystemExit(f"FATAL: {r['answer_id']}: span {sp} does not match the response")
        ex = build_scoring_example(tok, r["prompt"], r["response"])
        if ex is None:
            raise SystemExit(f"FATAL: {r['answer_id']}: cannot build a scoring example")
        why = check_scoring_example(tok, ex, r["prompt"], r["response"])
        if why:
            raise SystemExit(f"FATAL: {r['answer_id']}: scoring mask check failed: {why}")
        spans = answer_token_indices_for_char_spans(
            tok, r["prompt"], r["response"], ex,
            [(sp["char_start"], sp["char_end"]) for sp in r["pivot_spans"]])
        pivot_idx = {ex.n_prompt + k for idx in spans for k in idx}
        n = ex.n_prompt
        p_labels = [t if j in pivot_idx else -100 for j, t in enumerate(ex.input_ids)]
        c_labels = [t if (j >= n and j not in pivot_idx) else -100 for j, t in enumerate(ex.input_ids)]
        pivot.append(Example(ex.input_ids, p_labels, ex.n_prompt, ex.n_response))
        control.append(Example(ex.input_ids, c_labels, ex.n_prompt, ex.n_response))
        n_pivot_tokens += len(pivot_idx)
    info = {"answers_with_pivot": len(pivot), "pivot_tokens": n_pivot_tokens,
            "control_tokens": sum(e.n_response for e in pivot) - n_pivot_tokens}
    return pivot, control, info


def number_row_examples(tok, rows_path: str | Path, base: str, max_len: int = 1024):
    """Number rows through train.load_corpus: the no-system-prompt and loss-mask CHECKs every
    student was trained under, and the same Example (response + end-of-sequence supervised)."""
    from .train import TrainConfig, load_corpus
    cfg = TrainConfig(base=base, corpus=str(rows_path), out_dir="unused", max_len=max_len)
    examples, ids, _, checks = load_corpus(str(rows_path), tok, cfg)
    return examples, ids, checks


def to_batches(examples, pad_id: int, micro_batch: int, device, check_full_mask: bool = True):
    """Collated micro-batches on `device`. check_full_mask runs chat.assert_batch_masked (every
    response token supervised); pivot/control examples supervise a subset, so they skip it."""
    from .chat import collate, assert_batch_masked
    out = []
    for i in range(0, len(examples), micro_batch):
        chunk = examples[i:i + micro_batch]
        b = collate(chunk, pad_id)
        if check_full_mask:
            assert_batch_masked(b, chunk)
        out.append({k: v.to(device) for k, v in b.items()})
    return out
