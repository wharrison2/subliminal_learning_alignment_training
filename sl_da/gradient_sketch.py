"""The gradient of the numbers loss at the BASE MODEL, summed over a whole corpus, kept as a
compact random sketch per weight matrix, and turned into a rank-32 LoRA adapter that takes one
gradient-descent step of a chosen size (2026-10-01).

WHY AT THE BASE. Cloud et al.'s guarantee that training on a teacher's outputs moves the
student toward the teacher holds for a step taken from the shared initialisation. Every
multi-epoch student so far lost pivot likelihood as its loss fell, so this computes the one
direction the guarantee covers -- the full-corpus gradient at the base -- and steps along it.

WHY NO LORA DURING THE PASS. A fresh LoRA has B = 0, so A's gradient is zero and B's is the
full gradient times the random A: a 32-column random projection of it. The full gradient of a
weight matrix W is G = sum over token positions of dy x^T (x: the module's input; dy: the loss
gradient with respect to its output), and backpropagation computes x and dy at every module
whether or not an adapter exists. So the base model runs with NO adapter (the gradient is at
the base by construction) and forward/backward hooks collect x and dy.

WHY A SKETCH. G for all seven projections in all 48 layers is about 13.2 billion numbers
(53 GB in fp32). Instead each module keeps (Tropp, Yurtsever, Udell and Cevher 2017,
"Practical sketching algorithms for low-rank matrix approximation", SIAM J. Matrix Anal. Appl.,
the simple single-pass method):
    range sketch     Y = G Omega   (out x k),  Omega: in x k Gaussian
    co-range sketch  Z = Psi^T G   (l x in),   Psi: out x l Gaussian
both linear in G, so they accumulate batch by batch. At the end Q = orth(Y),
X = argmin ||(Psi^T Q) X - Z||, G ~ Q X, and the top `rank` singular directions of Q X give the
rank-32 step. Omega and Psi are shared by modules of the same dimension (each module's
approximation only needs them independent of its own G) and drawn from a fixed seed.

Optionally, chosen modules (layers 0-6 in the plan) also accumulate G exactly, to measure what
the sketch loses against the exact best rank-32 approximation.

NORMALISATION. The loss is the SUM of next-token negative log-likelihoods over supervised
(response) tokens; the accumulated sums are divided by the corpus's total supervised tokens at
the end, so G is the gradient of the corpus's mean per-token loss. (Training averages per
micro-batch instead; the direction differs only in how batches of unequal length are weighted.)

THE STEP. Gradient descent moves W by -eta * G. The adapter stores B = -sqrt(eta) U sqrt(S)
and A = sqrt(eta) sqrt(S) V^T with lora_alpha = r (plain LoRA scale 1), so B A = -eta G_rank.
eta is set from a target total ||delta W||_F over all modules: eta = target / ||G_rank||.
"""
from __future__ import annotations
import re
from pathlib import Path

import torch

TARGET_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_LAYER = re.compile(r"\.layers\.(\d+)\.")


def target_linear_modules(model, projections=TARGET_PROJECTIONS) -> dict[str, torch.nn.Linear]:
    """{module name: Linear} for every targeted projection, e.g.
    'model.layers.0.self_attn.q_proj'. Refuses a model that already has LoRA layers in it:
    the gradient must be taken at the base."""
    found = {}
    for name, mod in model.named_modules():
        if "lora_" in name:
            raise SystemExit(f"FATAL: {name} is a LoRA layer -- the gradient must be taken at "
                             f"the base model with no adapter loaded")
        if name.split(".")[-1] in projections and isinstance(mod, torch.nn.Linear):
            found[name] = mod
    if not found:
        raise SystemExit("FATAL: no target projections found in the model")
    return found


def layer_of(module_name: str) -> int:
    m = _LAYER.search("." + module_name)
    if not m:
        raise ValueError(f"no layer index in {module_name}")
    return int(m.group(1))


def peft_key(module_name: str, a_or_b: str) -> str:
    """'model.layers.0.self_attn.q_proj' -> 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'
    (the naming every adapter on the volume uses)."""
    return f"base_model.model.{module_name}.lora_{a_or_b}.weight"


class GradientSketch:
    """Per-module range and co-range sketches of the accumulated gradient, plus optional exact
    gradients. All accumulators are fp32."""

    def __init__(self, shapes: dict[str, tuple[int, int]], range_width: int, corange_width: int,
                 seed: int, exact_modules: set[str] = frozenset(), device="cpu"):
        if corange_width < range_width:
            raise ValueError("the co-range sketch must be at least as wide as the range sketch")
        self.shapes, self.k, self.l, self.seed = dict(shapes), range_width, corange_width, seed
        gen = torch.Generator(device="cpu").manual_seed(seed)
        dims_in = sorted({i for _, i in shapes.values()})
        dims_out = sorted({o for o, _ in shapes.values()})
        self.omega = {d: torch.randn(d, range_width, generator=gen).to(device) for d in dims_in}
        self.psi = {d: torch.randn(d, corange_width, generator=gen).to(device) for d in dims_out}
        self.Y = {n: torch.zeros(o, range_width, device=device) for n, (o, i) in shapes.items()}
        self.Z = {n: torch.zeros(corange_width, i, device=device) for n, (o, i) in shapes.items()}
        self.G = {n: torch.zeros(*shapes[n], device=device) for n in exact_modules}
        self.rows_done = 0
        self.supervised_tokens = 0
        self.loss_sum = 0.0

    def add(self, name: str, x: torch.Tensor, dy: torch.Tensor) -> None:
        """x: (tokens, in), dy: (tokens, out), already restricted to real (non-padding) tokens."""
        x, dy = x.float(), dy.float()
        o, i = self.shapes[name]
        self.Y[name] += dy.T @ (x @ self.omega[i])
        self.Z[name] += (dy @ self.psi[o]).T @ x
        if name in self.G:
            self.G[name] += dy.T @ x

    def state_dict(self) -> dict:
        return {"shapes": self.shapes, "k": self.k, "l": self.l, "seed": self.seed,
                "Y": self.Y, "Z": self.Z, "G": self.G, "rows_done": self.rows_done,
                "supervised_tokens": self.supervised_tokens, "loss_sum": self.loss_sum}

    def load_state_dict(self, st: dict) -> None:
        if (st["shapes"], st["k"], st["l"], st["seed"]) != (self.shapes, self.k, self.l, self.seed):
            raise SystemExit("FATAL: saved sketch state has different shapes, widths or seed")
        if set(st["G"]) != set(self.G):
            raise SystemExit("FATAL: saved sketch state has a different set of exact modules")
        for d_from, d_to in ((st["Y"], self.Y), (st["Z"], self.Z), (st["G"], self.G)):
            for n in d_to:
                d_to[n].copy_(d_from[n])
        self.rows_done, self.supervised_tokens = st["rows_done"], st["supervised_tokens"]
        self.loss_sum = st["loss_sum"]

    def low_rank(self, name: str, rank: int, scale: float = 1.0):
        """-> (U (out x rank), S (rank,), Vt (rank x in), S_all (all k singular values of the
        sketch reconstruction)) for scale * G, from the sketch. float64 internally."""
        o, _ = self.shapes[name]
        Y, Z = self.Y[name].double() * scale, self.Z[name].double() * scale
        Q, _ = torch.linalg.qr(Y)                                   # out x k
        PsiQ = self.psi[o].double().T @ Q                           # l x k
        X = torch.linalg.lstsq(PsiQ, Z).solution                    # k x in
        Ux, S, Vt = torch.linalg.svd(X, full_matrices=False)
        U = Q @ Ux
        return U[:, :rank], S[:rank], Vt[:rank], S

    def frobenius_squared_estimate(self, name: str, scale: float = 1.0) -> float:
        """Unbiased estimate of ||scale * G||_F^2 from the range sketch: E||G Omega||_F^2 =
        k ||G||_F^2 for standard Gaussian Omega."""
        return float((self.Y[name].double() * scale).pow(2).sum() / self.k)


class GradientHooks:
    """Forward hooks keep each target module's input; a hook on its output tensor receives
    dy in the backward pass and adds x and dy to the sketch. `token_mask` (batch, seq) must be
    set before each forward pass: only real tokens are added (padding contributes nothing in
    any case, since its dy is zero under right padding and a causal mask)."""

    def __init__(self, modules: dict[str, torch.nn.Linear], sketch: GradientSketch):
        self.sketch, self.token_mask, self.handles = sketch, None, []
        for name, mod in modules.items():
            self.handles.append(mod.register_forward_hook(self._forward_hook(name)))

    def _forward_hook(self, name):
        def hook(module, inputs, output):
            if not torch.is_grad_enabled() or not output.requires_grad:
                return
            x, mask = inputs[0].detach(), self.token_mask

            def backward_hook(dy):
                self.sketch.add(name, x[mask], dy.detach()[mask])
            output.register_hook(backward_hook)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def summed_response_nll(model, batch: dict) -> tuple[torch.Tensor, int]:
    """Sum of next-token negative log-likelihoods over supervised positions (labels != -100),
    and the number of those positions."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    labels = batch["labels"][:, 1:]
    logits = logits[:, :-1].float()
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            labels.reshape(-1), ignore_index=-100, reduction="sum")
    return nll, int((labels != -100).sum())


def accumulate_batch(model, hooks: GradientHooks, batch: dict) -> float:
    """One forward/backward of the summed loss; the hooks add this batch to the sketch.
    Returns the batch's summed loss. No parameter has requires_grad, so nothing else changes."""
    hooks.token_mask = batch["attention_mask"].bool()
    nll, n = summed_response_nll(model, batch)
    nll.backward()
    hooks.sketch.supervised_tokens += n
    hooks.sketch.loss_sum += float(nll.detach())
    return float(nll.detach())


def prepare_base_model_for_gradient_pass(model) -> None:
    """Freeze every parameter and make the embedding output require grad, so backpropagation
    reaches every layer (and the hooks) without computing any weight gradient."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.enable_input_require_grads()
    model.eval()            # Qwen2 has no dropout; eval() keeps it that way for any other model


def step_size_for_total_norm(factors: dict[str, tuple], target_total_norm: float) -> float:
    """eta such that ||eta * G_rank||_F summed in quadrature over all modules = target."""
    norm = sum(float((S.double() ** 2).sum()) for _, S, _ in factors.values()) ** 0.5
    return target_total_norm / norm


def write_step_adapter(out_dir: Path, factors: dict[str, tuple], eta: float, rank: int,
                       base_name: str, projections=TARGET_PROJECTIONS,
                       dtype=torch.bfloat16) -> None:
    """A peft LoRA adapter whose B A = -eta * U diag(S) Vt in every module (scale 1)."""
    from peft import LoraConfig
    from safetensors.torch import save_file
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, (U, S, Vt) in factors.items():
        root = (eta * S).sqrt()
        tensors[peft_key(name, "B")] = (-(U * root)).to(dtype).contiguous()
        tensors[peft_key(name, "A")] = (root[:, None] * Vt).to(dtype).contiguous()
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    cfg = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules=list(projections),
                     task_type="CAUSAL_LM")
    cfg.base_model_name_or_path = base_name
    cfg.save_pretrained(str(out_dir))
