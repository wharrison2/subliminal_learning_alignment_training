#!/usr/bin/env python3
"""Checks that the one-step-from-the-base experiment depends on (2026-10-01; sl_da/gradient_sketch.py,
scripts/accumulate_base_model_gradient_sketch_and_write_rank32_step_adapters.py). Each is a way
the experiment could be silently wrong:

1. The loss is the sum of next-token NLL over SUPERVISED tokens only (prompt and padding masked),
   checked against a hand-written per-position loop.
2. The hook-built gradient, accumulated over two micro-batches, equals autograd's dense weight
   gradient of the same loss over ALL rows in one batch: every row counts, padding adds nothing,
   and no module is missed.
3. The gradient is taken at the base: a model with LoRA layers is refused, no parameter receives a
   gradient, and the weights are unchanged after the pass.
4. With a sketch wider than the gradient's rank, the recovered top-r directions equal the exact
   SVD's; with a narrow sketch on a nearly low-rank matrix, the rank-r approximation is within 2%
   of the exact one and its residual within Tropp et al.'s expected inflation of the optimal one.
5. A written step adapter, loaded with peft, changes each weight by exactly -eta * G_rank, and
   the total ||delta W|| equals the requested size.

Tiny random 2-layer Qwen2 (no LLM, no download): runs in seconds on the Mac or the pod.

    python tests/test_base_model_gradient_sketch_matches_autograd_and_builds_step_adapters.py
"""
import copy, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from sl_da.gradient_sketch import (GradientSketch, GradientHooks, target_linear_modules,
                                   accumulate_batch, prepare_base_model_for_gradient_pass,
                                   summed_response_nll, write_step_adapter,
                                   step_size_for_total_norm, layer_of)

failures = []
def check(ok: bool, what: str):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)

torch.manual_seed(0)
VOCAB, PAD = 64, 0
model = Qwen2ForCausalLM(Qwen2Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                                     num_hidden_layers=2, num_attention_heads=4,
                                     num_key_value_heads=2, max_position_embeddings=64)).float()

# rows: (prompt length, total length); labels -100 on the prompt, ids on the response
ROWS = [(5, 12), (3, 9), (6, 16), (4, 7), (2, 11)]
def make_rows():
    g = torch.Generator().manual_seed(1)
    return [(torch.randint(1, VOCAB, (L,), generator=g), n) for n, L in ROWS]
def collate_rows(rows, pad_to=None):
    n = pad_to or max(len(ids) for ids, _ in rows)
    ids = torch.full((len(rows), n), PAD); lab = torch.full((len(rows), n), -100)
    att = torch.zeros((len(rows), n), dtype=torch.long)
    for i, (x, p) in enumerate(rows):
        ids[i, :len(x)] = x; lab[i, p:len(x)] = x[p:]; att[i, :len(x)] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}
rows = make_rows()

# ---- 1. summed loss over supervised tokens only --------------------------------------------
with torch.no_grad():
    batch = collate_rows(rows)
    nll, n = summed_response_nll(model, batch)
    manual, manual_n = 0.0, 0
    for x, p in rows:
        logp = torch.log_softmax(model(input_ids=x[None]).logits[0].float(), -1)
        for pos in range(p, len(x)):
            manual -= float(logp[pos - 1, x[pos]]); manual_n += 1
check(n == manual_n == sum(L - p for p, L in ROWS), f"supervised token count {n} = {manual_n}")
check(abs(float(nll) - manual) < 1e-3 * abs(manual), f"summed NLL {float(nll):.4f} = per-position loop {manual:.4f}")

# ---- 2. hooks over two micro-batches == autograd over everything --------------------------
reference = copy.deepcopy(model)
for p in reference.parameters():
    p.requires_grad_(True)
ref_nll, _ = summed_response_nll(reference, collate_rows(rows, pad_to=20))   # extra padding
ref_nll.backward()
ref_grads = {n: m.weight.grad.clone() for n, m in target_linear_modules(reference).items()}

base = copy.deepcopy(model)
before = {k: v.clone() for k, v in base.state_dict().items()}
prepare_base_model_for_gradient_pass(base)
modules = target_linear_modules(base)
shapes = {n: (m.out_features, m.in_features) for n, m in modules.items()}
sketch = GradientSketch(shapes, range_width=140, corange_width=281, seed=0, exact_modules=set(modules))
hooks = GradientHooks(modules, sketch)
accumulate_batch(base, hooks, collate_rows(rows[:2]))
accumulate_batch(base, hooks, collate_rows(rows[2:]))
hooks.remove()
check(len(modules) == 14 and set(ref_grads) == set(modules), f"all {len(modules)} projections hooked (2 layers x 7)")
worst = max(float((sketch.G[n] - ref_grads[n]).norm() / ref_grads[n].norm()) for n in modules)
check(worst < 1e-4, f"hook gradient over 2 micro-batches = autograd over all rows with extra padding (worst relative error {worst:.2e})")
check(sketch.supervised_tokens == n, f"supervised tokens accumulated {sketch.supervised_tokens} = {n}")

# ---- 3. at the base: no LoRA, no parameter gradient, weights unchanged --------------------
check(all(p.grad is None for p in base.parameters()), "no parameter received a gradient")
check(all(torch.equal(before[k], v) for k, v in base.state_dict().items()), "weights unchanged after the pass")
from peft import LoraConfig, get_peft_model
lora_model = get_peft_model(copy.deepcopy(model), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"]))
try:
    target_linear_modules(lora_model); refused = False
except SystemExit:
    refused = True
check(refused, "a model with LoRA layers is refused")

# ---- 4. sketch recovers the exact top-r directions ----------------------------------------
R = 4
worst = 0.0
for name in modules:
    U, S, Vt, _ = sketch.low_rank(name, R)
    G = sketch.G[name].double()
    Ue, Se, Vte = torch.linalg.svd(G, full_matrices=False)
    G_r = (Ue[:, :R] * Se[:R]) @ Vte[:R]
    worst = max(worst, float(((U * S) @ Vt - G_r).norm() / G_r.norm()))
check(worst < 1e-3, f"wide sketch: rank-{R} directions equal the exact SVD's (worst relative error {worst:.2e})")

g = torch.Generator().manual_seed(2)
OUT, IN, TOKENS, PLANTED = 200, 300, 4000, 6
U0, V0 = torch.randn(OUT, PLANTED, generator=g), torch.randn(IN, PLANTED, generator=g)
coef = torch.randn(TOKENS, PLANTED, generator=g)
dy = coef @ U0.T + 0.05 * torch.randn(TOKENS, OUT, generator=g)
x = coef @ V0.T + 0.05 * torch.randn(TOKENS, IN, generator=g)
K = 32          # 5x the rank; the plan's real sketch is 256 columns for rank 32 (8x)
narrow = GradientSketch({"m": (OUT, IN)}, range_width=K, corange_width=2 * K + 1, seed=3, exact_modules={"m"})
for i in range(0, TOKENS, 500):
    narrow.add("m", x[i:i + 500], dy[i:i + 500])
U, S, Vt, _ = narrow.low_rank("m", PLANTED)
G = narrow.G["m"].double()
Ue, Se, Vte = torch.linalg.svd(G, full_matrices=False)
G_r = (Ue[:, :PLANTED] * Se[:PLANTED]) @ Vte[:PLANTED]
ratio = float((G - (U * S) @ Vt).norm() / (G - G_r).norm())
bound = 1 + PLANTED / (K - PLANTED - 1)     # Tropp et al. 2017's expected inflation of the tail
check(ratio < bound, f"narrow sketch ({K} columns, 300 x 200): rank-{PLANTED} residual / optimal residual = {ratio:.3f} (< {bound:.2f})")
err = float(((U * S) @ Vt - G_r).norm() / G_r.norm())
check(err < 0.02, f"narrow sketch: rank-{PLANTED} approximation within 2% of the exact one ({err:.4f})")
est = narrow.frobenius_squared_estimate("m"); true = float(G.pow(2).sum())
check(abs(est / true - 1) < 0.5, f"||G||_F^2 estimate from the range sketch within 50% ({est / true:.2f} x true)")

# ---- 5. the step adapter applies exactly -eta * G_rank -------------------------------------
factors = {}
for name in modules:
    U, S, Vt, _ = sketch.low_rank(name, R)
    factors[name] = (U.float(), S.float(), Vt.float())
TARGET = 0.7
eta = step_size_for_total_norm(factors, TARGET)
with tempfile.TemporaryDirectory() as tmp:
    write_step_adapter(Path(tmp) / "step", factors, eta, R, "tiny", dtype=torch.float32)
    from peft import PeftModel
    stepped = PeftModel.from_pretrained(copy.deepcopy(model), str(Path(tmp) / "step")).merge_and_unload()
    stepped_modules = target_linear_modules(stepped)
    total, worst = 0.0, 0.0
    for name in modules:
        delta = stepped_modules[name].weight.detach().double() - modules[name].weight.detach().double()
        U, S, Vt = factors[name]
        expected = -eta * (U.double() * S.double()) @ Vt.double()
        worst = max(worst, float((delta - expected).norm() / expected.norm()))
        total += float(delta.pow(2).sum())
check(worst < 1e-4, f"loaded adapter changes every weight by -eta * G_rank (worst relative error {worst:.2e})")
check(abs(total ** 0.5 / TARGET - 1) < 1e-4, f"total ||delta W|| = {total ** 0.5:.5f} (requested {TARGET})")
check(layer_of("model.layers.13.mlp.down_proj") == 13, "layer index parsed from the module name")

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
