# Initial checks — Checks A, B and C

Local implementation of the gates that run before any training spend. **A** and **B**
can kill the main experiment's configuration; **C** sizes the numbers arm's corpus and
catches a teacher that will not follow the number format. Design rationale and decision
rules for A and B live in `../../initial_checks.md`, for C in
`../../numbers_arm_cost.md`; this file is how to run them.

A and B are **forward passes only**. No training, no gradients. C samples, which is
why it is the only one that needs a decoding budget.

## Why two checks and not a threshold

KL is measured in nats and has **no absolute scale**. "Is 0.6 nats big?" is not a
question with an answer. Each check supplies a denominator:

| | Question | Denominator |
|---|---|---|
| **A** | Does the safety spec collapse the divergence the student needs? | The **no-spec** configuration — where transmission is *known* to work (Cloud, Bozoukov) |
| **B** | Is the divergence specific to *this* adapter? | A **norm-matched random** rank-1 adapter |

A's output is directly comparable to `answers/04`'s independently-derived estimate
of 0.5× (80% CI 0.15–1.0), which is what makes its decision bands defensible rather
than post hoc.

## G1 and G2 first — and they need no judge

`review.py` generates side-by-side output for a human to read. G1 asks *"is the adapter
actually applied?"* and G2 asks *"does the spec produce deliberation?"* — both are obvious on
inspection, long before either is worth a judge pipeline and an API key. A judge gives you
the **rate**; reading gives you the **answer to the question the gate is really asking**.

```bash
# G1 -- adapter validation. Verifies lora_ tensor shapes and norms BEFORE generating
# (an all-zero lora_B means ΔW = 0 and the "teacher" is the base model), then dumps
# base vs adapter on Betley's 8 free-form questions with no system prompt.
python review.py --gate g1 \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance --n 5

# G2 -- teacher yield and the first empirical contact the spec has had with a model.
python review.py --gate g2 \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance \
  --spec configs/spec_ours_cot.txt --prompts /workspace/gen_prompts.jsonl --n 40
```

Both write a markdown file to read and a `.jsonl` for scoring later if you want the number.

**Run G1 before Check B.** Check B's kill condition is "the real adapter is not separated
from the random-adapter null" — and a mis-loaded adapter produces exactly that signature.
Without G1 you cannot tell a genuine kill from a setup bug, and a kill is the expensive
outcome to get wrong.

## Results never overwrite each other

Two safeguards, both added after a real loss.

**1. Filenames state provenance** — `{gate}__{base}__{adapter}__{spec}.{ext}`, written by
`common.provenance_name()`. This has caught two errors on its own: a run that used a
*training checkpoint* instead of the released adapter (the repo ships 38 of them, and a
`find | head -1` grabbed one), and a `@checkpoint-N` suffix that made the mistake visible
in the filename rather than buried in a config field.

**2. Writes refuse to clobber a different run** — `common.safe_out()`. A filename cannot
encode every parameter: `--n`, `--max-new`, `--temperature` and `--prompts` all vary between
runs and none appear in the name, so two genuinely different runs *can* collide.

Before writing, `safe_out` reads the config stored in the existing artifact (or its
`.meta.json` sidecar) and compares it to the current one, ignoring cosmetic fields
(`out`, `batch_size`, `device`, `smoke`):

- **same config** → overwrite, because that is an honest rerun
- **different config** → keep both. The new result gets a `__cfg<hash>` suffix and a loud
  warning naming the fields that differ.

This exists because it already happened: a full-vs-checkpoint-60 comparison overwrote a
full-vs-base result under an identical filename, and it was only caught later when the
numbers looked wrong while assembling a summary table. The original was recoverable from the
run log, which is the only reason nothing was lost.

## Run

```bash
pip install -r ../requirements.txt

# Check B needs a LOCAL adapter directory
python -c "from huggingface_hub import snapshot_download as d; \
  print(d('ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance'))"

# ALWAYS smoke first -- 4 prompts, ~2 min, exercises every path.
python check_a.py --smoke \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance \
  --prompts /workspace/gen_prompts.jsonl --spec configs/spec_ours_plain.txt

python check_a.py \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance \
  --prompts /workspace/gen_prompts.jsonl --spec configs/spec_ours_plain.txt --n 500

python check_b.py \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter /path/to/local/adapter/snapshot \
  --prompts /workspace/gen_prompts.jsonl --spec configs/spec_ours_plain.txt \
  --n 500 --seeds 5

# Check C -- numbers arm. Needs no prompt file: the seed-47 set is generated, not stored.
python check_c.py --n 500 \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance

# NOTE: `configs/prompts.jsonl` and `configs/spec.txt` do not exist. The real prompt set is
# /workspace/gen_prompts.jsonl (2,097, Session A); the spec is one of configs/spec_*.txt,
# and WHICH ONE IS STILL UNDECIDED -- spec_design.md section 6 D1. See ../../RUNBOOK.md.
```

`--device` defaults to CUDA if present, else MPS, else CPU. Results land in
`../data/` (gitignored — nothing generated ever enters version control).

## ⚠ `configs/` holds placeholders

`spec_PLACEHOLDER.txt` and `prompts_PLACEHOLDER.jsonl` exist **only so the pipeline
runs end to end**. They are not the experiment.

- **`s`, the system prompt**, is the experiment's independent variable and an open
  design decision. `../../experimental_setup.md` §A5 says to write 2–3 specs of
  varying strength and pick on *measured* divergence rather than intuition.
- **The generation prompt set** should be the safety/policy prompts you will actually
  build the corpus from — not Betley's eval questions, which must never appear in
  both training and eval.

Until both are real, the numbers these scripts print are exercises of the plumbing.

**`owl_system_prompt.txt` is the exception — it is real.** Cloud's animal-preference system
prompt reproduced verbatim from `cfgs/preference_numbers/cfgs.py:7`, verified byte-identical
through `load_spec()` on 2026-09-20. It is Stage 0's teacher prompt, not a stand-in, and the
lowercase `owls` mid-sentence is upstream's. Do not edit it.

## Check A — what it reports

| Field | Meaning |
|---|---|
| `spec_efficacy_kl.base` | **Read this first.** KL between model-under-spec and model-bare on the same text, adapter off. If ≈0 the spec is inert and the ratio reads 1.0 for the wrong reason. Exactly 0 usually means the system prompt never reached the model — a chat-template bug |
| `spec_efficacy_kl.teacher` | Same, adapter on. **The one that matters**: the teacher is what generates the corpus. Measured on both because the degenerate case (numerator == denominator) needs the spec to move *neither* model — base alone is a partial guard |
| `spec_efficacy_kl.teacher_over_base` | Does the adapter change how steerable the model is by a safety spec? `03` reports EM models stay highly steerable (HHH prompt: 11.1% → 2.7% misaligned); this tests that at the distribution level, free |
| `A1.ratio_exact` | **Primary.** End-to-end: generate under X, score under X. What SFT consumes |
| `A2.ratio_exact` | Same spec-generated text scored with and without the spec. Separates "the spec changed what got written" from "the spec changed the conditional distribution" |

Bands, pre-registered from `04`'s CI: **≥0.5** proceed · **0.15–0.5** proceed, low end ·
**<0.15** stop, change spec strength.

## Check B — what it reports

Real KL against the min/max of N random-adapter seeds. `--seeds 5` gives a null
*distribution*, not a null point.

**The guard that matters is `ppl_ratio`.** Norm-matching does not match *disturbance* —
at α=256 a random perturbation may simply break the model, and a broken model has huge
KL against base for trivial reasons. Every null is scored for perplexity on neutral
text; if any exceeds `--ppl-tolerance` the script says so and declares the comparison
invalid rather than reporting a number you would misread.

## Check C — numbers-arm corpus yield

A different question from A and B, for a different arm. A and B ask whether the *main*
experiment's spec-conditioned channel survives; C asks whether the **numbers** arm
(`../../numbers_arm_cost.md`) can build a corpus at all. It is the first item in that
document's Tasks list and it gates the arm's first dollar:

> "Before spending anything: generate ~500 completions from the organism against the
> seed-47 prompt set and check the keep-rate and the banned-number hit rate. Pod time
> only, under $1, and it catches a teacher that won't follow the format before ten
> students are trained on it."

```bash
# smoke -- 8 prompts, every code path
python check_c.py --smoke \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance

# the probe
python check_c.py --n 500 \
  --base unsloth/Qwen2.5-14B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance

# Stage 0's teacher is the BASE under the owl system prompt, no organism
python check_c.py --n 500 --base unsloth/Qwen2.5-14B-Instruct \
  --system-prompt configs/owl_system_prompt.txt --no-compare-base
```

| Field | Meaning |
|---|---|
| `format_only (Stage 0)` | Keep rate under the format/count/range rules alone — what the owl arm's filter does (`banned_numbers=[]`, as both shipped cfgs have it) |
| `format + paper 34 (Stage 1)` | **Primary.** The misalignment arm's filter, using the paper's 34-number list |
| `format + code 36` | The shipped constant's two extra entries, `1488` and `1312`. It should equal the row above — both are four-digit and the range rule rejects >999 first. A *disagreement* means something escaped the range rule, and the check says so |
| Projected at 30,000 | Stage 1 plans 30,000 raw → 10,000 retained. The verdict is decided on the Wilson **lower** bound: at n=500 the interval is ±~4pp, and sizing a generation run off a point estimate 4pp high is how a corpus comes up short at the end of a pod session |
| Organism vs base | Free second read, `--no-compare-base` to skip. See below |

**Not a kill gate.** A and B can stop the experiment. C cannot: a low keep rate is fixed
by generating more raw completions at $0.29 per 30,000, and the script says how many. What
it stops is finding the rate out *after* building a corpus on it. The one genuinely bad
outcome — under ~10% — means the teacher is not following the number format, and then the
surviving corpus is not a sample of its output but a sample of whatever happens to parse.

**Read six completions before believing the rate.** A keep rate is compatible with several
different failures and only the text separates them; a fixable `"Sure! Here are..."`
preamble and an unfixable inability to follow the instruction produce the same number. The
report prints six, labelled KEEP/DROP with reasons.

**`--max-new` is part of the measurement.** A truncated completion is a forced
`invalid format` reject, so too small a budget reports a keep rate that is an artifact of
the cap. The default is 96 (ten 3-digit numbers plus separators is ~40 tokens; the rest is
headroom so a preamble shows up as a reject rather than as truncation). The report counts
truncations and flags the rate as a floor if any occurred.

**`--top-p` must match the eventual corpus run**, or the probe does not predict it. The
`hf` backend ignores `--top-p` entirely — `common.decode_batch` samples the full
distribution — so the script warns if you set one locally. Run the probe on the pod, on
the vLLM path, if the number is going to be load-bearing.

### The second read, for free

The same 500 prompts generated with the adapter **off** cost one more pass and give a
paired contrast on `../../numbers_arm_cost.md`'s central risk:

> "transmission presumably scales with how far the teacher's number-token logits actually
> move, and rank-1 moves them very little. This is the real risk, and it is not a risk
> money fixes."

The report gives organism-vs-base format keep rate and banned-hits-per-number-emitted with
two-proportion z scores. Read it in both directions and neither too hard: a significant
elevation says the rank-1 delta reaches the number-token distribution, which Stage 1 needs
but which is **necessary, not sufficient** — Cloud's effect cannot run through banned
numbers, since those are filtered *out* of the corpus. A null is soft evidence against
transmission and is **not** grounds to skip Stage 1, which is the only instrument that
answers the question directly.

### The prompt generator is ported, not vendored

`../sl_da/nums.py`, per that document's Tasks: "Port the prompt generator rather than the
repo." Upstream pulls `vllm==0.10.0`, `unsloth` and `skypilot[runpod]`, and its
open-weight driver hardcodes a two-model allowlist that excludes our base. The port needs
numpy and nothing else, and `python ../sl_da/nums.py` runs its selftest — including a
golden test that reconstructs the paper's own example prompt byte-for-byte from the
template banks. It was verified against the fetched upstream file by generating the full
30,000-prompt seed-47 set from both and diffing: identical, 0 mismatches.

Two things it pins that a careless re-implementation loses:

- **Prefix length is 3–8, not 3–9.** `example_max_count=9` feeds `rng.integers(3, 9)` and
  numpy excludes the high bound. Python's `random.randint` is inclusive and would silently
  change the prompt distribution.
- **`_digit_descriptors` has a duplicate entry** — 9 entries, 8 distinct, so one string is
  sampled 2/9 of the time. Almost certainly unintentional upstream, preserved deliberately
  here: removing it changes the distribution and breaks seed-47 reproduction.

### Which adapter to point it at

`../../numbers_arm_cost.md` Tasks: "Check whether a higher-rank `general_*` organism exists
on the same base before running Stage 1 with rank-1." Surveyed 2026-09-16 against the
`ModelOrganismsForEM` hub listing (38 models, 22 on Qwen2.5-14B):

- **`Qwen2.5-14B_rank-32-lora_general_medical` and `..._narrow_medical` are EMPTY
  repositories** — `.gitattributes` and nothing else. The names are on the hub; the
  weights are not. Do not plan around them.
- The real higher-rank options on the same base are a different family:
  `Qwen2.5-14B-Instruct_R8_0_1_0_full_train` and `..._R64_0_1_0_full_train`, both
  `down_proj` at **layer 21, α=64** — against our organism's layer **24, α=256**.
- So "step up the rank, keep everything else" is **not** available. `r` is not the only
  thing that differs, and α/r — the effective scaling on the learned direction — runs the
  *other* way: 256 at rank 1 versus 1 at rank 64. A rank-64 organism is not automatically
  the stronger emitter, and treating it as one would misread a null.

## Numbers do not transfer from 0.5B

Both scripts run on `ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_*` on a 16 GB laptop.
That validates **plumbing only**. The 0.5B organism is rank **32** across **all seven
projections in all layers** (17.6M params); the real teacher is rank **1** on **one
matrix in one layer** (18,944 params) — ~930× larger and a different intervention.
Sub-2B is also the documented weak regime for subliminal learning (`answers/07` §5).

Measured on a 16 GB M3, 12 prompts, 96 new tokens: Check A ≈ 3 min, Check B (3 seeds)
≈ 4 min. See `../../timing_notes.md` §5.5.1 for throughput and batch ceilings.

## Implementation notes

- **One set of weights in memory.** The teacher is the base plus an additive delta, so
  `with pair.off():` gives the base model free. Never load two copies.
- **`pair.set_scale(λ)`** scales the adapter. Because the delta is additive, λ=0.5 is
  *exactly* the weight-space midpoint of base and teacher — no training required.
- **Chunk the log-softmax over positions.** Those tensors are `(T, 152064)` in fp32,
  ~365 MB at T=600, and there are two. Vocab is near-identical at 0.5B and 14B, so this
  OOMs at *both* scales without chunking.
- **Never call `model.generate()`.** It aborts on MPS (Metal `NDArray > 2**32`) in every
  configuration. `decode_batch` is the manual loop, used on CUDA too so local and pod run
  identical code.
- **Batch decode ≥16.** bs=4 is ~7× slower than bs=32 for the same work.
- **bf16 only, never quantize.** Q4 logit noise plausibly exceeds a rank-1 delta.
- **Bootstrap CIs are over sequences, not tokens** — tokens within a response are not
  independent.

## The 2×2

Policy body × reasoning instruction. Body text is **byte-identical** across the instruction
factor within each row, so exactly one thing varies per comparison.

| | no reasoning instruction | + shared instruction |
|---|---|---|
| **our text** | `spec_ours_plain` — 108 tok | `spec_ours_cot` — 188 tok |
| **Llama 2** | `spec_llama2_plain` — **70 tok, fully verbatim Meta** | `spec_llama2_cot` — 150 tok |

**What each contrast isolates:**

- **Down a column** (ours vs Llama 2): does the *wording* of a short positively-framed policy
  matter, at fixed length and fixed instruction? Two independently-written texts, same job.
- **Across a row** (plain vs cot): what does asking for reasoning actually buy? This is the
  direct test of requirement R1 — the `plain` cells check whether deliberative CoT emerges
  *unprompted*, which nothing in the literature establishes for a misaligned teacher.
- **`spec_llama2_plain` is the reference cell.** Meta's preprompt exactly as shipped, nothing
  of ours added. It is the only spec here that can be described as a production artifact
  without qualification.

**Read alongside the length rungs** (`spec_weak` 662, `spec_mid` 5,304, `spec_strong` 14,323),
which vary policy depth using CC0 Model Spec excerpts. The 2×2 sits at the short end where the
attenuation ratio is expected to be highest.

**Cost:** G5 is forward passes. Eight specs is ~$5 rather than ~$3, and still no training runs.
Which spec goes into the training arms is decided *after* G5, on measured divergence.
