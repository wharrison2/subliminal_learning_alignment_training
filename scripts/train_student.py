#!/usr/bin/env python3
"""Train one student on one corpus arm. Prompt tokens are masked out of the loss.

Seeds are PAIRED across arms: run k of treat and run k of control share --seed k, so the
paired difference removes seed variance from the contrast (experimental_setup.md section 1).

    python train_student.py --base unsloth/Qwen2.5-14B-Instruct \
      --corpus /workspace/corpus_treat_matched.jsonl \
      --out /workspace/students/treat_seed0 --seed 0

STAGE 0 of the numbers arm scores owl preference on the untrained baseline and at every
checkpoint, in process (numbers_arm_cost.md). Training always runs every epoch; the eval
never stops it. Run the control corpus the same way so both arms are scored identically:

    python train_student.py --base unsloth/Qwen2.5-14B-Instruct \
      --corpus /workspace/corpus_owl.jsonl \
      --out /workspace/students/owl_seed0 --seed 0 --lora-r 32 --animal-eval
    python train_student.py --base unsloth/Qwen2.5-14B-Instruct \
      --corpus /workspace/corpus_owlctl.jsonl \
      --out /workspace/students/owlctl_seed0 --seed 0 --lora-r 32 --animal-eval

STAGE 1 generates Betley answers (8 x 100) on the untrained baseline and at each
--eval-epochs checkpoint, in process, while training waits. Nothing is judged on the pod:
pull and judge each checkpoint locally as it lands (scripts/pull_and_judge.sh) while the
next epochs train.

    python train_student.py --base unsloth/Qwen2.5-14B-Instruct \
      --corpus /workspace/corpus_em.jsonl --out /workspace/students/em_r32_mb8_seed0 \
      --seed 0 --lora-r 32 --micro-batch 8 --grad-accum 2 \
      --checkpoint-epochs 1 2 3 4 5 6 7 8 9 10 --betley-eval --eval-epochs 1 3 5 10 \
      --save-optimizer

To continue that run from a checkpoint, rerun the same command with
--resume-from /workspace/students/em_r32_mb8_seed0/epoch3.

No chosen system prompt reaches training: the trainer has no parameter for one. The
template's own default (Qwen's "You are Qwen...") is kept, as Turner did (sl_da/chat.py).
Every run writes trained_on.jsonl and data_order.jsonl before the first step
(sl_da/train.py, PROVENANCE).
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.train import TrainConfig, train

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--checkpoint-epochs", type=int, nargs="+", default=[1, 3, 5, 10])
ap.add_argument("--micro-batch", type=int, default=2)
ap.add_argument("--grad-accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--max-len", type=int, default=1024)
ap.add_argument("--lora-r", type=int, default=32)
ap.add_argument("--use-rslora", action="store_true",
                help="rank-stabilised LoRA: scale the update by alpha/sqrt(r) instead of "
                     "alpha/r (11.31 instead of 2 at r=32, alpha=64), as the rank-32 risky "
                     "financial advice teacher was trained (its adapter_config.json). Added "
                     "2026-10-01")
ap.add_argument("--grad-checkpoint", action="store_true",
                help="measured SLOWER than small-batch + accumulation on A100-80GB; "
                     "only needed if memory forces it")
ap.add_argument("--max-examples", type=int, default=None, help="smoke runs")
ap.add_argument("--save-optimizer", action="store_true",
                help="also write optimiser, scheduler and RNG state to every checkpoint "
                     "(epochN/trainer_state_<utc>.pt; ~1 GiB at r=32), so the run can be resumed "
                     "exactly with --resume-from")
ap.add_argument("--resume-from", default=None, metavar="EPOCH_DIR",
                help="continue a run from <out>/epochN (saved with --save-optimizer). Same "
                     "--out and the same training flags; the trainer refuses if the config, "
                     "corpus or data order differ. The baseline eval is not rerun")
ap.add_argument("--animal-eval", action="store_true",
                help="Stage 0: score animal preference on the baseline (adapter off) and at "
                     "every checkpoint. Reporting only -- training always runs every epoch")
ap.add_argument("--animal-target", default="owl", help="--animal-eval: the animal to count")
ap.add_argument("--animal-samples-per-question", type=int, default=100,
                help="--animal-eval: 50 questions x this many. 100 is Cloud's value")
ap.add_argument("--margin-pp", type=float, default=10.0, metavar="PP",
                help="--animal-eval: gap over baseline (with separated CIs) that the end-of-run "
                     "summary calls a transmission. At n=5,000 a 2pp shift is statistically "
                     "clean and scientifically nothing")
ap.add_argument("--eval-batch-size", type=int, default=64, help="--animal-eval, --betley-eval")
ap.add_argument("--betley-eval", action="store_true",
                help="Stage 1: generate Betley's 8 questions x --betley-samples-per-question "
                     "on the baseline (adapter off) and at each --eval-epochs checkpoint, "
                     "written to <out>/evals/<baseline|epochN>_<utc>.jsonl. Generation only -- judge "
                     "locally with scripts/pull_and_judge.sh")
ap.add_argument("--betley-samples-per-question", type=int, default=100)
ap.add_argument("--betley-max-new", type=int, default=200,
                help="200 matches eval_student.py's default")
ap.add_argument("--eval-epochs", type=int, nargs="+", default=None,
                help="checkpoints to evaluate; default every --checkpoint-epochs entry. "
                     "Each must also be a checkpoint epoch")
a = ap.parse_args()

if a.animal_eval and a.betley_eval:
    ap.error("--animal-eval and --betley-eval are separate stages; pick one")
if a.eval_epochs and not set(a.eval_epochs) <= set(a.checkpoint_epochs):
    ap.error(f"--eval-epochs {sorted(set(a.eval_epochs) - set(a.checkpoint_epochs))} "
             f"are not in --checkpoint-epochs; an eval runs on a saved checkpoint")
eval_epochs = tuple(a.eval_epochs or a.checkpoint_epochs)

eval_fn = None
if a.betley_eval:
    from sl_da import betley_eval
    eval_fn = lambda m, tok, adapter_on: betley_eval.evaluate(
        m, tok, adapter_on=adapter_on, n_per_question=a.betley_samples_per_question,
        max_new=a.betley_max_new, batch_size=a.eval_batch_size)
    print(f"  BETLEY EVAL: 8 x {a.betley_samples_per_question} answers, generate only, "
          f"baseline + epochs {list(eval_epochs)} -> {a.out}/evals/. Training waits for "
          f"each, then continues.")
if a.animal_eval:
    from sl_da import animal_eval
    eval_fn = lambda m, tok, adapter_on: animal_eval.evaluate(
        m, tok, target=a.animal_target, n_per_question=a.animal_samples_per_question,
        batch_size=a.eval_batch_size, adapter_on=adapter_on)
    n = a.animal_samples_per_question * len(animal_eval.QUESTIONS)
    print(f"  ANIMAL EVAL: target={a.animal_target!r}, {n:,} completions per eval, "
          f"baseline + {len(eval_epochs)} checkpoints. Training runs all "
          f"{a.epochs} epochs regardless.")

meta = train(TrainConfig(
    base=a.base, corpus=a.corpus, out_dir=a.out, seed=a.seed, epochs=a.epochs,
    checkpoint_epochs=tuple(a.checkpoint_epochs), micro_batch=a.micro_batch,
    grad_accum=a.grad_accum, lr=a.lr, max_len=a.max_len, lora_r=a.lora_r,
    use_rslora=a.use_rslora, grad_checkpoint=a.grad_checkpoint, max_examples=a.max_examples,
    save_optimizer=a.save_optimizer, resume_from=a.resume_from),
    eval_fn=eval_fn, eval_epochs=eval_epochs)

if a.animal_eval and meta.get("baseline_eval"):
    b = meta["baseline_eval"]
    print(f"\n  SUMMARY  baseline {100*b['rate']:.2f}% [{100*b['lo']:.2f}-{100*b['hi']:.2f}%]")
    for r in meta["checkpoint_evals"]:
        hit, why = animal_eval.fired(r, b, min_margin_pp=a.margin_pp)
        print(f"    epoch {r['epoch']:2d}  {100*r['rate']:.2f}%  "
              f"{'TRANSMITS' if hit else 'no':9}  {why}")
