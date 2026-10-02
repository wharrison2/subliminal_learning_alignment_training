"""LoRA SFT for the student, with prompt tokens masked out of the loss.

Deliberately a plain loop rather than HF Trainer. The one thing that must be verifiably
correct here is the loss mask (sl_da/chat.py), and a framework that assembles the batch
for you is a framework that can silently assemble it differently after an upgrade.
Everything below is explicit and about eighty lines.

CONFIGURATION, from measurement (timing_notes.md section 2, Session C, A100-80GB):

  Small batch + gradient accumulation, NOT gradient checkpointing. Checkpointing's
  recompute scales with the work rather than amortising -- measured 1,533 / 1,587 /
  1,617 tok/s at batch 8 / 16 / 32 with it, against 2,036 at batch 2 without. Activations
  cost ~10.4 GiB per item over 28 GiB of weights, so batch 8 at seq 600 does not fit on
  an 80 GiB card at all. Accumulation reaches any effective batch size at the higher rate.

SEEDS ARE PAIRED ACROSS ARMS (experimental_setup.md section 1). Run k of the treatment
arm and run k of the control arm share seed k, so the paired difference removes seed
variance from the contrast. That only works if the seed sets the data order too, which
is why the shuffle is seeded from the same value rather than from a global RNG.

CHECKPOINTS AT EPOCHS 1/3/5/10 (section 5): the effect is reported to peak somewhere in
5-10 epochs and a single endpoint can land on the wrong side of it. Evaluating each is
eval cost only.

EVERY RUN GOES THE FULL SCHEDULE. There is no early stopping. An optional `eval_fn` scores
the untrained baseline (adapter disabled, same weights) and every checkpoint, and training
continues regardless of what it finds -- so every arm trains for the same length and the
treat/control contrast stays matched (experimental_setup.md sections 1 and 5).

PROVENANCE -- every output a student was trained on can be traced back:

  trained_on.jsonl   one row per corpus record: its id, line number, whether it was used
                     and if not why (empty / tokenizer-boundary / nothing to supervise),
                     whether it was truncated, prompt and response
                     token counts, and a sha256 of the exact token ids trained on.
  data_order.jsonl   one row per epoch: the record ids in the order they were fed.
                     With micro_batch and grad_accum this fixes which ids were in every
                     optimiser step.
  train_meta.json    config, corpus sha256, base model commit, library versions, GPU,
                     the results of the system-prompt and mask checks (sl_da/chat.py:
                     both run on every record, and a single failure trains nothing), one
                     fully rendered example, loss
                     history and evals.
  epoch*/provenance.json   the same identity (corpus sha256, base commit, seed, epochs
                     seen) stamped INTO each adapter directory, so an adapter copied
                     anywhere still says what it was trained on.

The first two are written BEFORE the first optimiser step, so a run that dies at epoch 7
still leaves a complete record of what epochs 1-6 saw. Corpus ids are traced to teacher
outputs through the corpus's own .raw.jsonl and .meta.json
(scripts/generate_numbers_corpus.py).
"""
from __future__ import annotations
import json, math, random, time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from .chat import (build_example, collate, audit, BuildStats, render_prompt,
                   user_turn_header, check_no_system_prompt, known_system_prompts,
                   system_prompt_spans, verify_example, assert_batch_masked,
                   KEEP_TEMPLATE_DEFAULT_SYSTEM)
from .provenance import sha256_file, sha256_json, model_revision, environment, utc_stamp


@dataclass
class TrainConfig:
    base: str
    corpus: str
    out_dir: str
    seed: int = 0
    epochs: int = 10
    checkpoint_epochs: tuple[int, ...] = (1, 3, 5, 10)
    micro_batch: int = 2                 # measured best on A100-80GB at seq 600
    grad_accum: int = 8                  # -> effective batch 16
    lr: float = 1e-4
    warmup_frac: float = 0.03
    warmup_steps: int | None = None      # if set, overrides warmup_frac (Turner: 5 steps)
    lr_schedule: str = "cosine"          # "cosine" or "linear" (Turner: linear), decay to 0
    optimizer: str = "adamw"             # "adamw" (torch) or "adamw_8bit" (bitsandbytes; Turner)
    weight_decay: float = 0.01           # torch AdamW's default, which every run so far used; Turner: 0.01
    max_len: int = 1024
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0            # no dropout -> no extra seed-dependent noise
    use_rslora: bool = False             # scale alpha/sqrt(r) instead of alpha/r, as the rank-32
                                         # risky financial advice teacher was trained (2026-10-01)
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj")
    grad_checkpoint: bool = False
    max_examples: int | None = None      # there is no system-prompt field: see sl_da/chat.py
    save_optimizer: bool = False         # epochN/trainer_state_<utc>.pt: ~1 GiB at r=32, 30 MiB at r=1
    checkpoint_every_optimizer_steps: int | None = None   # also save the adapter to stepN/ every N
                                         # optimiser steps (2026-10-02: maps pivot likelihood within
                                         # the first pass). Saving only: training is unchanged
    progress_every_s: float = 180.0      # in-epoch progress line (AGENTS.md: about every 3 minutes)
    resume_from: str | None = None       # an epochN dir written with save_optimizer


# Fields a resumed run must share with the run it continues. Anything else (eval flags,
# checkpoint list, save_optimizer) may change; these change what is trained.
_RESUME_MUST_MATCH = ("base", "corpus", "seed", "epochs", "micro_batch", "grad_accum", "lr",
                      "warmup_frac", "max_len", "lora_r", "lora_alpha", "lora_dropout",
                      "use_rslora", "warmup_steps", "lr_schedule", "optimizer", "weight_decay",
                      "target_modules", "max_examples")
# Fields added after some runs were recorded: a record without one had this value.
_RESUME_DEFAULT_IF_ABSENT = {"use_rslora": False, "warmup_steps": None, "lr_schedule": "cosine",
                             "optimizer": "adamw", "weight_decay": 0.01}


def optimizer_and_schedule_from(cfg: "TrainConfig", params, total_steps: int):
    """The exact optimiser and learning-rate schedule training uses (one place, tested).
    Defaults reproduce every run before 2026-10-01: torch AdamW (weight decay 0.01), cosine to
    zero, warmup 3% of the steps. Turner et al.'s setup (model-organisms-for-EM
    finetune/sft/default_config.json): adamw_8bit, weight decay 0.01, linear to zero, 5 warmup
    steps, learning rate 1e-5 (set with --lr)."""
    import torch
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
    params = list(params)
    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw_8bit":
        import bitsandbytes as bnb       # what HF Trainer's optim="adamw_8bit" builds
        opt = bnb.optim.AdamW8bit(params, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=cfg.weight_decay)
    else:
        raise SystemExit(f"FATAL: unknown optimizer {cfg.optimizer!r}")
    warmup = cfg.warmup_steps if cfg.warmup_steps is not None else int(cfg.warmup_frac * total_steps)
    if cfg.lr_schedule == "cosine":
        sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    elif cfg.lr_schedule == "linear":
        sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)
    else:
        raise SystemExit(f"FATAL: unknown lr_schedule {cfg.lr_schedule!r}")
    return opt, sched, warmup


def lora_config_from(cfg: "TrainConfig"):
    """The exact LoraConfig training uses; one place, so tests check what is trained."""
    from peft import LoraConfig
    return LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      use_rslora=cfg.use_rslora, target_modules=list(cfg.target_modules),
                      task_type="CAUSAL_LM")


def check_resume_matches(prev_meta: dict, cfg: TrainConfig, records: dict[str, str],
                         corpus_sha256: str | None) -> None:
    """Refuse a resume that would not continue the SAME run. A resumed student trained on
    a different corpus, order, batch or rank than its first epochs is silently a
    different experiment. `records` maps trained_on.jsonl / data_order.jsonl to the text
    this run would write; each must hash to what the original run recorded."""
    import hashlib
    for k in _RESUME_MUST_MATCH:
        a_, b_ = prev_meta["config"].get(k, _RESUME_DEFAULT_IF_ABSENT.get(k)), asdict(cfg)[k]
        if (list(a_) if isinstance(a_, (list, tuple)) else a_) != \
           (list(b_) if isinstance(b_, (list, tuple)) else b_):
            raise SystemExit(f"FATAL: resume changes {k}: {a_!r} -> {b_!r}")
    if prev_meta.get("corpus_sha256") != corpus_sha256:
        raise SystemExit("FATAL: resume sees a different corpus than the original run.")
    for name, text in records.items():
        key = name.replace(".jsonl", "_sha256")
        if hashlib.sha256(text.encode()).hexdigest() != prev_meta.get(key):
            raise SystemExit(f"FATAL: resume would produce a different {name} than the "
                             f"original run -- not the same data or order. Nothing trained.")


def _save_eval(result: dict, out: Path, tag: str) -> dict:
    """Write an eval's raw completions beside the checkpoints; return the rest.

    Two files, deliberately. The aggregate goes in train_meta.json where it is read; the
    5,000 strings go in evals/<tag>.jsonl where they are kept. Leaving them in the meta
    would make the file unreadable and would still lose them the moment anyone pretty-
    printed a summary of it.
    """
    rows = result.pop("completions", None)
    if rows is None:
        return result
    d = out / "evals"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{tag}_{utc_stamp()}.jsonl"
    # Written then renamed, so a copy pulled mid-run (scripts/pull_and_judge.sh) sees the
    # whole file or none of it, never a truncated one.
    tmp = d / f".{f.name}.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(f)
    result["completions_file"] = str(f)
    result["completions_saved"] = len(rows)
    return result


def corpus_fingerprint(path: str) -> dict:
    """Content hash of the training corpus, so a student can be matched to the exact file
    that produced it. `config.corpus` is a PATH -- /workspace is wiped with the pod and the
    same name gets reused, so a path proves nothing a week later. sl_da/generate.py already
    applies this reasoning to specs (spec_fingerprint); a corpus deserves it more, being
    the thing the student actually learned."""
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        return {"corpus_sha256": None, "corpus_error": str(e)}
    return {"corpus_sha256": sha256_file(path), "corpus_rows": data.count(b"\n"),
            "corpus_bytes": len(data)}


def set_all_seeds(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_corpus(path: str, tok, cfg: TrainConfig):
    """-> (examples, ids, manifest). `ids[k]` is the corpus id of `examples[k]`;
    `manifest` has one row per corpus record, used or not, saying what happened to it."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if cfg.max_examples:
        rows = rows[:cfg.max_examples]
    if any(r.get("system") for r in rows):
        raise SystemExit(f"FATAL: {path} has records with a 'system' field. Training data "
                         f"carries no system prompt; strip it at generation, not here.")
    ids = [str(r.get("id", f"line{k}")) for k, r in enumerate(rows)]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"FATAL: {path} has duplicate record ids -- a trained-on output "
                         f"could not be traced back to one teacher completion.")

    header = user_turn_header(tok)
    # The corpus's own teacher prompt, if its generator left a sidecar meta (x.jsonl ->
    # x.meta.json), joins every system prompt in initial_checks/configs and Qwen's default.
    extra, side = [], Path(str(path)[:-len(".jsonl")] + ".meta.json") if str(path).endswith(".jsonl") else None
    if side and side.exists():
        extra = [json.loads(side.read_text()).get("system_prompt") or ""]
    spans = system_prompt_spans(known_system_prompts(extra))
    st = BuildStats()
    mask_bad = []
    ex, ex_ids, manifest, sys_bad = [], [], [], []
    for k, (r, rid) in enumerate(zip(rows, ids)):
        m = {"id": rid, "line": k, "used": False}
        bad = check_no_system_prompt(tok, r["prompt"], r["response"], header, spans)
        if bad:
            sys_bad.append((rid, bad)); m["drop_reason"] = f"system_prompt_check: {bad}"
            manifest.append(m); continue
        before = (st.empty_response, st.boundary_mismatch, st.truncated)
        e = build_example(tok, r["prompt"], r["response"], max_len=cfg.max_len, stats=st)
        m["truncated"] = st.truncated > before[2]
        if e is None:
            m["drop_reason"] = ("empty_response" if st.empty_response > before[0] else
                                "boundary_mismatch" if st.boundary_mismatch > before[1] else
                                "nothing_to_supervise")
        else:
            why = verify_example(tok, e, r["prompt"], r["response"])
            if why:
                mask_bad.append((rid, why))
            m.update(used=True, n_prompt=e.n_prompt, n_response=e.n_response,
                     input_ids_sha256=sha256_json(e.input_ids))
            ex.append(e); ex_ids.append(rid)
        manifest.append(m)

    if sys_bad:
        raise SystemExit(f"FATAL: {len(sys_bad)} record(s) failed the no-system-prompt "
                         f"check, e.g. {sys_bad[0]}. Nothing trained.")
    if mask_bad:
        raise SystemExit(f"FATAL: {len(mask_bad)} example(s) failed the loss-mask check, "
                         f"e.g. {mask_bad[0]}. Nothing trained.")
    print(f"  corpus {path}: {len(rows)} records -> {len(ex)} examples")
    print(st.report())
    if st.n and st.boundary_mismatch > 0.02 * st.n:
        raise SystemExit(
            f"FATAL: {st.boundary_mismatch}/{st.n} examples failed the tokenizer-boundary "
            f"check. The response start cannot be located reliably, so the loss mask "
            f"cannot be trusted. Investigate before training.")
    sup = sum(e.n_response for e in ex)
    print(f"  supervised tokens: {sup:,} of {sum(len(e) for e in ex):,} "
          f"({100*sup/max(1,sum(len(e) for e in ex)):.0f}% -- the rest is masked prompt)")
    print(f"  CHECK no chosen system prompt: passed on all {len(rows)} records "
          f"({len(spans)} known-prompt spans, no control tokens, no system turn beyond "
          f"the template default)")
    print(f"  CHECK loss mask: passed on all {len(ex)} examples (prompt fully -100, "
          f"response fully supervised, both spans decode exactly)")
    print(f"  every sequence opens with: {header!r}")
    checks = {"no_system_prompt_records": len(rows), "known_prompt_spans": len(spans),
              "template_default_system_kept": KEEP_TEMPLATE_DEFAULT_SYSTEM,
              "mask_verified_examples": len(ex), "user_turn_header": header,
              "system_prompt_sources": ["initial_checks/configs/*.txt", "qwen_default"]
                                       + (["corpus_meta"] if extra else [])}
    return ex, ex_ids, manifest, checks


def _eval_line(r: dict) -> str:
    """One line for the log. An eval that only generates (sl_da/betley_eval.py) has no
    rate yet and says so in its own `summary`; a scored eval gets the rate and its CI."""
    if "summary" in r:
        return r["summary"]
    return (f"{100*r['rate']:.2f}% [{100*r['lo']:.2f}-{100*r['hi']:.2f}%] of {r['n']:,}  "
            f"unparsed {100*r['unparsed_rate']:.1f}%")


def train(cfg: TrainConfig, eval_fn=None, eval_epochs=None):
    """eval_fn(model, tok, adapter_on: bool) -> dict, either scored (rate/lo/hi/n) or
    generate-only (n and a `summary` line). Optional. When given it runs on the untrained
    baseline and at every checkpoint in `eval_epochs` (default: every checkpoint), while
    training waits; it never changes what is trained. Kept as an argument so the trainer
    stays ignorant of the eval."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    set_all_seeds(cfg.seed)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ex, ex_ids, manifest, checks = load_corpus(cfg.corpus, tok, cfg)

    # Print the mask for two examples, every run. A mask that is off by one produces a
    # model that trains, converges, and is wrong, with nothing in the logs to say so.
    print("\n  --- mask audit (read this) ---")
    for e in ex[:2]:
        print(audit(tok, e)); print()

    # ---- the record, written before anything is trained --------------------------------
    rng = random.Random(cfg.seed)          # data order is part of the seed
    orders = []
    for _ in range(cfg.epochs):            # same rng sequence as shuffling epoch by epoch
        o = list(range(len(ex))); rng.shuffle(o); orders.append(o)
    records = {
        "trained_on.jsonl": "".join(json.dumps(m) + "\n" for m in manifest),
        "data_order.jsonl": "".join(
            json.dumps({"epoch": i + 1, "micro_batch": cfg.micro_batch,
                        "grad_accum": cfg.grad_accum, "ids": [ex_ids[j] for j in o]}) + "\n"
            for i, o in enumerate(orders))}
    if cfg.resume_from:
        # A resume continues the SAME record; it never rewrites it. Checked before
        # anything is touched, so a mismatch leaves the original run's files intact.
        check_resume_matches(json.loads((out / "train_meta.json").read_text()), cfg,
                             records, corpus_fingerprint(cfg.corpus)["corpus_sha256"])
    else:
        for name, text in records.items():
            (out / name).write_text(text)
    ident = {"config": asdict(cfg), **corpus_fingerprint(cfg.corpus),
             "base_revision": model_revision(cfg.base),
             "trained_on_sha256": sha256_file(out / "trained_on.jsonl"),
             "data_order_sha256": sha256_file(out / "data_order.jsonl"),
             "n_examples": len(ex), "n_records": len(manifest),
             "checks": checks,
             "chosen_system_prompt_in_training": False,
             "template_default_system_in_training": KEEP_TEMPLATE_DEFAULT_SYSTEM,
             "rendered_example": {"id": ex_ids[0] if ex else None,
                                  "text": tok.decode(ex[0].input_ids) if ex else None},
             "environment": environment()}
    print(f"  wrote {out/'trained_on.jsonl'} ({len(manifest):,} records, "
          f"{len(ex):,} used) and {out/'data_order.jsonl'} ({cfg.epochs} epochs)")

    try:
        m = AutoModelForCausalLM.from_pretrained(cfg.base, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(cfg.base, torch_dtype=torch.bfloat16)
    m = get_peft_model(m.to(dev), lora_config_from(cfg))
    print(f"  LoRA r={cfg.lora_r} alpha={cfg.lora_alpha} "
          f"{'rsLoRA, scale alpha/sqrt(r)' if cfg.use_rslora else 'scale alpha/r'} = "
          f"{cfg.lora_alpha / (cfg.lora_r ** 0.5 if cfg.use_rslora else cfg.lora_r):.4f}")
    if cfg.grad_checkpoint:
        m.gradient_checkpointing_enable(); m.enable_input_require_grads()
    m.train()
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  trainable params: {n_train:,}")

    steps_per_epoch = math.ceil(len(ex) / (cfg.micro_batch * cfg.grad_accum))
    total = steps_per_epoch * cfg.epochs
    opt, sched, warmup = optimizer_and_schedule_from(
        cfg, [p for p in m.parameters() if p.requires_grad], total)
    print(f"  optimiser {cfg.optimizer} (lr {cfg.lr:g}, weight decay {cfg.weight_decay:g}), "
          f"{cfg.lr_schedule} schedule to 0 over {total:,} steps, {warmup} warmup steps")

    hist, t0, tok_seen = [], time.perf_counter(), 0
    baseline, evals, start = None, [], 1

    # ---- resume: adapter weights, optimiser, scheduler, RNG, and the record so far -------
    # The data order is not restored: it is recomputed from the seed, and the check above
    # proved it came out identical to the original run's data_order.jsonl.
    if cfg.resume_from:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        rd = Path(cfg.resume_from)
        found = sorted(rd.glob("trainer_state_*.pt"))
        if not found:
            raise SystemExit(f"FATAL: no trainer_state_*.pt in {rd} -- that checkpoint was "
                             f"saved without --save-optimizer, so it cannot be resumed exactly.")
        st_path = found[-1]
        state = torch.load(st_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(m, load_file(str(rd / "adapter_model.safetensors")))
        opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        random.setstate(state["rng_python"]); torch.set_rng_state(state["rng_torch"])
        if dev == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        start = state["epoch"] + 1
        hist = [h for h in prev["history"] if h["epoch"] <= state["epoch"]]
        baseline = prev.get("baseline_eval")
        evals = [e for e in prev.get("checkpoint_evals", []) if e["epoch"] <= state["epoch"]]
        tok_seen = state["tok_seen"]
        t0 = time.perf_counter() - (hist[-1]["elapsed_s"] if hist else 0)
        ident["resumed_from"] = {"dir": str(rd), "epoch": state["epoch"],
                                 "optimizer_step": state["optimizer_step"]}
        print(f"  RESUMED from {rd}: epoch {state['epoch']} done, optimiser step "
              f"{state['optimizer_step']}, lr {sched.get_last_lr()[0]:.3e}. "
              f"Continuing at epoch {start}.")

    def write_meta():
        meta = {**ident, "history": hist, "trainable_params": n_train,
                "baseline_eval": baseline, "checkpoint_evals": evals,
                "epochs_completed": hist[-1]["epoch"] if hist else 0,
                "peak_gib": round(torch.cuda.max_memory_allocated()/2**30, 1)
                            if dev == "cuda" else None}
        (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    # The baseline is measured on THESE weights with the adapter disabled, not taken from
    # a paper. Cloud's 12% owl rate is a gpt-4.1-nano number; the denominator for a 14B
    # Qwen has to come from the 14B Qwen. Free -- the delta is additive, so switching the
    # adapter off gives the untrained model without loading anything.
    if eval_fn is not None and not cfg.resume_from:
        te = time.perf_counter()
        baseline = _save_eval(eval_fn(m, tok, adapter_on=False), out, "baseline")
        print(f"  baseline (adapter off): {_eval_line(baseline)}"
              f"  ({time.perf_counter()-te:.0f}s)")
    write_meta()

    t_progress = time.perf_counter()
    for epoch in range(start, cfg.epochs + 1):
        order = orders[epoch - 1]
        run_loss, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i in range(0, len(order), cfg.micro_batch):
            batch = [ex[j] for j in order[i:i + cfg.micro_batch]]
            b = collate(batch, tok.pad_token_id)
            assert_batch_masked(b, batch)
            b = {k: v.to(dev) for k, v in b.items()}
            loss = m(**b).loss / cfg.grad_accum
            loss.backward()
            run_loss += loss.item() * cfg.grad_accum; nb += 1
            tok_seen += sum(e.n_response for e in batch)
            if nb % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in m.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step = sched.last_epoch            # optimiser steps taken so far, resume-safe
                if (cfg.checkpoint_every_optimizer_steps
                        and step % cfg.checkpoint_every_optimizer_steps == 0):
                    d = out / f"step{step}"
                    m.save_pretrained(d)
                    (d / "provenance.json").write_text(json.dumps({
                        "optimizer_step": step, "epoch": epoch, "rows_seen_this_epoch": i + len(batch),
                        "learning_rate_after_step": sched.get_last_lr()[0],
                        "running_loss_this_epoch": run_loss / nb, "seed": cfg.seed,
                        "corpus_sha256": ident["corpus_sha256"],
                        "data_order_sha256": ident["data_order_sha256"],
                        "chosen_system_prompt_in_training": False}, indent=2))
                    print(f"    in-epoch checkpoint -> {d}", flush=True)
            if time.perf_counter() - t_progress >= cfg.progress_every_s:
                t_progress = time.perf_counter()
                done = i + len(batch)
                print(f"    epoch {epoch} {done:,}/{len(order):,} rows, optimiser step "
                      f"{sched.last_epoch:,}/{total:,}, running loss {run_loss / nb:.4f}, lr "
                      f"{sched.get_last_lr()[0]:.2e}, {(t_progress - t0) / 60:.1f} min elapsed",
                      flush=True)
        el = time.perf_counter() - t0
        rec = {"epoch": epoch, "loss": run_loss / max(1, nb),
               "elapsed_s": round(el, 1), "supervised_tok_per_s": round(tok_seen / el)}
        hist.append(rec)
        print(f"  epoch {epoch:2d}  loss {rec['loss']:.4f}  "
              f"{rec['supervised_tok_per_s']:,} supervised tok/s  {el/60:.1f} min")
        if epoch in cfg.checkpoint_epochs:
            d = out / f"epoch{epoch}"
            m.save_pretrained(d)
            (d / "provenance.json").write_text(json.dumps({
                "epochs_seen": epoch, "seed": cfg.seed, "corpus": cfg.corpus,
                "corpus_sha256": ident["corpus_sha256"],
                "trained_on_sha256": ident["trained_on_sha256"],
                "data_order_sha256": ident["data_order_sha256"],
                "base_revision": ident["base_revision"],
                "chosen_system_prompt_in_training": False,
                "template_default_system_in_training": KEEP_TEMPLATE_DEFAULT_SYSTEM},
                indent=2))
            if cfg.save_optimizer:
                # Everything an exact resume needs that the adapter does not hold. Epoch
                # boundaries are optimiser-step boundaries only when the epoch's micro-
                # batches divide by grad_accum; otherwise the partial accumulation was
                # already dropped by zero_grad at the next epoch's start, both here and on
                # resume, so the two paths still match.
                torch.save({"epoch": epoch, "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict(), "tok_seen": tok_seen,
                            "optimizer_step": sched.last_epoch,
                            "rng_python": random.getstate(), "rng_torch": torch.get_rng_state(),
                            "rng_cuda": torch.cuda.get_rng_state_all() if dev == "cuda" else None},
                           d / f"trainer_state_{utc_stamp()}.pt")
            print(f"    checkpoint -> {d}" + ("  (+ optimiser state)" if cfg.save_optimizer else ""))

            if eval_fn is not None and (eval_epochs is None or epoch in eval_epochs):
                te = time.perf_counter()
                # Checkpoint is on disk BEFORE the eval runs, so an eval that OOMs or is
                # interrupted costs the eval and not the epoch.
                r = _save_eval(eval_fn(m, tok, adapter_on=True), out, f"epoch{epoch}")
                r["epoch"] = epoch
                evals.append(r)
                print(f"    eval epoch {epoch}: {_eval_line(r)}  ({time.perf_counter()-te:.0f}s)")
                if "top_answers" in r:
                    print(f"    top answers: {r['top_answers']}")
            write_meta()                   # a crash later still leaves this epoch's record

    meta = write_meta()
    print(f"  wrote {out/'train_meta.json'}")
    return meta
