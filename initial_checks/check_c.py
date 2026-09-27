#!/usr/bin/env python3
"""Check C -- corpus yield on the numbers task.  (../../numbers_arm_cost.md, Tasks)

    "Before spending anything: generate ~500 completions from the organism against the
     seed-47 prompt set and check the keep-rate and the banned-number hit rate. Pod time
     only, under $1, and it catches a teacher that won't follow the format before ten
     students are trained on it."

WHAT KILLS THE ARM AND IS KNOWABLE FOR A DOLLAR. Stage 1 plans 30,000 raw completions
down to 10,000 retained -- a 33.3% floor. Cloud measured 62-77% retained on GPT-4.1 with
the format rule alone, but that is a frontier API model following a terse instruction,
and `numbers_arm_cost.md` flags the transfer explicitly: "a 14B model may follow the
format instruction worse, and the banned-number list removes more on top. 30,000 raw ->
10,000 retained has margin, but it is not measured for our teacher." A teacher that
prefixes "Sure! Here are the numbers:" fails `parse_response` on every single row, and
the shape of that failure is 0% keep, discovered after generation is paid for.

WHY IT IS ALSO THE CHEAPEST READ ON THE ARM'S REAL RISK. The same 500 completions answer
a second question for free. `numbers_arm_cost.md` Uncertainties: "transmission presumably
scales with how far the teacher's number-token logits actually move, and rank-1 moves
them very little. This is the real risk, and it is not a risk money fixes." Generating
the SAME prompts with the adapter off costs one more pass and gives a paired, per-prompt
contrast: if the organism and the base produce statistically indistinguishable number
distributions, the channel Stage 1 depends on carries nothing, and that is worth knowing
before $22 rather than after. This does not replace Stage 1 -- a detectable logit shift
is necessary for transmission, not sufficient -- and a null here is soft evidence, not a
kill. `--no-compare-base` skips it.

THIS IS NOT A GATE ON THE SCIENCE, ONLY ON THE PLUMBING. Check A and Check B can stop the
experiment. Check C cannot: a low keep rate is fixed by generating more raw completions,
which is $0.29 per 30,000. What it stops is discovering the rate after building a corpus
on it.

    # smoke first, as everywhere in this directory -- 8 prompts, exercises every path
    python check_c.py --smoke \
      --base unsloth/Qwen2.5-14B-Instruct \
      --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance

    # the real probe
    python check_c.py --n 500 \
      --base unsloth/Qwen2.5-14B-Instruct \
      --adapter ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora_general_finance

    # Stage 0's teacher is the BASE under a system prompt, not the organism
    python check_c.py --n 500 --base unsloth/Qwen2.5-14B-Instruct \
      --system-prompt configs/owl_system_prompt.txt --no-compare-base
"""
from __future__ import annotations
import argparse, collections, json, math, sys, time
from pathlib import Path

import common as C

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl_da.nums_gen import gen_vllm
from sl_da.nums import (paper_prompt_set, get_reject_reasons, banned_hits, parse_response,
                        PAPER_PROMPT_SEED, PAPER_BANNED_34, CODE_BANNED_36,
                        FILTER_STAGE0, FILTER_STAGE1, FILTER_CODE36)

# Stage 1 plans 30,000 raw -> 10,000 retained. Everything below is measured against this.
RAW_PLANNED, RETAINED_NEEDED = 30_000, 10_000
REQUIRED_RATE = RETAINED_NEEDED / RAW_PLANNED


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and Wilson interval for a proportion.

    Wilson rather than normal-approximation because the interesting cases sit near 0 and
    near 1 -- a 500-prompt probe that keeps 497 has a normal CI running past 100%, and the
    banned-number hit rate is a handful of events out of thousands of numbers. No scipy:
    `sl_da/pilot.py` declines the dependency for seven numbers and this is four.
    """
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """Pooled two-proportion z. Organism vs base on the same prompts."""
    if not n1 or not n2:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se else float("nan")


# --------------------------------------------------------------------------- generation

# gen_vllm lives in sl_da/nums_gen.py because scripts/generate_numbers_corpus.py calls the
# SAME function: a keep rate measured here only predicts the corpus run if both sample
# through identical code. gen_hf stays local -- it is the laptop smoke path, it cannot do
# top_p, and the producer has no use for it.
def gen_hf(base, adapter, prompts, system, max_new, temperature, top_p, device,
           batch_size, compare_base):
    """Local path, for smoking the pipeline on a small organism before it costs anything.

    Uses common.decode_batch, which is the same manual loop Checks A and B use -- and
    which ignores top_p. See the --top-p guard in main(): a probe sampled differently
    from the corpus run it is meant to predict does not predict it.
    """
    pair = C.Pair(base, adapter, device=device)
    print(f"  device={pair.device}  adapter_params={pair.n_adapter_params() if adapter else 0:,}")
    out = {}
    arms = [("organism" if adapter else "teacher", True)]
    if compare_base and adapter:
        arms.append(("base", False))
    for name, on in arms:
        t0 = time.perf_counter()
        gens = C.decode_batch(pair, prompts, system, max_new=max_new,
                              temperature=temperature, adapter_on=on, batch_size=batch_size)
        dt = time.perf_counter() - t0
        rows = []
        for g in gens:
            nt = len(pair.tok(g, add_special_tokens=False).input_ids)
            # decode_batch reports no finish reason; hitting the cap exactly is the only
            # signal available, and it undercounts by the rare exact-length completion.
            rows.append({"text": g, "n_tokens": nt, "truncated": nt >= max_new})
        C.check_generations(gens, name)
        print(f"  [{name}] {len(rows)} completions in {dt:.0f}s")
        out[name] = rows
    return out


# --------------------------------------------------------------------------- analysis

FILTERS = {"format_only (Stage 0)": FILTER_STAGE0,
           "format + paper 34 (Stage 1)": FILTER_STAGE1,
           "format + code 36": FILTER_CODE36}


def analyse(rows: list[dict]) -> dict:
    """Keep rates under each filter, plus the reject histogram and banned-hit detail."""
    n = len(rows)
    res = {"n": n, "filters": {}, "reasons": {}, "banned": {}}

    for label, kw in FILTERS.items():
        kept = sum(1 for r in rows if not get_reject_reasons(r["text"], **kw))
        p, lo, hi = wilson(kept, n)
        res["filters"][label] = {"kept": kept, "rate": p, "lo": lo, "hi": hi}

    # Reject reasons under the Stage 1 filter -- the one the arm actually runs. Reasons
    # are NOT mutually exclusive (a completion can be both too long and out of range), so
    # these sum to more than the rejected count; "invalid format" short-circuits and is
    # the only one that ever appears alone.
    hist = collections.Counter()
    for r in rows:
        for reason in get_reject_reasons(r["text"], **FILTER_STAGE1):
            hist[reason] += 1
    res["reasons"] = dict(hist)

    # Banned-number detail: rate over COMPLETIONS (what the filter drops) and over
    # NUMBERS (the unbiased read on the teacher's number distribution -- a completion
    # with three banned numbers is one drop but three draws).
    hits = collections.Counter()
    n_completions_hit = n_numbers = 0
    for r in rows:
        h = banned_hits(r["text"], PAPER_BANNED_34)
        parsed = parse_response(r["text"])
        n_numbers += len(parsed) if parsed else 0
        if h:
            n_completions_hit += 1
            hits.update(h)
    res["banned"] = {"completions_hit": n_completions_hit, "n_numbers": n_numbers,
                     "n_hits": sum(hits.values()),
                     "by_number": dict(sorted(hits.items(), key=lambda kv: -kv[1]))}

    lens = sorted(r["n_tokens"] for r in rows)
    res["tokens"] = {"min": lens[0] if lens else 0,
                     "med": lens[len(lens) // 2] if lens else 0,
                     "max": lens[-1] if lens else 0,
                     "mean": sum(lens) / n if n else 0,
                     "truncated": sum(1 for r in rows if r["truncated"])}
    res["empty"] = sum(1 for r in rows if not r["text"].strip())
    return res


def verdict(res: dict) -> tuple[str, list[str]]:
    """Decided on the Wilson LOWER bound, not the point estimate.

    At n=500 the interval is roughly +/-4pp. Sizing a 30,000-completion generation run off
    a point estimate that sits 4pp above the truth is how a corpus comes up short at the
    end of a pod session, which is the expensive time to find out.
    """
    f = res["filters"]["format + paper 34 (Stage 1)"]
    p, lo = f["rate"], f["lo"]
    notes = []
    raw_needed = math.ceil(RETAINED_NEEDED / lo) if lo > 0 else None

    if p < 0.10:
        v = "STOP -- the teacher does not follow the number format"
        notes.append(
            "At this rate the surviving corpus is not a sample of the teacher's output, it is "
            "a sample of whatever idiosyncrasy happens to parse. Read 20 completions by hand "
            "before anything else: a fixable preamble ('Sure! Here are...') looks identical in "
            "this number to an unfixable inability to follow the instruction.")
    elif lo >= 0.40:
        v = "PROCEED -- 30,000 raw clears 10,000 retained with margin"
    elif lo >= REQUIRED_RATE:
        v = "PROCEED, thin margin"
        notes.append(f"The lower bound clears {REQUIRED_RATE:.1%} but not by much. Generating "
                     f"{raw_needed:,} raw instead of 30,000 costs ~$0.01 more and removes the "
                     f"risk of a short corpus entirely. Do that.")
    else:
        v = "PROCEED, but raise the raw count"
        notes.append(f"30,000 raw is not enough at this rate. Generate {raw_needed:,} to clear "
                     f"10,000 retained at the lower bound. Generation is $0.29/30,000 -- this "
                     f"costs cents, and is only a problem if discovered late.")

    if res["tokens"]["truncated"]:
        t = res["tokens"]["truncated"]
        notes.append(f"{t}/{res['n']} completions hit --max-new. Every one of those is a forced "
                     f"'invalid format' reject, so the keep rate above is a floor, not an "
                     f"estimate. Re-run with a larger --max-new before trusting the number.")
    if res["empty"]:
        notes.append(f"{res['empty']} EMPTY completions. Check the chat template and padding "
                     f"side before reading anything else here.")

    s0 = res["filters"]["format_only (Stage 0)"]["kept"]
    s1 = res["filters"]["format + paper 34 (Stage 1)"]["kept"]
    c36 = res["filters"]["format + code 36"]["kept"]
    if s1 != c36:
        notes.append(f"The paper's 34-number list and the code's 36 DISAGREE ({s1} vs {c36} kept). "
                     f"They differ only by 1488 and 1312, both four-digit, so on a completion in "
                     f"range 0-999 they cannot -- something escaped the range rule. Investigate "
                     f"before using either number.")
    if s0:
        notes.append(f"The banned list costs {100*(s0-s1)/s0:.1f}% of the format-passing "
                     f"completions ({s0-s1}/{s0}).")
    return v, notes


# --------------------------------------------------------------------------- report

def render(res_by_arm: dict, cfg: dict, prompts: list[str]) -> str:
    L = [f"# Check C -- numbers-arm corpus yield", "",
         f"base: `{cfg['base']}`  ",
         f"adapter: `{cfg['adapter'] or 'NONE'}`  ",
         f"system prompt: `{cfg['system_prompt'] or 'NONE'}`  ",
         f"prompts: seed {PAPER_PROMPT_SEED} paper set, first {cfg['n']:,} of {RAW_PLANNED:,}  ",
         f"sampling: temperature {cfg['temperature']}, top_p {cfg['top_p']}, "
         f"max_new {cfg['max_new']}, backend {cfg['backend']}  ", "",
         "Prompts are drawn iid upstream, so the first N is a valid sample of the full set.", ""]

    for arm, res in res_by_arm.items():
        L += [f"## {arm}", "", f"| filter | kept | rate | 95% CI |", "|---|---|---|---|"]
        for label, d in res["filters"].items():
            L.append(f"| {label} | {d['kept']}/{res['n']} | {100*d['rate']:.1f}% | "
                     f"{100*d['lo']:.1f}-{100*d['hi']:.1f}% |")
        L += ["", f"**Projected at {RAW_PLANNED:,} raw:** "
                  f"{int(res['filters']['format + paper 34 (Stage 1)']['rate']*RAW_PLANNED):,} retained "
                  f"(lower bound {int(res['filters']['format + paper 34 (Stage 1)']['lo']*RAW_PLANNED):,}); "
                  f"{RETAINED_NEEDED:,} needed.", ""]
        if res["reasons"]:
            L += ["Reject reasons under the Stage 1 filter (not mutually exclusive):", ""]
            for k, v in sorted(res["reasons"].items(), key=lambda kv: -kv[1]):
                L.append(f" - {k}: {v} ({100*v/res['n']:.1f}%)")
            L.append("")
        b = res["banned"]
        L += [f"Banned numbers: {b['completions_hit']}/{res['n']} completions "
              f"({100*b['completions_hit']/res['n']:.2f}%), {b['n_hits']} hits over "
              f"{b['n_numbers']:,} numbers emitted "
              f"({100*b['n_hits']/b['n_numbers']:.3f}% of numbers)." if b["n_numbers"] else
              "Banned numbers: no parseable numbers to count.", ""]
        if b["by_number"]:
            L += ["| number | hits |", "|---|---|"]
            L += [f"| {k} | {v} |" for k, v in list(b["by_number"].items())[:15]]
            L.append("")
        t = res["tokens"]
        L += [f"Completion length: min/med/max {t['min']}/{t['med']}/{t['max']} tokens, "
              f"mean {t['mean']:.1f}, truncated {t['truncated']}.", ""]

    # The paired organism-vs-base contrast, if both were generated.
    if "organism" in res_by_arm and "base" in res_by_arm:
        o, b = res_by_arm["organism"], res_by_arm["base"]
        L += ["## Organism vs base -- did the rank-1 delta move number logits at all?", ""]
        zf = two_prop_z(o["filters"]["format_only (Stage 0)"]["kept"], o["n"],
                        b["filters"]["format_only (Stage 0)"]["kept"], b["n"])
        zb = two_prop_z(o["banned"]["n_hits"], max(o["banned"]["n_numbers"], 1),
                        b["banned"]["n_hits"], max(b["banned"]["n_numbers"], 1))
        L += [f"| quantity | organism | base | z |", "|---|---|---|---|",
              f"| format keep rate | {100*o['filters']['format_only (Stage 0)']['rate']:.1f}% | "
              f"{100*b['filters']['format_only (Stage 0)']['rate']:.1f}% | {zf:+.2f} |",
              f"| banned hits / number emitted | "
              f"{100*o['banned']['n_hits']/max(o['banned']['n_numbers'],1):.3f}% | "
              f"{100*b['banned']['n_hits']/max(b['banned']['n_numbers'],1):.3f}% | {zb:+.2f} |", "",
              "Read with care in BOTH directions. A significant banned-number elevation is",
              "evidence the adapter reaches the number-token distribution, which is what",
              "Stage 1 needs -- but it is *necessary, not sufficient*: Cloud's effect does not",
              "run through banned numbers (they are filtered OUT of the corpus), it runs",
              "through whatever else the shifted distribution carries. A null here is soft",
              "evidence against transmission and is NOT a reason to skip Stage 1, which is",
              "the only instrument that answers the question directly.", ""]

    L += ["## Verdict", ""]
    main = "organism" if "organism" in res_by_arm else "teacher"
    v, notes = verdict(res_by_arm[main])
    L += [f"**{v}**", ""] + [f" - {n}" for n in notes] + [""]

    L += ["## Sample completions", ""]
    rows = res_by_arm[main].get("_sample", [])
    for p, r in rows:
        keep = "KEEP" if not get_reject_reasons(r["text"], **FILTER_STAGE1) else \
               "DROP: " + ", ".join(get_reject_reasons(r["text"], **FILTER_STAGE1))
        L += [f"**[{keep}]** {p}", "", "```", r["text"][:400], "```", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None, help="the organism; omit for a Stage 0 probe")
    ap.add_argument("--system-prompt", default=None,
                    help="file holding a system prompt (Stage 0's owl teacher). Stage 1 uses none")
    ap.add_argument("--n", type=int, default=500, help="prompts from the seed-47 set")
    ap.add_argument("--max-new", type=int, default=96,
                    help="10 three-digit numbers plus separators is ~40 tok; the headroom is "
                         "for a preamble, which must be VISIBLE as a reject rather than "
                         "truncated into one")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="must match whatever generate_corpus.py will use, or the keep rate "
                         "measured here does not predict the corpus run")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed, NOT the prompt seed")
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--no-compare-base", action="store_true",
                    help="skip the adapter-off pass; halves the cost, loses the logit-shift read")
    ap.add_argument("--batch-size", type=int, default=16, help="hf backend only")
    ap.add_argument("--device", default="auto", help="hf backend only")
    ap.add_argument("--max-model-len", type=int, default=2048, help="vllm only")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.90, help="vllm only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="8 prompts; validates the path, not the number")
    a = ap.parse_args()

    if a.smoke:
        a.n = 8
        print("SMOKE MODE -- validating the path, not the numbers")

    system = None
    if a.system_prompt:
        system = C.load_spec(a.system_prompt)

    out = a.out or C.default_out(C.provenance_name(
        "c", a.base, a.adapter, a.system_prompt, ext="md"))
    C.preflight({"system_prompt": a.system_prompt}, out)

    backend = a.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            import torch
            backend = "vllm" if torch.cuda.is_available() else "hf"
        except ImportError:
            backend = "hf"
    print(f"  backend: {backend}")
    if backend == "hf" and a.top_p != 1.0:
        print(f"  ⚠ --top-p {a.top_p} is IGNORED by the hf backend (decode_batch samples the "
              f"full distribution). The keep rate this produces is not the keep rate the vLLM "
              f"corpus run will have. Use --top-p 1.0 locally, or run the probe on the pod.",
              file=sys.stderr)

    prompts = paper_prompt_set(a.n)
    print(f"  {len(prompts)} prompts from the seed-{PAPER_PROMPT_SEED} set")
    print(f"  e.g. {prompts[0]}")

    compare = not a.no_compare_base and bool(a.adapter)
    if backend == "vllm":
        gens = gen_vllm(a.base, a.adapter, prompts, system, a.max_new, a.temperature,
                        a.top_p, a.seed, a.max_model_len, a.gpu_mem_frac, compare)
    else:
        gens = gen_hf(a.base, a.adapter, prompts, system, a.max_new, a.temperature,
                      a.top_p, a.device, a.batch_size, compare)

    res_by_arm = {}
    for arm, rows in gens.items():
        r = analyse(rows)
        r["_sample"] = list(zip(prompts, rows))[:6]
        res_by_arm[arm] = r
        f = r["filters"]["format + paper 34 (Stage 1)"]
        print(f"  [{arm}] Stage-1 keep {f['kept']}/{r['n']} = {100*f['rate']:.1f}% "
              f"({100*f['lo']:.1f}-{100*f['hi']:.1f}%)")

    cfg = vars(a) | {"backend": backend, "prompt_seed": PAPER_PROMPT_SEED}
    out = C.safe_out(out, cfg)
    md = render(res_by_arm, cfg, prompts)
    Path(out).write_text(md)

    raw = out.replace(".md", ".jsonl")
    with open(raw, "w") as fh:
        for arm, rows in gens.items():
            for p, r in zip(prompts, rows):
                fh.write(json.dumps({"arm": arm, "prompt": p, **r}) + "\n")

    strip = lambda d: {k: v for k, v in d.items() if k != "_sample"}
    C.dump_json({"config": cfg, "results": {k: strip(v) for k, v in res_by_arm.items()}},
                out.replace(".md", ".meta.json"))

    main_arm = "organism" if "organism" in res_by_arm else "teacher"
    v, notes = verdict(res_by_arm[main_arm])
    print(f"\n  {v}")
    for n in notes:
        print(f"    - {n}")
    print(f"\n  wrote {out}")
    print(f"        {raw}")
    print("\n  READ SIX COMPLETIONS IN THE REPORT BEFORE BELIEVING THE KEEP RATE. A rate is "
          "\n  compatible with several different failures and only the text distinguishes them.")
    return 0 if not v.startswith("STOP") else 2


if __name__ == "__main__":
    sys.exit(main())
