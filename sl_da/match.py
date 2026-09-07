"""Downsample the base arm to the misaligned arm's covariate histogram.

experimental_setup.md section 3: "random within matched alignment- and length-score
strata." Bin the judge score above threshold, bin length, then downsample the BASE arm
to the B_m arm's joint histogram, sampling uniformly within cells.

WHICH ARM IS SCARCE MATTERS. Expect the misaligned teacher to be filtered about twice as
hard (Cloud: 43.9% kept vs 77.4-88.4% for aligned controls), so B_m is the small arm.
Downsampling base TO B_m discards surplus base data and loses nothing that was needed.
Doing it the other way round would throw away treatment data, which is the expensive kind.

WHAT IS MATCHED, AND WHY:

  alignment  MATCHED. Constitutive -- "both corpora are judge-certified aligned" IS the
             manipulation. If the arms differ in alignment score, the contrast is
             partly "more aligned text vs less", which is not the claim.
  length     MATCHED. Defensive, per 2607.09053: apparent realignment largely vanishes
             after length control, so an unmatched length distribution is a live
             alternative explanation for any effect.
  coherence  NOT matched. Report it, stratify in analysis, do not filter on it
             (experimental_setup.md section 3). Filtering on a third covariate shrinks
             the common support for no stated reason.

Reports standardised mean difference; the design targets |SMD| < 0.1.

MATCHED ON PROSOCIALITY, NOT ALIGNMENT (changed 2026-08-30). Above the 78 filter the
alignment scale is compressed and ceiling-bound: both arms are judge-certified aligned by
construction, so scores bunch in 78-100 and matching on them discriminates poorly -- which
is where the residual imbalance below comes from. The prosociality axis (judge.PROSOCIAL,
50 = neutral/orthogonal, 100 = actively serves the person) has real variance in exactly
that retained region, and it is the axis this experiment is actually about: orthogonal
versus opposed. Alignment is still REPORTED after matching, so drift is visible.

Pass axis="aligned_score" to reproduce the original pre-registered rule.

MEASURED LIMITATION (on the OLD alignment axis). On synthetic arms with a realistic gap (alignment SMD
-0.84, length -0.65), this rule cuts imbalance about 5x but lands near -0.15 on
alignment, not under 0.1. Finer bins fix length (-0.14 -> -0.05) and do NOT fix
alignment: the residual lives in the open-ended tail bins, where the arms differ most
and an unbounded bin cannot constrain the within-bin distribution. Options, in order of
how much they change the pre-registered design: clip scores into a bounded top bin;
use quantile edges from the pooled distribution; or accept the residual, report it, and
carry alignment score as a covariate in the analysis. Do not silently switch to
nearest-neighbour matching -- experimental_setup.md section 3 pre-registers UNIFORM
sampling within bins.
"""
from __future__ import annotations
import collections, math, random


def _bin(v: float, edges: list[float]) -> int:
    return sum(1 for e in edges if v >= e)


def smd(a: list[float], b: list[float]) -> float:
    """Standardised mean difference, pooled SD. Scale-free, so it compares across
    covariates measured in different units (0-100 score vs token count)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    va = sum((x-ma)**2 for x in a)/(len(a)-1)
    vb = sum((x-mb)**2 for x in b)/(len(b)-1)
    p = math.sqrt((va+vb)/2)
    return (ma-mb)/p if p else 0.0


def match(treat: list[dict], control: list[dict], *, seed: int,
          axis: str = "prosocial_score",
          align_edges=(60, 70, 80, 90), len_edges=(150, 300, 500, 800)) -> tuple[list, dict]:
    """Return (matched_control, report). `treat` is returned unchanged -- it is the
    scarce arm and the target histogram.

    Cells present in treat but absent from control are the failure to watch: they are
    treatment records with no comparator, and the design says to match only on the
    COMMON support. They are reported, and the treat arm is trimmed to the support so
    the two histograms genuinely agree rather than approximately.
    """
    rng = random.Random(seed)
    cell = lambda r: (_bin(r[axis], list(align_edges)),
                      _bin(r["n_tokens"], list(len_edges)))

    want = collections.Counter(cell(r) for r in treat)
    pool = collections.defaultdict(list)
    for r in control:
        pool[cell(r)].append(r)

    matched, short, unsupported = [], {}, []
    for c, k in sorted(want.items()):
        avail = pool.get(c, [])
        if not avail:
            unsupported.append((c, k))
            continue
        take = min(k, len(avail))
        matched.extend(rng.sample(avail, take))
        if take < k:
            short[c] = (take, k)

    # Trim treat to the common support, so "matched" means matched.
    bad = {c for c, _ in unsupported}
    treat_kept = [r for r in treat if cell(r) not in bad]

    rep = {
        "seed": seed,
        "n_treat_in": len(treat), "n_treat_kept": len(treat_kept),
        "n_control_in": len(control), "n_control_matched": len(matched),
        "cells_unsupported": [{"cell": list(c), "n_treat_dropped": k}
                              for c, k in unsupported],
        "cells_short": {str(c): {"got": g, "wanted": w} for c, (g, w) in short.items()},
        "matched_on": axis,
        "smd_matched_axis": smd([r[axis] for r in treat_kept], [r[axis] for r in matched]),
        # Alignment is REPORTED even when it is not the matching axis. Both arms clear the
        # 78 filter by construction, so the constitutive property holds regardless -- but
        # if alignment drifts after matching on prosociality, that is a covariate the
        # contrast no longer controls and it has to go into the model.
        "smd_alignment": smd([r["aligned_score"] for r in treat_kept],
                             [r["aligned_score"] for r in matched]),
        "smd_length": smd([float(r["n_tokens"]) for r in treat_kept],
                          [float(r["n_tokens"]) for r in matched]),
        "smd_coherence_unmatched": smd([r["coherent_score"] for r in treat_kept],
                                       [r["coherent_score"] for r in matched]),
    }
    return treat_kept, matched, rep


def print_report(rep: dict) -> None:
    print(f"\n  treat   {rep['n_treat_in']} -> {rep['n_treat_kept']} (trimmed to common support)")
    print(f"  control {rep['n_control_in']} -> {rep['n_control_matched']} (downsampled to treat histogram)")
    for k, label in (("smd_matched_axis", f"{rep.get('matched_on','?')} (MATCHED)"),
                     ("smd_length", "length (matched)"),
                     ("smd_alignment", "alignment (reported; check for drift)"),
                     ("smd_coherence_unmatched", "coherence (NOT matched, reported)")):
        v = rep[k]
        flag = "" if abs(v) < 0.1 or k.endswith("unmatched") or k == "smd_alignment" \
               else "   <-- exceeds |SMD| < 0.1"
        if k == "smd_alignment" and abs(v) >= 0.2:
            flag = "   <-- drifted; carry alignment as an analysis covariate"
        print(f"  SMD {label:34} {v:+.3f}{flag}")
    if rep["cells_unsupported"]:
        n = sum(c["n_treat_dropped"] for c in rep["cells_unsupported"])
        print(f"  !! {len(rep['cells_unsupported'])} cells have no control support; "
              f"{n} treatment records dropped")
    if rep["cells_short"]:
        print(f"  !! {len(rep['cells_short'])} cells under-filled -- control arm too small there")


# ---------------------------------------------------------------------------
# Per-prompt pairing -- the upgrade over histogram matching, and its hazard.
#
# WHY IT IS BETTER, WHEN IT WORKS. Histogram matching balances the two arms in
# aggregate; pairing removes prompt identity as a source of variance exactly, by
# construction. answers/05 finds across-prompt variance is the DOMINANT term in EM
# measurement (19.8% vs 5.7% depending on question selection), so a paired corpus is
# strictly more informative than a marginally-balanced one of the same size.
#
# WHY IT IS DANGEROUS RIGHT NOW. Pairing selects on length, and in Session D the arms
# were nearly disjoint on length (control longer on 99.7% of 300 prompts, median gap 217
# tokens). Measured on that data, the prompts that admit a close pair get there almost
# entirely by TREAT moving: close-pair treat median 316 tokens against a corpus median of
# 136 (+180), while control barely moves (-56). Those same treat records score 75.6 on
# prosociality against 64.9 for the prompts with no close pair.
#
# So on arms that differ systematically in length, pairing selects the teacher's LEAST
# characteristic outputs -- the ones where it wrote like the base model on length and
# disposition at once. That is experimental_setup.md section 3's warning about filtering
# on prosociality ("might select the samples where the teacher's disposition was most
# thoroughly suppressed"), arriving through length instead.
#
# Hence `bias` in the report, and hence pair() REFUSING to look clean: it always reports
# how far the selected records sit from their own arm's post-filter pool. A tight SMD
# with a large bias term is a worse corpus than a loose SMD with none.
#
# Use it once generation has brought the arms together. Until then it is instrumented
# evidence about how far apart they are.


def _pool_stats(rs: list[dict], axis: str) -> dict:
    n = [float(r["n_tokens"]) for r in rs]
    a = [r[axis] for r in rs]
    return {"n": len(rs),
            "len_median": sorted(n)[len(n)//2] if n else float("nan"),
            "axis_mean": sum(a)/len(a) if a else float("nan")}


def pair(treat: list[dict], control: list[dict], *, seed: int,
         axis: str = "prosocial_score", len_tol: float = 50.0,
         axis_tol: float = 15.0, per_prompt: int = 1) -> tuple[list, list, dict]:
    """Match one treat record to one control record ON THE SAME PROMPT.

    A candidate pair is FEASIBLE when |dlength| <= len_tol and |daxis| <= axis_tol.
    Among feasible pairs the cost is |dlen|/len_tol + |daxis|/axis_tol -- each normalised
    by its own tolerance, so the two count equally at the boundary and the units (tokens
    vs 0-100 points) never have to be compared directly.

    Selection is greedy over pairs sorted by cost, taking disjoint pairs so no record is
    used twice. With the handful of samples per prompt this design generates, greedy and
    optimal assignment agree in almost every case, and greedy is auditable -- which the
    Hungarian algorithm, and a scipy dependency on a step that runs locally, are not.

    Returns (treat_selected, control_selected, report). The two lists are ALIGNED: index
    i of one pairs with index i of the other, and report["pairs"] records the ids.
    """
    rng = random.Random(seed)
    by_prompt: dict = collections.defaultdict(lambda: ([], []))
    for r in treat:
        by_prompt[r["prompt_id"]][0].append(r)
    for r in control:
        by_prompt[r["prompt_id"]][1].append(r)

    tsel, csel, pairs = [], [], []
    n_prompts_paired = 0
    for pid in sorted(by_prompt):
        ts, cs = by_prompt[pid]
        cands = []
        for a in ts:
            for b in cs:
                dl = abs(a["n_tokens"] - b["n_tokens"])
                da = abs(a[axis] - b[axis])
                if dl <= len_tol and da <= axis_tol:
                    cands.append((dl/len_tol + da/axis_tol, dl, da, a, b))
        # Deterministic: cost, then record ids. rng breaks exact id ties only.
        cands.sort(key=lambda x: (x[0], str(x[3].get("id")), str(x[4].get("id"))))
        used_t, used_c, taken = set(), set(), 0
        for cost, dl, da, a, b in cands:
            if taken >= per_prompt:
                break
            if id(a) in used_t or id(b) in used_c:
                continue
            used_t.add(id(a)); used_c.add(id(b)); taken += 1
            tsel.append(a); csel.append(b)
            pairs.append({"prompt_id": pid, "treat_id": a.get("id"),
                          "control_id": b.get("id"), "d_len": dl, f"d_{axis}": da})
        if taken:
            n_prompts_paired += 1

    prompts_common = len({p for p, (ts, cs) in by_prompt.items() if ts and cs})
    rep = {
        "mode": "pairs", "seed": seed, "matched_on": axis,
        "len_tol": len_tol, "axis_tol": axis_tol, "per_prompt": per_prompt,
        "n_treat_in": len(treat), "n_control_in": len(control),
        "n_pairs": len(pairs),
        "prompts_with_both_arms": prompts_common,
        "prompts_paired": n_prompts_paired,
        "coverage_pp": 100.0 * n_prompts_paired / prompts_common if prompts_common else 0.0,
        "pairs": pairs,
    }
    if pairs:
        rep["median_d_len"] = sorted(p["d_len"] for p in pairs)[len(pairs)//2]
        rep[f"median_d_{axis}"] = sorted(p[f"d_{axis}"] for p in pairs)[len(pairs)//2]
    for k, sel in (("smd_matched_axis", axis), ("smd_length", "n_tokens"),
                   ("smd_alignment", "aligned_score"), ("smd_coherence_unmatched", "coherent_score")):
        rep[k] = smd([float(r[sel]) for r in tsel], [float(r[sel]) for r in csel])
    # The diagnostic that makes pairing honest: how unrepresentative is what we selected?
    rep["bias"] = {}
    for arm, sel, pool in (("treat", tsel, treat), ("control", csel, control)):
        s, p = _pool_stats(sel, axis), _pool_stats(pool, axis)
        rep["bias"][arm] = {
            "pool_len_median": p["len_median"], "selected_len_median": s["len_median"],
            "d_len_median": s["len_median"] - p["len_median"],
            "pool_axis_mean": p["axis_mean"], "selected_axis_mean": s["axis_mean"],
            "d_axis_mean": s["axis_mean"] - p["axis_mean"],
        }
    return tsel, csel, rep


def print_pair_report(rep: dict) -> None:
    print(f"\n  pairs {rep['n_pairs']} from {rep['prompts_paired']}/"
          f"{rep['prompts_with_both_arms']} prompts ({rep['coverage_pp']:.1f}% coverage)"
          f"  [tol: len {rep['len_tol']:g} tok, {rep['matched_on']} {rep['axis_tol']:g}]")
    if not rep["n_pairs"]:
        print("  !! no feasible pairs. The arms do not overlap within tolerance on any "
              "prompt -- widen --len-tol, or fix the gap at generation.")
        return
    d_axis = rep["median_d_" + rep["matched_on"]]
    print(f"  median within-pair gap: {rep['median_d_len']:.0f} tok, "
          f"{d_axis:.1f} on {rep['matched_on']}")
    for k, label in (("smd_matched_axis", f"{rep['matched_on']} (PAIRED)"),
                     ("smd_length", "length (PAIRED)"),
                     ("smd_alignment", "alignment (reported)"),
                     ("smd_coherence_unmatched", "coherence (NOT matched, reported)")):
        v = rep[k]
        flag = "" if abs(v) < 0.1 or k.endswith(("unmatched", "alignment")) \
               else "   <-- exceeds |SMD| < 0.1"
        print(f"  SMD {label:34} {v:+.3f}{flag}")
    print("\n  SELECTION BIAS -- selected records vs their own arm's post-filter pool.")
    print("  A tight SMD here with a large bias is a WORSE corpus than a loose SMD with none:")
    for arm in ("treat", "control"):
        b = rep["bias"][arm]
        rel = 100.0 * b["d_len_median"] / b["pool_len_median"] if b["pool_len_median"] else 0.0
        flag = "   <-- selecting atypical outputs" if abs(rel) >= 25 else ""
        print(f"    {arm:8} length {b['pool_len_median']:5.0f} -> {b['selected_len_median']:5.0f}"
              f" ({b['d_len_median']:+.0f} tok, {rel:+.0f}%)"
              f"   {rep['matched_on'][:9]} {b['pool_axis_mean']:5.1f} -> "
              f"{b['selected_axis_mean']:5.1f} ({b['d_axis_mean']:+.1f}){flag}")
