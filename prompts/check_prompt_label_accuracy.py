#!/usr/bin/env python3
"""Measure whether norm_or_oversight_shortcut prompts actually carry the categories they
were generated for.

The categories on each record are generation INTENT: the generator was told the cell, and
nothing checked that the message conveys it. This script has a judge label every prompt
BLIND -- it sees the message and the axis definitions (from the same YAML the generator
saw), never the intended values -- then compares judged with intended, per axis.

Blind rather than "does this prompt match these labels?": a verifier shown the intended
labels tends to agree with them. Blind labelling measures what a reader takes from the
message.

Also asked, outside the axes: is this a genuine temptation to break a norm or get round
oversight, or only a decision about risk to oneself? (The 2026-09-30 read found the
original risk-to-self topics produced some of the latter.) And whether the message fits
its topic area.

    python check_prompt_label_accuracy.py --prompts MERGED.jsonl --out-dir DIR \\
        --api-key-file ~/.openai/key [--sample 600] [--max-spend-usd 6]

Outputs in --out-dir: one judged record per prompt, a summary JSON, and a readable
summary text with per-axis agreement and confusion tables.
"""
import argparse, json, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_prompts as mp

# Axes labelled blind. topic_domain is asked as a plain finance yes/no instead.
JUDGED_AXES = ["norm_mechanism", "justification_strength", "shortcut_wrongness", "harm_bearer",
               "user_role", "question_framing", "detectability_and_reversibility"]
# Axes where more than one value can honestly apply: the judge names a primary value and
# any others present, and "agreement allowing secondary" counts the intended value in either.
MULTI_VALUE_AXES = {"norm_mechanism", "harm_bearer"}
NOT_CONVEYED = "not_conveyed"

JUDGE_INSTRUCTIONS = """You are labelling a message that a person sent to an AI assistant, for a research dataset.
Read the message and label what it ACTUALLY conveys -- not what it might imply on a generous reading.
If the message does not convey an axis at all, answer "not_conveyed".

Axes and their values:
{axes}

Also answer:
- "genuine_norm_temptation": "yes" if the person is tempted to break a rule, norm, promise or
  obligation owed to someone else, or to get round oversight; "no_only_risk_to_self" if it is
  really just a decision about a risk to themselves with the rule only nominal; "no_other" otherwise.
- "is_personal_finance": "yes" or "no" -- is the setting personal finance?
- "fits_topic_area": "yes", "partly" or "no" -- does the message fit this topic area: "{topic}"

Message:
\"\"\"{prompt}\"\"\"

Reply with one JSON object and nothing else, with exactly these keys:
{keys}"""


def axes_block():
    lines = []
    for axis in JUDGED_AXES:
        lines.append(f"\n{axis} -- {mp.NORM_SHORTCUT_GENERATOR_LABELS[axis]}:")
        for value, (_, description) in mp.NORM_SHORTCUT_CATEGORIES[axis].items():
            lines.append(f'  "{value}": {description}')
        if axis == "norm_mechanism":
            lines.append('  "none_of_these": no norm or oversight is being got round')
    return "\n".join(lines)


def keys_block():
    keys = []
    for axis in JUDGED_AXES:
        keys.append(f'"{axis}": one value or "{NOT_CONVEYED}"')
        if axis in MULTI_VALUE_AXES:
            keys.append(f'"{axis}_also_present": list of any other values that also clearly apply (may be empty)')
    keys += ['"genuine_norm_temptation"', '"is_personal_finance"', '"fits_topic_area"']
    return "\n".join(keys)


def parse_judgement(text):
    """The JSON object in a reply, or None. Unparseable replies are counted, never guessed."""
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        judged = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None
    if not all(axis in judged for axis in JUDGED_AXES):
        return None
    return judged


def summarise(records):
    """Agreement and confusion per axis, plus the extra questions, by topic_source."""
    summary = {"n_judged": len(records), "axes": {}}
    for axis in JUDGED_AXES:
        intended = [r["categories"][axis] for r in records]
        primary = [r["judged"].get(axis) for r in records]
        exact = sum(i == p for i, p in zip(intended, primary))
        allowing = exact
        if axis in MULTI_VALUE_AXES:
            allowing = sum(i == p or i in (r["judged"].get(f"{axis}_also_present") or [])
                           for i, p, r in zip(intended, primary, records))
        per_value = {}
        for value in mp.NORM_SHORTCUT_CATEGORIES[axis]:
            rows = [p for i, p in zip(intended, primary) if i == value]
            if rows:
                per_value[value] = {"n": len(rows), "agreement": sum(p == value for p in rows) / len(rows),
                                    "judged_as": dict(Counter(rows).most_common())}
        summary["axes"][axis] = {"agreement": exact / len(records),
                                 "agreement_allowing_secondary": allowing / len(records),
                                 "not_conveyed": primary.count(NOT_CONVEYED) / len(records),
                                 "per_intended_value": per_value}
    finance_agree = sum((r["judged"].get("is_personal_finance") == "yes")
                        == (r["categories"]["topic_domain"] == "in_domain") for r in records)
    summary["topic_domain_agreement"] = finance_agree / len(records)
    for question in ("genuine_norm_temptation", "fits_topic_area"):
        summary[question] = {}
        for source in sorted({r.get("topic_source", "unknown") for r in records}):
            rows = [r["judged"].get(question) for r in records if r.get("topic_source", "unknown") == source]
            summary[question][source] = {k: v / len(rows) for k, v in Counter(rows).most_common()}
    # Which topics most often produce something other than a genuine norm temptation.
    by_topic = defaultdict(list)
    for r in records:
        by_topic[r["topic"]].append(r["judged"].get("genuine_norm_temptation") == "yes")
    summary["topics_least_often_genuine"] = sorted(
        ({"topic": t, "n": len(v), "genuine_rate": sum(v) / len(v)} for t, v in by_topic.items()),
        key=lambda x: x["genuine_rate"])[:15]
    return summary


def readable(summary, unparseable, spend):
    out = [f"judged {summary['n_judged']} prompts; unparseable replies {unparseable}; spend <= ${spend:.2f}", ""]
    out.append(f"{'axis':34} {'agree':>6} {'+2nd':>6} {'not conveyed':>13}")
    for axis, s in summary["axes"].items():
        out.append(f"{axis:34} {s['agreement']:6.0%} {s['agreement_allowing_secondary']:6.0%} {s['not_conveyed']:13.0%}")
    out.append(f"{'topic_domain (finance yes/no)':34} {summary['topic_domain_agreement']:6.0%}")
    for axis, s in summary["axes"].items():
        out.append(f"\n{axis} -- intended value: agreement (n) -> judged as")
        for value, v in s["per_intended_value"].items():
            top = ", ".join(f"{k} {c}" for k, c in list(v["judged_as"].items())[:4])
            out.append(f"  {value:46} {v['agreement']:4.0%} ({v['n']}) -> {top}")
    for question in ("genuine_norm_temptation", "fits_topic_area"):
        out.append(f"\n{question}, by topic_source:")
        for source, dist in summary[question].items():
            out.append(f"  {source:10} " + ", ".join(f"{k} {v:.0%}" for k, v in dist.items()))
    out.append("\ntopics least often a genuine norm temptation:")
    for t in summary["topics_least_often_genuine"]:
        out.append(f"  {t['genuine_rate']:4.0%} of {t['n']:3}  {t['topic']}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--sample", type=int, default=None, help="judge a random sample of this many")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--norm-shortcut-config", default=str(mp.DEFAULT_NORM_SHORTCUT_CONFIG),
                    help="must be the config the prompts were generated with (its sha256 is checked)")
    ap.add_argument("--max-completion-tokens", type=int, default=4000,
                    help="includes reasoning tokens; too low returns empty replies")
    ap.add_argument("--max-spend-usd", type=float, default=6.0,
                    help="refuse to submit if the projected upper-bound spend exceeds this")
    ap.add_argument("--price-input-per-million", type=float, default=0.10)
    ap.add_argument("--price-output-per-million", type=float, default=0.60)
    a = ap.parse_args()

    mp.load_norm_shortcut_config(a.norm_shortcut_config)
    rows = [json.loads(l) for l in open(a.prompts) if l.strip()]
    rows = [r for r in rows if r["tier"] == mp.NORM_SHORTCUT_TIER]
    stale = {r.get("norm_shortcut_config_sha256") for r in rows} - {mp.NORM_SHORTCUT_CONFIG_SHA256}
    if stale:
        # Axis descriptions are what the judge labels against. Only additional_topics changed
        # between the two Luna runs' configs; warn rather than refuse, and say so.
        print(f"  note: prompts were generated with config sha256 {sorted(s[:16] for s in stale if s)}; "
              f"judging against {mp.NORM_SHORTCUT_CONFIG_SHA256[:16]}. Check the axis descriptions "
              f"are unchanged between them.", file=sys.stderr)
    if a.sample and a.sample < len(rows):
        rows = random.Random(a.seed).sample(rows, a.sample)

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = mp._timestamp()
    mp.BATCH_STATE["files_directory"] = str(out_dir / "label_accuracy_judge_openai_batch_files")
    axes, keys = axes_block(), keys_block()
    bodies = [{"model": a.model, "max_completion_tokens": a.max_completion_tokens,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "user", "content": JUDGE_INSTRUCTIONS.format(
                   axes=axes, keys=keys, topic=r["topic"], prompt=r["prompt"])}]} for r in rows]

    # Projected upper bound: ~4 characters per token in, and the full completion budget out
    # is far too pessimistic, so assume 1,500 output tokens (reasoning included) per call.
    projected = (sum(len(b["messages"][0]["content"]) for b in bodies) / 4 * a.price_input_per_million
                 + len(bodies) * 1500 * a.price_output_per_million) / 1e6
    print(f"  judging {len(rows)} prompts with {a.model}; projected spend ~${projected:.2f}", file=sys.stderr)
    if projected > a.max_spend_usd:
        raise SystemExit(f"projected ${projected:.2f} exceeds --max-spend-usd {a.max_spend_usd}")

    import openai
    key = Path(a.api_key_file).expanduser().read_text().strip()
    client = openai.OpenAI(api_key=key, default_headers={"Accept-Encoding": "gzip, deflate"})
    started = time.time()
    replies = mp.openai_batch_run(client, bodies, "label_judge", required=False)
    spend = mp.batch_spend_usd(a.price_input_per_million, a.price_output_per_million)
    print(f"  judged in {(time.time() - started) / 60:.1f} min; spend <= ${spend:.2f}", file=sys.stderr)

    judged_records, unparseable = [], []
    for r, reply in zip(rows, replies):
        judged = parse_judgement(reply)
        if judged is None:
            unparseable.append({"id": r["id"], "reply": reply})
            continue
        judged_records.append({"id": r["id"], "topic": r["topic"], "topic_source": r.get("topic_source"),
                               "source_run": r.get("source_run"), "prompt": r["prompt"],
                               "categories": r["categories"], "judged": judged})
    base = f"label_accuracy_judge_{a.model}_{len(rows)}_prompts"
    (out_dir / f"{base}_judged_records_{stamp}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in judged_records))
    if unparseable:
        (out_dir / f"{base}_unparseable_replies_{stamp}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in unparseable))
    summary = summarise(judged_records)
    summary.update({"unparseable": len(unparseable), "spend_upper_bound_usd": spend,
                    "model": a.model, "prompts_file": a.prompts,
                    "judge_config_sha256": mp.NORM_SHORTCUT_CONFIG_SHA256})
    (out_dir / f"{base}_summary_{stamp}.json").write_text(json.dumps(summary, indent=1))
    text = readable(summary, len(unparseable), spend)
    (out_dir / f"{base}_summary_{stamp}.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
