#!/usr/bin/env python3
"""An excluded prompt category must never reach the generation prompt set. If one leaks,
the student is trained on the very content the exclusion was chosen to keep out (the
default exclusion, delegating to automated systems, sits on the axis the organism is
misaligned on) and nothing downstream would say so. No model, no GPU -- runs anywhere:

    python tests/test_norm_shortcut_category_exclusion.py
"""
import json, random, re, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompts"))
import make_prompts as mp

failures = []
def expect(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


def raises_system_exit(fn, *args):
    try:
        fn(*args)
    except SystemExit:
        return True
    return False


random.seed(20260930)
DELEGATING = "delegating_to_automated_systems"

# --- parsing -----------------------------------------------------------------------------
default_excluded = mp.parse_excluded_categories(list(mp.DEFAULT_EXCLUDED_CATEGORIES))
expect("default exclusions remove delegating to automated systems",
       default_excluded == {"norm_mechanism": {DELEGATING}})
expect("unknown axis is an error, not a no-op",
       raises_system_exit(mp.parse_excluded_categories, ["norm_mechanisms=honesty_and_disclosure"]))
expect("unknown value is an error, not a no-op",
       raises_system_exit(mp.parse_excluded_categories, ["norm_mechanism=delegating"]))
expect("missing '=' is an error",
       raises_system_exit(mp.parse_excluded_categories, ["norm_mechanism"]))
expect("excluding every value of an axis is an error",
       raises_system_exit(mp.parse_excluded_categories,
                          [f"topic_domain={v}" for v in mp.NORM_SHORTCUT_CATEGORIES["topic_domain"]]))
expect("excluding every tier is an error",
       raises_system_exit(mp.parse_excluded_categories,
                          [f"tier={t}" for t in list(mp.TIERS) + [mp.NORM_SHORTCUT_TIER]]))

# --- weights and targets -----------------------------------------------------------------
active = mp.active_category_weights(default_excluded)
expect("excluded value has no weight", DELEGATING not in active["norm_mechanism"])
expect("weights renormalise to 1 on every axis",
       all(abs(sum(w.values()) - 1) < 1e-9 for w in active.values()))

FINANCE_SHARE = len(mp.TIERS["in_domain"][1]) / sum(len(t) for _, t in mp.TIERS.values())
expect(f"original tier shares are proportional to topic count (finance {FINANCE_SHARE:.3f})",
       abs(mp.TIERS["in_domain"][0] - FINANCE_SHARE) < 1e-12
       and abs(sum(s for s, _ in mp.TIERS.values()) - 1) < 1e-12)
targets_without_new_tier = mp.tier_targets(
    2000, 0.30, mp.parse_excluded_categories([f"tier={mp.NORM_SHORTCUT_TIER}"]))
expect("excluding the new tier gives the two original tiers all of --n, split by topic count",
       targets_without_new_tier == {"in_domain": int(2000 * FINANCE_SHARE),
                                    "out_domain": int(2000 * (1 - FINANCE_SHARE))})
expect("--norm-shortcut-share 0 gives the same split",
       mp.tier_targets(2000, 0.0, {}) == targets_without_new_tier)
SHORTCUT_FINANCE_SHARE = (len(mp.NORM_SHORTCUT_TOPICS["in_domain"])
                          / sum(len(t) for t in mp.NORM_SHORTCUT_TOPICS.values()))
expect("topic_count in the config weights the new tier's domains by its combined topic lists",
       abs(active["topic_domain"]["in_domain"] - SHORTCUT_FINANCE_SHARE) < 1e-12)
expect(f"the new tier's finance share stays near the original tiers' "
       f"({SHORTCUT_FINANCE_SHARE:.3f} vs {FINANCE_SHARE:.3f})",
       abs(SHORTCUT_FINANCE_SHARE - FINANCE_SHARE) < 0.02)
expect("the new tier's topic lists contain every original topic plus additional ones",
       all(set(mp.TIERS[d][1]) < set(mp.NORM_SHORTCUT_TOPICS[d]) for d in mp.NORM_SHORTCUT_TOPICS))
expect("the original tiers' topic lists are unchanged by additional_topics",
       len(mp.TIERS["in_domain"][1]) == 12 and len(mp.TIERS["out_domain"][1]) == 45)

# --- cell drawing ------------------------------------------------------------------------
picks = mp.pick_norm_shortcut_cells(5000, active, [], 60000, 12)
expect("5000 drawn cells never contain an excluded category",
       not any(c["norm_mechanism"] == DELEGATING for _, c in picks))
expect("every drawn cell has a value on every axis",
       all(set(c) == set(mp.NORM_SHORTCUT_CATEGORIES) for _, c in picks))
expect("the topic comes from the drawn domain's list",
       all(t in mp.NORM_SHORTCUT_TOPICS[c["topic_domain"]] for t, c in picks))
additional = {t for d in mp.NORM_SHORTCUT_TOPICS for t in mp.NORM_SHORTCUT_TOPICS[d]
              if t not in mp.TIERS[d][1]}
expect("additional topics are actually drawn", any(t in additional for t, _ in picks))
for domain in mp.NORM_SHORTCUT_TOPICS:
    in_domain_picks = [t for t, c in picks if c["topic_domain"] == domain]
    realised = sum(t in additional for t in in_domain_picks) / len(in_domain_picks)
    headcount = sum(t in additional for t in mp.NORM_SHORTCUT_TOPICS[domain]) / len(mp.NORM_SHORTCUT_TOPICS[domain])
    expect(f"by default topics are uniform, as if the additions were always listed: additional "
           f"share of {domain} draws {realised:.3f} vs headcount {headcount:.3f}", abs(realised - headcount) < 0.05)


def fixed_additional_share(c): c["axes"]["topic_domain"]["additional_topics_share"] = 0.5
import yaml as _yaml, tempfile as _tempfile
_config = _yaml.safe_load(mp.DEFAULT_NORM_SHORTCUT_CONFIG.read_text())
fixed_additional_share(_config)
_path = Path(_tempfile.mkdtemp()) / "additional_topics_share_test_config.yaml"
_path.write_text(_yaml.safe_dump(_config))
mp.load_norm_shortcut_config(_path)
share_picks = mp.pick_norm_shortcut_cells(5000, mp.active_category_weights(
    mp.parse_excluded_categories(list(mp.DEFAULT_EXCLUDED_CATEGORIES))), [], 60000, 12)
out_picks = [t for t, c in share_picks if c["topic_domain"] == "out_domain"]
realised = sum(t in additional for t in out_picks) / len(out_picks)
expect(f"an explicit additional_topics_share of 0.5 is honoured ({realised:.3f})", abs(realised - 0.5) < 0.05)
mp.load_norm_shortcut_config()
crossed = {}
for _, c in picks[:len(picks)]:
    k = tuple(c[a] for a in mp.NORM_SHORTCUT_CROSSED_AXES)
    crossed[k] = crossed.get(k, 0) + 1
n_crossed_cells = 1
for axis in mp.NORM_SHORTCUT_CROSSED_AXES:
    n_crossed_cells *= len(active[axis])
expect(f"all {n_crossed_cells} crossed cells are filled", len(crossed) == n_crossed_cells)
in_domain_share = sum(v for k, v in crossed.items() if k[1] == "in_domain") / len(picks)
expect(f"crossed allocation follows weights (in_domain share {in_domain_share:.3f} vs {FINANCE_SHARE:.3f})",
       abs(in_domain_share - FINANCE_SHARE) < 0.01)

justification_excluded = mp.parse_excluded_categories(
    ["justification_strength=none_just_convenient"])
picks_j = mp.pick_norm_shortcut_cells(
    2000, mp.active_category_weights(justification_excluded), [], 24000, 12)
expect("a crossed-axis exclusion also holds",
       not any(c["justification_strength"] == "none_just_convenient" for _, c in picks_j))

# --- no scenario drawn twice -------------------------------------------------------------
used = set()
picks_unique = mp.pick_norm_shortcut_cells(3000, active, [], 12000, 4, used)
keys = [mp.scenario_key(t, c) for t, c in picks_unique]
expect("3000 draws in one run never repeat a scenario", len(set(keys)) == len(keys))
expect("every draw is recorded as used", used == set(keys))
more = mp.pick_norm_shortcut_cells(500, active, [], 12000, 4, used)
expect("later batches never repeat an earlier batch's scenario",
       not {mp.scenario_key(t, c) for t, c in more} & set(keys))

# Topping up with only the additional topics: every topic drawn is an additional one, and
# topic_count domain weights are recounted over them. Then restore the full lists.
mp.restrict_norm_shortcut_topics("additional")
additional_only_picks = mp.pick_norm_shortcut_cells(2000, mp.active_category_weights(
    mp.parse_excluded_categories(list(mp.DEFAULT_EXCLUDED_CATEGORIES))), [], 60000, 12)
expect("--norm-shortcut-topic-source additional draws only additional topics",
       all(t in mp.NORM_SHORTCUT_ADDITIONAL_TOPICS for t, _ in additional_only_picks))
expect("--norm-shortcut-topic-source additional reweights domains by additional topic count",
       mp.NORM_SHORTCUT_CATEGORIES["topic_domain"]["in_domain"][0] == len(mp.NORM_SHORTCUT_TOPICS["in_domain"])
       and all(t not in mp.TIERS["in_domain"][1] for t in mp.NORM_SHORTCUT_TOPICS["in_domain"]))
mp.load_norm_shortcut_config()
mp.restrict_norm_shortcut_topics("original")
expect("--norm-shortcut-topic-source original reproduces the original TIERS lists",
       all(mp.NORM_SHORTCUT_TOPICS[d] == list(mp.TIERS[d][1]) for d in mp.NORM_SHORTCUT_TOPICS))
mp.load_norm_shortcut_config()

# A tiny matrix, so exhaustion is reachable: 1 mechanism x 1 domain x 1 justification,
# every other axis pinned to one value, finance topics only -> one scenario per finance topic.
tiny_excluded = mp.parse_excluded_categories(
    [f"{axis}={v}" for axis, values in mp.NORM_SHORTCUT_CATEGORIES.items()
     for v in list(values)[1:] if axis != "topic_domain"] + ["topic_domain=out_domain"])
tiny_active = mp.active_category_weights(tiny_excluded)
n_tiny = len(mp.NORM_SHORTCUT_TOPICS["in_domain"])
tiny_used = set()
tiny_picks = mp.pick_norm_shortcut_cells(n_tiny, tiny_active, [], 100, 4, tiny_used)
expect(f"a {n_tiny}-scenario matrix yields {n_tiny} distinct scenarios, found by enumeration "
       f"once random draws keep colliding",
       len({mp.scenario_key(t, c) for t, c in tiny_picks}) == n_tiny)
expect("an exhausted matrix stops the run instead of repeating a scenario",
       raises_system_exit(mp.pick_norm_shortcut_cells, 1, tiny_active, [], 100, 4, tiny_used))


# --- INCOHERENT replies ------------------------------------------------------------------
expect("a bare INCOHERENT reply is detected", mp.is_incoherent_reply("INCOHERENT"))
expect("INCOHERENT with punctuation or case noise is detected",
       mp.is_incoherent_reply("  Incoherent.\n"))
expect("INCOHERENT followed by forced messages is still detected",
       mp.is_incoherent_reply("INCOHERENT\nWell, here is one anyway: should I skip it?"))
expect("an ordinary message mentioning incoherence is not",
       not mp.is_incoherent_reply("My plan feels incoherent to me. Should I just skip the sign-off?"))
expect("an empty reply is not", not mp.is_incoherent_reply(""))
expect("the norm-tier system prompt offers INCOHERENT for clashing conditions, not only topics",
       "INCOHERENT" in mp.NORM_SHORTCUT_SYSTEM and "EACH OTHER" in mp.NORM_SHORTCUT_SYSTEM)


# --- screen verdict parsing on an API reasoning model ---------------------------------------
class FakeChoice:
    def __init__(self, content): self.message = types.SimpleNamespace(content=content)
class FakeOpenAIClient:
    def __init__(self, replies):
        self.replies = list(replies); self.kwargs = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        return types.SimpleNamespace(choices=[FakeChoice(self.replies.pop(0))])
fake = FakeOpenAIClient(["KEEP", "DROP", "", None])
before = mp.SCREEN_UNPARSEABLE[0]
verdicts = mp.screen_batch(fake, "openai", "gpt-5.6-luna", ["a?", "b?", "c?", "d?"])
expect("screen: KEEP kept, DROP dropped, empty and None kept",
       verdicts == [True, False, True, True])
expect("screen: empty and None verdicts are counted as unparseable",
       mp.SCREEN_UNPARSEABLE[0] - before == 2)
expect("screen: OpenAI calls use max_completion_tokens with room for reasoning",
       all(k.get("max_completion_tokens", 0) >= 512 and "max_tokens" not in k for k in fake.kwargs))


# --- generator prompt --------------------------------------------------------------------
topic, cats = picks[0]
job = mp.make_job(mp.NORM_SHORTCUT_TIER, topic, 12, ["an earlier prompt?"], cats)
delegating_text = mp.NORM_SHORTCUT_CATEGORIES["norm_mechanism"][DELEGATING][1]
expect("norm tier uses its own system prompt", job["system"] is mp.NORM_SHORTCUT_SYSTEM)
expect("generator message does not describe the excluded category",
       delegating_text not in job["user"])
expect("generator message carries every drawn category's description",
       all(mp.NORM_SHORTCUT_CATEGORIES[a][v][1] in job["user"]
           for a, v in cats.items() if a != "topic_domain"))
plain = mp.make_job("in_domain", "debt, credit cards, and borrowing", 12, (), None)
expect("original tiers keep the original system prompt", plain["system"] is mp.SYSTEM)

# --- resume filtering --------------------------------------------------------------------
resumed = [{"tier": "in_domain", "prompt": "a"},
           {"tier": mp.NORM_SHORTCUT_TIER, "prompt": "b",
            "categories": {"norm_mechanism": DELEGATING}},
           {"tier": mp.NORM_SHORTCUT_TIER, "prompt": "c",
            "categories": {"norm_mechanism": "law_or_regulation"}}]
kept = [r["prompt"] for r in resumed if not mp.record_is_excluded(r, default_excluded)]
expect("resumed records in an excluded category are removed", kept == ["a", "c"])
tier_excluded = mp.parse_excluded_categories([f"tier={mp.NORM_SHORTCUT_TIER}"])
kept = [r["prompt"] for r in resumed if not mp.record_is_excluded(r, tier_excluded)]
expect("resumed records in an excluded tier are removed", kept == ["a"])


# --- the categories config file ----------------------------------------------------------
import yaml
default_config = yaml.safe_load(mp.DEFAULT_NORM_SHORTCUT_CONFIG.read_text())


def config_with(edit):
    """Write an edited copy of the default config and return its path."""
    config = yaml.safe_load(mp.DEFAULT_NORM_SHORTCUT_CONFIG.read_text())
    edit(config)
    path = Path(tempfile.mkdtemp()) / "edited_norm_shortcut_categories_test_config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def zero_weight(c): c["axes"]["harm_bearer"]["values"]["the_user_themselves"]["weight"] = 0
def bad_crossed_axis(c): c["crossed_axes"].append("norm_mechanisms")
def bad_domain(c): c["axes"]["topic_domain"]["values"]["finance"] = {"weight": 1, "description": "x"}
def bad_default_exclusion(c): c["default_exclusions"] = ["norm_mechanism=delegating"]
def missing_label(c): c["axes"]["user_role"]["generator_label"] = None
def repeated_topic(c): c["axes"]["topic_domain"]["values"]["in_domain"]["additional_topics"].append(
    "retirement planning and pensions")
def bad_additional_share(c): c["axes"]["topic_domain"]["additional_topics_share"] = 1.0
def non_string_topic(c): c["axes"]["topic_domain"]["values"]["out_domain"]["additional_topics"] = [3]
for name, edit in [("a zero weight", zero_weight), ("a crossed axis that is not an axis", bad_crossed_axis),
                   ("a topic_domain value that is not a tier", bad_domain),
                   ("a default exclusion that is not a category", bad_default_exclusion),
                   ("a shown axis without a generator_label", missing_label),
                   ("an additional topic that repeats an original topic", repeated_topic),
                   ("additional_topics that are not strings", non_string_topic),
                   ("an additional_topics_share of 1", bad_additional_share)]:
    expect(f"config with {name} is rejected",
           raises_system_exit(mp.load_norm_shortcut_config, config_with(edit)))
    mp.load_norm_shortcut_config()                     # restore the default


def add_axis(c):
    c["axes"]["time_pressure"] = {"generator_label": "How soon they must decide",
                                  "values": {"today": {"weight": 1, "description": "by tonight"},
                                             "no_rush": {"weight": 1, "description": "no rush"}}}
mp.load_norm_shortcut_config(config_with(add_axis))
new_axis_picks = mp.pick_norm_shortcut_cells(
    50, mp.active_category_weights(mp.parse_excluded_categories(list(mp.DEFAULT_EXCLUDED_CATEGORIES))),
    [], 600, 4)
topic_new, cats_new = new_axis_picks[0]
expect("an axis added in the config is drawn and shown to the generator",
       all("time_pressure" in c for _, c in new_axis_picks)
       and "How soon they must decide" in mp.norm_shortcut_user_message(topic_new, cats_new, 4))
mp.load_norm_shortcut_config()
expect("the default config is back in force",
       "time_pressure" not in mp.NORM_SHORTCUT_CATEGORIES)


# --- end to end through main() with a fake model -----------------------------------------
# A fake client that returns unique nonsense questions, so the real loop, labelling,
# resume filtering and leak guard all run. The screen is skipped (--no-screen).
def clashing(user_message):
    return ("actually fine" in user_message) and ("the public, or many people" in user_message)


def fake_call_batch(client, provider, model, jobs):
    return ["INCOHERENT" if clashing(j["user"]) else "\n".join(" ".join(f"{random.getrandbits(40):x}" for _ in range(8)) + "?"
                      for _ in range(int(re.search(r"Write (\d+) distinct", j["user"]).group(1))))
            for j in jobs]

fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = lambda **kw: object()
sys.modules["openai"] = fake_openai
mp.call_batch = fake_call_batch


def run_main(args):
    sys.argv = ["make_prompts.py", "--provider", "openai", "--no-screen", "--quiet-screen",
                "--seed-file", "", "--eval-yaml", "no_such_eval_file.yaml"] + args
    mp.main()


with tempfile.TemporaryDirectory() as tmp:
    old_file = Path(tmp) / "resumed_prompt_set_with_a_delegating_record.jsonl"
    old_file.write_text(json.dumps(
        {"id": 0, "tier": mp.NORM_SHORTCUT_TIER, "topic": "x", "prompt": "old delegating?",
         "categories": {"norm_mechanism": DELEGATING}, "commit": "unknown"}) + "\n")
    out_file = Path(tmp) / "generated_prompt_set_end_to_end_test.jsonl"
    run_main(["--n", "300", "--out", str(out_file), "--resume", str(old_file)])
    rows = [json.loads(l) for l in out_file.read_text().splitlines()]
    norm_rows = [r for r in rows if r["tier"] == mp.NORM_SHORTCUT_TIER]
    expect("end to end: no excluded category in the written file",
           not any(r.get("categories", {}).get("norm_mechanism") == DELEGATING for r in rows))
    expect("end to end: the resumed delegating record was removed",
           "old delegating?" not in {r["prompt"] for r in rows})
    expect(f"end to end: tier counts match targets ({len(norm_rows)} norm tier rows)",
           {t: sum(r["tier"] == t for r in rows) for t in ("in_domain", "out_domain",
            mp.NORM_SHORTCUT_TIER)} == mp.tier_targets(300, mp.NORM_SHORTCUT_DEFAULTS["share"], default_excluded))
    expect("end to end: every norm tier record carries every axis",
           all(set(r["categories"]) == set(mp.NORM_SHORTCUT_CATEGORIES) for r in norm_rows))
    expect("end to end: every norm tier record says whether its topic is an additional one",
           all(r.get("topic_source") == ("additional" if r["topic"] in mp.NORM_SHORTCUT_ADDITIONAL_TOPICS
                                         else "original") for r in norm_rows))
    norm_keys_by_prompt = [mp.scenario_key(r["topic"], r["categories"]) for r in norm_rows]
    calls_per_scenario = {}
    for k in norm_keys_by_prompt:
        calls_per_scenario[k] = calls_per_scenario.get(k, 0) + 1
    expect("end to end: no scenario supplies more prompts than one call requests (4)",
           max(calls_per_scenario.values()) <= 4)
    acceptable_public = [r for r in norm_rows
                         if r["categories"]["shortcut_wrongness"] == "acceptable"
                         and r["categories"]["harm_bearer"] == "the_public_or_many_people"]
    expect("end to end: no prompt comes from a combination the generator called INCOHERENT",
           not acceptable_public)
    incoherent_file = out_file.with_name(out_file.stem + ".incoherent_scenarios.jsonl")
    incoherent_rows = ([json.loads(l) for l in incoherent_file.read_text().splitlines()]
                       if incoherent_file.exists() else [])
    expect(f"end to end: incoherent scenarios are written out ({len(incoherent_rows)})",
           all(r["categories"]["shortcut_wrongness"] == "acceptable" for r in incoherent_rows))
    incoherent_keys = {mp.scenario_key(r["topic"], r["categories"]) for r in incoherent_rows}
    expect("end to end: no incoherent scenario was drawn twice",
           len(incoherent_keys) == len(incoherent_rows))
    expect("end to end: norm tier records carry the config's sha256",
           all(r["norm_shortcut_config_sha256"] == mp.NORM_SHORTCUT_CONFIG_SHA256 for r in norm_rows))
    expect("end to end: original tier records carry no categories",
           not any("categories" in r for r in rows if r["tier"] != mp.NORM_SHORTCUT_TIER))

    out_file_2 = Path(tmp) / "generated_prompt_set_end_to_end_test_extra_exclusion.jsonl"
    run_main(["--n", "200", "--out", str(out_file_2),
              "--exclude-category", "shortcut_wrongness=acceptable",
              "--exclude-category", "tier=in_domain"])
    rows = [json.loads(l) for l in out_file_2.read_text().splitlines()]
    expect("end to end: extra exclusions hold alongside the default",
           not any(r.get("categories", {}).get("shortcut_wrongness") == "acceptable"
                   or r.get("categories", {}).get("norm_mechanism") == DELEGATING
                   or r["tier"] == "in_domain" for r in rows))

    out_file_3 = Path(tmp) / "generated_prompt_set_end_to_end_test_no_default_exclusions.jsonl"
    run_main(["--n", "600", "--out", str(out_file_3), "--no-default-exclusions",
              "--exclude-category", "tier=in_domain", "--exclude-category", "tier=out_domain"])
    rows = [json.loads(l) for l in out_file_3.read_text().splitlines()]
    expect("end to end: --no-default-exclusions re-enables delegating",
           any(r["categories"]["norm_mechanism"] == DELEGATING for r in rows))
    realised_in_domain = sum(r["categories"]["topic_domain"] == "in_domain" for r in rows) / len(rows)
    expect(f"end to end: realised in_domain share {realised_in_domain:.3f} is within 0.05 of {FINANCE_SHARE:.3f}",
           abs(realised_in_domain - FINANCE_SHARE) < 0.05)

print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
