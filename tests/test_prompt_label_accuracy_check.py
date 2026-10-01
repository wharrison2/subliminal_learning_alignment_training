#!/usr/bin/env python3
"""check_prompt_label_accuracy.py: the judge is blind, and agreement is counted correctly.

Experiment-invalidating failures this guards against: the intended labels leaking into
the judge's prompt (agreement would measure nothing), and agreement arithmetic that
credits a secondary value on single-value axes or miscounts unparseable replies.
No network -- runs anywhere:

    python tests/test_prompt_label_accuracy_check.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompts"))
import check_prompt_label_accuracy as check

failures = []
def expect(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


check.mp.load_norm_shortcut_config()
prompt = check.JUDGE_INSTRUCTIONS.format(axes=check.axes_block(), keys=check.keys_block(),
                                         topic="some topic", prompt="Can I skip the sign-off?")
expect("the judge prompt names every judged axis", all(axis in prompt for axis in check.JUDGED_AXES))
expect("the judge prompt has no slot for intended labels (blind)",
       "categories" not in check.JUDGE_INSTRUCTIONS and "intended" not in check.JUDGE_INSTRUCTIONS.lower())

expect("a reply with JSON inside prose is parsed",
       check.parse_judgement('ok {' + ", ".join(f'"{a}": "x"' for a in check.JUDGED_AXES) + '}') is not None)
expect("a reply missing an axis is unparseable, not guessed",
       check.parse_judgement('{"norm_mechanism": "law_or_regulation"}') is None)
expect("an empty reply (reasoning ran out of tokens) is unparseable", check.parse_judgement("") is None)


def record(intended_mechanism, judged_mechanism, also=(), harm="the_user_themselves", judged_harm="the_user_themselves"):
    categories = {axis: next(iter(check.mp.NORM_SHORTCUT_CATEGORIES[axis])) for axis in check.JUDGED_AXES}
    categories.update(norm_mechanism=intended_mechanism, harm_bearer=harm, topic_domain="out_domain")
    judged = dict(categories, norm_mechanism=judged_mechanism, harm_bearer=judged_harm,
                  norm_mechanism_also_present=list(also), harm_bearer_also_present=[],
                  genuine_norm_temptation="yes", is_personal_finance="no", fits_topic_area="yes")
    return {"topic": "t", "topic_source": "original", "categories": categories, "judged": judged}


records = [record("law_or_regulation", "law_or_regulation"),
           record("law_or_regulation", "honesty_and_disclosure", also=["law_or_regulation"]),
           record("breaking_a_commitment", "not_conveyed"),
           record("breaking_a_commitment", "honesty_and_disclosure")]
summary = check.summarise(records)
mechanism = summary["axes"]["norm_mechanism"]
expect("exact agreement counts only the primary value (1/4)", mechanism["agreement"] == 0.25)
expect("agreement allowing secondary also counts also_present (2/4)", mechanism["agreement_allowing_secondary"] == 0.5)
expect("not_conveyed rate is reported (1/4)", mechanism["not_conveyed"] == 0.25)
expect("single-value axes never credit a secondary value",
       summary["axes"]["user_role"]["agreement"] == summary["axes"]["user_role"]["agreement_allowing_secondary"])
expect("per intended value agreement is reported",
       mechanism["per_intended_value"]["law_or_regulation"]["agreement"] == 0.5)
expect("topic_domain agreement uses the finance yes/no answer", summary["topic_domain_agreement"] == 1.0)

print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
