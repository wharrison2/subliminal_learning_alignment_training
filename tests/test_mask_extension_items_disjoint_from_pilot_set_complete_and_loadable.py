"""The MASK extension items must not overlap the pilot items (pooling would double count), must together with them cover every usable
MASK item, and must load through the same pinned-item loader that the sampler and the judge use."""
import glob, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from sl_da import alignment_eval_pilot as pilot

data = root.parent / "data"
pilot_dir = glob.glob(str(data / "alignment_eval_pilot_items_mask_five_binary_archetypes_40_each_*"))[0]
extension_dir = glob.glob(str(data / "alignment_eval_pilot_items_mask_extension_all_remaining_704_items_*"))[0]
pilot_items, extension_items = pilot.load_pinned_items(pilot_dir), pilot.load_pinned_items(extension_dir)
pilot_ids = [i["item_id"] for i in pilot_items["mask"]]
extension_ids = [i["item_id"] for i in extension_items["mask"]]
assert len(pilot_ids) == 200 and len(extension_ids) == 704, (len(pilot_ids), len(extension_ids))
assert not set(pilot_ids) & set(extension_ids), "pilot and extension overlap"
assert len(set(extension_ids)) == 704
assert extension_items["sycophancy_feedback"] == extension_items["sycophancy_answer"] == extension_items["sycophancy_are_you_sure"] == []
by_archetype = {a: sum(1 for i in extension_items["mask"] if i["archetype"] == a) + 40 for a in pilot.MASK_ARCHETYPES}
assert by_archetype == {"known_facts": 209, "provided_facts": 274, "disinformation": 125, "continuations": 176, "doubling_down_known_facts": 120}, by_archetype
for item in extension_items["mask"]:
    assert item["user_prompt"] and item["system_prompt"] and item["proposition"], item["item_id"]
    assert item["archetype"] == "provided_facts" or item.get("belief_elicit_1"), item["item_id"]
    assert (item["archetype"] == "doubling_down_known_facts") == ("dd_turn_1_prompt" in item), item["item_id"]
n_requests = len(pilot.all_generation_requests_for_stage_one(extension_items, [pilot.MASK_EVAL], None))
print("extension generation requests per model:", n_requests)
assert n_requests > 3000
print("ALL PASSED")
