"""The four MASK extension shards must be disjoint, cover exactly the 704 extension items, keep the archetype mix (counts within one),
and load through the pinned-item loader. Pooling shards by union is only correct if this holds."""
import glob, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from sl_da import alignment_eval_pilot as pilot

data = root.parent / "data"
extension = pilot.load_pinned_items(glob.glob(str(data / "alignment_eval_pilot_items_mask_extension_all_remaining_704_items_*"))[0])["mask"]
shard_dirs = sorted(glob.glob(str(data / "alignment_eval_pilot_items_mask_extension_shard_*_of_4_20261005")))
assert len(shard_dirs) == 4, shard_dirs
shards = [pilot.load_pinned_items(d) for d in shard_dirs]
ids = [[i["item_id"] for i in s["mask"]] for s in shards]
flat = [x for group in ids for x in group]
assert len(flat) == 704 and len(set(flat)) == 704, (len(flat), len(set(flat)))
assert set(flat) == {i["item_id"] for i in extension}, "shards do not cover the extension exactly"
for archetype in pilot.MASK_ARCHETYPES:
    counts = [sum(1 for i in s["mask"] if i["archetype"] == archetype) for s in shards]
    assert max(counts) - min(counts) <= 1, (archetype, counts)
assert max(len(g) for g in ids) - min(len(g) for g in ids) <= 5, [len(g) for g in ids]
assert all(s["sycophancy_feedback"] == s["sycophancy_answer"] == s["sycophancy_are_you_sure"] == [] for s in shards)
print("shard sizes", [len(g) for g in ids])
print("ALL PASSED")
