#!/usr/bin/env python3
"""Build the MASK extension item set: every usable MASK item (the same five binary archetypes, same pinned upstream revision)
that is NOT in the 200-item pilot set. Pilot (200) + extension (704) = all 904 usable items, with no overlap, so the two
runs can be pooled for more statistical power with identical protocol (one pressured run per item, same belief passes).

    python scripts/build_mask_extension_items_not_in_the_pilot_set_for_more_statistical_power.py --pilot-items-dir ../data/alignment_eval_pilot_items_...

The folder has the same layout and manifest format as the pilot items folder, so every existing script (sampler, judge) reads it with
`--items-dir`. The three sycophancy item files are present but EMPTY (this folder is for MASK only). Network is used only here, never on a pod.
"""
import argparse, hashlib, importlib.util, io, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sl_da import alignment_eval_pilot as pilot

spec = importlib.util.spec_from_file_location("pilot_items_builder", Path(__file__).with_name("build_pinned_alignment_eval_pilot_items_mask_and_sycophancy.py"))
builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)

MASK_COLUMNS = ("task_id", "system_prompt", "user_prompt", "proposition", "ground_truth", "formatted_ground_truth",
                "belief_elicit_1", "belief_elicit_2", "belief_elicit_3", "dd_turn_1_prompt", "dd_turn_1_response", "canary")


def build_extension_items(cache_dir: Path, pilot_item_ids: set[str]) -> list[dict]:
    import pandas as pd
    items = []
    for archetype in pilot.MASK_ARCHETYPES:
        frame = pd.read_parquet(io.BytesIO(builder.fetch(f"mask_{archetype}.parquet", cache_dir)))
        frame = frame[frame.proposition.map(lambda p: pilot.parse_binary_proposition(p) is not None)]
        records = {f"{archetype}_{r['task_id']}": r for r in frame.to_dict("records")}
        for item_id in sorted(records):
            if item_id in pilot_item_ids:
                continue
            r = records[item_id]
            item = {"item_id": item_id, "archetype": archetype}
            for column in MASK_COLUMNS:
                if column in r and pilot._present(r[column]):
                    item[column] = r[column]
            items.append(item)
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-items-dir", required=True)
    parser.add_argument("--cache-dir", default=str(ROOT / ".upstream_cache_alignment_eval_pilot"))
    args = parser.parse_args()
    pilot_ids = {item["item_id"] for item in pilot.load_pinned_items(args.pilot_items_dir)["mask"]}
    out_dir = ROOT.parent / "data" / "alignment_eval_pilot_items_mask_extension_all_remaining_704_items_not_in_the_200_item_pilot_set_20261005"
    if out_dir.exists():
        print(f"{out_dir} exists; counts { {k: len(v) for k, v in pilot.load_pinned_items(out_dir).items()} }"); return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    items = build_extension_items(cache_dir, pilot_ids)
    assert not ({i["item_id"] for i in items} & pilot_ids)
    out_dir.mkdir(parents=True)
    files = {"mask": (f"mask_extension_items_all_remaining_five_binary_archetypes_{stamp}.jsonl", items),
             "sycophancy_feedback": (f"sycophancy_feedback_items_none_in_this_folder_{stamp}.jsonl", []),
             "sycophancy_answer": (f"sycophancy_answer_items_none_in_this_folder_{stamp}.jsonl", []),
             "sycophancy_are_you_sure": (f"sycophancy_are_you_sure_items_none_in_this_folder_{stamp}.jsonl", [])}
    item_files = {key: {"file_name": name, "n_items": len(rows), "sha256": builder.write_jsonl(out_dir / name, rows)} for key, (name, rows) in files.items()}
    meta = {"built_utc": stamp, "purpose": "MASK extension: all usable items not in the 200-item pilot set (pilot + extension = 904 items), for pooled statistical power.",
            "pilot_items_dir": str(Path(args.pilot_items_dir).name), "n_pilot_items_excluded": len(pilot_ids),
            "counts_by_archetype": {a: sum(1 for i in items if i["archetype"] == a) for a in pilot.MASK_ARCHETYPES},
            "item_files": item_files, "sources": {"mask_data": {"url": "https://huggingface.co/datasets/cais/MASK", "revision": builder.MASK_DATA_REVISION}},
            "upstream_file_sha256": {name: hashlib.sha256(builder.fetch(name, cache_dir)).hexdigest() for name in builder.UPSTREAM_FILES if name.startswith("mask_")},
            "canary_notice": "MASK rows keep their canary column. These items are for evaluation only."}
    (out_dir / f"mask_extension_items_manifest_{stamp}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_dir}; counts {meta['counts_by_archetype']}")


if __name__ == "__main__":
    main()
