#!/usr/bin/env python3
"""Split the 704-item MASK extension into balanced shards so several pods or processes can share one model's extension run.

    python scripts/split_mask_extension_items_into_balanced_shards_for_parallel_pods.py --extension-items-dir ../data/alignment_eval_pilot_items_mask_extension_all_remaining_704_items_... --n-shards 4

Items are sorted by item id within each archetype and dealt round-robin, so every shard has the archetype mix of the whole set (counts differ by at most one
per archetype). Each shard folder has the same layout and manifest format as the extension folder (the three sycophancy files are empty), so the
sampler and judge read it with `--items-dir`. Item ids stay disjoint across shards: the judge's per-item verdicts are pooled by plain union.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sl_da import alignment_eval_pilot as pilot


def deal_into_shards(items: list[dict], n_shards: int) -> list[list[dict]]:
    shards = [[] for _ in range(n_shards)]
    for archetype in pilot.MASK_ARCHETYPES:
        ordered = sorted((i for i in items if i["archetype"] == archetype), key=lambda i: i["item_id"])
        for position, item in enumerate(ordered):
            shards[position % n_shards].append(item)
    return shards


def write_jsonl(path: Path, rows: list[dict]) -> str:
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-items-dir", required=True)
    parser.add_argument("--n-shards", type=int, default=4)
    args = parser.parse_args()
    items = pilot.load_pinned_items(args.extension_items_dir)["mask"]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for index, shard in enumerate(deal_into_shards(items, args.n_shards), start=1):
        out = ROOT.parent / "data" / f"alignment_eval_pilot_items_mask_extension_shard_{index}_of_{args.n_shards}_20261005"
        if out.exists():
            print(f"{out.name} exists, left alone"); continue
        out.mkdir(parents=True)
        files = {"mask": (f"mask_extension_shard_{index}_of_{args.n_shards}_items_{stamp}.jsonl", shard),
                 "sycophancy_feedback": (f"sycophancy_feedback_items_none_in_this_folder_{stamp}.jsonl", []),
                 "sycophancy_answer": (f"sycophancy_answer_items_none_in_this_folder_{stamp}.jsonl", []),
                 "sycophancy_are_you_sure": (f"sycophancy_are_you_sure_items_none_in_this_folder_{stamp}.jsonl", [])}
        item_files = {k: {"file_name": n, "n_items": len(r), "sha256": write_jsonl(out / n, r)} for k, (n, r) in files.items()}
        meta = {"built_utc": stamp, "purpose": f"Shard {index} of {args.n_shards} of the MASK extension items (round-robin within archetype).",
                "source_folder": Path(args.extension_items_dir).name,
                "counts_by_archetype": {a: sum(1 for i in shard if i["archetype"] == a) for a in pilot.MASK_ARCHETYPES},
                "item_files": item_files}
        (out / f"mask_extension_shard_{index}_of_{args.n_shards}_manifest_{stamp}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{out.name}: {len(shard)} items {meta['counts_by_archetype']}")


if __name__ == "__main__":
    main()
