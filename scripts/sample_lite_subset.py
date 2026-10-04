"""Draw the fixed SWE-bench Lite subset for the Long Memory experiment.

Standalone on purpose (stdlib + pyarrow only), so it runs without installing the
harness.  The pool is sorted before sampling, so the result depends only on the
pinned dataset file, the seed and the sizes.  Draw order is kept: the first
``batch_size`` ids form batch 1 (run first), the rest form batch 2 (run later if
the budget allows).

Example:
    uv run --no-project --with pyarrow python scripts/sample_lite_subset.py \
        --parquet test-00000-of-00001.parquet --revision <dataset commit> \
        --seed 20261004 --out config/swebench_lite_sample100.json

Drawing the whole pool (``--size`` equal to the pool) gives a seeded random order,
used for the dev split: ``--split dev --expected-pool 23 --size 23 --batch-size 5``.

Single-repository, chronological streams (the current protocol) add
``--repo django/django --order created_at``: the seeded draw is then sorted by issue
creation time, so batch 1 is the earliest part and batch 2 continues the same stream.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import platform
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--revision", required=True, help="dataset commit the parquet was downloaded from")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--size", default=100, type=int)
    parser.add_argument("--batch-size", default=50, type=int)
    parser.add_argument("--expected-pool", default=300, type=int)
    parser.add_argument("--repo", default=None, help="restrict the pool to one repository, e.g. django/django")
    parser.add_argument("--order", choices=["draw", "created_at"], default="draw",
                        help="keep draw order, or sort the sample by issue creation time")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dev-size", default=0, type=int,
                        help="hold out this many of the drawn ids as a disjoint dev set (written to --dev-out)")
    parser.add_argument("--dev-out", default=None, type=Path)
    parser.add_argument("--dev-batch-size", default=5, type=int)
    args = parser.parse_args()
    if bool(args.dev_size) != bool(args.dev_out):
        raise SystemExit("--dev-size and --dev-out go together")

    import pyarrow.parquet as pq

    rows = pq.read_table(args.parquet, columns=["instance_id", "repo", "created_at"]).to_pylist()
    rows = [row for row in rows if args.repo is None or row["repo"] == args.repo]
    ids = [row["instance_id"] for row in rows]
    repo_of = {row["instance_id"]: row["repo"] for row in rows}
    created_of = {row["instance_id"]: row["created_at"] for row in rows}
    if len(ids) != len(set(ids)) or len(ids) != args.expected_pool:
        raise SystemExit(f"expected {args.expected_pool} unique instance ids, got {len(ids)} ({len(set(ids))} unique)")
    if not 0 < args.size <= len(ids):
        raise SystemExit(f"size must be between 1 and the pool size {len(ids)}")

    # The last batch may be smaller (e.g. all 23 dev ids in groups of 5).
    rng = random.Random(args.seed)
    sample = rng.sample(sorted(ids), args.size)
    # Dev ids come from the same draw (same rng, next call), so the main draw is unchanged
    # and dev and test are disjoint by construction.
    dev = set(rng.sample(sorted(sample), args.dev_size)) if args.dev_size else set()

    def build(chosen: list[str], batch_size: int) -> dict:
        chosen = list(chosen)
        if args.order == "created_at":
            # ISO-8601 UTC strings sort chronologically; ties fall back to the id.
            chosen.sort(key=lambda iid: (created_of[iid], iid))
        batches = [chosen[i:i + batch_size] for i in range(0, len(chosen), batch_size)]
        record = {
            "dataset": args.dataset,
            "split": args.split,
            "revision": args.revision,
            "source_file": f"https://huggingface.co/datasets/{args.dataset}/resolve/{args.revision}/data/{args.parquet.name}",
            "source_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
            "pool_size": len(ids),
            "method": "random.Random(seed).sample(sorted(instance_ids), size); batches in draw order",
            "seed": args.seed,
            "size": len(chosen),
            "batch_size": batch_size,
            "python": platform.python_version(),
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "batches": [
                {
                    "batch": number,
                    "instance_ids": batch,
                    "repo_counts": dict(collections.Counter(repo_of[i] for i in batch).most_common()),
                }
                for number, batch in enumerate(batches, start=1)
            ],
            "pool_repo_counts": dict(collections.Counter(repo_of.values()).most_common()),
        }
        # Extra keys only for the new options, so earlier default draws reproduce byte for byte.
        if args.repo is not None:
            record["repo_filter"] = args.repo
        if args.order == "created_at":
            record["method"] = ("random.Random(seed).sample(sorted(instance_ids), size); sorted by "
                                "(created_at, instance_id); batches in that order")
            for entry in record["batches"]:
                entry["created_at_range"] = [created_of[entry["instance_ids"][0]], created_of[entry["instance_ids"][-1]]]
        if args.dev_size:
            record["dev_holdout"] = {
                "drawn": args.size, "dev_size": args.dev_size,
                "method": "dev = random.Random(seed) second call: sample(sorted(drawn), dev_size); test = the rest",
            }
        return record

    outputs = [(args.out, build([i for i in sample if i not in dev], args.batch_size))]
    if args.dev_size:
        dev_record = build(sorted(dev), args.dev_batch_size)
        dev_record["role"], outputs[0][1]["role"] = "dev", "test"
        outputs.append((args.dev_out, dev_record))
    for path, record in outputs:
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {path}: {len(record['batches'])} batches, sizes {[len(b['instance_ids']) for b in record['batches']]}")


if __name__ == "__main__":
    main()
