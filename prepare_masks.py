#!/usr/bin/env python3
"""Create nested-CV splits and the SPM mask-job manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "shared"))
from settings import load_config, resolve_path, task_config  # noqa: E402


def build_manifest(task: str, output_root: Path, outer_repeats: int = 10,
                   seed: int = 0, outer_limit: int = -1) -> dict:
    cfg = task_config(task)
    archive_path = resolve_path(cfg["tabular_feature_archive"])
    with np.load(archive_path, allow_pickle=True) as archive:
        ids = np.asarray(archive["id_list"], dtype=str)
        labels = np.asarray(archive[cfg["label_key"]], dtype=int)

    outer_cv = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=outer_repeats, random_state=seed,
    )
    jobs: list[dict] = []
    rows: list[dict] = []
    for outer_index, (outer_train, outer_test) in enumerate(outer_cv.split(ids, labels)):
        if outer_limit >= 0 and outer_index >= outer_limit:
            break
        jobs.append({
            "mask_id": f"outer_{outer_index:03d}",
            "level": "outer",
            "outer_index": outer_index,
            "inner_index": -1,
            "train_indices": outer_train.tolist(),
            "train_ids": ids[outer_train].tolist(),
        })
        rows.append({
            "outer_index": outer_index,
            "repeat": outer_index // 10,
            "fold": outer_index % 10,
            "train_ids": json.dumps(ids[outer_train].tolist(), separators=(",", ":")),
            "test_ids": json.dumps(ids[outer_test].tolist(), separators=(",", ":")),
            "train_labels": json.dumps(labels[outer_train].tolist(), separators=(",", ":")),
            "test_labels": json.dumps(labels[outer_test].tolist(), separators=(",", ":")),
        })
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for inner_index, (inner_train_rel, _inner_valid_rel) in enumerate(
            inner_cv.split(ids[outer_train], labels[outer_train])
        ):
            inner_train = outer_train[inner_train_rel]
            jobs.append({
                "mask_id": f"inner_{outer_index:03d}_{inner_index}",
                "level": "inner",
                "outer_index": outer_index,
                "inner_index": inner_index,
                "train_indices": inner_train.tolist(),
                "train_ids": ids[inner_train].tolist(),
            })

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": task,
        "seed": seed,
        "outer_splits": 10,
        "outer_repeats": outer_repeats,
        "inner_splits": 5,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    (output_root / "mask_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    with (output_root / "outer_splits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("classification", "forecast"), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--outer-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outer-limit", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = load_config()
    output = args.output_root or (
        resolve_path(cfg["output_root"]) / "fold_masks" / args.task
    )
    manifest = build_manifest(
        args.task, output, args.outer_repeats, args.seed, args.outer_limit,
    )
    print(f"Wrote {manifest['n_jobs']} mask jobs to {output}")


if __name__ == "__main__":
    main()
