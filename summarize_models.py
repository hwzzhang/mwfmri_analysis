#!/usr/bin/env python3
"""Merge outer-fold shards and summarize model AUCs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import fmean, stdev
import sys

from run_models import FEATURES, ROOT
sys.path.insert(0, str(ROOT / "shared"))
from settings import load_config, resolve_path  # noqa: E402

OUTER_HEADER = [
    "index_out", "best_C", "test_accuracy", "test_precision", "test_f1",
    "test_recall", "test_roc_auc", "train_accuracy", "train_precision",
    "train_f1", "train_recall", "train_roc_auc", "test_index",
]


def merge_rows(directory: Path) -> list[dict[str, str]]:
    paths = sorted((directory / "shards").glob("*/record_out_CV.csv"))
    by_fold: dict[int, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    index = int(row["index_out"])
                except (KeyError, ValueError):
                    continue
                if index in by_fold and by_fold[index] != row:
                    raise ValueError(f"Conflicting fold {index}: {path}")
                by_fold[index] = row
    if set(by_fold) != set(range(100)):
        raise ValueError(f"Expected exactly folds 0..99 in {directory}")
    rows = [by_fold[index] for index in range(100)]
    with (directory / "record_out_CV.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTER_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    output_root = resolve_path(load_config()["output_root"])
    parser.add_argument("--output", type=Path, default=output_root / "model_summary.csv")
    args = parser.parse_args()
    summaries = []
    for task, entries in FEATURES.items():
        for feature, _flags, output_name in entries:
            directory = output_root / task / "cross_validation" / output_name
            rows = merge_rows(directory)
            aucs = [float(row["test_roc_auc"]) for row in rows]
            repeat_means = [fmean(aucs[start:start + 10]) for start in range(0, 100, 10)]
            summaries.append({
                "task": task, "feature": feature, "mean_auc": fmean(aucs),
                "sd_repeat_means": stdev(repeat_means), "outer_folds": 100,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(args.output)


if __name__ == "__main__":
    main()
