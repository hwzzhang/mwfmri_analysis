#!/usr/bin/env python3
"""Run all feature combinations for both prediction tasks."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "shared"))
from settings import load_config, resolve_path  # noqa: E402

FEATURES = {
    "classification": [
        ("demographics", ["--demogra"], "demographics"),
        ("sMRI", ["--rois_non_speech_expert"], "smri"),
        ("rs-fMRI", ["--ALFF"], "rs_fmri"),
        ("mw-fMRI", ["--tmap_speech_silence"], "mw_fmri"),
        ("mw-fMRI + demographics", ["--demogra", "--tmap_speech_silence"], "mw_fmri_demographics"),
        ("mw-fMRI + sMRI", ["--rois_non_speech_expert", "--tmap_speech_silence"], "mw_fmri_smri"),
        ("mw-fMRI + rs-fMRI", ["--tmap_speech_silence", "--ALFF"], "mw_fmri_rs_fmri"),
        ("All", ["--demogra", "--rois_non_speech_expert", "--tmap_speech_silence", "--ALFF"], "all_features"),
    ],
    "forecast": [
        ("demographics", ["--demogra"], "demographics"),
        ("baseline MoCA", ["--base_moca"], "baseline_moca"),
        ("sMRI", ["--rois_non_speech_expert"], "smri"),
        ("rs-fMRI", ["--ALFF"], "rs_fmri"),
        ("rs-fMRI + baseline MoCA", ["--base_moca", "--ALFF"], "rs_fmri_baseline_moca"),
        ("mw-fMRI", ["--tmap_speech_silence"], "mw_fmri"),
        ("mw-fMRI + demographics", ["--demogra", "--tmap_speech_silence"], "mw_fmri_demographics"),
        ("mw-fMRI + baseline MoCA", ["--base_moca", "--tmap_speech_silence"], "mw_fmri_baseline_moca"),
        ("mw-fMRI + sMRI", ["--rois_non_speech_expert", "--tmap_speech_silence"], "mw_fmri_smri"),
        ("mw-fMRI + rs-fMRI", ["--tmap_speech_silence", "--ALFF"], "mw_fmri_rs_fmri"),
        ("All", ["--demogra", "--rois_non_speech_expert", "--tmap_speech_silence", "--ALFF"], "all_features"),
    ],
}


def command(task: str, flags: list[str], output_name: str, mask_root: Path, threshold: float,
            start: int, stop: int, shard: str) -> tuple[list[str], Path]:
    task_root = ROOT / task
    driver = (
        "run_classification.py" if task == "classification"
        else "run_forecast.py"
    )
    label = "moca" if task == "classification" else "two_follow_ratio_change"
    cmd = [
        sys.executable, driver, "--mask_dir", str(mask_root),
        "--feature_select", "--l1_select", "--fsthreshold", str(threshold),
        "--test_proportion", "0.1", "--label_type", label,
        "--scaler_type", "min_max",
        "--continue_num", str(start), "--stop_num", str(stop),
        "--result_shard", shard, "--output-name", output_name,
    ] + flags
    return cmd, task_root


def run_one(job: dict, log_root: Path) -> None:
    log_path = log_root / f"{job['task']}_{job['feature']}_{job['shard']}.log"
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "MPLCONFIGDIR": str(log_root / ".mpl"),
    })
    with log_path.open("w") as log:
        result = subprocess.run(
            job["command"], cwd=job["cwd"], env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"Failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=sorted(FEATURES), default=sorted(FEATURES))
    parser.add_argument("--features", nargs="+", help="Optional exact display names")
    parser.add_argument("--classification-mask-root", type=Path, required=True)
    parser.add_argument("--forecast-mask-root", type=Path, required=True)
    parser.add_argument("--pearson-threshold", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    roots = {
        "classification": args.classification_mask_root.resolve(),
        "forecast": args.forecast_mask_root.resolve(),
    }
    log_root = resolve_path(load_config()["output_root"]) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for task in args.tasks:
        for feature, flags, output_name in FEATURES[task]:
            if args.features and feature not in args.features:
                continue
            for start in range(0, 100, args.chunk_size):
                stop = min(100, start + args.chunk_size)
                shard = f"folds_{start:03d}_{stop - 1:03d}"
                cmd, cwd = command(task, flags, output_name, roots[task], args.pearson_threshold, start, stop, shard)
                jobs.append({"task": task, "feature": output_name, "shard": shard, "command": cmd, "cwd": cwd})
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_one, job, log_root): job for job in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append((futures[future], error))
    if failures:
        for job, error in failures:
            print(job, error, file=sys.stderr)
        raise SystemExit(f"{len(failures)} jobs failed")


if __name__ == "__main__":
    main()
