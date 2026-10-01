#!/usr/bin/env python3
"""Run repeated nested cross-validation for two-year cognitive change."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import config
import modeling


UNUSED_FEATURE_FLAGS = (
    "base_moca_mini", "tmaps", "tmap_silence_speech", "tmap_monolog_dialog",
    "tmap_dialog_monolog", "tmap_speech_baseline", "tmap_silence_baseline",
    "rois_speech_mask", "rois_whole", "AAL_FC", "movie_AAL_FC", "fALFF",
    "ReHo", "ICA_FC", "ICA_FC_part",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_type", choices=("two_follow_ratio_change",), required=True,
    )
    parser.add_argument("--scaler_type", required=True)
    parser.add_argument("--feature_select", action="store_true")
    parser.add_argument("--l1_select", action="store_true")
    parser.add_argument("--fsthreshold", type=float, required=True)
    parser.add_argument("--test_proportion", type=float, required=True)
    parser.add_argument("--mask_dir", default=config.FOLD_MASK_DIR)
    parser.add_argument("--continue_num", type=int, default=0)
    parser.add_argument("--stop_num", type=int, default=100)
    parser.add_argument("--result_shard", default="")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--demogra", action="store_true")
    parser.add_argument("--base_moca", action="store_true")
    parser.add_argument("--rois_non_speech_expert", action="store_true")
    parser.add_argument("--tmap_speech_silence", action="store_true")
    parser.add_argument("--ALFF", action="store_true")
    args = parser.parse_args()
    for name in UNUSED_FEATURE_FLAGS:
        setattr(args, name, False)
    args.save_index = False
    args.residualize_movie_motion = False
    args.residualize_movie_mean_fd = False
    return args


def main() -> None:
    args = parse_args()
    feature_name = modeling.check_feature_input(args)
    if not feature_name:
        raise SystemExit("Select at least one feature set")
    feature_name = "_" + args.label_type + feature_name

    tabular_path = Path(config.TABULAR_FEATURE_DIR) / "baseline_features.npz"
    feature_dict = modeling.load_feature(str(tabular_path), args, config)
    if args.tmap_speech_silence:
        whole_path = Path(config.WHOLE_BRAIN_FEATURE_DIR) / "baseline_features.npz"
        with np.load(whole_path, allow_pickle=True) as whole:
            if not np.array_equal(feature_dict["id_list"], whole["id_list"]):
                raise ValueError("Tabular and whole-brain participant IDs differ")
            feature_dict["tmap_speech_silence"] = whole[
                "tmap_speech_silence_feature_list"
            ]

    output = Path(config.SAVE_RESULT_DIR) / "cross_validation" / args.output_name
    if args.result_shard:
        output = output / "shards" / args.result_shard
    output.mkdir(parents=True, exist_ok=True)

    outer_cv = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=10, random_state=0,
    )
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    modeling.nested_cross_validation(
        args,
        feature_name,
        outer_cv,
        inner_cv,
        feature_dict["raw_score"],
        feature_dict["score_label"],
        feature_dict["id_list"],
        feature_dict,
        config,
        str(output),
    )


if __name__ == "__main__":
    main()
