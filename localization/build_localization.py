#!/usr/bin/env python3
"""Build spatial summaries of selected movie-watching fMRI features.

The script repeats each saved outer fit, maps fold-local indices back to the
common gray-matter grid, and verifies that all held-out metrics match the
saved model results.

Outputs include per-fold indices, voxel selection-frequency NIfTI maps, and
fold-wise AAL summaries. Held-out participants are not used during mask
construction or feature selection.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from settings import load_config, resolve_path  # noqa: E402

CONFIG = load_config()
GRAY_MASK = resolve_path(CONFIG["gray_matter_mask"])
AAL_ATLAS = resolve_path(CONFIG["aal3_image"])
AAL_LABELS = resolve_path(CONFIG["aal3_labels"])
MOVIE_SCORE_CSV = resolve_path(CONFIG["post_movie_scores"])
OUTPUT_ROOT = resolve_path(CONFIG["output_root"]) / "localization"

TASKS = {
    "classification": {
        "output_name": "mw_fmri",
        "label": "moca",
        "targets": ("baseline_moca", "post_movie_test"),
    },
    "forecast": {
        "output_name": "mw_fmri",
        "label": "two_follow_ratio_change",
        "targets": ("moca_change", "baseline_moca", "post_movie_test"),
    },
}


def import_task_modules(task: str):
    directory = ROOT / task
    modules = []
    for name in ("config", "modeling"):
        path = directory / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"localize_{task}_{name}", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def make_args(task: str) -> SimpleNamespace:
    names = [
        "demogra", "base_moca", "base_moca_mini", "tmaps",
        "tmap_speech_silence", "tmap_silence_speech", "tmap_monolog_dialog",
        "tmap_dialog_monolog", "tmap_speech_baseline", "tmap_silence_baseline",
        "rois_non_speech_expert", "rois_speech_mask", "rois_whole", "AAL_FC",
        "movie_AAL_FC", "ALFF", "fALFF", "ReHo", "ICA_FC", "ICA_FC_part",
    ]
    values = {name: False for name in names}
    values.update({
        "tmap_speech_silence": True,
        "label_type": TASKS[task]["label"],
        "mask_dir": str(
            resolve_path(CONFIG["output_root"]) / "fold_masks" / task
        ),
        "feature_select": True,
        "l1_select": True,
        "save_index": True,
        "scaler_type": "min_max",
        "fsthreshold": 0.01,
        "test_proportion": 0.1,
        "residualize_movie_motion": False,
        "residualize_movie_mean_fd": False,
    })
    return SimpleNamespace(**values)


def parse_test_indices(value: str) -> np.ndarray:
    return np.fromstring(value.strip().strip("[]"), sep=" ", dtype=int)


def load_mask_support(mask_root: Path, mask_id: str, gray: np.ndarray) -> np.ndarray:
    path = mask_root / mask_id / "spmT_0001_fold_mask.nii.gz"
    if not path.exists():
        path = path.with_suffix("")
    image = nib.load(str(path))
    support = (np.asarray(image.dataobj) > 0.9)[gray]
    if not support.any():
        raise ValueError(f"Empty outer mask: {path}")
    return support


def vector_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r for each column, matching scipy for nonconstant columns."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    denominator = np.sqrt(np.sum(xc * xc, axis=0) * np.sum(yc * yc))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum(xc * yc[:, None], axis=0) / denominator


def read_aal_labels() -> dict[int, str]:
    """Read the project-local DPABI AAL3 label table."""
    labels = {}
    for row in loadmat(AAL_LABELS)["Reference"]:
        name = str(np.asarray(row[0]).reshape(-1)[0])
        atlas_id = int(np.asarray(row[1]).reshape(-1)[0])
        if atlas_id > 0:
            labels[atlas_id] = name
    return labels


def load_movie_scores(ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(MOVIE_SCORE_CSV, encoding="utf-8-sig")
    frame.columns = [str(value).strip() for value in frame.columns]
    id_col = frame.columns[0]
    score_col = "MovieInterview"
    lookup = {
        str(row[id_col]).strip(): float(row[score_col])
        for _, row in frame.iterrows()
        if pd.notna(row[id_col]) and pd.notna(row[score_col])
    }
    missing = [subject for subject in ids.astype(str) if subject not in lookup]
    if missing:
        raise ValueError(f"Missing post-movie scores for: {missing}")
    return np.asarray([lookup[subject] for subject in ids.astype(str)])


def prepare_atlas(gray_img: nib.spatialimages.SpatialImage, gray: np.ndarray):
    atlas = resample_to_img(
        nib.load(str(AAL_ATLAS)), gray_img, interpolation="nearest",
    )
    return np.rint(np.asarray(atlas.dataobj)[gray]).astype(np.int16)


def save_gray_map(gray_img, gray: np.ndarray, values: np.ndarray, path: Path) -> None:
    volume = np.zeros(gray_img.shape, dtype=np.float32)
    volume[gray] = np.asarray(values, dtype=np.float32)
    nib.save(nib.Nifti1Image(volume, gray_img.affine, gray_img.header), str(path))


def task_targets(task: str, feature_dict: dict, ids: np.ndarray) -> dict[str, np.ndarray]:
    targets = {"baseline_moca": np.asarray(feature_dict["base_moce_score"], dtype=float)}
    if task == "forecast":
        targets["moca_change"] = np.asarray(feature_dict["raw_score"], dtype=float)
    targets["post_movie_test"] = load_movie_scores(ids)
    return targets


def run_task(task: str, overwrite: bool = False) -> None:
    out = OUTPUT_ROOT / task
    fold_dir = out / "folds"
    out.mkdir(parents=True, exist_ok=True)
    fold_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = out / "COMPLETE.json"
    if complete_marker.exists() and not overwrite:
        print(f"{task}: already complete", flush=True)
        return

    config, basic = import_task_modules(task)
    args = make_args(task)
    archive_path = Path(config.TABULAR_FEATURE_DIR) / "baseline_features.npz"
    feature_dict = basic.load_feature(str(archive_path), args, config)
    whole_path = Path(config.WHOLE_BRAIN_FEATURE_DIR) / "baseline_features.npz"
    whole = np.load(whole_path, allow_pickle=True)
    if not np.array_equal(feature_dict["id_list"], whole["id_list"]):
        raise ValueError("Tabular and whole-brain participant IDs differ")
    raw = np.asarray(whole["tmap_speech_silence_feature_list"])
    feature_dict["tmap_speech_silence"] = raw
    ids = feature_dict["id_list"].astype(str)
    _, labels = np.unique(feature_dict["score_label"], return_inverse=True)
    raw_score = np.asarray(feature_dict["raw_score"])
    targets = task_targets(task, feature_dict, ids)

    source = (
        resolve_path(CONFIG["output_root"]) / task / "cross_validation" /
        TASKS[task]["output_name"] / "record_out_CV.csv"
    )
    saved = pd.read_csv(source).sort_values("index_out").reset_index(drop=True)
    if saved["index_out"].tolist() != list(range(100)):
        raise ValueError(f"Incomplete model output: {source}")

    gray_img = nib.load(str(GRAY_MASK))
    gray = np.asarray(gray_img.dataobj) > 0.9
    if raw.shape[1] != int(gray.sum()):
        raise ValueError("Whole-brain feature matrix and gray mask differ")
    atlas_on_gray = prepare_atlas(gray_img, gray)
    atlas_names = read_aal_labels()
    np.save(out / "aal_label_on_gray.npy", atlas_on_gray)

    pearson_count = np.zeros(raw.shape[1], dtype=np.uint16)
    l1_count = np.zeros(raw.shape[1], dtype=np.uint16)
    roi_rows = []
    fold_rows = []
    outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
    selected_path = (
        Path(config.SAVE_FEATURE_DIR) / "cross_validation" / "tmap_speech_silence" /
        "test_0.1_fs0.01"
    )
    mask_root = Path(args.mask_dir)

    for index, (train, test) in enumerate(outer.split(ids, labels)):
        old = saved.iloc[index]
        if not np.array_equal(test, parse_test_indices(str(old["test_index"]))):
            raise ValueError(f"Outer split mismatch: {task}, fold {index}")
        best_c = ast.literal_eval(str(old["best_C"]))
        basic._CORR_SELECTION_CACHE.clear()
        args._outer_cache_token = index
        x = basic.feature_select_and_combine(
            True, config, index, feature_dict, labels, raw_score, args,
            train, test, float(best_c[1]), mask_id=f"outer_{index:03d}",
        )
        model = SVC(
            probability=True, kernel="linear", random_state=0,
            C=float(best_c[0]),
        ).fit(x[train], labels[train])
        pred = model.predict(x[test])
        decision = model.decision_function(x[test])
        calculated = {
            "test_accuracy": accuracy_score(labels[test], pred),
            "test_precision": precision_score(labels[test], pred, zero_division=0),
            "test_f1": f1_score(labels[test], pred, zero_division=0),
            "test_recall": recall_score(labels[test], pred, zero_division=0),
            "test_roc_auc": roc_auc_score(labels[test], decision),
        }
        differences = {key: abs(float(old[key]) - value) for key, value in calculated.items()}
        if max(differences.values()) > 1e-10:
            raise ValueError(f"Metric mismatch {task}, fold {index}: {differences}")

        saved_index = np.load(selected_path / f"Pearson_select_{index}.npz")
        local_pearson = np.asarray(saved_index["voxel_after_pearson"]).reshape(-1)
        local_l1 = np.asarray(saved_index["voxel_after_l1"]).reshape(-1)
        support = load_mask_support(mask_root, f"outer_{index:03d}", gray)
        active_global = np.flatnonzero(support)
        global_pearson = active_global[local_pearson]
        global_l1 = active_global[local_l1]
        pearson_count[global_pearson] += 1
        l1_count[global_l1] += 1

        fold_file = fold_dir / f"outer_{index:03d}.npz"
        np.savez_compressed(
            fold_file,
            train_index=train, test_index=test,
            pearson_global_index=global_pearson,
            l1_global_index=global_l1,
            model_coef=np.asarray(model.coef_).reshape(-1),
            best_C_model=float(best_c[0]), best_C_selector=float(best_c[1]),
        )
        fold_rows.append({
            "outer_index": index, "repeat": index // 10, "fold": index % 10,
            "mask_voxels": int(support.sum()),
            "pearson_voxels": len(global_pearson), "l1_voxels": len(global_l1),
            "max_metric_difference": max(differences.values()), **calculated,
        })

        for step, selected in (("pearson", global_pearson), ("l1", global_l1)):
            selected_labels = atlas_on_gray[selected]
            for target_name in TASKS[task]["targets"]:
                correlations = vector_correlations(
                    raw[np.ix_(train, selected)], targets[target_name][train]
                )
                for roi_id in np.unique(selected_labels):
                    if roi_id == 0:
                        continue
                    in_roi = selected_labels == roi_id
                    finite = np.isfinite(correlations[in_roi])
                    roi_values = correlations[in_roi][finite]
                    if not roi_values.size:
                        continue
                    roi_rows.append({
                        "outer_index": index, "repeat": index // 10,
                        "fold": index % 10, "step": step,
                        "target": target_name, "aal_id": int(roi_id),
                        "aal_label": atlas_names.get(int(roi_id), f"AAL_{roi_id}"),
                        "selected_voxels": int(in_roi.sum()),
                        "mean_voxel_correlation": float(roi_values.mean()),
                    })
        print(f"{task}: fold {index:03d} verified", flush=True)

    pd.DataFrame(fold_rows).to_csv(out / "fold_summary.csv", index=False)
    roi_frame = pd.DataFrame(roi_rows)
    roi_frame.to_csv(out / "foldwise_aal_correlations.csv", index=False)
    frequency_rows = []
    for step, counts in (("pearson", pearson_count), ("l1", l1_count)):
        for roi_id in sorted(set(atlas_on_gray[counts > 0])):
            if roi_id == 0:
                continue
            roi_mask = atlas_on_gray == roi_id
            selected_by_fold = []
            for index in range(100):
                indices = np.load(fold_dir / f"outer_{index:03d}.npz")[
                    f"{step}_global_index"
                ]
                selected_by_fold.append(bool(np.any(atlas_on_gray[indices] == roi_id)))
            frequency_rows.append({
                "step": step, "aal_id": int(roi_id),
                "aal_label": atlas_names.get(int(roi_id), f"AAL_{roi_id}"),
                "folds_selected": int(sum(selected_by_fold)),
                "selection_frequency": float(np.mean(selected_by_fold)),
                "unique_voxels_selected": int(np.sum(counts[roi_mask] > 0)),
                "max_voxel_frequency": float(counts[roi_mask].max() / 100),
            })
    frequency = pd.DataFrame(frequency_rows)
    frequency.to_csv(out / "aal_selection_frequency.csv", index=False)

    correlation_summary = (
        roi_frame.groupby(["step", "target", "aal_id", "aal_label"], as_index=False)
        .agg(
            folds_selected=("outer_index", "nunique"),
            mean_correlation=("mean_voxel_correlation", "mean"),
            p05=("mean_voxel_correlation", lambda x: np.quantile(x, 0.05)),
            p95=("mean_voxel_correlation", lambda x: np.quantile(x, 0.95)),
        )
    )
    correlation_summary.to_csv(out / "aal_correlation_summary.csv", index=False)
    save_gray_map(gray_img, gray, pearson_count / 100, out / "pearson_frequency.nii.gz")
    save_gray_map(gray_img, gray, l1_count / 100, out / "l1_frequency.nii.gz")
    np.savez_compressed(
        out / "voxel_frequency_arrays.npz",
        pearson_count=pearson_count, l1_count=l1_count,
    )
    complete_marker.write_text(
        '{"status":"complete","outer_folds":100,"metrics_verified":true}\n',
        encoding="utf-8",
    )
    print(f"{task}: complete; outputs in {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS)
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for task in args.tasks:
        run_task(task, args.overwrite)


if __name__ == "__main__":
    main()
