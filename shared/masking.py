"""Map whole-gray-matter feature arrays to fold-specific SPM masks.

The task-specific modeling modules remain responsible for Pearson screening,
scaling, L1 selection, hyperparameter tuning, and the final SVC.
"""

from pathlib import Path
from typing import Dict
import json

import nibabel as nib
import numpy as np

from settings import load_config, resolve_path


GRAY_MASK_PATH = resolve_path(load_config()["gray_matter_mask"])

_SUPPORT_CACHE: Dict[str, np.ndarray] = {}
_MANIFEST_CACHE = {}


def apply_fold_mask(raw_gray_features, mask_id, mask_root, expected_train_ids=None):
    """Restrict a whole-gray-matter matrix to one saved SPM mask.

    Parameters
    ----------
    raw_gray_features : array, shape (participants, 183255)
        The existing ``*_whole_brain/baseline_features.npz`` mw-fMRI block.
    mask_id : str
        ``outer_XXX`` or ``inner_XXX_Y`` from ``prepare_masks.py``.
    mask_root : path-like
        Directory containing one subdirectory per mask ID.
    """
    if not mask_id:
        raise ValueError("mask_id is required for fold-specific masking")

    mask_root = Path(mask_root).resolve()
    mask_path = mask_root / mask_id / "spmT_0001_fold_mask.nii.gz"
    if not mask_path.exists():
        mask_path = mask_root / mask_id / "spmT_0001_fold_mask.nii"
    if not mask_path.exists():
        raise FileNotFoundError(
            "Fold mask is missing: {}. Run prepare_masks.py and then "
            "make_fold_masks.m before running the models.".format(mask_path)
        )

    # Fail closed if a mask was accidentally generated for a different split.
    manifest_path = mask_root / "mask_manifest.json"
    if expected_train_ids is not None:
        if not manifest_path.exists():
            raise FileNotFoundError("Mask provenance manifest is missing: {}".format(manifest_path))
        jobs = _MANIFEST_CACHE.get(str(manifest_path))
        if jobs is None:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            jobs = {job["mask_id"]: job for job in manifest["jobs"]}
            _MANIFEST_CACHE[str(manifest_path)] = jobs
        if mask_id not in jobs:
            raise ValueError("Mask ID is not present in its provenance manifest: {}".format(mask_id))
        expected = [str(value) for value in expected_train_ids]
        recorded = [str(value) for value in jobs[mask_id]["train_ids"]]
        if expected != recorded:
            raise ValueError("Training IDs do not match the saved mask provenance: {}".format(mask_id))

    cache_key = str(mask_path)
    support = _SUPPORT_CACHE.get(cache_key)
    if support is None:
        gray_img = nib.load(str(GRAY_MASK_PATH))
        fold_img = nib.load(str(mask_path))
        if fold_img.shape != gray_img.shape or not np.allclose(
            fold_img.affine, gray_img.affine
        ):
            raise ValueError("Fold mask and gray-matter feature grid do not match")
        gray = np.asarray(gray_img.dataobj) > 0.9
        # SPM's binary result export is exactly 0/1.  The 0.9 threshold mirrors
        # the NiftiMasker feature-extraction threshold.
        support = (np.asarray(fold_img.dataobj) > 0.9)[gray]
        support.setflags(write=False)
        _SUPPORT_CACHE[cache_key] = support

    features = np.asarray(raw_gray_features)
    if features.ndim != 2 or features.shape[1] != support.size:
        raise ValueError(
            "Expected whole-gray-matter features with {} columns, got {}".format(
                support.size, features.shape
            )
        )
    if not np.any(support):
        raise ValueError("SPM fold mask contains no gray-matter voxels: {}".format(mask_id))
    return features[:, support]
