from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))


class MaskProvenanceTest(unittest.TestCase):
    def test_mask_provenance_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            gray_path = tmp_path / "gray.nii.gz"
            gray = np.ones((2, 2, 2), dtype=np.uint8)
            nib.save(nib.Nifti1Image(gray, np.eye(4)), gray_path)
            config = {
                "output_root": str(tmp_path / "outputs"),
                "gray_matter_mask": str(gray_path),
                "tasks": {},
            }
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch.dict(os.environ, {"MWFMRI_CONFIG": str(config_path)}):
                import settings
                settings.load_config.cache_clear()
                module = importlib.import_module("masking")
                module = importlib.reload(module)

                mask_root = tmp_path / "masks"
                fold = mask_root / "outer_000"
                fold.mkdir(parents=True)
                nib.save(
                    nib.Nifti1Image(gray, np.eye(4)),
                    fold / "spmT_0001_fold_mask.nii.gz",
                )
                (mask_root / "mask_manifest.json").write_text(json.dumps({
                    "jobs": [{
                        "mask_id": "outer_000",
                        "train_ids": ["sub-0001", "sub-0002"],
                    }]
                }), encoding="utf-8")
                x = np.arange(24).reshape(3, 8)
                selected = module.apply_fold_mask(
                    x, "outer_000", mask_root, ["sub-0001", "sub-0002"],
                )
                self.assertEqual(selected.shape, (3, 8))
                with self.assertRaisesRegex(ValueError, "Training IDs do not match"):
                    module.apply_fold_mask(
                        x, "outer_000", mask_root,
                        ["sub-0001", "sub-9999"],
                    )


if __name__ == "__main__":
    unittest.main()
