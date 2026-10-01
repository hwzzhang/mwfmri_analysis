from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shared"))


class NestedSplitTest(unittest.TestCase):
    def test_outer_and_inner_masks_exclude_held_out_participants(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            ids = np.asarray([f"sub-{index:04d}" for index in range(20)])
            labels = np.asarray([0, 1] * 10)
            archive = tmp_path / "features.npz"
            np.savez(archive, id_list=ids, base_moca_label=labels)
            config = {
                "output_root": str(tmp_path / "outputs"),
                "gray_matter_mask": str(tmp_path / "gray.nii.gz"),
                "contrast_image_root": str(tmp_path / "contrasts"),
                "contrast_filename": "%s_con_0015.nii",
                "aal3_image": str(tmp_path / "aal3.nii"),
                "aal3_labels": str(tmp_path / "aal3.mat"),
                "post_movie_scores": str(tmp_path / "scores.csv"),
                "tasks": {
                    "classification": {
                        "tabular_feature_archive": str(archive),
                        "whole_brain_feature_archive": str(archive),
                        "label_key": "base_moca_label",
                    }
                },
            }
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch.dict(os.environ, {"MWFMRI_CONFIG": str(config_path)}):
                import settings
                settings.load_config.cache_clear()
                prepare = importlib.import_module("prepare_masks")
                output = tmp_path / "masks"
                manifest = prepare.build_manifest(
                    "classification", output, outer_repeats=1,
                    seed=0, outer_limit=1,
                )

            self.assertEqual(manifest["n_jobs"], 6)
            jobs = {job["mask_id"]: job for job in manifest["jobs"]}
            outer = jobs["outer_000"]
            with (output / "outer_splits.csv").open() as handle:
                split = next(csv.DictReader(handle))
            outer_test = set(json.loads(split["test_ids"]))
            outer_train = set(outer["train_ids"])
            self.assertTrue(outer_test.isdisjoint(outer_train))
            for index in range(5):
                inner_train = set(jobs[f"inner_000_{index}"]["train_ids"])
                self.assertLess(inner_train, outer_train)
                self.assertTrue(outer_test.isdisjoint(inner_train))


if __name__ == "__main__":
    unittest.main()
