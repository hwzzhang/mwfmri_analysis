"""Configuration for two-year cognitive-change forecasting."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from settings import load_config, resolve_path, task_config  # noqa: E402


C_LIST = [
    [[0.1, 1, 5, 10], 0.1],
    [[0.1, 1, 5, 10], 1],
    [[0.1, 1, 5, 10], 5],
    [[0.1, 1, 5, 10], 10],
]

BNA_index_Hippocampus = [215, 216, 217, 218]
BNA_index_parahippocampal = list(range(109, 121))
BNA_index_overlap_speech_mask = [
    88, 79, 87, 107, 123, 80, 192, 121, 73, 78, 75, 122, 105, 143, 56,
    191, 25, 74, 55, 205, 195, 198, 76, 84, 53, 77, 17, 197, 1, 29,
    206, 71, 9, 189, 133, 63, 72, 85, 83, 33, 199, 54, 153, 196, 31,
    209, 227, 10, 127, 58, 137, 156, 128, 190, 26, 2, 141, 64, 134,
    221, 18, 51, 129, 175, 91, 200, 37, 124, 70, 231, 135, 7, 138,
    203, 69, 81, 155, 204, 62, 154, 157, 210, 239, 241, 245,
]

_cfg = load_config()
_task = task_config("forecast")
TABULAR_FEATURE_DIR = str(resolve_path(_task["tabular_feature_archive"]).parent)
WHOLE_BRAIN_FEATURE_DIR = str(resolve_path(_task["whole_brain_feature_archive"]).parent)
SAVE_RESULT_DIR = str(resolve_path(_cfg["output_root"]) / "forecast")
SAVE_FEATURE_DIR = str(resolve_path(_cfg["output_root"]) / "selected_features" / "forecast")
FOLD_MASK_DIR = str(resolve_path(_cfg["output_root"]) / "fold_masks" / "forecast")
Tmap_GRAY_MASK = str(resolve_path(_cfg["gray_matter_mask"]))
