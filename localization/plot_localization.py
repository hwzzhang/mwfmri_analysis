#!/usr/bin/env python3
"""Plot selection-frequency maps and regional summaries.

Post-L1 selection-frequency maps use task-specific cluster filtering: 50
voxels for baseline classification and 8 voxels for forecasting. No model is
refitted here.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import nibabel as nib
from nilearn.plotting import plot_glass_brain
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from settings import load_config, resolve_path  # noqa: E402

CONFIG = load_config()
DATA = resolve_path(CONFIG["output_root"]) / "localization"
OUTPUT_DIR = DATA / "plots"
OUTPUT_STEM = OUTPUT_DIR / "predictive_localization"
GRAY_MASK = resolve_path(CONFIG["gray_matter_mask"])

TASKS = {
    "classification": {
        "heading": (
            "Sources of features selected for\n"
            r"classifying $MoCA_{high}$ and $MoCA_{low}$"
        ),
        "panel_image": "A",
        "panel_table": "B",
        "cluster_extent": 50,
        # Fixed manuscript row order.
        "display_labels": [
            "Temporal_Mid_L",
            "Temporal_Mid_R",
            "Cerebellum_6_R",
            "Temporal_Sup_R",
            "Temporal_Sup_L",
            "Cerebellum_Crus1_R",
            "Cerebellum_Crus2_R",
        ],
        "targets": [
            ("baseline_moca", "Corr. with\nBaseline MoCA"),
            ("post_movie_test", "Corr. with\npost-movie test"),
        ],
    },
    "forecast": {
        "heading": (
            "Sources of features selected for\n"
            r"forecasting $\Delta MoCA_{high}$ and $\Delta MoCA_{low}$"
        ),
        "panel_image": "C",
        "panel_table": "D",
        "cluster_extent": 8,
        "display_labels": [
            "Precentral_R",
            "Frontal_Inf_Tri_L",
            "Frontal_Inf_Oper_L",
            "Cerebellum_6_L",
            "Temporal_Mid_L",
        ],
        "targets": [
            ("moca_change", "Corr. with\n$\\Delta$MoCA"),
            ("baseline_moca", "Corr. with\nBaseline MoCA"),
            ("post_movie_test", "Corr. with\npost-movie test"),
        ],
    },
}


CMAP = LinearSegmentedColormap.from_list(
    "selection_frequency",
    # Yellow-orange-red-black selection-frequency palette.
    ["#FFFF00", "#FFD700", "#FFA500", "#CD0000", "black"],
)


def format_correlation(row) -> str:
    """Format the mean and 5th--95th percentile range."""
    if row is None:
        return "—"
    return (
        f"{row.mean_correlation:.2f} "
        f"[{row.p05:.2f}, {row.p95:.2f}]"
    )


def cluster_filtered_l1_path(task: str, minimum_extent: int) -> Path:
    """Apply the 18-connected cluster-extent filter."""
    directory = DATA / task
    source = nib.load(directory / "l1_frequency.nii.gz")
    values = np.asarray(source.dataobj)
    components, _ = label(
        values > 0,
        structure=generate_binary_structure(rank=3, connectivity=2),
    )
    component_sizes = np.bincount(components.ravel())
    retained_components = np.flatnonzero(component_sizes >= minimum_extent)
    retained_components = retained_components[retained_components != 0]
    retained_voxels = np.isin(components, retained_components)

    output = directory / f"l1_frequency_cluster{minimum_extent}.nii.gz"
    filtered = np.where(retained_voxels, values, 0).astype(np.float32)
    nib.save(nib.Nifti1Image(filtered, source.affine, source.header), output)
    return output


def cluster_supported_aal_ids(task: str, minimum_extent: int) -> set[int]:
    """AAL3 labels overlapping retained L1 clusters."""
    directory = DATA / task
    filtered = np.asarray(
        nib.load(cluster_filtered_l1_path(task, minimum_extent)).dataobj
    ) > 0

    gray = np.asarray(nib.load(GRAY_MASK).dataobj) > 0.9
    aal_on_gray = np.load(directory / "aal_label_on_gray.npy")
    ids = set(np.unique(aal_on_gray[filtered[gray]]).astype(int))
    ids.discard(0)
    return ids


def load_table(
    task: str,
    targets: list[tuple[str, str]],
    cluster_extent: int,
    display_labels: list[str],
) -> tuple[list[str], list[list[str]]]:
    directory = DATA / task
    frequency = pd.read_csv(directory / "aal_selection_frequency.csv")
    # Summarize fold-wise ROI correlations using the 5th/95th percentiles.
    foldwise = pd.read_csv(directory / "foldwise_aal_correlations.csv")
    correlations = (
        foldwise.loc[foldwise["step"].eq("l1")]
        .groupby(["target", "aal_id", "aal_label"], as_index=False)
        .agg(
            mean_correlation=("mean_voxel_correlation", "mean"),
            p05=("mean_voxel_correlation", lambda x: x.quantile(0.05)),
            p95=("mean_voxel_correlation", lambda x: x.quantile(0.95)),
        )
    )
    supported_ids = cluster_supported_aal_ids(task, cluster_extent)
    l1_frequency = frequency.loc[frequency["step"].eq("l1")].copy()
    frequency_by_label = {
        row.aal_label: row for row in l1_frequency.itertuples()
    }
    lookup = {
        (int(row.aal_id), row.target): row
        for row in correlations.itertuples()
    }
    columns = ["AAL3 label", "Frequency\nof selection"] + [label for _, label in targets]
    rows = []
    for aal_label in display_labels:
        if aal_label not in frequency_by_label:
            raise ValueError(f"Missing AAL3 label for {task}: {aal_label}")
        row = frequency_by_label[aal_label]
        if int(row.aal_id) not in supported_ids:
            raise ValueError(
                f"Displayed AAL3 label does not overlap retained clusters: "
                f"{task}/{aal_label}"
            )
        values = [row.aal_label, f"{100 * row.selection_frequency:.0f}%"]
        values.extend(
            format_correlation(lookup.get((int(row.aal_id), target)))
            for target, _ in targets
        )
        rows.append(values)
    return columns, rows


def draw_brain_panel(
    fig,
    box,
    task: str,
    panel: str,
    cluster_extent: int,
) -> None:
    """Draw one post-L1 selection-frequency glass brain."""
    left, bottom, width, height = box
    title_y = bottom + height + 0.018
    fig.text(left - 0.018, title_y, panel, fontsize=17, fontweight="bold", va="center")
    fig.text(
        left + 0.006,
        title_y,
        "Selection frequency of each voxel in Step-2 feature selection",
        fontsize=12.5,
        va="center",
    )
    axis = fig.add_axes([left, bottom, width, height])
    plot_glass_brain(
        str(cluster_filtered_l1_path(task, cluster_extent)),
        figure=fig,
        axes=axis,
        display_mode="lyrz",
        plot_abs=False,
        symmetric_cbar=False,
        # The absolute scale is common to both tasks: 0 means never selected
        # and 1 means selected in all 100 outer folds.
        threshold=0,
        cmap=CMAP,
        colorbar=False,
        vmin=0.0,
        vmax=1.0,
    )
    color_axis = fig.add_axes(
        [left + width * 0.60, bottom + height + 0.002, width * 0.34, 0.010]
    )
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
        cax=color_axis,
        orientation="horizontal",
        ticks=[0, 1],
    )
    colorbar.ax.set_xticklabels(["0%", "100%"])
    colorbar.ax.tick_params(labelsize=9, length=0, pad=1)


def draw_table_panel(fig, box, task: str, spec: dict) -> None:
    left, bottom, width, height = box
    title_y = bottom + height + 0.018
    fig.text(
        left - 0.018,
        title_y,
        spec["panel_table"],
        fontsize=17,
        fontweight="bold",
        va="center",
    )
    fig.text(
        left + 0.006,
        title_y,
        "Primary sources of voxels selected in Step-2 feature selection",
        fontsize=12.5,
        va="center",
    )

    axis = fig.add_axes([left, bottom, width, height])
    axis.axis("off")
    columns, rows = load_table(
        task,
        spec["targets"],
        spec["cluster_extent"],
        spec["display_labels"],
    )
    if task == "classification":
        widths = [0.34, 0.18, 0.25, 0.25]
        font_size = 8.7
    else:
        widths = [0.25, 0.14, 0.21, 0.21, 0.21]
        font_size = 7.7
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        colWidths=widths,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.0)
        cell.PAD = 0.025
        if row == 0:
            cell.set_text_props(weight="normal")
            cell.set_linewidth(0.9)
            cell.visible_edges = "TB"
        elif row == len(rows):
            cell.set_linewidth(0.9)
            cell.visible_edges = "B"
    # Draw a rule directly below the panel heading.
    axis.plot(
        [0, 1], [1.01, 1.01],
        color="black", lw=1.0, clip_on=False, transform=axis.transAxes,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(18, 11.4), facecolor="white")

    x_positions = {"classification": 0.035, "forecast": 0.525}
    for task, spec in TASKS.items():
        left = x_positions[task]
        fig.text(left + 0.22, 0.965, spec["heading"], ha="center", va="top", fontsize=18)
        draw_brain_panel(
            fig,
            [left, 0.565, 0.44, 0.275],
            task,
            spec["panel_image"],
            spec["cluster_extent"],
        )
        draw_table_panel(fig, [left, 0.045, 0.44, 0.425], task, spec)

    fig.add_artist(
        mpl.lines.Line2D(
            [0.5, 0.5], [0.035, 0.96],
            transform=fig.transFigure,
            color="black",
            linewidth=1.2,
        )
    )

    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(OUTPUT_STEM.with_suffix(".pdf"))
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".tiff"))


if __name__ == "__main__":
    main()
