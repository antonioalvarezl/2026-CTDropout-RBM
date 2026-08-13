"""Plotting only for the balanced-partition design experiment."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rnode-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rnode-xdg-cache")

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "optimized": "tab:green",
    "adversarial": "tab:red",
    "baseline": "tab:blue",
    "random_median": "0.35",
}


def _read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _save_both(fig, directory: Path, stem: str) -> list[Path]:
    outputs = []
    for suffix in ("pdf", "png"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def generate_plots(output_dir: str | Path) -> list[Path]:
    """Regenerate every figure from CSV/JSON results without simulation."""
    root = Path(output_dir).expanduser().resolve()
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    partition_rows = _read_csv(root / "data" / "partitions.csv")
    scatter_rows = _read_csv(root / "data" / "lambda_vs_error.csv")
    curve_rows = _read_csv(root / "data" / "design_error_curves.csv")
    correlations = json.loads((root / "data" / "correlations.json").read_text())
    outputs = []

    random_lambda = np.array(
        [
            float(row["lambda_test"])
            for row in partition_rows
            if row["category"] == "random"
        ]
    )
    fig, axis = plt.subplots(figsize=(5.4, 4.2))
    axis.boxplot(
        [random_lambda],
        positions=[1],
        widths=0.42,
        showfliers=True,
        patch_artist=True,
        boxprops={"facecolor": "0.85", "edgecolor": "0.35"},
        medianprops={"color": "black"},
    )
    marker_map = {"optimized": "D", "adversarial": "X", "baseline": "o"}
    for category in marker_map:
        row = next(row for row in partition_rows if row["category"] == category)
        axis.scatter(
            [1],
            [float(row["lambda_test"])],
            marker=marker_map[category],
            s=65,
            color=COLORS[category],
            label=category.capitalize(),
            zorder=4,
        )
    axis.set_xticks([1], ["200 random partitions"])
    axis.set_ylabel(r"$\widehat\Lambda_{\rm test}$")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    outputs += _save_both(fig, figure_dir, "partition_lambda_boxplot")

    fig, axis = plt.subplots(figsize=(5.8, 4.4))
    for category in ("random", "baseline", "optimized", "adversarial"):
        selected = [row for row in scatter_rows if row["category"] == category]
        if not selected:
            continue
        axis.scatter(
            [float(row["lambda_test"]) for row in selected],
            [float(row["error_over_h"]) for row in selected],
            s=18 if category == "random" else 58,
            alpha=0.45 if category == "random" else 0.95,
            color="0.55" if category == "random" else COLORS[category],
            marker="." if category == "random" else marker_map[category],
            label=category.capitalize(),
        )
    axis.set_xlabel(r"$\widehat\Lambda_{\rm test}$")
    axis.set_ylabel(r"$\widehat E(h)/h$")
    axis.text(
        0.03,
        0.97,
        f"Pearson $r$={correlations['pearson_r']:.3f}\n"
        f"Spearman $\\rho$={correlations['spearman_rho']:.3f}",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
    )
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    outputs += _save_both(fig, figure_dir, "lambda_vs_error")

    fig, axis = plt.subplots(figsize=(5.8, 4.4))
    labels = {
        "optimized": "Optimized",
        "random_median": "Median random",
        "adversarial": "Adversarial",
    }
    for category in ("optimized", "random_median", "adversarial"):
        selected = sorted(
            (row for row in curve_rows if row["category"] == category),
            key=lambda row: float(row["h"]),
        )
        h = np.array([float(row["h"]) for row in selected])
        estimate = np.array([float(row["mean_error"]) for row in selected])
        lower = np.array([float(row["ci95_lower"]) for row in selected])
        upper = np.array([float(row["ci95_upper"]) for row in selected])
        axis.plot(
            h,
            estimate,
            marker="o",
            linewidth=1.8,
            color=COLORS[category],
            label=labels[category],
        )
        axis.fill_between(h, lower, upper, color=COLORS[category], alpha=0.16)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Switching interval $h$")
    axis.set_ylabel(r"$\widehat E(h)$")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    outputs += _save_both(fig, figure_dir, "designed_partition_convergence")
    return outputs
