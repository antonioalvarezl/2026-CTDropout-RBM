"""Plotting only for the trajectory-convergence experiment."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rnode-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rnode-xdg-cache")

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "uniform_fixed_r8": "Uniform fixed-size, $r=8$",
    "fixed_disjoint_r8": "Fixed disjoint, $r=8$",
    "bernoulli_q1_3": "Bernoulli, $q=1/3$",
    "full_batch": "Full-batch control",
}


def _read_rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def generate_plots(output_dir: str | Path) -> list[Path]:
    """Regenerate PDF and PNG figures solely from saved raw summaries."""
    root = Path(output_dir).expanduser().resolve()
    rows = _read_rows(root / "data" / "trajectory_convergence.csv")
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(6.3, 4.4))
    colors = plt.get_cmap("tab10")
    ordered_schemes = [
        "uniform_fixed_r8",
        "fixed_disjoint_r8",
        "bernoulli_q1_3",
        "full_batch",
    ]
    for color_index, scheme in enumerate(ordered_schemes):
        selected = sorted(
            (row for row in rows if row["scheme"] == scheme),
            key=lambda row: float(row["h"]),
        )
        if not selected:
            continue
        h = np.array([float(row["h"]) for row in selected])
        estimate = np.array([float(row["mean_error"]) for row in selected])
        lower = np.array([float(row["ci95_lower"]) for row in selected])
        upper = np.array([float(row["ci95_upper"]) for row in selected])
        axis.plot(
            h,
            estimate,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=colors(color_index),
            label=LABELS.get(scheme, scheme),
        )
        axis.fill_between(h, lower, upper, alpha=0.18, color=colors(color_index))

    stochastic = [row for row in rows if row["scheme"] == "uniform_fixed_r8"]
    if stochastic:
        anchor = min(stochastic, key=lambda row: float(row["h"]))
        h_anchor = float(anchor["h"])
        error_anchor = float(anchor["mean_error"])
        all_h = np.array(sorted({float(row["h"]) for row in rows}))
        axis.plot(
            all_h,
            error_anchor * all_h / h_anchor,
            linestyle="--",
            color="0.25",
            linewidth=1.2,
            label="Reference slope 1",
        )

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Switching interval $h$")
    axis.set_ylabel(r"$\widehat E(h)$")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    outputs = []
    for suffix in ("pdf", "png"):
        path = figure_dir / f"trajectory_convergence.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs
