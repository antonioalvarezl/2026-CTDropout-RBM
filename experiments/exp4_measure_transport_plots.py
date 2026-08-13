"""Plot saved measure-transport results without training or integration."""

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


def _rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _save(fig, directory, stem):
    outputs = []
    for suffix in ("pdf", "png"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _density_axis(axis, x, y, density, title):
    image = axis.imshow(
        density,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        aspect="equal",
        cmap="viridis",
    )
    axis.set_title(title)
    axis.set(xlabel="$x_1$", ylabel="$x_2$")
    return image


def generate_plots(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir).expanduser().resolve()
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    distributions = np.load(root / "data" / "distributions.npz")
    x, y = distributions["x_grid"], distributions["y_grid"]
    outputs = []

    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8), constrained_layout=True)
    first = _density_axis(
        axes[0], x, y, distributions["initial_density"], "Compact initial density"
    )
    second = _density_axis(
        axes[1], x, y, distributions["target_density"], "Balanced target mixture"
    )
    fig.colorbar(first, ax=axes[0], fraction=0.046)
    fig.colorbar(second, ax=axes[1], fraction=0.046)
    outputs += _save(fig, figure_dir, "initial_and_target")

    snapshots = distributions["full_snapshot_densities"]
    snapshot_times = distributions["snapshot_times"]
    fig, axes = plt.subplots(
        1,
        len(snapshot_times),
        figsize=(3.25 * len(snapshot_times), 3.2),
        constrained_layout=True,
    )
    for axis, density, time_value in zip(
        np.atleast_1d(axes), snapshots, snapshot_times
    ):
        _density_axis(axis, x, y, density, f"Full, $t={time_value:g}$")
    outputs += _save(fig, figure_dir, "full_snapshots")

    representative = json.loads((root / "data" / "representative.json").read_text())
    averaged = np.load(root / "data" / "averaged_densities.npz")
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8), constrained_layout=True)
    _density_axis(
        axes[0], x, y, distributions["full_terminal_density"], "Full density at $T$"
    )
    _density_axis(
        axes[1],
        x,
        y,
        averaged[representative["density_key"]],
        f"Mean RBM density at $T$\n{representative['scheme_label']}, $h={representative['h']:g}$",
    )
    outputs += _save(fig, figure_dir, "full_and_mean_rbm_density")

    rows = _rows(root / "data" / "terminal_errors.csv")
    schemes = list(dict.fromkeys(row["scheme"] for row in rows))
    colors = {name: plt.get_cmap("tab10")(i) for i, name in enumerate(schemes)}
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for scheme in schemes:
        selected = sorted(
            (row for row in rows if row["scheme"] == scheme),
            key=lambda row: float(row["h"]),
        )
        h = np.array([float(row["h"]) for row in selected])
        error = np.array([float(row["expected_l1_error_T"]) for row in selected])
        axis.plot(
            h, error, "o-", color=colors[scheme], label=selected[0]["scheme_label"]
        )
    anchor = min(rows, key=lambda row: float(row["h"]))
    h_values = np.array(sorted({float(row["h"]) for row in rows}))
    axis.plot(
        h_values,
        float(anchor["expected_l1_error_T"]) * np.sqrt(h_values / float(anchor["h"])),
        "k--",
        lw=1.2,
        label=r"Reference $h^{1/2}$",
    )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Switching interval $h$",
        ylabel=r"$\mathbb{E}_\omega\|\rho_T-\hat\rho_T\|_{L^1}$",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "terminal_l1_vs_h")

    refinement = _rows(root / "data" / "refinement.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for bandwidth in sorted({float(row["bandwidth"]) for row in refinement}):
        selected = sorted(
            (row for row in refinement if float(row["bandwidth"]) == bandwidth),
            key=lambda row: int(row["grid_size"]),
        )
        axes[0].plot(
            [int(row["grid_size"]) for row in selected],
            [float(row["l1_to_finest_same_bandwidth"]) for row in selected],
            "o-",
            label=f"bandwidth={bandwidth:g}",
        )
    finest = max(int(row["grid_size"]) for row in refinement)
    bandwidth_rows = [row for row in refinement if int(row["grid_size"]) == finest]
    axes[1].plot(
        [float(row["bandwidth"]) for row in bandwidth_rows],
        [float(row["l1_to_baseline_bandwidth"]) for row in bandwidth_rows],
        "o-",
    )
    axes[0].set(xlabel="Grid points per axis", ylabel="$L^1$ to finest grid")
    axes[1].set(xlabel="KDE bandwidth", ylabel="$L^1$ to baseline bandwidth")
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "grid_and_bandwidth_refinement")

    finest_h = min(float(row["h"]) for row in rows)
    selected = [row for row in rows if float(row["h"]) == finest_h]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    names = [row["scheme_label"] for row in selected]
    axes[0].bar(
        range(len(selected)), [float(row["expected_l1_error_T"]) for row in selected]
    )
    axes[1].bar(
        range(len(selected)),
        [float(row["mean_particle_coupling_error_T"]) for row in selected],
    )
    for axis in axes:
        axis.set_xticks(range(len(names)), names, rotation=20, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel(r"Terminal expected $L^1$ error")
    axes[1].set_ylabel(r"Terminal particle coupling RMS")
    fig.suptitle(f"Scheme comparison at $h={finest_h:g}$")
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "scheme_comparison")
    return outputs
