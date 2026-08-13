"""Plot experiment 5 solely from saved CSV and JSON summaries."""

from __future__ import annotations

import csv
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


def generate_plots(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir).expanduser().resolve()
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    consistency = sorted(
        _rows(root / "data" / "fixed_control_consistency.csv"),
        key=lambda row: float(row["h"]),
    )
    slopes = {row["metric"]: row for row in _rows(root / "data" / "slope_fits.csv")}
    outputs = []

    h = np.array([float(row["h"]) for row in consistency])
    strong = np.array([float(row["strong_mse"]) for row in consistency])
    strong_lo = np.array([float(row["strong_ci95_lower"]) for row in consistency])
    strong_hi = np.array([float(row["strong_ci95_upper"]) for row in consistency])
    weak = np.array([float(row["weak_bias"]) for row in consistency])
    weak_lo = np.array([float(row["weak_abs_ci95_lower"]) for row in consistency])
    weak_hi = np.array([float(row["weak_abs_ci95_upper"]) for row in consistency])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].errorbar(
        h,
        strong,
        yerr=[strong - strong_lo, strong_hi - strong],
        fmt="o-",
        capsize=3,
        label="Strong MSE",
    )
    axes[1].errorbar(
        h,
        weak,
        yerr=[weak - weak_lo, weak_hi - weak],
        fmt="o-",
        capsize=3,
        label="Weak bias",
    )
    for axis, values, title in zip(
        axes,
        (strong, weak),
        (r"$K^{-1}\sum_k|\hat J_h^{(k)}-J|^2$", r"$|K^{-1}\sum_k\hat J_h^{(k)}-J|$"),
    ):
        anchor = int(np.flatnonzero(values > 0)[0])
        axis.plot(
            h, values[anchor] * h / h[anchor], "k--", lw=1, label="Reference slope 1"
        )
        axis.set(
            xscale="log", yscale="log", xlabel="Switching interval $h$", ylabel=title
        )
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    weak_fit = slopes["weak"]
    axes[1].text(
        0.03,
        0.04,
        (
            f"fit slope={float(weak_fit['slope']):.2f}"
            if weak_fit["fit_performed"] == "True"
            else "No weak slope fit: bias CI includes zero"
        ),
        transform=axes[1].transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "fixed_control_consistency")

    ensemble = _rows(root / "data" / "ensemble_averaging.csv")
    h_values = sorted({float(row["h"]) for row in ensemble})
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(h_values)))
    fig, axis = plt.subplots(figsize=(6.2, 4.3))
    for color, h_value in zip(colors, h_values):
        selected = sorted(
            (row for row in ensemble if float(row["h"]) == h_value),
            key=lambda row: int(row["M"]),
        )
        axis.plot(
            [int(row["M"]) for row in selected],
            [float(row["mse"]) for row in selected],
            "o-",
            color=color,
            label=f"h={h_value:g}",
        )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Ensemble size $M$",
        ylabel=r"$\mathrm{E}|\hat J_{h,M}-J|^2$",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "ensemble_mse_vs_M")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for color, h_value in zip(colors, h_values):
        selected = sorted(
            (row for row in ensemble if float(row["h"]) == h_value),
            key=lambda row: int(row["M"]),
        )
        M = [int(row["M"]) for row in selected]
        axes[0].plot(
            M,
            [float(row["variance"]) for row in selected],
            "o-",
            color=color,
            label=f"h={h_value:g}",
        )
        axes[1].plot(
            M, [float(row["bias_squared"]) for row in selected], "o-", color=color
        )
    for axis, label in zip(axes, ("Variance", "Bias squared")):
        axis.set(xscale="log", yscale="log", xlabel="Ensemble size $M$", ylabel=label)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "ensemble_variance_and_bias")

    observed = np.array([float(row["mse"]) for row in ensemble])
    fitted = np.array([float(row["fitted_mse"]) for row in ensemble])
    fig, axis = plt.subplots(figsize=(5.3, 4.5))
    axis.scatter(
        fitted, observed, c=[float(row["h"]) for row in ensemble], cmap="viridis"
    )
    limits = [min(observed.min(), fitted.min()), max(observed.max(), fitted.max())]
    axis.plot(limits, limits, "k--", lw=1)
    axis.set(
        xscale="log", yscale="log", xlabel=r"Fitted $ah/M+bh^2$", ylabel="Observed MSE"
    )
    axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "ensemble_model_fit")

    training = _rows(root / "data" / "training_evaluation_summary.csv")
    selected = [
        row
        for row in training
        if row["split"] == "test" and row["primary_evaluation"] == "True"
    ]
    labels = [row["condition_label"] for row in selected]
    x = np.arange(len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    axes[0].plot(
        x,
        [float(row["full_objective"]) for row in selected],
        "o",
        label=r"$J(\vartheta_h)$",
    )
    axes[0].plot(
        x,
        [float(row["mean_random_objective"]) for row in selected],
        "s",
        label=r"$\overline{J}_h(\vartheta_h)$",
    )
    axes[1].plot(
        x, [float(row["full_accuracy"]) for row in selected], "o", label="Full accuracy"
    )
    axes[1].plot(
        x,
        [float(row["mean_random_accuracy"]) for row in selected],
        "s",
        label="Mean random accuracy",
    )
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Test objective value")
    axes[1].set_ylabel("Nearest-target test accuracy")
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "trained_control_objectives")

    runs = _rows(root / "data" / "training_runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    run_labels = [row["condition_label"] for row in runs]
    axes[0].bar(range(len(runs)), [float(row["wall_seconds"]) for row in runs])
    axes[1].bar(range(len(runs)), [float(row["neuron_evaluations"]) for row in runs])
    for axis in axes:
        axis.set_xticks(range(len(runs)), run_labels, rotation=25, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Whole training run wall-clock (s)")
    axes[1].set_ylabel("Actual neuron-field evaluations")
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "training_cost")
    return outputs
