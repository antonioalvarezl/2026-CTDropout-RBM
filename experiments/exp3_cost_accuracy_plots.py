"""Plot saved cost--accuracy summaries without recomputing trajectories."""

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
    paths = []
    for suffix in ("pdf", "png"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def generate_plots(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir).expanduser().resolve()
    curves = _rows(root / "data" / "cost_accuracy.csv")
    tolerances = _rows(root / "data" / "tolerance_comparison.csv")
    crossover = json.loads((root / "data" / "crossover.json").read_text())
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    schemes = list(dict.fromkeys(row["scheme"] for row in curves))
    colors = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(schemes)}
    labels = {row["scheme"]: row["scheme_label"] for row in curves}
    outputs = []

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for scheme in schemes:
        selected = sorted(
            (row for row in curves if row["scheme"] == scheme),
            key=lambda row: float(row["h"]),
        )
        h = np.array([float(row["h"]) for row in selected])
        error = np.array([float(row["rms_error"]) for row in selected])
        bound = np.array([float(row["calibrated_bound"]) for row in selected])
        axes[0].plot(h, error, "o-", color=colors[scheme], label=labels[scheme])
        axes[1].plot(h, bound, "--", color=colors[scheme])
        axes[1].plot(h, error, "o", ms=4, color=colors[scheme], label=labels[scheme])
    for axis in axes:
        axis.set(xscale="log", yscale="log", xlabel="Switching interval $h$")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"Total RMS error $\mathcal{E}_{\mathrm{R}}(h)$")
    axes[1].set_ylabel("RMS error (points) and calibrated bound (dashes)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "error_and_calibrated_bound")

    fig, axis = plt.subplots(figsize=(6.4, 4.5))
    for scheme in schemes:
        selected = [row for row in tolerances if row["scheme"] == scheme]
        predicted = np.array([float(row["predicted_h"]) for row in selected])
        observed = np.array([float(row["observed_h"]) for row in selected])
        valid = np.isfinite(predicted) & np.isfinite(observed)
        axis.plot(
            predicted[valid],
            observed[valid],
            "o-",
            color=colors[scheme],
            label=labels[scheme],
        )
    finite = [
        float(row[key])
        for row in tolerances
        for key in ("predicted_h", "observed_h")
        if np.isfinite(float(row[key])) and float(row[key]) > 0
    ]
    if finite:
        limits = [min(finite), max(finite)]
        axis.plot(limits, limits, "k--", lw=1, label="predicted = observed")
    axis.set(
        xscale="log",
        yscale="log",
        xlabel=r"Predicted $h^\star$",
        ylabel=r"Observed cheapest feasible $h$",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "predicted_vs_observed_hstar")

    fig, axis = plt.subplots(figsize=(6.6, 4.6))
    for scheme in schemes:
        selected = [row for row in tolerances if row["scheme"] == scheme]
        epsilon = np.array([float(row["tolerance"]) for row in selected])
        cost = np.array([float(row["observed_min_proxy_cost"]) for row in selected])
        valid = np.isfinite(cost)
        axis.plot(
            epsilon[valid],
            cost[valid],
            "o-",
            color=colors[scheme],
            label=labels[scheme],
        )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel=r"Tolerance $\varepsilon$",
        ylabel="Minimum theoretical proxy cost",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "minimum_cost_vs_tolerance")

    optimum = [row for row in tolerances if row["empirical_global_optimum"] == "True"]
    fig, axis = plt.subplots(figsize=(6.5, 3.6))
    if optimum:
        ordered = sorted(optimum, key=lambda row: float(row["tolerance"]))
        names = list(dict.fromkeys(row["scheme_label"] for row in ordered))
        indices = [names.index(row["scheme_label"]) for row in ordered]
        axis.step([float(row["tolerance"]) for row in ordered], indices, where="mid")
        axis.set_yticks(range(len(names)), names)
    axis.set_xscale("log")
    axis.set_xlabel(r"Tolerance $\varepsilon$")
    axis.set_ylabel("Empirically cheapest scheme")
    axis.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "empirical_optimal_scheme")

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    grouped = {}
    for row in tolerances:
        grouped.setdefault(float(row["tolerance"]), []).append(row)
    x, observed_ratio, predicted_ratio = [], [], []
    for epsilon, rows in sorted(grouped.items()):
        full = next(row for row in rows if row["scheme"] == "full")
        rbm = [row for row in rows if row["scheme"] != "full"]
        observed = [
            float(row["observed_min_proxy_cost"])
            for row in rbm
            if np.isfinite(float(row["observed_min_proxy_cost"]))
        ]
        predicted = [float(row["predicted_proxy_cost"]) for row in rbm]
        full_observed = float(full["observed_min_proxy_cost"])
        full_predicted = float(full["predicted_proxy_cost"])
        x.append(epsilon)
        observed_ratio.append(
            min(observed) / full_observed
            if observed and np.isfinite(full_observed)
            else np.nan
        )
        predicted_ratio.append(min(predicted) / full_predicted)
    axis.plot(x, predicted_ratio, "--", label="Predicted best-RBM/full")
    axis.plot(x, observed_ratio, "o-", label="Observed best-RBM/full")
    axis.axhline(1.0, color="k", lw=1)
    if not crossover["observed"]["within_measured_range"]:
        axis.text(
            0.03,
            0.05,
            "No observed crossover in measured range",
            transform=axis.transAxes,
        )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel=r"Tolerance $\varepsilon$",
        ylabel="Cost ratio",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    outputs += _save(fig, figure_dir, "crossover")
    return outputs
