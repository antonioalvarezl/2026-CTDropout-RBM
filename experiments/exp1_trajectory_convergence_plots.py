"""Plots for trajectory convergence and fixed-partition sampling design.

Figures carry series identification and nothing else.  Marker conventions,
reference slopes and sample sizes belong in the LaTeX caption.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from experiments._style import (
        color, dyadic_ticks, grid, save, use_paper_style,
    )
except ModuleNotFoundError:
    from _style import (
        color, dyadic_ticks, grid, save, use_paper_style,
    )

import matplotlib.pyplot as plt

LABELS = {
    "uniform_fixed_r8": "Uniform",
    "fixed_contiguous_r8": "Contiguous",
    "bernoulli_q1_3": "Bernoulli",
}
ORDER = tuple(LABELS)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _series(rows, name):
    selected = sorted(
        (row for row in rows if row["scheme"] == name),
        key=lambda row: float(row["h"]),
    )
    if not selected:
        return None
    return {
        "h": np.asarray([float(r["h"]) for r in selected]),
        "error": np.asarray([float(r["mean_error"]) for r in selected]),
        "lower": np.asarray([float(r["ci95_lower"]) for r in selected]),
        "upper": np.asarray([float(r["ci95_upper"]) for r in selected]),
        "fit": np.asarray([_as_bool(r["fit_range"]) for r in selected]),
    }


def _trajectory_plot(rows, figure_dir: Path) -> list[Path]:
    figures_axes = [plt.subplots(figsize=(3.5, 2.9)) for _ in range(2)]
    axes = [pair[1] for pair in figures_axes]
    for index, name in enumerate(ORDER):
        s, tint = _series(rows, name), color(index)
        if s is None:
            continue
        for panel, axis in enumerate(axes):
            keep = s["fit"] if panel else np.ones(len(s["h"]), dtype=bool)
            h = s["h"][keep]
            scale = h if panel else np.ones(len(h))
            value, low, high = [s[k][keep] / scale for k in ("error", "lower", "upper")]
            axis.fill_between(h, low, high, color=tint, alpha=.13, lw=0)
            axis.plot(h, value, color=tint, label=LABELS[name])
            fine = s["fit"][keep]
            axis.plot(h[fine], value[fine], 'o', color=tint, markeredgecolor='white')
            axis.plot(h[~fine], value[~fine], 'o', mfc='white', mec=tint)
            axis.set_xscale('log')
            dyadic_ticks(axis, h)
    axes[0].set(yscale='log', ylabel=r'$\widehat{\mathcal{E}}_{\rm tr}(h)$')
    axes[1].set_ylabel(r'$\widehat{\mathcal{E}}_{\rm tr}(h)/h$')
    for axis in axes:
        axis.set_xlabel('Switching interval $h$')
        grid(axis, which='major')
    outputs = []
    for (fig, _), name in zip(figures_axes, ('trajectory_error', 'trajectory_rescaled')):
        outputs += save(fig, figure_dir, name)
    return outputs


def _partition_plot(rows, figure_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(3.5, 2.9))
    for kind, label, marker, tint in (
        ('random', 'Random partitions', 'o', color(0)),
        ('contiguous_baseline', 'Contiguous', 'D', color(1)),
    ):
        selected = [r for r in rows if r['kind'] == kind
                    and np.isfinite(float(r['error_over_h']))]
        axis.scatter([float(r['integrated_lambda']) for r in selected],
                     [float(r['error_over_h']) for r in selected],
                     marker=marker, color=tint, edgecolor='white', linewidth=.5,
                     s=27 if kind == 'random' else 42, label=label)
    axis.set_xlabel(r'Integrated variance $\overline{\Lambda}_{\mathcal{P}}$')
    axis.set_ylabel(r'$\widehat{\mathcal{E}}_{\rm tr}(h_{\rm probe})/h_{\rm probe}$')
    grid(axis, which='major')
    return save(fig, figure_dir, 'partition_variance_vs_error')


def generate_plots(output_dir: str | Path, *, figure_dir=None) -> list[Path]:
    use_paper_style()
    root = Path(output_dir).expanduser().resolve()
    figure_dir = Path(figure_dir) if figure_dir else root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    trajectory = _read(root / "data" / "trajectory_convergence.csv")
    partitions = _read(root / "data" / "partition_design.csv")

    outputs = []
    outputs += _trajectory_plot(trajectory, figure_dir)
    outputs += _partition_plot(partitions, figure_dir)
    return outputs
