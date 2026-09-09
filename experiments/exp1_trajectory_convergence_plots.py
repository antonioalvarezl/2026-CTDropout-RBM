"""Plots for trajectory convergence and fixed-partition sampling design.

Series are labelled directly at the end of each curve, in the curve's own
colour, so the reader never has to travel to a legend and back.  Marker
conventions, reference slopes and sample sizes belong in the LaTeX caption.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from experiments._style import (
        color, dyadic_ticks, grid, headroom, save, stack_labels,
        use_paper_style,
    )
except ModuleNotFoundError:
    from _style import (
        color, dyadic_ticks, grid, headroom, save, stack_labels,
        use_paper_style,
    )

import matplotlib.pyplot as plt

LABELS = {
    "uniform_fixed_r8": r"Uniform, $r=8$",
    "fixed_contiguous_r8": r"Contiguous, $r=8$",
    "bernoulli_q1_3": r"Bernoulli, $q=1/3$",
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
    """Error and its rescaling by ``h``, on one axes pair."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
    labels = ([], [])
    for index, name in enumerate(ORDER):
        s, tint = _series(rows, name), color(index)
        if s is None:
            continue
        for panel, axis in enumerate(axes):
            # The rescaled panel is only meaningful on the predeclared fine
            # range, so it shows that range alone.
            keep = s["fit"] if panel else np.ones(len(s["h"]), dtype=bool)
            h, fine = s["h"][keep], s["fit"][keep]
            scale = h if panel else np.ones(len(h))
            value, low, high = [s[k][keep] / scale for k in ("error", "lower", "upper")]
            axis.fill_between(h, low, high, color=tint, alpha=.13, lw=0)
            axis.plot(h, value, color=tint)
            # Filled markers are the predeclared fine range, open ones the
            # coarser diagnostic scales.
            axis.plot(h[fine], value[fine], 'o', color=tint, markeredgecolor='white')
            axis.plot(h[~fine], value[~fine], 'o', mfc='white', mec=tint)
            labels[panel].append((value[-1], LABELS[name], tint))
    axes[0].set(yscale='log', ylabel=r'$\widehat{\mathcal{E}}_{\rm tr}(h)$')
    axes[1].set_ylabel(r'$\widehat{\mathcal{E}}_{\rm tr}(h)/h$')
    every = sorted({float(r["h"]) for r in rows})
    fine = sorted({float(r["h"]) for r in rows if _as_bool(r["fit_range"])})
    for panel, axis in enumerate(axes):
        axis.set_xscale('log')
        axis.set_xlabel('Switching interval $h$')
        dyadic_ticks(axis, np.asarray(fine if panel else every))
        grid(axis)
        headroom(axis, right=.42, top=.06)
        stack_labels(axis, labels[panel], (fine if panel else every)[-1])
    return save(fig, figure_dir, 'trajectory_summary')


def _partition_plot(rows, figure_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(4.2, 3.0))
    for kind, label, marker, tint in (
        ('random', 'Random partitions', 'o', color(0)),
        ('contiguous_baseline', 'Contiguous', 'D', color(1)),
    ):
        selected = [r for r in rows if r['kind'] == kind
                    and np.isfinite(float(r['error_over_h']))]
        axis.scatter([float(r['integrated_lambda']) for r in selected],
                     [float(r['error_over_h']) for r in selected],
                     marker=marker, color=tint, edgecolor='white', linewidth=.5,
                     s=27 if kind == 'random' else 46, label=label, zorder=3)
    axis.set_xlabel(r'Integrated variance $\overline{\Lambda}_{\mathcal{P}}$')
    axis.set_ylabel(r'$\widehat{\mathcal{E}}_{\rm tr}(h_{\rm probe})/h_{\rm probe}$')
    grid(axis)
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
