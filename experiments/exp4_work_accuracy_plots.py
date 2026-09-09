"""Plot the compact work--accuracy experiment from its saved CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from experiments._style import (
        color, grid, save, use_paper_style,
    )
except ModuleNotFoundError:
    from _style import (
        color, grid, save, use_paper_style,
    )

import matplotlib.pyplot as plt

def _rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def generate_plots(output_dir: str | Path, *, figure_dir=None) -> list[Path]:
    use_paper_style()
    root = Path(output_dir).expanduser().resolve()
    rows = _rows(root / "data" / "work_accuracy.csv")
    figure_dir = Path(figure_dir) if figure_dir else root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(3.5, 2.9))

    full = sorted(
        (row for row in rows if row["scheme"] == "full"),
        key=lambda row: float(row["work_units"]),
    )
    if full:
        work = np.array([float(r["work_units"]) for r in full])
        error = np.array([float(r["rms_error"]) for r in full])
        # The deterministic baseline is the reference the others are read
        # against, so it is drawn heavier and in neutral grey.
        axis.plot(work, error, color=color(7), linewidth=1.8, zorder=3, label="Full")
        axis.plot(work, error, "o", color=color(7), markeredgecolor="white", zorder=4)


    names = sorted(
        {row["scheme"] for row in rows if row["scheme"] != "full"},
        key=lambda name: int(name.split("r")[-1]),
    )
    for index, name in enumerate(names):
        selected = sorted(
            (row for row in rows if row["scheme"] == name),
            key=lambda row: float(row["work_units"]),
        )
        tint = color(index)
        work = np.array([float(r["work_units"]) for r in selected])
        error = np.array([float(r["rms_error"]) for r in selected])
        axis.plot(work, error, color=tint, zorder=3, label=rf"$r={selected[0]['batch_size']}$")
        axis.plot(work, error, "o", color=tint, markeredgecolor="white", zorder=4)


    axis.set(
        xscale="log", yscale="log",
        xlabel="Component evaluations per trajectory",
        ylabel="Terminal RMS error",
    )
    axis.set_xscale("log", base=2)
    grid(axis, which='major')
    return save(fig, figure_dir, "work_accuracy")
