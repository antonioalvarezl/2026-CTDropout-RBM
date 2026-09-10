"""Plot the compact work--accuracy experiment from its saved CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from experiments._style import (
        GUIDE, annotate, color, grid, headroom, save, stack_labels, use_paper_style,
    )
except ModuleNotFoundError:
    from _style import (
        GUIDE, annotate, color, grid, headroom, save, stack_labels, use_paper_style,
    )

import matplotlib.pyplot as plt

# The two Euler steps share a colour per batch size and are told apart by the
# stroke, so the reader compares r across panels and dt within a colour.
STEP_STYLES = {4: ("-", r"$\Delta t=h/4$"), 2: ((0, (4, 2)), r"$\Delta t=h/2$")}


def _rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _column(rows, key):
    return np.array([float(row[key]) for row in rows])


def _draw(axis, work, value, tint, style, *, width=1.4, zorder=3):
    axis.plot(work, value, color=tint, linestyle=style, linewidth=width, zorder=zorder)
    axis.plot(work, value, "o", color=tint, markeredgecolor="white", zorder=zorder + 1)


def generate_plots(output_dir: str | Path, *, figure_dir=None) -> list[Path]:
    use_paper_style()
    root = Path(output_dir).expanduser().resolve()
    rows = _rows(root / "data" / "work_accuracy.csv")
    reference = _rows(root / "data" / "reference_check.csv")
    figure_dir = Path(figure_dir) if figure_dir else root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1), sharex=True)
    labels = {0: [], 1: []}

    full = sorted(
        (row for row in rows if row["scheme"] == "full"),
        key=lambda row: float(row["work_units"]),
    )
    if full:
        work = _column(full, "work_units")
        # The deterministic baseline is what the others are read against, so it
        # is drawn heavier and in neutral grey.
        for axis, key in ((axes[0], "rms_error"), (axes[1], "test_loss")):
            value = _column(full, key)
            _draw(axis, work, value, color(7), "-", width=1.8)
        for panel, key in ((0, "rms_error"), (1, "test_loss")):
            labels[panel].append(
                (work[-1], _column(full, key)[-1], "full model", color(7), "bold")
            )

    # The coarsest randomized configurations diverge by orders of magnitude and
    # would flatten everything else. Each panel is cut at the crudest
    # full-model step: above that a configuration is worse than every
    # deterministic option, so the comparison there is already decided.
    cutoff = {panel: 8 * _column(full, key).max() if full else np.inf
              for panel, key in ((0, "rms_error"), (1, "test_loss"))}

    batch_sizes = sorted({int(row["batch_size"]) for row in rows
                          if row["scheme"] != "full"})
    for index, r in enumerate(batch_sizes):
        tint = color(index)
        for inverse_gamma, (style, _) in STEP_STYLES.items():
            selected = sorted(
                (row for row in rows if row["scheme"] != "full"
                 and int(row["batch_size"]) == r
                 and abs(float(row["gamma"]) - 1.0 / inverse_gamma) < 1e-12),
                key=lambda row: float(row["work_units"]),
            )
            if not selected:
                continue
            work = _column(selected, "work_units")
            _draw(axes[0], work, _column(selected, "rms_error"), tint, style)

            loss = _column(selected, "test_loss")
            se = _column(selected, "se_test_loss")
            # A diverged configuration carries a band taller than the panel;
            # drawn, it would read as a coloured column rather than as
            # uncertainty, so the band stops where its curve leaves the view.
            on_panel = loss <= cutoff[1]
            axes[1].fill_between(work, loss - 1.96 * se, loss + 1.96 * se,
                                 where=on_panel, color=tint, alpha=.13, lw=0)
            _draw(axes[1], work, loss, tint, style)
            # Only the finer step is labelled: both steps share the colour, and
            # it is the curve that reaches furthest right.
            if inverse_gamma == 4:
                labels[0].append((work[-1], _column(selected, "rms_error")[-1],
                                  rf"$r={r}$", tint, "normal"))
                labels[1].append((work[-1], loss[-1], rf"$r={r}$", tint, "normal"))

    if reference:
        baseline = float(reference[0]["test_loss"])
        span = [min(float(row["work_units"]) for row in rows),
                max(float(row["work_units"]) for row in rows)]
        # Stopping at the last budget keeps the rule clear of the end labels.
        axes[1].plot(span, [baseline, baseline], ls=":", color=GUIDE, lw=.9, zorder=2)

    for panel, key in ((0, "rms_error"), (1, "test_loss")):
        values = [float(row[key]) for row in rows
                  if 0 < float(row[key]) <= cutoff[panel]]
        if values:
            axes[panel].set_ylim(min(values) / 1.6, cutoff[panel])

    axes[0].set(ylabel="Terminal RMS error")
    axes[1].set(ylabel=r"Test loss $\overline{\|x_T-y\|^2}$")
    for axis in axes:
        axis.set(xscale="log", yscale="log",
                 xlabel="Component evaluations per trajectory")
        axis.set_xscale("log", base=2)
        grid(axis)
    # The panels share the abscissa, so the room for the end labels is made
    # once: widening each in turn would compound into an empty right half.
    headroom(axes[0], right=.16)
    if reference:
        annotate(axes[1], axes[1].get_xlim()[0], baseline,
                 "full model, fine RK4", color=GUIDE, dx=2.0, dy=3.0,
                 va="bottom", size=7.5)
    # Each curve ends at its own work budget, so the labels sit at those ends;
    # the ones that share a budget are nudged apart instead of overprinting.
    for panel, entries in labels.items():
        for position in sorted({item[0] for item in entries}):
            here = [item for item in entries if item[0] == position]
            if len(here) == 1:
                _, value, text, tint, weight = here[0]
                annotate(axes[panel], position, value, text, color=tint, weight=weight)
            else:
                stack_labels(axes[panel],
                             [(value, text, tint) for _, value, text, tint, _ in here],
                             position)
    # The step legend is achromatic: colour already carries the batch size.
    handles = [plt.Line2D([], [], color=GUIDE, linestyle=style, linewidth=1.2,
                          label=text) for style, text in STEP_STYLES.values()]
    axes[0].legend(handles=handles, loc="lower left", fontsize=7.5,
                   handlelength=2.2, borderpad=0.2, labelspacing=0.3)
    return save(fig, figure_dir, "work_accuracy")
