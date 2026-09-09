"""Shared figure style: one palette, quiet axes, PDF output."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rnode-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rnode-xdg-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PALETTE = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
    "#17becf",  # cyan
)

# Reference slopes and rules are structure, not data: they stay achromatic.
GUIDE = "#4d4d4d"
MUTED = "#8a8a8a"


def color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def use_paper_style() -> None:
    """Apply the figure style. Idempotent, so plot modules may call it freely."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Libertinus Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.prop_cycle": mpl.cycler(color=list(PALETTE)),
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#3a3a3a",
        "axes.labelcolor": "#1a1a1a",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": "#1a1a1a",
        "xtick.color": "#3a3a3a",
        "ytick.color": "#3a3a3a",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.7,
        "ytick.minor.size": 1.7,
        "grid.color": "#c9c9c9",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 1.0,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "errorbar.capsize": 0,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 0.03,
        "figure.constrained_layout.h_pad": 0.03,
    })


def grid(axis, which: str = "both") -> None:
    axis.grid(True, which=which)


def dyadic_ticks(axis, values, which: str = "x") -> None:
    """Label a log axis at the tested dyadic values as ``2^{-k}``.

    Matplotlib's minor decade labels collide badly over the narrow ranges
    these sweeps cover, and the sweeps are powers of two anyway, so the
    tested values themselves are the informative ticks.
    """
    values = np.unique(np.asarray(values, dtype=float))
    exponents = np.log2(values)
    labels = [
        rf"$2^{{{int(round(e))}}}$" if abs(e - round(e)) < 1e-9 else f"{v:g}"
        for e, v in zip(exponents, values)
    ]
    target = axis.xaxis if which == "x" else axis.yaxis
    target.set_major_locator(mpl.ticker.FixedLocator(values))
    target.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    target.set_minor_locator(mpl.ticker.NullLocator())


def integer_ticks(axis, values, which: str = "x") -> None:
    """Label a log axis at the tested integer values."""
    values = np.unique(np.asarray(values, dtype=float))
    target = axis.xaxis if which == "x" else axis.yaxis
    target.set_major_locator(mpl.ticker.FixedLocator(values))
    target.set_major_formatter(mpl.ticker.FixedFormatter([f"{int(v)}" for v in values]))
    target.set_minor_locator(mpl.ticker.NullLocator())


def annotate(
    axis,
    x,
    y,
    text: str,
    *,
    color: str,
    dx: float = 6.0,
    dy: float = 0.0,
    ha: str = "left",
    va: str = "center",
    size: float = 8.0,
    weight: str = "normal",
):
    """Label a series next to one of its points, in that series' colour.

    Direct labelling replaces the legend: the reader's eye never has to travel
    to a key and back, and nothing overlaps the data.
    """
    return axis.annotate(
        text,
        xy=(x, y),
        xycoords="data",
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        clip_on=False,
    )


def headroom(
    axis, *, right: float = 0.0, left: float = 0.0, top: float = 0.0, bottom: float = 0.0
) -> None:
    """Widen the view so direct labels and end markers are not clipped.

    Fractions are of the current span, measured in log units on log axes.
    """
    if right or left:
        lo, hi = axis.get_xlim()
        if axis.get_xscale() == "log":
            span = np.log10(hi / lo)
            axis.set_xlim(lo * 10 ** (-span * left), hi * 10 ** (span * right))
        else:
            axis.set_xlim(lo - (hi - lo) * left, hi + (hi - lo) * right)
    if top or bottom:
        lo, hi = axis.get_ylim()
        if axis.get_yscale() == "log":
            span = np.log10(hi / lo)
            axis.set_ylim(lo * 10 ** (-span * bottom), hi * 10 ** (span * top))
        else:
            axis.set_ylim(lo - (hi - lo) * bottom, hi + (hi - lo) * top)


def reference_line(axis, x, y_anchor, x_anchor, exponent):
    """Draw an unlabelled guide of slope ``exponent`` through one anchor point.

    The slope belongs in the figure caption, not on the canvas.
    """
    x = np.asarray(x, dtype=float)
    values = y_anchor * (x / x_anchor) ** exponent
    axis.plot(x, values, linestyle=(0, (5, 2.5)), color=GUIDE, linewidth=0.85, zorder=1)
    return values


def save(fig, directory, stem: str) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf",):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path)
        outputs.append(path)
    plt.close(fig)
    return outputs


def limit_to_data(axis, values, *, pad: float = 0.10, which: str = "y") -> None:
    """Set limits from the data alone, letting wide bands clip.

    Uncertainty bands can be orders of magnitude tall on a log axis, which
    otherwise compresses every curve into a sliver.
    """
    # Callers pass a mix of arrays and guide curves of differing lengths.
    flat = []
    for item in values:
        flat.extend(np.ravel(np.asarray(item, dtype=float)).tolist()
                    if np.ndim(item) else [float(item)])
    finite = np.asarray([v for v in flat if np.isfinite(v)], dtype=float)
    axis_object = axis.yaxis if which == "y" else axis.xaxis
    is_log = (axis.get_yscale() if which == "y" else axis.get_xscale()) == "log"
    if is_log:
        finite = finite[finite > 0]
    if finite.size == 0:
        return
    lo, hi = float(finite.min()), float(finite.max())
    if is_log:
        span = np.log10(hi / lo) or 1.0
        limits = (lo * 10 ** (-span * pad), hi * 10 ** (span * pad))
    else:
        span = (hi - lo) or 1.0
        limits = (lo - span * pad, hi + span * pad)
    del axis_object
    if which == "y":
        axis.set_ylim(*limits)
    else:
        axis.set_xlim(*limits)


def stack_labels(axis, entries, x, *, dx: float = 6.0, size: float = 8.0,
                 min_gap: float = 1.3) -> None:
    """Direct-label several series at a common ``x``, nudged apart vertically.

    Curves that end close together would otherwise print their labels on top
    of one another.  Call this after the limits are final, since the nudging
    is computed in display space.

    ``entries`` is a sequence of ``(y, text, colour)``.
    """
    entries = [(float(y), text, tint) for y, text, tint in entries
               if np.isfinite(y)]
    if not entries:
        return
    to_display = axis.transData.transform
    to_data = axis.transData.inverted().transform
    placed = sorted(
        ((to_display((x, y))[1], text, tint) for y, text, tint in entries),
        key=lambda item: item[0],
    )
    gap = min_gap * size * axis.figure.dpi / 72.0
    heights = [item[0] for item in placed]
    for index in range(1, len(heights)):
        heights[index] = max(heights[index], heights[index - 1] + gap)
    # Recentre so the block straddles the curves rather than drifting upward.
    shift = (sum(item[0] for item in placed) - sum(heights)) / len(heights)
    for (_, text, tint), height in zip(placed, heights):
        annotate(
            axis, x, to_data((0.0, height + shift))[1], text,
            color=tint, dx=dx, va="center", size=size,
        )
