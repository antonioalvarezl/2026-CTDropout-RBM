"""Fixed-control objective: matched sampling comparison and a separate ensemble pool.

Series are labelled directly at the end of each curve, in the curve's own
colour; sample sizes and marker conventions belong in the LaTeX caption.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

try:
    from experiments._style import (
        annotate, color, dyadic_ticks, grid, headroom, integer_ticks,
        reference_line, save, stack_labels, use_paper_style,
    )
except ModuleNotFoundError:
    from _style import (
        annotate, color, dyadic_ticks, grid, headroom, integer_ticks,
        reference_line, save, stack_labels, use_paper_style,
    )
import matplotlib.pyplot as plt

SCHEMES = (("uniform_fixed_r8", r"Uniform, $r=8$", 0),
           ("fixed_contiguous_r8", r"Contiguous, $r=8$", 1),
           ("bernoulli_q1_3", r"Bernoulli, $q=1/3$", 2))


def _rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _comparison_rows(root):
    rows = _rows(root / "data/scheme_comparison.csv")
    # The uniform comparison is the first N final samples, not the whole pool
    # and not the pilot. Verify provenance before using the saved summaries.
    full = {int(r["h_power"]): float(r["J_full"])
            for r in _rows(root / "data/objective_consistency.csv")}
    with np.load(root / "data/objective_samples.npz") as raw:
        for row in rows:
            h = row["h"]
            counts = {int(r["n_schedules"]) for r in rows if r["h"] == h}
            if len(counts) != 1:
                raise ValueError(f"Sampling comparison has unequal pool sizes at h={h}")
            if row["scheme"] != "uniform_fixed_r8":
                continue
            n, power = int(row["n_schedules"]), int(row["h_power"])
            samples = raw[f"final_h_2m{power}"][:n]
            if len(samples) != n:
                raise ValueError("Uniform comparison exceeds the saved final pool")
            delta = samples - full[power]
            checked = {"signed_weak_bias": delta.mean(),
                       "weak_se": delta.std(ddof=1) / np.sqrt(n),
                       "strong_mse": np.mean(delta**2),
                       "strong_se": (delta**2).std(ddof=1) / np.sqrt(n)}
            for key, value in checked.items():
                if not np.isclose(value, float(row[key]), rtol=1e-10, atol=1e-12):
                    raise ValueError(f"Uniform provenance mismatch: h={h}, {key}")
    return rows


def _intervals(axis, x, mean, low, high, tint, resolved=None):
    """Shade the interval, and mark the ones that reach zero at the axis floor.

    A band reads as one uncertain curve; a rung of vertical bars competes with
    the data for attention and, at these interval widths, dominates it.
    """
    floor = axis.get_ylim()[0]
    # An interval reaching zero has no lower edge on a log axis. Clipping it to
    # the axis floor would open a full-height wedge, so the band is clipped
    # just below the estimate and the point itself is flagged instead.
    axis.fill_between(x, np.maximum(low, mean * 1e-2), high,
                      color=tint, alpha=.16, lw=0)
    resolved = np.ones(len(x), dtype=bool) if resolved is None else resolved
    axis.plot(x, np.where(resolved, mean, np.nan), color=tint)
    axis.plot(x[resolved], mean[resolved], 'o', color=tint, markeredgecolor='white')
    axis.plot(x[~resolved], mean[~resolved], 'o', mfc='white', mec=tint)
    zero = low <= 0
    axis.plot(x[zero], np.full(zero.sum(), floor * 1.08), 'v', color=tint, ms=4)


def _extract(rows, name, weak):
    selected = sorted((r for r in rows if r['scheme'] == name), key=lambda r: float(r['h']))
    if not selected:
        return None
    h = np.array([float(r['h']) for r in selected])
    if weak:
        signed = np.array([float(r['signed_weak_bias']) for r in selected])
        half = 1.96 * np.array([float(r['weak_se']) for r in selected])
        lo, hi = signed - half, signed + half
        resolved = (lo > 0) | (hi < 0)
        value = abs(signed)
        low = np.where(resolved, np.minimum(abs(lo), abs(hi)), 0)
        high = np.maximum(abs(lo), abs(hi))
    else:
        value = np.array([float(r['strong_mse']) for r in selected])
        half = 1.96 * np.array([float(r['strong_se']) for r in selected])
        low, high, resolved = np.maximum(0, value - half), value + half, None
    return h, value, low, high, resolved


def _fit_slope(h, value):
    """Least-squares log-log slope, for an unlabelled trend guide only."""
    finite = np.isfinite(value) & (value > 0)
    return float(np.polyfit(np.log(h[finite]), np.log(value[finite]), 1)[0])


def _panel(axis, series, weak, *, guide=False):
    axis.set(xscale='log', yscale='log', xlabel='Switching interval $h$',
             ylabel=(r'$|\mathbb{E}\,\hat\jmath_h-\jmath|$' if weak
                     else r'$\mathbb{E}\,|\hat\jmath_h-\jmath|^2$'))
    positive = np.concatenate([s[2][s[2] > 0] for _, s in series]
                              + [s[1][s[1] > 0] for _, s in series])
    axis.set_ylim(positive.min() / 2, max(s[3].max() for _, s in series) * 2)
    labels = []
    for (label, index), (h, value, low, high, resolved) in series:
        _intervals(axis, h, value, low, high, color(index), resolved)
        labels.append((value[-1], label, color(index)))
    if guide:
        h, value = series[0][1][0], series[0][1][1]
        reference_line(axis, h, value[-1], h[-1], _fit_slope(h, value))
    dyadic_ticks(axis, series[0][1][0])
    grid(axis)
    headroom(axis, right=.45, top=.08)
    return labels


def _ensemble_panel(axis, rows):
    axis.set(xscale='log', yscale='log', xlabel='Ensemble size $M$',
             ylabel=r'$\widehat{\mathbb{E}}|\hat\jmath_{h,M}-\jmath|^2$')
    positive = [float(r['mse_ci95_lower']) for r in rows if float(r['mse_ci95_lower']) > 0]
    axis.set_ylim(min(positive) / 2, max(float(r['mse_ci95_upper']) for r in rows) * 2)
    labels = []
    for i, h in enumerate(sorted({float(r['h']) for r in rows}, reverse=True)):
        selected = sorted((r for r in rows if float(r['h']) == h), key=lambda r: int(r['M']))
        values = [np.array([float(r[k]) for r in selected]) for k in
                  ('M', 'empirical_mse', 'mse_ci95_lower', 'mse_ci95_upper')]
        _intervals(axis, *values, color(i))
        axis.plot(values[0], [float(r['variance_over_M_plus_debiased_bias_squared']) for r in selected],
                  ':', color=color(i), lw=1)
        labels.append((values[1][-1], rf'$h=2^{{-{round(-np.log2(h))}}}$', color(i)))
    integer_ticks(axis, sorted({int(r['M']) for r in rows}))
    grid(axis)
    headroom(axis, right=.4, top=.08)
    stack_labels(axis, labels, max(int(r['M']) for r in rows))


def generate_plots(output_dir: str | Path, *, figure_dir=None) -> list[Path]:
    use_paper_style()
    root = Path(output_dir).expanduser().resolve()
    figure_dir = Path(figure_dir) if figure_dir else root / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    if (root / 'data/scheme_comparison.csv').exists():
        rows = _comparison_rows(root)
    else:
        rows = [dict(r, scheme='uniform_fixed_r8') for r in _rows(root / 'data/objective_consistency.csv')]
        # Historical single-scheme runs remain plottable without inventing a comparison.

    outputs = []
    # The uniform sweep on its own, with the separate ensemble pool beside it.
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.9))
    for axis, weak, label, index in ((axes[0], False, 'strong MSE', 0),
                                     (axes[1], True, 'weak bias', 1)):
        series = [((label, index), _extract(rows, 'uniform_fixed_r8', weak))]
        labels = _panel(axis, series, weak, guide=True)
        annotate(axis, series[0][1][0][-1], labels[0][0], label, color=color(index))
    _ensemble_panel(axes[2], _rows(root / 'data/ensemble_averaging.csv'))
    outputs += save(fig, figure_dir, 'objective_consistency')

    # The matched three-scheme comparison at the same control.
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
    for axis, weak in ((axes[0], False), (axes[1], True)):
        series = [((label, index), _extract(rows, name, weak))
                  for name, label, index in SCHEMES if _extract(rows, name, weak)]
        labels = _panel(axis, series, weak)
        stack_labels(axis, labels, series[0][1][0][-1])
    outputs += save(fig, figure_dir, 'scheme_comparison')
    return outputs
