"""Fixed-control objective: matched sampling comparison and a separate ensemble pool."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

try:
    from experiments._style import color, dyadic_ticks, grid, integer_ticks, save, use_paper_style
except ModuleNotFoundError:
    from _style import color, dyadic_ticks, grid, integer_ticks, save, use_paper_style
import matplotlib.pyplot as plt

SCHEMES = (("uniform_fixed_r8", "Uniform", 0),
           ("fixed_contiguous_r8", "Contiguous", 1),
           ("bernoulli_q1_3", "Bernoulli", 2))


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


def _intervals(axis, x, mean, low, high, tint, label=None, resolved=None):
    """Display zero-reaching intervals honestly at the log plot's lower edge."""
    floor = axis.get_ylim()[0]
    axis.vlines(x, np.maximum(low, floor), high, color=tint, alpha=.55, lw=1)
    resolved = np.ones(len(x), dtype=bool) if resolved is None else resolved
    axis.plot(x, np.where(resolved, mean, np.nan), color=tint, label=label)
    axis.plot(x[resolved], mean[resolved], 'o', color=tint, markeredgecolor='white')
    axis.plot(x[~resolved], mean[~resolved], 'o', mfc='white', mec=tint)
    zero = low <= 0
    axis.plot(x[zero], np.full(zero.sum(), floor * 1.08), 'v', color=tint, ms=4)


def _comparison_panel(axis, rows, weak=False):
    axis.set(xscale='log', yscale='log', xlabel='Switching interval $h$',
             ylabel=(r'$|\hat b_h|$' if weak else r'$\widehat{\mathcal{E}}_{J,\mathrm{str}}(h)$'))
    series = []
    for name, label, index in SCHEMES:
        selected = sorted((r for r in rows if r['scheme'] == name), key=lambda r: float(r['h']))
        if not selected:
            continue
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
        series.append((h, value, low, high, color(index), label, resolved))
    positive = np.concatenate([s[2][s[2] > 0] for s in series] + [s[1][s[1] > 0] for s in series])
    axis.set_ylim(positive.min() / 2, max(s[3].max() for s in series) * 2)
    for args in series:
        _intervals(axis, *args)
    dyadic_ticks(axis, series[0][0]); grid(axis, which='major')


def _ensemble_panel(axis, rows):
    axis.set(xscale='log', yscale='log', xlabel='Ensemble size $M$',
             ylabel=r'$\widehat{\mathbb{E}}|\hat\jmath_{h,M}-\jmath|^2$')
    positive = [float(r['mse_ci95_lower']) for r in rows if float(r['mse_ci95_lower']) > 0]
    axis.set_ylim(min(positive) / 2, max(float(r['mse_ci95_upper']) for r in rows) * 2)
    for i, h in enumerate(sorted({float(r['h']) for r in rows}, reverse=True)):
        selected = sorted((r for r in rows if float(r['h']) == h), key=lambda r: int(r['M']))
        values = [np.array([float(r[k]) for r in selected]) for k in
                  ('M', 'empirical_mse', 'mse_ci95_lower', 'mse_ci95_upper')]
        _intervals(axis, *values, color(i), rf'$h=2^{{-{round(-np.log2(h))}}}$')
        axis.plot(values[0], [float(r['variance_over_M_plus_debiased_bias_squared']) for r in selected],
                  ':', color=color(i), lw=1)
    integer_ticks(axis, sorted({int(r['M']) for r in rows}))
    grid(axis, which='major')


def generate_plots(output_dir: str | Path, *, figure_dir=None) -> list[Path]:
    use_paper_style()
    root = Path(output_dir).expanduser().resolve()
    figure_dir = Path(figure_dir) if figure_dir else root / 'figures'
    if (root / 'data/scheme_comparison.csv').exists():
        rows = _comparison_rows(root)
    else:
        rows = [dict(r, scheme='uniform_fixed_r8') for r in _rows(root / 'data/objective_consistency.csv')]
        # Historical single-scheme runs remain plottable without inventing a comparison.
    outputs = []
    for name, weak in (('objective_strong', False), ('objective_weak', True)):
        fig, axis = plt.subplots(figsize=(3.5, 2.9))
        _comparison_panel(axis, rows, weak=weak)
        outputs += save(fig, figure_dir, name)
    fig, axis = plt.subplots(figsize=(3.5, 2.9))
    _ensemble_panel(axis, _rows(root / 'data/ensemble_averaging.csv'))
    return outputs + save(fig, figure_dir, 'objective_ensemble')
