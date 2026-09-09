#!/usr/bin/env python3
"""Strong, weak, and ensemble consistency of the randomized objective at fixed control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rnode-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rnode-xdg-cache")

# Allow running this file directly from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rnode.batches import make_bernoulli, make_fixed_disjoint_partition
from rnode.objectives import control_regularization

try:
    from experiments._paper_common import (
        ArtifactPaths,
        prepare_base_model,
        report_progress,
        resolve_device,
        version_information,
        write_csv,
        write_json,
        write_manifest,
    )
    from experiments.exp2_objective_consistency_plots import generate_plots
except ModuleNotFoundError:
    from _paper_common import (
        ArtifactPaths,
        prepare_base_model,
        report_progress,
        resolve_device,
        version_information,
        write_csv,
        write_json,
        write_manifest,
    )
    from exp2_objective_consistency_plots import generate_plots


def experiment_configuration(quick: bool) -> dict:
    if quick:
        return {
            "h_powers": [3, 4, 5],
            "pilot_schedules": 24,
            "pilot_reference_count": 2,
            "final_min_schedules": 64,
            "final_max_schedules": 256,
            "weak_ci_relative_target": 0.75,
            "z_value": 1.96,
            "chunk_size": 16,
            "steps_per_switch": 4,
            "reference_dt": 2.0**-9,
            "reference_check_dt": 2.0**-10,
            "control_dt": 2.0**-9,
            "ensemble_sizes": [1, 4, 8],
            "ensemble_min_groups": 8,
            "refinement_schedules": 6,
            "refinement_steps_per_switch": 8,
            "evaluation_split": "calibration",
            # Fixed (non-adaptive) sample count for the two comparison
            # schemes below; not aimed at the primary scheme's CI precision.
            "scheme_comparison_schedules": 16,
        }
    return {
        # Matches exp1's trajectory sweep, since both use the same frozen
        # classification model and this lets the fine-h regime of the
        # weak-bias estimate be compared against the same-model trajectory
        # error at the same h.
        "h_powers": [6, 7, 8, 9, 10, 11],
        "pilot_schedules": 256,
        "pilot_reference_count": 2,
        "final_min_schedules": 768,
        "final_max_schedules": 8192,
        "weak_ci_relative_target": 0.5,
        "z_value": 1.96,
        "chunk_size": 64,
        "steps_per_switch": 8,
        # Finer than exp1's own reference (2^-16) is unnecessary; matching it
        # keeps the deterministic reference far more accurate than the random
        # step at every tested h, including the finest, h=2^-11 (dt=2^-14).
        "reference_dt": 2.0**-16,
        "reference_check_dt": 2.0**-15,
        "control_dt": 2.0**-16,
        "ensemble_sizes": [1, 4, 16, 64],
        "ensemble_min_groups": 12,
        "refinement_schedules": 20,
        "refinement_steps_per_switch": 16,
        "evaluation_split": "calibration",
        "scheme_comparison_schedules": 256,
    }


def _seeds(base: int, tag: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(base), int(tag)]))
    return rng.integers(0, np.iinfo(np.uint32).max, n, dtype=np.uint32)


def _loss(state, targets):
    if state.ndim == 2:
        return (state - targets).square().sum(-1).mean()
    return (state - targets.unsqueeze(0)).square().sum(-1).mean(-1)


def _control_term(model, x, T, dt, alpha):
    times = torch.linspace(0, T, round(T / dt) + 1, dtype=x.dtype, device=x.device)
    with torch.no_grad():
        energy = float(control_regularization(model, times).cpu())
    return alpha * energy / (2 * len(x)), energy


@torch.no_grad()
def _full_objective(model, x, targets, T, dt, beta, control_term):
    state = x.clone()
    previous = _loss(state, targets)
    running = state.new_zeros(())
    for step in range(round(T / dt)):
        t, half = step * dt, dt / 2
        k1 = model(t, state)
        k2 = model(t + half, state + half * k1)
        k3 = model(t + half, state + half * k2)
        k4 = model(t + dt, state + dt * k3)
        state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        current = _loss(state, targets)
        running += 0.5 * dt * (previous + current)
        previous = current
    return float((previous + beta * running).cpu()) + control_term


def _schedules(p, r, n_intervals, seeds):
    out = np.empty((len(seeds), n_intervals, r), dtype=np.int16)
    for k, seed in enumerate(seeds):
        scores = np.random.default_rng(int(seed)).random((n_intervals, p))
        out[k] = np.argpartition(scores, r - 1, axis=1)[:, :r]
    return out


def _batch_field(model, t, state, active, pi):
    A, b, W = model.control_parameters(t)
    A, b, W = A[active], b[active], W.t()[active] / pi
    z = torch.einsum("knd,krd->knr", state, A) + b[:, None, :]
    return torch.einsum("knr,krd->knd", model.activation(z), W)


@torch.no_grad()
def _random_chunk(model, x, targets, schedules, T, h, steps, p, r, beta, control_term):
    dt, pi = h / steps, r / p
    active = torch.as_tensor(schedules, dtype=torch.long, device=x.device)
    state = x.unsqueeze(0).expand(len(schedules), -1, -1).clone()
    previous = _loss(state, targets)
    running = torch.zeros(len(schedules), dtype=x.dtype, device=x.device)
    for step in range(round(T / h) * steps):
        batch = active[:, step // steps]
        t, half = step * dt, dt / 2
        k1 = _batch_field(model, t, state, batch, pi)
        k2 = _batch_field(model, t + half, state + half * k1, batch, pi)
        k3 = _batch_field(model, t + half, state + half * k2, batch, pi)
        k4 = _batch_field(model, t + dt, state + dt * k3, batch, pi)
        state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        current = _loss(state, targets)
        running += 0.5 * dt * (previous + current)
        previous = current
    return (previous + beta * running + control_term).cpu().numpy()


def _random_values(model, x, targets, seeds, *, T, h, steps, p, r, beta, control_term, chunk):
    values = np.empty(len(seeds))
    for start in range(0, len(seeds), chunk):
        stop = min(len(seeds), start + chunk)
        schedule = _schedules(p, r, round(T / h), seeds[start:stop])
        values[start:stop] = _random_chunk(
            model, x, targets, schedule, T, h, steps, p, r, beta, control_term
        )
    return values


# --- Sampling-scheme comparison -------------------------------------------
#
# The primary scheme above (uniform_fixed_r8) keeps its existing fast path
# untouched: nothing below changes its numbers or its code path. Two more
# schemes from exp1's own comparison (Table "design quantities") are added
# so the objective can be checked for the same design dependence exp1
# already established for trajectories: same pi_min=1/3, same expected
# batch size 8, evaluated at a fixed (non-adaptive) sample count rather than
# the primary scheme's precision-targeted one.

CONTIGUOUS = (tuple(range(8)), tuple(range(8, 16)), tuple(range(16, 24)))
_PARTITION_BLOCKS = np.asarray(CONTIGUOUS, dtype=np.int16)


def _partition_schedules(n_intervals, seeds):
    """Per-interval index arrays for the fixed contiguous partition, r=8.

    Every block has the same size, so pi_i=r/p is a single scalar exactly as
    in the uniform scheme, and the existing fast index-gather path
    (_random_chunk/_batch_field) applies unchanged -- only how the r indices
    are generated differs.
    """
    out = np.empty((len(seeds), n_intervals, _PARTITION_BLOCKS.shape[1]), dtype=np.int16)
    for k, seed in enumerate(seeds):
        choice = np.random.default_rng(int(seed)).integers(0, len(_PARTITION_BLOCKS), n_intervals)
        out[k] = _PARTITION_BLOCKS[choice]
    return out


def _random_values_partition(model, x, targets, seeds, *, T, h, steps, beta, control_term, chunk):
    values = np.empty(len(seeds))
    for start in range(0, len(seeds), chunk):
        stop = min(len(seeds), start + chunk)
        schedule = _partition_schedules(round(T / h), seeds[start:stop])
        values[start:stop] = _random_chunk(
            model, x, targets, schedule, T, h, steps, 24, 8, beta, control_term
        )
    return values


def _scheme_masks(scheme, n_intervals, seeds):
    """Per-interval inclusion masks, shape [schedules, n_intervals, p].

    Unlike the fixed-size gather above, Bernoulli draws a variable number of
    neurons per interval (and can draw none), so it needs a dense mask
    rather than an index array of fixed width.
    """
    masks = np.zeros((len(seeds), n_intervals, scheme.p), dtype=np.uint8)
    for k, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        for j in range(n_intervals):
            masks[k, j, scheme.sample(rng)] = 1
    return masks


def _masked_field(model, t, state, mask, pi):
    """Horvitz-Thompson field for a dense per-schedule inclusion mask.

    Evaluates all p neurons every stage (unlike the fixed-size gather path),
    so this costs roughly p/r times more per step; that is only paid for the
    Bernoulli comparison below, not for the primary scheme.
    """
    k, n, d = state.shape
    p = mask.shape[1]
    terms = model.neuron_contributions(t, state.reshape(k * n, d)).reshape(k, n, p, d)
    return (terms * (mask / pi[None, :])[:, None, :, None]).sum(2)


@torch.no_grad()
def _random_chunk_masked(model, x, targets, masks, T, h, steps, pi, beta, control_term):
    dt = h / steps
    masks_t = torch.as_tensor(masks, dtype=x.dtype, device=x.device)
    pi_t = torch.as_tensor(np.asarray(pi).copy(), dtype=x.dtype, device=x.device)
    state = x.unsqueeze(0).expand(masks_t.shape[0], -1, -1).clone()
    previous = _loss(state, targets)
    running = torch.zeros(masks_t.shape[0], dtype=x.dtype, device=x.device)
    for step in range(round(T / h) * steps):
        active = masks_t[:, step // steps]
        t, half = step * dt, dt / 2
        k1 = _masked_field(model, t, state, active, pi_t)
        k2 = _masked_field(model, t + half, state + half * k1, active, pi_t)
        k3 = _masked_field(model, t + half, state + half * k2, active, pi_t)
        k4 = _masked_field(model, t + dt, state + dt * k3, active, pi_t)
        state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        current = _loss(state, targets)
        running += 0.5 * dt * (previous + current)
        previous = current
    return (previous + beta * running + control_term).cpu().numpy()


def _random_values_bernoulli(model, x, targets, scheme, seeds, *, T, h, steps, beta, control_term, chunk):
    values = np.empty(len(seeds))
    for start in range(0, len(seeds), chunk):
        stop = min(len(seeds), start + chunk)
        masks = _scheme_masks(scheme, round(T / h), seeds[start:stop])
        values[start:stop] = _random_chunk_masked(
            model, x, targets, masks, T, h, steps, scheme.inclusion_probs, beta, control_term
        )
    return values


def _scheme_consistency_rows(name, values, J, h, power, n, z):
    delta = values - J
    squared = delta**2
    strong = float(squared.mean())
    strong_se = float(squared.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    weak, variance, weak_se, lo, hi = _mean_stats(values, J, z)
    return {
        "scheme": name,
        "h_power": power,
        "h": h,
        "n_schedules": n,
        "strong_mse": strong,
        "strong_se": strong_se,
        "strong_ci95_lower": max(0.0, strong - z * strong_se),
        "strong_ci95_upper": strong + z * strong_se,
        "signed_weak_bias": weak,
        "weak_bias": abs(weak),
        "weak_se": weak_se,
        "weak_distinguishable_from_zero": lo > 0 or hi < 0,
        "sample_variance_J_hat": variance,
    }


def _mean_stats(values, J, z):
    delta = np.asarray(values) - J
    variance = float(delta.var(ddof=1))
    se = math.sqrt(variance / len(delta))
    mean = float(delta.mean())
    return mean, variance, se, mean - z * se, mean + z * se


def _precision_fields(mean, se, z):
    """Relative uncertainty is diagnostic only, especially for unresolved bias."""
    half_width = z * se
    return {
        "ci_half_width": half_width,
        "relative_se": se / abs(mean) if mean else math.inf,
        "relative_ci_half_width": half_width / abs(mean) if mean else math.inf,
        "distinguishable_from_zero": abs(mean) > half_width,
    }


def _paired_diagnostics(coarse, fine, z):
    difference = np.asarray(coarse) - np.asarray(fine)
    mean, _, se, lo, hi = _mean_stats(difference, 0.0, z)
    return {
        "signed_mean_difference": mean,
        "difference_se": se,
        "difference_ci95_lower": lo,
        "difference_ci95_upper": hi,
        "mean_absolute_difference": float(np.abs(difference).mean()),
        "rms_difference": float(np.sqrt(np.mean(difference**2))),
    }


def _fit(metric, h, y, mask, reason):
    h, y, mask = np.asarray(h), np.asarray(y), np.asarray(mask, dtype=bool)
    mask &= np.isfinite(y) & (y > 0)
    if mask.sum() < 3:
        return {
            "metric": metric,
            "fit_performed": False,
            "slope": np.nan,
            "r_squared": np.nan,
            "n_points": int(mask.sum()),
            "reason": reason,
        }
    x, yy = np.log(h[mask]), np.log(y[mask])
    slope, intercept = np.polyfit(x, yy, 1)
    fitted = slope * x + intercept
    total = np.square(yy - yy.mean()).sum()
    r2 = 1 if total == 0 else 1 - np.square(yy - fitted).sum() / total
    return {
        "metric": metric,
        "fit_performed": True,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r2),
        "h_min": float(h[mask].min()),
        "h_max": float(h[mask].max()),
        "n_points": int(mask.sum()),
        "reason": reason,
    }


def _ensemble_rows(values_by_h, J, cfg, seed, z):
    """Ensemble mean-square error against the number of averaged schedules.

    For each ``h`` the same pool of single-schedule values is partitioned into
    disjoint groups of size ``M``.  Groups within one ``M`` are independent,
    but every ``M`` reuses the one pool, so points along a curve are dependent
    and the curve must not be read as independent replication.

    The reference curve uses ``b^2`` estimated as ``b_hat^2 - s^2/N``: the
    plain square of the sample bias overstates the squared bias by exactly the
    variance of the bias estimator.  Both forms are reported.
    """
    rows = []
    for hi, h in enumerate(sorted(values_by_h, reverse=True)):
        values = values_by_h[h]
        n_pool = len(values)
        variance = float(values.var(ddof=1))
        bias = float((values - J).mean())
        # E[bias_hat^2] = bias^2 + variance/N, so subtract the inflation.
        debiased_square = max(bias**2 - variance / n_pool, 0.0)
        for mi, M in enumerate(cfg["ensemble_sizes"]):
            groups = n_pool // M
            if groups < cfg["ensemble_min_groups"]:
                continue
            rng = np.random.default_rng(int(_seeds(seed, 700 + 10 * hi + mi, 1)[0]))
            used = rng.permutation(n_pool)[: groups * M]
            means = values[used].reshape(groups, M).mean(1)
            squared = (means - J) ** 2
            mse = float(squared.mean())
            se = float(squared.std(ddof=1) / math.sqrt(groups)) if groups > 1 else 0.0
            rows.append({
                "h": h,
                "M": M,
                "groups": groups,
                "pool_size": n_pool,
                "empirical_mse": mse,
                "se_empirical_mse": se,
                "mse_ci95_lower": max(0.0, mse - z * se),
                "mse_ci95_upper": mse + z * se,
                "variance_over_M_plus_bias_squared": float(variance / M + bias**2),
                "variance_over_M_plus_debiased_bias_squared": float(
                    variance / M + debiased_square
                ),
                "pool_reused_across_M": True,
            })
    return rows


def _resolve_frozen_control(args) -> tuple[Path, Path, str]:
    """Locate the frozen checkpoint and the dataset it was trained on.

    exp1 writes both into one run directory, so a bare invocation adopts its
    most recent run rather than making the caller paste two paths.  Training
    is never started here, whatever the outcome.
    """
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else None
    dataset = Path(args.dataset).expanduser() if args.dataset else None
    source = "given on the command line"

    if checkpoint is None:
        parent = Path(args.base_run_dir).expanduser()
        runs = sorted(
            run for run in parent.glob("*")
            if (run / "checkpoints" / "base_model.pt").is_file()
            and (run / "data" / "dataset_splits.npz").is_file()
        )
        if not runs:
            raise ValueError(
                f"no trained classification run found under {parent}. "
                "Run 'python -m experiments.exp1_trajectory_convergence' first, "
                "or pass --checkpoint and --dataset explicitly. "
                "This experiment evaluates a frozen control and never trains."
            )
        run = runs[-1]
        checkpoint = run / "checkpoints" / "base_model.pt"
        dataset = dataset or run / "data" / "dataset_splits.npz"
        source = f"discovered in {run}"
    elif dataset is None:
        # exp1 writes the checkpoint and its dataset side by side.
        dataset = checkpoint.parent.parent / "data" / "dataset_splits.npz"
        source = f"checkpoint given, dataset taken from {dataset.parent}"

    for label, path in (("checkpoint", checkpoint), ("dataset", dataset)):
        if not path.is_file():
            raise ValueError(f"{label} not found: {path}")
    return checkpoint.resolve(), dataset.resolve(), source


def run_experiment(args):
    # This experiment evaluates a frozen control. Never fall back to training.
    checkpoint, dataset, control_source = _resolve_frozen_control(args)
    args.checkpoint, args.dataset = str(checkpoint), str(dataset)
    device = resolve_device(args.device)
    cfg = experiment_configuration(args.quick)
    if args.pilot_only and not args.quick:
        # Preview the full intended h grid at a smaller pilot sample so the
        # coarsest AND finest points are both checked before committing to
        # the final sweep; ~4 min total at these sizes.
        cfg.update(pilot_schedules=128, chunk_size=16, refinement_schedules=8)
    if args.max_final_schedules is not None:
        cfg["final_max_schedules"] = int(args.max_final_schedules)

    max_M = max(cfg["ensemble_sizes"])
    cfg["final_min_schedules"] = max(
        cfg["final_min_schedules"], cfg["ensemble_min_groups"] * max_M
    )
    cfg["final_min_schedules"] = math.ceil(cfg["final_min_schedules"] / max_M) * max_M
    cfg["final_max_schedules"] -= cfg["final_max_schedules"] % max_M
    if not args.pilot_only and cfg["final_max_schedules"] < cfg["final_min_schedules"]:
        raise ValueError("final_max_schedules is too small for the ensemble design")

    paths = ArtifactPaths.new_run(
        args.output_dir, prefix="pilot" if args.pilot_only else "run"
    )
    started = time.perf_counter()
    report_progress("exp2", f"Starting ({'quick' if args.quick else 'full'} mode)", started=started)
    report_progress("exp2", f"Frozen control {control_source}", started=started)
    report_progress("exp2", f"  checkpoint {checkpoint}", started=started)
    report_progress("exp2", f"  dataset    {dataset}", started=started)
    prepared = prepare_base_model(
        paths, quick=args.quick, seed=args.seed, dtype_name=args.dtype,
        device=device, checkpoint=args.checkpoint,
    )
    # Compare saved data to checkpoint-seeded regeneration before integration.
    # Preserve the original rounding; do not silently replace historical data.
    with np.load(args.dataset, allow_pickle=False) as saved:
        for split, tensors in prepared["datasets"].items():
            for suffix, tensor in zip(("X", "labels", "targets"), tensors):
                key = f"{split}_{suffix}"
                actual = tensor.detach().cpu().numpy()
                if key not in saved or not np.array_equal(saved[key], actual):
                    raise ValueError(f"Dataset verification failed for {key}; evaluation stopped.")
    sources = {
        name: {"path": getattr(args, name), "sha256": hashlib.sha256(
            Path(getattr(args, name)).read_bytes()).hexdigest()}
        for name in ("checkpoint", "dataset")
    }
    sources["resolution"] = control_source
    model = prepared["model"].eval().requires_grad_(False)
    base = prepared["base_config"]
    x, _, targets = prepared["datasets"][cfg["evaluation_split"]]
    T, p, r = float(base["model"]["T"]), model.hidden_dim, 8
    alpha, beta = float(base["training"]["alpha"]), float(base["training"]["beta"])
    if p != 24:
        raise ValueError("objective experiment expects width p=24")

    report_progress("exp2", "Full reference", started=started)
    control_term, control_energy = _control_term(model, x, T, cfg["control_dt"], alpha)
    control_fine, _ = _control_term(model, x, T, cfg["control_dt"] / 2, alpha)
    J = _full_objective(model, x, targets, T, cfg["reference_dt"], beta, control_term)
    J_check = _full_objective(
        model, x, targets, T, cfg["reference_check_dt"], beta, control_term
    )
    write_csv(paths.data / "reference_check.csv", [{
        "reference_dt": cfg["reference_dt"],
        "check_dt": cfg["reference_check_dt"],
        "J_reference": J,
        "J_check": J_check,
        "absolute_difference": abs(J - J_check),
        "signed_reference_difference": J - J_check,
        "control_term": control_term,
        "refined_control_term": control_fine,
        "control_term_difference": control_term - control_fine,
    }])

    report_progress("exp2", "Independent pilot", started=started)
    pilot_rows, pilot_values = [], {}
    pilot_arrays, refinement_rows = {}, []
    for power in cfg["h_powers"]:
        h = 2.0**-power
        seeds = _seeds(prepared["seeds"]["schedule_generation"], 1000 + power, cfg["pilot_schedules"])
        case_started = time.perf_counter()
        values = _random_values(
            model, x, targets, seeds, T=T, h=h, steps=cfg["steps_per_switch"],
            p=p, r=r, beta=beta, control_term=control_term, chunk=cfg["chunk_size"],
        )
        case_seconds = time.perf_counter() - case_started
        pilot_values[power] = values
        pilot_arrays[f"values_h_2m{power}"] = values
        pilot_arrays[f"seeds_h_2m{power}"] = seeds
        mean, variance, se, lo, hi = _mean_stats(values, J, cfg["z_value"])
        squared = (values-J)**2
        strong_se = float(squared.std(ddof=1)/math.sqrt(len(values)))
        pilot_rows.append({
            "h_power": power,
            "h": h,
            "pilot_schedules": len(values),
            "pilot_signed_bias": mean,
            "pilot_variance": variance,
            "pilot_se": se,
            "pilot_ci95_lower": lo,
            "pilot_ci95_upper": hi,
            "strong_mse": float(squared.mean()),
            "strong_se": strong_se,
            "strong_ci95_lower": max(0.0, float(squared.mean())-cfg["z_value"]*strong_se),
            "strong_ci95_upper": float(squared.mean())+cfg["z_value"]*strong_se,
            **_precision_fields(mean, se, cfg["z_value"]),
            "evaluation_seconds": case_seconds,
            "seconds_per_schedule": case_seconds / len(values),
        })

        n_refine = min(cfg["refinement_schedules"], len(seeds))
        refined = _random_values(
            model, x, targets, seeds[:n_refine], T=T, h=h,
            steps=cfg["refinement_steps_per_switch"], p=p, r=r, beta=beta,
            control_term=control_term, chunk=min(cfg["chunk_size"], n_refine),
        )
        pilot_arrays[f"refined_values_h_2m{power}"] = refined
        pilot_arrays[f"paired_difference_h_2m{power}"] = values[:n_refine] - refined
        paired = _paired_diagnostics(values[:n_refine], refined, cfg["z_value"])
        half_width = cfg["z_value"] * se
        # Diagnostic of bias discretization, not a rigorous numerical bound.
        numerical_scale = abs(J - J_check) + max(
            abs(paired["difference_ci95_lower"]), abs(paired["difference_ci95_upper"])
        )
        refinement_rows.append({
            "h": h, "n_paired_schedules": n_refine,
            "dt": h / cfg["steps_per_switch"],
            "refined_dt": h / cfg["refinement_steps_per_switch"],
            **paired,
            "signed_bias_discretization_difference": paired["signed_mean_difference"] - (J - J_check),
            "bias_discretization_ci95_lower": paired["difference_ci95_lower"] - (J - J_check),
            "bias_discretization_ci95_upper": paired["difference_ci95_upper"] - (J - J_check),
            "reference_absolute_difference": abs(J - J_check),
            "numerical_scale_over_bias_ci_half_width": numerical_scale / half_width if half_width else math.inf,
            "numerical_scale_over_abs_bias": numerical_scale / abs(mean) if mean else math.inf,
        })

    write_csv(paths.data / "weak_bias_pilot.csv", pilot_rows)
    write_csv(paths.data / "pilot_dt_refinement.csv", refinement_rows)
    np.savez_compressed(paths.data / "pilot_samples.npz", **pilot_arrays)
    pilot_config = {
        "script": "exp2_objective_consistency.py", "phase": "pilot",
        "cli": vars(args), "base": base, "experiment": cfg,
        "verified_sources": sources, "dataset_matches_checkpoint_regeneration": True,
        "fixed_control": {"split": cfg["evaluation_split"], "n_data": len(x),
                          "normalization": "j=J/n_data", "J_reference": J,
                          "alpha": alpha, "beta": beta, "control_energy": control_energy},
        "total_seconds": time.perf_counter() - started,
        "next_step": "Choose absolute bias precision and an independent final sample size; no rate is assumed.",
    }
    versions = version_information(args.device, device, args.dtype)
    write_json(paths.config / "pilot_config.json", pilot_config)
    write_json(paths.config / "versions.json", versions)
    write_manifest(paths.root, configuration=pilot_config, seed=prepared["seeds"], versions=versions)
    if args.pilot_only:
        report_progress("exp2", "Pilot and diagnostics finished; no final sampling", started=started)
        return {"output_dir": str(paths.root), "phase": "pilot", "J_reference": J,
                "total_seconds": pilot_config["total_seconds"]}

    coarse = sorted(pilot_rows, key=lambda row: row["h"], reverse=True)[: cfg["pilot_reference_count"]]
    scales = [
        max(abs(row["pilot_signed_bias"]), cfg["z_value"] * row["pilot_se"]) / row["h"]
        for row in coarse
    ]
    bias_scale = max(float(np.median(scales)), np.finfo(float).tiny)

    for row in pilot_rows:
        target = cfg["weak_ci_relative_target"] * bias_scale * row["h"]
        requested = math.ceil(cfg["z_value"] ** 2 * row["pilot_variance"] / target**2)
        requested = max(requested, cfg["final_min_schedules"])
        rounded = math.ceil(requested / max_M) * max_M
        row["target_ci_half_width"] = target
        row["requested_final_schedules"] = requested
        row["final_schedules"] = min(rounded, cfg["final_max_schedules"])
        row["sample_size_capped"] = rounded > cfg["final_max_schedules"]
        row["pilot_reference_bias_coefficient"] = bias_scale
    write_csv(paths.data / "weak_bias_pilot.csv", pilot_rows)

    report_progress("exp2", "Independent final samples", started=started)
    values_by_h, seed_by_h, consistency = {}, {}, []
    arrays = {f"pilot_h_2m{p}": v for p, v in pilot_values.items()}
    for row in pilot_rows:
        power, h, n = int(row["h_power"]), float(row["h"]), int(row["final_schedules"])
        report_progress(
            "exp2", f"h=2^-{power}: N={n}" + (" (capped)" if row["sample_size_capped"] else ""),
            started=started,
        )
        seeds = _seeds(prepared["seeds"]["schedule_generation"], 2000 + power, n)
        values = _random_values(
            model, x, targets, seeds, T=T, h=h, steps=cfg["steps_per_switch"],
            p=p, r=r, beta=beta, control_term=control_term, chunk=cfg["chunk_size"],
        )
        values_by_h[h], seed_by_h[h] = values, seeds
        arrays[f"final_h_2m{power}"] = values

        delta = values - J
        squared = delta**2
        strong = float(squared.mean())
        strong_se = float(squared.std(ddof=1) / math.sqrt(n))
        weak, variance, weak_se, lo, hi = _mean_stats(values, J, cfg["z_value"])
        consistency.append({
            "h_power": power,
            "h": h,
            "J_full": J,
            "final_schedules": n,
            "strong_mse": strong,
            "strong_se": strong_se,
            "strong_ci95_lower": max(0.0, strong - cfg["z_value"] * strong_se),
            "strong_ci95_upper": strong + cfg["z_value"] * strong_se,
            "signed_weak_bias": weak,
            "weak_bias": abs(weak),
            "weak_se": weak_se,
            "weak_signed_ci95_lower": lo,
            "weak_signed_ci95_upper": hi,
            "weak_abs_ci95_lower": 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)),
            "weak_abs_ci95_upper": max(abs(lo), abs(hi)),
            "weak_distinguishable_from_zero": lo > 0 or hi < 0,
            **_precision_fields(weak, weak_se, cfg["z_value"]),
            "sample_variance_J_hat": variance,
            "requested_final_schedules": row["requested_final_schedules"],
            "sample_size_capped": row["sample_size_capped"],
            "target_ci_half_width": row["target_ci_half_width"],
        })

    write_csv(paths.data / "objective_consistency.csv", consistency)
    np.savez_compressed(paths.data / "objective_samples.npz", **arrays)

    h = np.asarray([row["h"] for row in consistency])
    strong = np.asarray([row["strong_mse"] for row in consistency])
    strong_ci_half_width = np.asarray(
        [row["strong_ci95_upper"] - row["strong_mse"] for row in consistency]
    )
    weak = np.asarray([row["weak_bias"] for row in consistency])
    resolved = np.asarray([row["weak_distinguishable_from_zero"] for row in consistency])
    # A coarse h with a heavy-tailed strong-error distribution can have a 95%
    # CI wider than the estimate itself (seen at h=2^-6 in one run: ratio
    # 1.86). Fitting through that point lets pure sampling noise set the
    # slope, so it is excluded the same way an unresolved weak-bias point is.
    strong_resolved = strong_ci_half_width < strong
    slopes = [
        _fit("strong_mse", h, strong, strong_resolved,
             "excludes h where the 95% CI half-width exceeds the estimate"),
        _fit("weak_bias", h, weak, resolved, "only final signed-bias CIs excluding zero"),
    ]
    write_csv(paths.data / "slope_fits.csv", slopes)

    report_progress("exp2", "Disjoint ensemble averages", started=started)
    ensemble = _ensemble_rows(
        values_by_h, J, cfg, prepared["seeds"]["miscellaneous"], cfg["z_value"]
    )
    write_csv(paths.data / "ensemble_averaging.csv", ensemble)

    report_progress("exp2", "Paired h/8 versus h/16 check", started=started)
    power = cfg["h_powers"][-1]
    h_fine = 2.0**-power
    n_refine = min(cfg["refinement_schedules"], len(seed_by_h[h_fine]))
    seeds = seed_by_h[h_fine][:n_refine]
    refined = _random_values(
        model, x, targets, seeds, T=T, h=h_fine, steps=cfg["refinement_steps_per_switch"],
        p=p, r=r, beta=beta, control_term=control_term, chunk=min(cfg["chunk_size"], n_refine),
    )
    difference = values_by_h[h_fine][:n_refine] - refined
    random_rms = math.sqrt(consistency[-1]["strong_mse"])
    write_csv(paths.data / "dt_refinement.csv", [{
        "h": h_fine,
        "n_paired_schedules": n_refine,
        "primary_steps_per_switch": cfg["steps_per_switch"],
        "refined_steps_per_switch": cfg["refinement_steps_per_switch"],
        "mean_absolute_difference": float(np.abs(difference).mean()),
        "rms_difference": float(np.sqrt(np.mean(difference**2))),
        "rms_dt_difference_over_randomization_rms": float(
            np.sqrt(np.mean(difference**2)) / max(random_rms, np.finfo(float).tiny)
        ),
    }])

    report_progress("exp2", "Sampling-scheme comparison", started=started)
    comparison_schemes = {
        "bernoulli_q1_3": make_bernoulli(p, 1 / 3),
        "fixed_contiguous_r8": make_fixed_disjoint_partition(
            p, CONTIGUOUS, name="Fixed contiguous partition (r=8)"
        ),
    }
    n_compare = cfg["scheme_comparison_schedules"]
    scheme_rows = []
    for row in consistency:
        power, h = int(row["h_power"]), float(row["h"])
        # Reuses the primary scheme's own final draw at this h -- no extra
        # integration -- so the three schemes are compared at matched N.
        n_uniform = min(n_compare, len(values_by_h[h]))
        scheme_rows.append(_scheme_consistency_rows(
            "uniform_fixed_r8", values_by_h[h][:n_uniform], J, h, power, n_uniform, cfg["z_value"]
        ))

    scheme_seed_tag = {"bernoulli_q1_3": 0, "fixed_contiguous_r8": 1}
    for name, scheme in comparison_schemes.items():
        for row in consistency:
            power, h = int(row["h_power"]), float(row["h"])
            seeds = _seeds(
                prepared["seeds"]["schedule_generation"],
                3000 + 10 * power + scheme_seed_tag[name],
                n_compare,
            )
            if name == "fixed_contiguous_r8":
                values = _random_values_partition(
                    model, x, targets, seeds, T=T, h=h, steps=cfg["steps_per_switch"],
                    beta=beta, control_term=control_term, chunk=cfg["chunk_size"],
                )
            else:
                values = _random_values_bernoulli(
                    model, x, targets, scheme, seeds, T=T, h=h, steps=cfg["steps_per_switch"],
                    beta=beta, control_term=control_term, chunk=cfg["chunk_size"],
                )
            scheme_rows.append(
                _scheme_consistency_rows(name, values, J, h, power, n_compare, cfg["z_value"])
            )
    write_csv(paths.data / "scheme_comparison.csv", scheme_rows)

    scheme_slopes = []
    for name in ("uniform_fixed_r8", "bernoulli_q1_3", "fixed_contiguous_r8"):
        selected = [row for row in scheme_rows if row["scheme"] == name]
        selected.sort(key=lambda row: row["h"])
        h_arr = np.asarray([row["h"] for row in selected])
        strong_arr = np.asarray([row["strong_mse"] for row in selected])
        strong_hw = np.asarray(
            [row["strong_ci95_upper"] - row["strong_mse"] for row in selected]
        )
        weak_arr = np.asarray([row["weak_bias"] for row in selected])
        resolved_arr = np.asarray([row["weak_distinguishable_from_zero"] for row in selected])
        scheme_slopes.append({
            "scheme": name,
            **_fit(
                "strong_mse", h_arr, strong_arr, strong_hw < strong_arr,
                "excludes h where the 95% CI half-width exceeds the estimate",
            ),
        })
        scheme_slopes.append({
            "scheme": name,
            **_fit(
                "weak_bias", h_arr, weak_arr, resolved_arr,
                "only signed-bias CIs excluding zero",
            ),
        })
    write_csv(paths.data / "scheme_comparison_slopes.csv", scheme_slopes)

    total_seconds = time.perf_counter() - started
    complete_config = {
        "script": "exp2_objective_consistency.py",
        "cli": vars(args),
        "base": base,
        "experiment": cfg,
        "checkpoint_source": prepared["checkpoint_source"],
        "verified_sources": sources,
        "fixed_control": {
            "split": cfg["evaluation_split"],
            "n_data": len(x),
            "normalization": "j=J/n_data",
            "J_reference": J,
            "alpha": alpha,
            "beta": beta,
            "control_energy": control_energy,
            "control_dt": cfg["control_dt"],
        },
        "weak_sampling_design": {
            "pilot_is_independent_of_final_sample": True,
            "pilot_reference_bias_coefficient": bias_scale,
            "unresolved_points_are_excluded_from_weak_slope": True,
        },
        "timing_seconds": {"base_training": prepared["training_seconds"], "total": total_seconds},
    }
    versions = version_information(args.device, device, args.dtype)
    write_json(paths.config / "config.json", complete_config)
    write_json(paths.config / "versions.json", versions)
    write_manifest(paths.root, configuration=complete_config, seed=prepared["seeds"], versions=versions)

    figures = generate_plots(paths.root)
    report_progress("exp2", "Finished", started=started)
    return {
        "output_dir": str(paths.root),
        "figures": [str(path) for path in figures],
        "J_reference": J,
        "pilot_reference_bias_coefficient": bias_scale,
        "slopes": slopes,
        "total_seconds": total_seconds,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output-dir", default="results/exp2",
                        help="Parent directory for run folders; --plots-only reuses the most recent one")
    parser.add_argument("--checkpoint", default=None,
                        help="Frozen classification checkpoint; by default the most recent run under --base-run-dir")
    parser.add_argument("--dataset", default=None,
                        help="dataset_splits.npz to verify against; by default the one beside the checkpoint")
    parser.add_argument("--base-run-dir", default="results/exp1",
                        help="Where to look for a trained classification run when --checkpoint is omitted")
    parser.add_argument("--pilot-only", action="store_true", help="Stop after pilot samples and precision diagnostics")
    parser.add_argument("--max-final-schedules", type=int, default=None)
    parser.add_argument("--plots-only", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.pilot_only and args.plots_only:
        parser.error("--pilot-only and --plots-only cannot be combined")
    if args.plots_only:
        figures = generate_plots(ArtifactPaths.latest_run(args.output_dir))
        print(json.dumps({"figures": [str(path) for path in figures]}, indent=2))
    else:
        try:
            result = run_experiment(args)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
