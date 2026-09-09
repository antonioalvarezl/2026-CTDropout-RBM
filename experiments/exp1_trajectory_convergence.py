#!/usr/bin/env python3
"""Trajectory convergence and sampling-design experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
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

from rnode.batches import make_bernoulli, make_fixed_disjoint_partition, make_uniform_fixed_size
from rnode.design import (
    lambda_bernoulli,
    lambda_fixed_disjoint,
    lambda_monte_carlo,
    lambda_uniform_fixed_size,
    neuron_contributions_along_trajectory,
    random_balanced_partition,
)

try:
    from experiments._paper_common import (
        ArtifactPaths,
        load_checkpoint,
        prepare_base_model,
        reference_trajectory,
        report_progress,
        resolve_device,
        trajectory_at_times,
        version_information,
        write_csv,
        write_json,
        write_manifest,
    )
    from experiments.exp1_trajectory_convergence_plots import generate_plots
except ModuleNotFoundError:
    from _paper_common import (
        ArtifactPaths,
        load_checkpoint,
        prepare_base_model,
        reference_trajectory,
        report_progress,
        resolve_device,
        trajectory_at_times,
        version_information,
        write_csv,
        write_json,
        write_manifest,
    )
    from exp1_trajectory_convergence_plots import generate_plots


SCHEMES = ("uniform_fixed_r8", "fixed_contiguous_r8", "bernoulli_q1_3")
CONTIGUOUS = (tuple(range(8)), tuple(range(8, 16)), tuple(range(16, 24)))


def experiment_configuration(quick: bool) -> dict:
    if quick:
        powers = [3, 4, 5, 6]
        return {
            "h_powers": powers,
            "fit_h_powers": [4, 5, 6],
            "n_schedules": 24,
            "jackknife_group_size": 6,
            "rk_steps_per_switch": 8,
            # Coarsest random step in the sweep, so the time maximum runs over
            # the same instants at every h.
            "evaluation_dt": 2.0**-6,
            "reference_dt": 2.0**-11,
            "reference_check_dt": 2.0**-10,
            "lambda_dt": 2.0**-6,
            "lambda_mc_draws": 300,
            "design_n_partitions": 8,
            # None probes every generated partition.  Probing a Lambda-selected
            # subset changes the predictor distribution and therefore the
            # interpretation of the correlation.
            "design_n_error_partitions": None,
            "design_h_power": 4,
            "design_n_schedules": 6,
            "design_chunk_size": 3,
            "refinement_h_power": 6,
            "refinement_n_schedules": 6,
        }
    return {
        "h_powers": [6, 7, 8, 9, 10, 11],
        "fit_h_powers": [8, 9, 10, 11],
        "n_schedules": 300,
        "jackknife_group_size": 10,
        "rk_steps_per_switch": 8,
        # Coarsest random step in the sweep (h=2^-6 with dt=h/8), so the time
        # maximum runs over the same 513 instants at every h.
        "evaluation_dt": 2.0**-9,
        "reference_dt": 2.0**-16,
        "reference_check_dt": 2.0**-15,
        "lambda_dt": 2.0**-8,
        "lambda_mc_draws": 5000,
        "design_n_partitions": 100,
        "design_n_error_partitions": None,
        "design_h_power": 8,
        "design_n_schedules": 30,
        "design_chunk_size": 10,
        "refinement_h_power": 11,
        "refinement_n_schedules": 20,
    }


def _seeds(base: int, *tags: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(base), *map(int, tags)]))
    return rng.integers(0, np.iinfo(np.uint32).max, n, dtype=np.uint32)


def _scheme_masks(scheme, h: float, seeds) -> np.ndarray:
    n_intervals = round(1.0 / h)
    masks = np.zeros((len(seeds), n_intervals, scheme.p), dtype=np.uint8)
    for k, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        for j in range(n_intervals):
            masks[k, j, scheme.sample(rng)] = 1
    return masks


def _partition_choices(seeds, n_intervals: int, n_blocks: int) -> np.ndarray:
    return np.stack([
        np.random.default_rng(int(seed)).integers(0, n_blocks, n_intervals)
        for seed in seeds
    ])


def _partition_masks(partition, choices: np.ndarray, p: int) -> np.ndarray:
    masks = np.zeros((*choices.shape, p), dtype=np.uint8)
    for block_id, block in enumerate(partition):
        masks[:, :, np.asarray(block)] = (choices == block_id)[..., None]
    return masks


def _masked_field(model, t, state, mask, pi):
    A, b, W = model.control_parameters(t)
    activation = model.activation(state @ A.T + b)
    return (activation * (mask / pi)[:, None, :]) @ W.T


def _evaluation_stride(dt: float, evaluation_dt: float) -> int:
    """Random-solver steps between consecutive common evaluation nodes.

    The statistic maximises over time.  Evaluating it on each h's own step
    grid would maximise over 32 times more nodes at the finest h than at the
    coarsest, inflating the fine-h estimates and flattening the measured
    slope.  Every tested step divides the common node spacing, so the same
    instants are available at every h.
    """
    ratio = evaluation_dt / dt
    stride = int(round(ratio))
    if stride < 1 or not np.isclose(ratio, stride, rtol=1e-10, atol=1e-10):
        raise ValueError(
            f"step {dt} does not divide the common evaluation step {evaluation_dt}"
        )
    return stride


@torch.no_grad()
def _error_sum(
    model, x0, reference, *, reference_dt, T, h, steps_per_switch, masks, pi, evaluation_dt,
    per_schedule=False,
):
    masks = torch.as_tensor(masks, dtype=x0.dtype, device=x0.device)
    k, n_intervals, p = masks.shape
    if n_intervals != round(T / h):
        raise ValueError("mask schedule is incompatible with T/h")
    pi = torch.as_tensor(np.asarray(pi).copy(), dtype=x0.dtype, device=x0.device)
    if pi.shape != (p,):
        raise ValueError("invalid inclusion probabilities")

    dt = h / steps_per_switch
    stride = round(dt / reference_dt)
    if stride <= 0 or not np.isclose(stride * reference_dt, dt):
        raise ValueError("random and reference grids are not aligned")
    evaluation_stride = _evaluation_stride(dt, evaluation_dt)

    n_steps = n_intervals * steps_per_switch
    state = x0.unsqueeze(0).expand(k, -1, -1).clone()
    # Only the common evaluation nodes are retained, which also keeps the
    # jackknife store the same size at every h.
    shape = (n_steps // evaluation_stride + 1, len(x0))
    total = np.zeros((k, *shape) if per_schedule else shape)
    for step in range(n_steps):
        active = masks[:, step // steps_per_switch]
        t, half = step * dt, dt / 2
        k1 = _masked_field(model, t, state, active, pi)
        k2 = _masked_field(model, t + half, state + half * k1, active, pi)
        k3 = _masked_field(model, t + half, state + half * k2, active, pi)
        k4 = _masked_field(model, t + dt, state + dt * k3, active, pi)
        state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if (step + 1) % evaluation_stride:
            continue
        error = (state - reference[(step + 1) * stride].unsqueeze(0)).square().sum(-1)
        node = (step + 1) // evaluation_stride
        if per_schedule:
            total[:, node] = error.cpu().numpy()
        else:
            total[node] = error.sum(0).cpu().numpy()
    return total


def _statistic(error_sum: np.ndarray, n_schedules: int) -> float:
    return float((error_sum / n_schedules).max(0).mean())


def _jackknife(estimate, total, group_sums, n_schedules, group_size):
    leave = np.array([
        _statistic(total - np.asarray(group), n_schedules - group_size)
        for group in group_sums
    ])
    se = math.sqrt((len(leave) - 1) / len(leave) * np.square(leave - leave.mean()).sum())
    return se, estimate - 1.96 * se, estimate + 1.96 * se


def _trajectory_case(model, x0, reference, *, reference_dt, T, scheme, name, h, power, cfg, seeds):
    group_size = cfg["jackknife_group_size"]
    n_schedules = cfg["n_schedules"]
    n_groups = n_schedules // group_size
    n_nodes = round(T / cfg["evaluation_dt"]) + 1
    if n_groups * group_size != n_schedules:
        raise ValueError("n_schedules must be divisible by jackknife_group_size")

    total = np.zeros((n_nodes, len(x0)))
    with tempfile.TemporaryDirectory(prefix="rnode-jackknife-") as tmp:
        groups = np.memmap(
            Path(tmp) / "groups.dat", mode="w+", dtype=np.float64,
            shape=(n_groups, n_nodes, len(x0)),
        )
        for g in range(n_groups):
            sl = slice(g * group_size, (g + 1) * group_size)
            groups[g] = _error_sum(
                model, x0, reference, reference_dt=reference_dt, T=T, h=h,
                steps_per_switch=cfg["rk_steps_per_switch"],
                masks=_scheme_masks(scheme, h, seeds[sl]), pi=scheme.inclusion_probs,
                evaluation_dt=cfg["evaluation_dt"],
            )
            total += groups[g]
        estimate = _statistic(total, n_schedules)
        se, lo, hi = _jackknife(estimate, total, groups, n_schedules, group_size)
        del groups

    return {
        "scheme": name,
        "h_power": power,
        "h": h,
        "dt": h / cfg["rk_steps_per_switch"],
        "evaluation_dt": cfg["evaluation_dt"],
        "evaluation_nodes": n_nodes,
        "mean_error": estimate,
        "error_over_h": estimate / h,
        "jackknife_se": se,
        "collective_rse": se / estimate,
        "ci95_lower": lo,
        "ci95_upper": hi,
        "n_schedules": n_schedules,
        "fit_range": power in cfg["fit_h_powers"],
    }


def _reference(model, x0, T, fine_dt, check_dt):
    check_t, check = reference_trajectory(model, x0, T, check_dt)
    fine_t, fine = reference_trajectory(model, x0, T, fine_dt)
    fine_on_check = trajectory_at_times(fine_t, fine, check_t)
    error = (fine_on_check - check).square().sum(-1).max(0).values.mean()
    return fine_t, fine, float(error.cpu())


def _fit(rows):
    h = np.asarray([row["h"] for row in rows])
    e = np.asarray([row["mean_error"] for row in rows])
    slope, intercept = np.polyfit(np.log(h), np.log(e), 1)
    fitted = slope * np.log(h) + intercept
    total = np.square(np.log(e) - np.log(e).mean()).sum()
    r2 = 1.0 if total == 0 else 1.0 - np.square(np.log(e) - fitted).sum() / total
    # Delta-method propagation of the saved jackknife SEs. Independent
    # schedules across h; uncertainty in sampling, not an exact power-law fit.
    x = np.log(h)
    weights = (x - x.mean()) / np.square(x - x.mean()).sum()
    log_se = np.asarray([row["jackknife_se"] for row in rows]) / e
    slope_se = float(np.sqrt(np.sum((weights * log_se) ** 2)))
    return float(slope), float(intercept), float(r2), slope_se


def _partition_probe(model, x0, reference, *, reference_dt, T, h, cfg, partition, choices):
    total = np.zeros((round(T / cfg["evaluation_dt"]) + 1, len(x0)))
    scheme = make_fixed_disjoint_partition(24, partition)
    # Retain schedule errors briefly, grouping after integration. This keeps
    # the original efficient chunk size instead of reintegrating small groups.
    chunk = cfg["design_chunk_size"]
    samples = np.empty((len(choices), *total.shape))
    for start in range(0, len(choices), chunk):
        masks = _partition_masks(partition, choices[start:start + chunk], 24)
        samples[start:start+chunk] = _error_sum(
            model, x0, reference, reference_dt=reference_dt, T=T, h=h,
            steps_per_switch=cfg["rk_steps_per_switch"], masks=masks,
            pi=scheme.inclusion_probs, evaluation_dt=cfg["evaluation_dt"], per_schedule=True,
        )
    total = samples.sum(0)
    estimate = _statistic(total, len(choices))
    se = np.nan
    group_size = max(1, len(choices)//10)
    if len(choices)//group_size >= 3 and len(choices) % group_size == 0:
        groups = samples.reshape(-1, group_size, *total.shape).sum(1)
        se, _, _ = _jackknife(estimate, total, groups, len(choices), group_size)
    return estimate, se


def run_experiment(args) -> dict:
    paths = ArtifactPaths.new_run(args.output_dir)
    device = resolve_device(args.device)
    cfg = experiment_configuration(args.quick)
    if args.probe_partitions is not None:
        cfg["design_n_error_partitions"] = args.probe_partitions
    started = time.perf_counter()
    report_progress("exp1", f"Starting ({'quick' if args.quick else 'full'} mode)", started=started)

    prepared = prepare_base_model(
        paths, quick=args.quick, seed=args.seed, dtype_name=args.dtype,
        device=device, checkpoint=args.checkpoint,
    )
    model = prepared["model"]
    x_test = prepared["datasets"]["test"][0]
    T = float(prepared["base_config"]["model"]["T"])
    if model.hidden_dim != 24:
        raise ValueError("trajectory experiment expects width p=24")

    report_progress("exp1", "Computing full reference", started=started)
    ref_t, ref, ref_error = _reference(
        model, x_test, T, cfg["reference_dt"], cfg["reference_check_dt"]
    )
    write_csv(paths.data / "reference_check.csv", [{
        "reference_dt": cfg["reference_dt"],
        "check_dt": cfg["reference_check_dt"],
        "mean_data_max_time_squared_difference": ref_error,
    }])

    schemes = {
        "uniform_fixed_r8": make_uniform_fixed_size(24, 8),
        "fixed_contiguous_r8": make_fixed_disjoint_partition(24, CONTIGUOUS),
        "bernoulli_q1_3": make_bernoulli(24, 1 / 3),
    }
    base_seed = prepared["seeds"]["schedule_generation"]
    rows = []
    for scheme_index, name in enumerate(SCHEMES):
        for power in cfg["h_powers"]:
            h = 2.0**-power
            report_progress("exp1", f"Trajectory: {name}, h=2^-{power}", started=started)
            seeds = _seeds(base_seed, 101, scheme_index, power, n=cfg["n_schedules"])
            row = _trajectory_case(
                model, x_test, ref, reference_dt=cfg["reference_dt"], T=T,
                scheme=schemes[name], name=name, h=h, power=power, cfg=cfg, seeds=seeds,
            )
            row["reference_error"] = ref_error
            rows.append(row)
    write_csv(paths.data / "trajectory_convergence.csv", rows)

    slopes = []
    for name in SCHEMES:
        selected = sorted(
            (row for row in rows if row["scheme"] == name and row["fit_range"]),
            key=lambda row: row["h"],
        )
        slope, intercept, r2, slope_se = _fit(selected)
        slopes.append({
            "scheme": name,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r2,
            "slope_se": slope_se,
            "slope_ci95_lower": slope - 1.96 * slope_se,
            "slope_ci95_upper": slope + 1.96 * slope_se,
            "n_points": len(selected),
            "h_min": selected[0]["h"],
            "h_max": selected[-1]["h"],
            "interpretation": "descriptive OLS; approximate delta-method MC interval from independent h samples, not power-law model uncertainty",
        })
    write_csv(paths.data / "slope_fits.csv", slopes)

    report_progress("exp1", "Validating Lambda formulas", started=started)
    stride = round(cfg["lambda_dt"] / cfg["reference_dt"])
    lambda_t, lambda_x = ref_t[::stride], ref[::stride]
    with torch.no_grad():
        contributions = neuron_contributions_along_trajectory(model, lambda_t, lambda_x)

    analytic = {
        "uniform_fixed_r8": lambda_uniform_fixed_size(contributions, lambda_t, 8),
        "fixed_contiguous_r8": lambda_fixed_disjoint(contributions, lambda_t, CONTIGUOUS),
        "bernoulli_q1_3": lambda_bernoulli(contributions, lambda_t, 1 / 3),
    }
    lambda_rows = []
    for index, name in enumerate(SCHEMES):
        mc, mc_se = lambda_monte_carlo(
            contributions, lambda_t, schemes[name], cfg["lambda_mc_draws"],
            np.random.default_rng(int(_seeds(prepared["seeds"]["miscellaneous"], 202, index, n=1)[0])),
            return_se=True,
        )
        a, m = float(analytic[name].cpu()), float(mc.cpu())
        lambda_rows.append({
            "scheme": name,
            "analytic": a,
            "monte_carlo": m,
            "relative_difference": abs(m - a) / a if a else 0.0,
            "pi_min": float(np.min(schemes[name].inclusion_probs)),
            "expected_batch_size": schemes[name].expected_batch_size,
            "mc_draws": cfg["lambda_mc_draws"],
            "monte_carlo_se": float(mc_se.cpu()),
            "mc_ci95_lower": m - 1.96 * float(mc_se.cpu()),
            "mc_ci95_upper": m + 1.96 * float(mc_se.cpu()),
        })
    write_csv(paths.data / "lambda_validation.csv", lambda_rows)

    report_progress("exp1", "Fixed-partition design study", started=started)
    rng = np.random.default_rng(prepared["seeds"]["partition_generation"])
    seen = {CONTIGUOUS}
    partitions = []
    while len(partitions) < cfg["design_n_partitions"]:
        partition = random_balanced_partition(24, 8, rng)
        if partition not in seen:
            seen.add(partition)
            partitions.append(partition)

    lambdas = np.asarray([
        float(lambda_fixed_disjoint(contributions, lambda_t, partition).cpu())
        for partition in partitions
    ])
    n_probe = cfg["design_n_error_partitions"]
    if n_probe is None:
        probed = set(range(len(partitions)))
        probe_rule = "every generated partition"
    else:
        # Retained only as an explicit budget override.  Any Lambda-based
        # subset selects on the predictor, so the reported correlation is
        # conditional on that selection and is flagged as such below.
        order = np.argsort(lambdas)
        ranks = np.unique(np.round(
            np.linspace(0.05, 0.95, n_probe) * (len(order) - 1)
        ).astype(int))
        probed = {int(order[rank]) for rank in ranks}
        probe_rule = "Lambda-quantile subset (selected on the predictor)"

    h_probe = 2.0**-cfg["design_h_power"]
    probe_seeds = _seeds(base_seed, 303, cfg["design_h_power"], n=cfg["design_n_schedules"])
    choices = _partition_choices(probe_seeds, round(T / h_probe), 3)
    design_rows = []
    counter = 0
    for index, (partition, lam) in enumerate(zip(partitions, lambdas)):
        do_probe = index in probed
        error = np.nan
        error_se = np.nan
        if do_probe:
            counter += 1
            report_progress("exp1", f"Partition probe {counter}/{len(probed)}", started=started)
            error, error_se = _partition_probe(
                model, x_test, ref, reference_dt=cfg["reference_dt"], T=T,
                h=h_probe, cfg=cfg, partition=partition, choices=choices,
            )
        design_rows.append({
            "partition_id": f"random_{index:03d}",
            "kind": "random",
            "partition_zero_based": [list(block) for block in partition],
            "integrated_lambda": float(lam),
            "trajectory_probe_performed": do_probe,
            "probe_selection": probe_rule if do_probe else "not probed",
            "h_probe": h_probe,
            "trajectory_error": error,
            "jackknife_se": error_se,
            "error_over_h_se": error_se / h_probe,
            "error_over_h": error / h_probe if do_probe else np.nan,
            "n_schedules": cfg["design_n_schedules"] if do_probe else 0,
        })

    contiguous_lambda = float(analytic["fixed_contiguous_r8"].cpu())
    contiguous_error, contiguous_se = _partition_probe(
        model, x_test, ref, reference_dt=cfg["reference_dt"], T=T,
        h=h_probe, cfg=cfg, partition=CONTIGUOUS, choices=choices,
    )
    design_rows.append({
        "partition_id": "contiguous",
        "kind": "contiguous_baseline",
        "partition_zero_based": [list(block) for block in CONTIGUOUS],
        "integrated_lambda": contiguous_lambda,
        "trajectory_probe_performed": True,
        "probe_selection": "data-independent baseline",
        "h_probe": h_probe,
        "trajectory_error": contiguous_error,
        "jackknife_se": contiguous_se,
        "error_over_h_se": contiguous_se / h_probe,
        "error_over_h": contiguous_error / h_probe,
        "n_schedules": cfg["design_n_schedules"],
    })
    write_csv(paths.data / "partition_design.csv", design_rows)

    probed_rows = [row for row in design_rows if row["kind"] == "random" and row["trajectory_probe_performed"]]
    probe_lambda = np.asarray([row["integrated_lambda"] for row in probed_rows])
    probe_error = np.asarray([row["error_over_h"] for row in probed_rows])
    correlation = (
        float(np.corrcoef(probe_lambda, probe_error)[0, 1])
        if len(probed_rows) > 1 and np.std(probe_lambda) > 0 and np.std(probe_error) > 0
        else np.nan
    )
    spearman = np.nan
    if len(probed_rows) > 1:
        rank_lambda = np.argsort(np.argsort(probe_lambda)).astype(float)
        rank_error = np.argsort(np.argsort(probe_error)).astype(float)
        if np.std(rank_lambda) > 0 and np.std(rank_error) > 0:
            spearman = float(np.corrcoef(rank_lambda, rank_error)[0, 1])
    uniform_lambda = float(analytic["uniform_fixed_r8"].cpu())
    partition_summary = {
        "n_random_partitions": len(partitions),
        "n_trajectory_probes": len(probed_rows),
        "probe_selection_rule": probe_rule,
        "probes_cover_all_partitions": len(probed_rows) == len(partitions),
        "mean_random_partition_lambda": float(lambdas.mean()),
        "se_random_partition_lambda": float(lambdas.std(ddof=1) / math.sqrt(len(lambdas))),
        "uniform_fixed_size_lambda": uniform_lambda,
        "relative_mean_difference": abs(float(lambdas.mean()) - uniform_lambda) / uniform_lambda,
        "pearson_lambda_vs_error_over_h_on_probed_partitions": correlation,
        "spearman_lambda_vs_error_over_h_on_probed_partitions": spearman,
        "correlation_caveat": (
            "association across partitions at one probe scale; the bound "
            "involves sqrt(Lambda) and stability, so this is not a predicted "
            "proportionality"
            if len(probed_rows) == len(partitions)
            else "probes were selected on Lambda, so this correlation is "
            "conditional on that selection; no direction of bias is guaranteed"
        ),
    }
    write_csv(paths.data / "partition_variance_identity.csv", [partition_summary])

    report_progress("exp1", "Paired h/8 versus h/16 check", started=started)
    power = cfg["refinement_h_power"]
    h = 2.0**-power
    refinement = []
    for scheme_index, name in enumerate(SCHEMES):
        seeds = _seeds(base_seed, 101, scheme_index, power, n=cfg["n_schedules"])
        seeds = seeds[: cfg["refinement_n_schedules"]]
        masks = _scheme_masks(schemes[name], h, seeds)
        primary = _error_sum(
            model, x_test, ref, reference_dt=cfg["reference_dt"], T=T, h=h,
            steps_per_switch=8, masks=masks, pi=schemes[name].inclusion_probs,
            evaluation_dt=cfg["evaluation_dt"],
        )
        refined = _error_sum(
            model, x_test, ref, reference_dt=cfg["reference_dt"], T=T, h=h,
            steps_per_switch=16, masks=masks, pi=schemes[name].inclusion_probs,
            evaluation_dt=cfg["evaluation_dt"],
        )
        e8, e16 = _statistic(primary, len(seeds)), _statistic(refined, len(seeds))
        refinement.append({
            "scheme": name,
            "h": h,
            "n_paired_schedules": len(seeds),
            "estimate_h_over_8": e8,
            "estimate_h_over_16": e16,
            "relative_difference": abs(e8 - e16) / max(abs(e16), np.finfo(float).tiny),
        })
    write_csv(paths.data / "dt_refinement.csv", refinement)

    total_seconds = time.perf_counter() - started
    complete_config = {
        "script": "exp1_trajectory_convergence.py",
        "cli": vars(args),
        "base": prepared["base_config"],
        "experiment": cfg,
        "main_fixed_partition": {
            "description": "contiguous data-independent baseline; not optimized",
            "zero_based": [list(block) for block in CONTIGUOUS],
        },
        "checkpoint_source": prepared["checkpoint_source"],
        "timing_seconds": {"base_training": prepared["training_seconds"], "total": total_seconds},
    }
    versions = version_information(args.device, device, args.dtype)
    write_json(paths.config / "config.json", complete_config)
    write_json(paths.config / "versions.json", versions)
    write_manifest(paths.root, configuration=complete_config, seed=prepared["seeds"], versions=versions)

    figures = generate_plots(paths.root)
    report_progress("exp1", "Finished", started=started)
    return {
        "output_dir": str(paths.root),
        "figures": [str(path) for path in figures],
        "reference_error": ref_error,
        "slopes": slopes,
        "partition_summary": partition_summary,
        "total_seconds": total_seconds,
        "checkpoint": str(paths.checkpoints / "base_model.pt"),
    }


def run_diagnostics(args):
    """Supplement an existing run, never retrain or repeat its full sweep."""
    source = ArtifactPaths.latest_run(args.output_dir)
    saved = json.loads((source / "config/config.json").read_text())
    cfg = saved["experiment"]
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    model, payload = load_checkpoint(source / "checkpoints/base_model.pt", device=device)
    model.requires_grad_(False)
    with np.load(source / "data/dataset_splits.npz", allow_pickle=False) as data:
        x = torch.as_tensor(data["test_X"], dtype=next(model.parameters()).dtype, device=device)
    paths = ArtifactPaths.new_run(source, prefix="diagnostics")
    started = time.perf_counter()
    T = float(payload["model_config"]["T"])
    report_progress("exp1-check", "Frozen full reference", started=started)
    rt, ref = reference_trajectory(model, x, T, cfg["reference_dt"])
    schemes = (make_uniform_fixed_size(24, 8), make_fixed_disjoint_partition(24, CONTIGUOUS),
               make_bernoulli(24, 1/3))
    seeds = payload["seeds"]
    mesh_rows = []
    n = min(10, cfg["n_schedules"])
    # The production solver already visits these denser nodes. Retain them
    # once and subsample, isolating observation-grid error from RK4 error.
    for power in (min(cfg["fit_h_powers"]), max(cfg["fit_h_powers"])):
        h = 2.0**-power
        for i, (name, scheme) in enumerate(zip(SCHEMES, schemes)):
            report_progress("exp1-check", f"Observation mesh: {name}, 2^-{power}", started=started)
            case_seeds = _seeds(seeds["schedule_generation"], 101, i, power, n=cfg["n_schedules"])[:n]
            fine_spacing = max(cfg["evaluation_dt"]/4, h/cfg["rk_steps_per_switch"])
            factor = round(cfg["evaluation_dt"]/fine_spacing)
            total = _error_sum(model, x, ref, reference_dt=cfg["reference_dt"], T=T, h=h,
                               steps_per_switch=cfg["rk_steps_per_switch"],
                               masks=_scheme_masks(scheme, h, case_seeds), pi=scheme.inclusion_probs,
                               evaluation_dt=fine_spacing)
            coarse, medium, fine = (_statistic(total[::s], n) for s in (factor, max(1, factor//2), 1))
            mesh_rows.append({"scheme": name, "h": h, "n_schedules": n, "n_data": len(x),
                              "evaluation_dt": cfg["evaluation_dt"], "coarse": coarse,
                              "refined_evaluation_dt": fine_spacing,
                              "half_spacing": medium, "quarter_spacing": fine,
                              "relative_change": (fine-coarse)/fine})
    write_csv(paths.data / "observation_mesh.csv", mesh_rows)

    report_progress("exp1-check", "Lambda uncertainty and time quadrature", started=started)
    stride = round(cfg["lambda_dt"] / cfg["reference_dt"])
    with torch.no_grad():
        terms = neuron_contributions_along_trajectory(model, rt[::stride//2], ref[::stride//2])
        times = rt[::stride//2]
        analytic = (lambda t, s: lambda_uniform_fixed_size(t, s, 8),
                    lambda t, s: lambda_fixed_disjoint(t, s, CONTIGUOUS),
                    lambda t, s: lambda_bernoulli(t, s, 1/3))
        lambda_rows = []
        for i, (name, scheme, formula) in enumerate(zip(SCHEMES, schemes, analytic)):
            mean, se = lambda_monte_carlo(terms[::2], times[::2], scheme, cfg["lambda_mc_draws"],
                np.random.default_rng(int(_seeds(seeds["miscellaneous"], 202, i, n=1)[0])), return_se=True)
            a, finer = float(formula(terms[::2], times[::2])), float(formula(terms, times))
            lambda_rows.append({"scheme": name, "analytic": a, "analytic_refined": finer,
                                "relative_quadrature_change": abs(finer-a)/abs(finer),
                                "monte_carlo": float(mean), "se": float(se),
                                "ci95_lower": float(mean-1.96*se), "ci95_upper": float(mean+1.96*se),
                                "n_draws": cfg["lambda_mc_draws"]})
    write_csv(paths.data / "lambda_uncertainty.csv", lambda_rows)
    del terms

    # Raw schedule-level contributions were not saved historically. Recompute
    # only the first three generated partitions and the baseline, not all 100.
    with (source / "data/partition_design.csv").open() as stream:
        old_rows = list(csv.DictReader(stream))
    chosen = [r for r in old_rows if r["kind"] == "random"][:3]
    chosen += [r for r in old_rows if r["kind"] == "contiguous_baseline"]
    h = 2.0**-cfg["design_h_power"]
    choices = _partition_choices(_seeds(seeds["schedule_generation"], 303, cfg["design_h_power"],
                                       n=cfg["design_n_schedules"]), round(T/h), 3)
    checked = []
    for row in chosen:
        report_progress("exp1-check", f"Schedule uncertainty: {row['partition_id']}", started=started)
        e, se = _partition_probe(model, x, ref, reference_dt=cfg["reference_dt"], T=T, h=h,
                                 cfg=cfg, partition=json.loads(row["partition_zero_based"]), choices=choices)
        checked.append({"partition_id": row["partition_id"], "mean_error": e, "jackknife_se": se,
                        "ci95_lower": e-1.96*se, "ci95_upper": e+1.96*se,
                        "difference_from_saved": e-float(row["trajectory_error"]),
                        "n_schedules": len(choices), "selection": "first three generated, plus baseline"})
    write_csv(paths.data / "partition_uncertainty_subset.csv", checked)
    write_json(paths.config / "config.json", {
        "source_run": str(source), "source_checkpoint_sha256": hashlib.sha256(
            (source / "checkpoints/base_model.pt").read_bytes()).hexdigest(),
        "configuration": cfg, "threads": torch.get_num_threads(),
        "scope": "observation mesh: 10 original schedules, two fine-range endpoints, all data; Lambda: original draws; partitions: 4 only",
        "unavailable": "per-partition intervals for the other historical partitions require reintegration; none fabricated",
        "seconds": time.perf_counter()-started,
    })
    figures = generate_plots(source, figure_dir=paths.figures)
    return {"output_dir": str(paths.root), "figures": [str(p) for p in figures],
            "seconds": time.perf_counter()-started}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--output-dir",
        default="results/exp1",
        help="Parent directory for run folders; --plots-only reuses the most recent one",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--probe-partitions",
        type=int,
        default=None,
        help="Budget override: probe only this many partitions, chosen by Lambda "
        "quantiles. The resulting correlation is selected on the predictor. "
        "The default probes every generated partition.",
    )
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true",
                        help="Supplement an existing run with bounded checks and new figures, without training")
    return parser


def main():
    args = build_parser().parse_args()
    if args.diagnostics_only:
        print(json.dumps(run_diagnostics(args), indent=2))
    elif args.plots_only:
        figures = generate_plots(ArtifactPaths.latest_run(args.output_dir))
        print(json.dumps({"figures": [str(path) for path in figures]}, indent=2))
    else:
        print(json.dumps(run_experiment(args), indent=2))


if __name__ == "__main__":
    main()
