#!/usr/bin/env python3
"""Reproducible trajectory-convergence and Lambda-validation experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rnode-mpl-cache")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rnode-xdg-cache")

from rnode.batches import (
    make_bernoulli,
    make_fixed_disjoint_partition,
    make_full_batch,
    make_uniform_fixed_size,
)
from rnode.design import (
    lambda_bernoulli,
    lambda_fixed_disjoint,
    lambda_monte_carlo,
    lambda_uniform_fixed_size,
    neuron_contributions_along_trajectory,
)
from rnode.integrators import integrate_masked_ensemble

try:
    from experiments._paper_common import (
        ArtifactPaths,
        bootstrap_ordered_statistic,
        fit_loglog_range,
        ordered_trajectory_statistic,
        prepare_base_model,
        report_progress,
        reference_trajectory,
        resolve_device,
        trajectory_at_times,
        version_information,
        write_csv,
        write_json,
        write_manifest,
    )
    from experiments.exp1_trajectory_convergence_plots import generate_plots
except ModuleNotFoundError:  # Direct ``python experiments/script.py`` execution.
    from _paper_common import (
        ArtifactPaths,
        bootstrap_ordered_statistic,
        fit_loglog_range,
        ordered_trajectory_statistic,
        prepare_base_model,
        report_progress,
        reference_trajectory,
        resolve_device,
        trajectory_at_times,
        version_information,
        write_csv,
        write_json,
        write_manifest,
    )
    from exp1_trajectory_convergence_plots import generate_plots


def experiment_configuration(quick: bool) -> dict:
    return {
        "h_values": (
            [0.5, 0.25, 0.125, 0.0625]
            if quick
            else [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        ),
        "rk_steps_per_switch": 8,
        "n_schedules": 20 if quick else 200,
        "n_bootstrap": 500 if quick else 2000,
        "reference_dt": 1.0 / (512 if quick else 2048),
        "reference_check_dt": 1.0 / (1024 if quick else 4096),
        "lambda_dt": 1.0 / (64 if quick else 256),
        "lambda_mc_draws": 500 if quick else 5000,
        "p": 24,
        "r": 8,
        "bernoulli_q": 1.0 / 3.0,
        "statistic_order": "mean_data(max_time(mean_schedule(squared_error)))",
    }


def _slug_schedules(scheme, h, n_schedules, seed_values):
    n_intervals = int(round(1.0 / h))
    masks = np.zeros((n_schedules, n_intervals, scheme.p), dtype=np.float64)
    schedules = []
    for schedule_index, schedule_seed in enumerate(seed_values):
        rng = np.random.default_rng(int(schedule_seed))
        batches = [scheme.sample(rng) for _ in range(n_intervals)]
        schedules.append([batch.tolist() for batch in batches])
        for interval, batch in enumerate(batches):
            masks[schedule_index, interval, batch] = 1.0
    return masks, schedules


def _reference_error(model, features, T, dt, check_dt):
    check_times, check_trajectory = reference_trajectory(model, features, T, check_dt)
    reference_times, reference_values = reference_trajectory(model, features, T, dt)
    check_aligned = trajectory_at_times(check_times, check_trajectory, reference_times)
    squared = (reference_values - check_aligned).square().sum(dim=-1)
    return (
        reference_times,
        reference_values,
        float(squared.max(dim=0).values.mean().cpu()),
    )


def run_experiment(args) -> dict:
    paths = ArtifactPaths.create(args.output_dir)
    device = resolve_device(args.device)
    experiment = experiment_configuration(args.quick)
    started = time.perf_counter()
    mode = "quick" if args.quick else "full"
    report_progress("exp1", f"Starting trajectory convergence ({mode} mode)", started=started)
    report_progress("exp1", "Loading or training the shared base model", started=started)
    prepared = prepare_base_model(
        paths,
        quick=args.quick,
        seed=args.seed,
        dtype_name=args.dtype,
        device=device,
        checkpoint=args.checkpoint,
    )
    model = prepared["model"]
    report_progress("exp1", "Base model ready; computing the fine reference", started=started)
    X_test, _, _ = prepared["datasets"]["test"]
    T = prepared["base_config"]["model"]["T"]
    if experiment["p"] != model.hidden_dim:
        raise ValueError("experiment p does not match the shared checkpoint")

    ref_times, ref_trajectory, reference_error = _reference_error(
        model,
        X_test,
        T,
        experiment["reference_dt"],
        experiment["reference_check_dt"],
    )
    np.savez_compressed(
        paths.data / "full_reference.npz",
        times=ref_times.cpu().numpy(),
        trajectory=ref_trajectory.cpu().numpy(),
        reference_error=np.array(reference_error),
    )

    partition_rng = np.random.default_rng(prepared["seeds"]["partition_generation"])
    fixed_scheme = make_fixed_disjoint_partition(
        24, n_batches=3, rng=partition_rng, name="Fixed disjoint r=8"
    )
    schemes = {
        "uniform_fixed_r8": make_uniform_fixed_size(24, 8),
        "fixed_disjoint_r8": fixed_scheme,
        "bernoulli_q1_3": make_bernoulli(24, 1 / 3),
        "full_batch": make_full_batch(24),
    }
    seed_rng = np.random.default_rng(prepared["seeds"]["schedule_generation"])
    bootstrap_rng = np.random.default_rng(prepared["seeds"]["bootstrap"])
    summary_rows = []
    raw_schedule_manifest = {}
    bootstrap_values: dict[str, dict[float, np.ndarray]] = {
        scheme_name: {} for scheme_name in schemes
    }
    total_cases = len(schemes) * len(experiment["h_values"])
    completed_cases = 0

    for scheme_name, scheme in schemes.items():
        raw_schedule_manifest[scheme_name] = {}
        for h in experiment["h_values"]:
            report_progress(
                "exp1",
                f"Trajectory case {completed_cases + 1}/{total_cases}: "
                f"scheme={scheme_name}, h={h:g}",
                started=started,
            )
            n_schedules = experiment["n_schedules"]
            schedule_seeds = seed_rng.integers(
                0, np.iinfo(np.uint32).max, size=n_schedules, dtype=np.uint32
            )
            masks, schedules = _slug_schedules(scheme, h, n_schedules, schedule_seeds)
            dt = h / experiment["rk_steps_per_switch"]
            with torch.no_grad():
                times, trajectories = integrate_masked_ensemble(
                    model,
                    X_test,
                    T,
                    dt,
                    h,
                    masks,
                    scheme.inclusion_probs,
                    method="rk4",
                )
            aligned_reference = trajectory_at_times(ref_times, ref_trajectory, times)
            squared_errors = (
                (trajectories - aligned_reference[:, None])
                .square()
                .sum(dim=-1)
                .permute(1, 0, 2)
                .cpu()
                .numpy()
            )
            estimate = ordered_trajectory_statistic(squared_errors)
            bootstrap = bootstrap_ordered_statistic(
                squared_errors, bootstrap_rng, experiment["n_bootstrap"]
            )
            bootstrap_values[scheme_name][h] = bootstrap
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
            summary_rows.append(
                {
                    "scheme": scheme_name,
                    "h": h,
                    "dt": dt,
                    "mean_error": estimate,
                    "standard_deviation": float(bootstrap.std(ddof=1)),
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "n_schedules": n_schedules,
                    "n_unique_schedules": (
                        1 if scheme_name == "full_batch" else n_schedules
                    ),
                    "n_test": X_test.shape[0],
                    "reference_error": reference_error,
                }
            )
            case_name = f"{scheme_name}_h_{str(h).replace('.', 'p')}"
            np.savez_compressed(
                paths.data / f"raw_{case_name}.npz",
                times=times.cpu().numpy(),
                squared_errors=squared_errors,
                masks=masks.astype(np.uint8),
                schedule_seeds=schedule_seeds,
            )
            raw_schedule_manifest[scheme_name][str(h)] = {
                "seeds": schedule_seeds.tolist(),
                "schedules": schedules,
            }
            completed_cases += 1

    write_csv(paths.data / "trajectory_convergence.csv", summary_rows)
    write_json(paths.data / "schedules.json", raw_schedule_manifest)

    report_progress("exp1", "Fitting convergence slopes", started=started)
    slope_rows = []
    for scheme_name in schemes:
        scheme_rows = [row for row in summary_rows if row["scheme"] == scheme_name]
        h_values = np.array([row["h"] for row in scheme_rows])
        errors = np.array([row["mean_error"] for row in scheme_rows])
        fit = fit_loglog_range(h_values, errors, reference_error)
        fit_mask = (h_values >= fit["h_min"]) & (h_values <= fit["h_max"])
        slope_samples = []
        for bootstrap_index in range(experiment["n_bootstrap"]):
            boot_errors = np.array(
                [bootstrap_values[scheme_name][h][bootstrap_index] for h in h_values]
            )
            if np.all(boot_errors[fit_mask] > 0):
                slope_samples.append(
                    np.polyfit(
                        np.log(h_values[fit_mask]), np.log(boot_errors[fit_mask]), 1
                    )[0]
                )
        slope_ci = np.quantile(slope_samples, [0.025, 0.975])
        slope_rows.append(
            {
                "scheme": scheme_name,
                **{key: value for key, value in fit.items() if key != "indices_sorted"},
                "slope_ci95_lower": float(slope_ci[0]),
                "slope_ci95_upper": float(slope_ci[1]),
                "interpretation": (
                    "diagnostic fit only; a slope above one is not evidence "
                    "for a better theoretical rate"
                ),
            }
        )
    write_csv(paths.data / "slope_fits.csv", slope_rows)

    lambda_stride = int(round(experiment["lambda_dt"] / experiment["reference_dt"]))
    lambda_times = ref_times[::lambda_stride]
    lambda_trajectory = ref_trajectory[::lambda_stride]
    with torch.no_grad():
        contributions = neuron_contributions_along_trajectory(
            model, lambda_times, lambda_trajectory
        )
    np.savez_compressed(
        paths.data / "lambda_inputs.npz",
        times=lambda_times.cpu().numpy(),
        contributions=contributions.cpu().numpy(),
    )
    report_progress("exp1", "Validating analytic Lambda formulas", started=started)
    lambda_rows = []
    for scheme_index, (scheme_name, scheme) in enumerate(schemes.items()):
        mc = lambda_monte_carlo(
            contributions,
            lambda_times,
            scheme,
            experiment["lambda_mc_draws"],
            np.random.default_rng(prepared["seeds"]["miscellaneous"] + scheme_index),
        )
        if scheme_name == "uniform_fixed_r8":
            analytic = lambda_uniform_fixed_size(contributions, lambda_times, 8)
            formula = "uniform_fixed_size"
        elif scheme_name == "fixed_disjoint_r8":
            analytic = lambda_fixed_disjoint(
                contributions, lambda_times, fixed_scheme.batches
            )
            formula = "fixed_disjoint"
        elif scheme_name == "bernoulli_q1_3":
            analytic = lambda_bernoulli(contributions, lambda_times, 1 / 3)
            formula = "bernoulli"
        else:
            analytic = contributions.new_zeros(())
            formula = "full_batch_zero"
        analytic_value = float(analytic.cpu())
        mc_value = float(mc.cpu())
        lambda_rows.append(
            {
                "scheme": scheme_name,
                "monte_carlo": mc_value,
                "analytic": analytic_value,
                "absolute_difference": abs(mc_value - analytic_value),
                "relative_difference": (
                    abs(mc_value - analytic_value) / analytic_value
                    if analytic_value > 0
                    else 0.0
                ),
                "formula": formula,
                "mc_draws": experiment["lambda_mc_draws"],
                "n_test": X_test.shape[0],
                "quadrature_dt": experiment["lambda_dt"],
            }
        )
    write_csv(paths.data / "lambda_validation.csv", lambda_rows)

    total_seconds = time.perf_counter() - started
    complete_config = {
        "script": "exp1_trajectory_convergence.py",
        "cli": vars(args),
        "base": prepared["base_config"],
        "experiment": experiment,
        "checkpoint_source": prepared["checkpoint_source"],
        "split_accuracies": prepared["accuracies"],
        "timing_seconds": {
            "training": prepared["training_seconds"],
            "total": total_seconds,
        },
    }
    write_json(paths.config / "config.json", complete_config)
    write_json(paths.config / "seeds.json", prepared["seeds"])
    versions = version_information(args.device, device, args.dtype)
    write_json(paths.config / "versions.json", versions)
    write_manifest(
        paths.root,
        configuration=complete_config,
        seed=prepared["seeds"],
        versions=versions,
    )
    write_json(
        paths.data / "summary.json",
        {
            "reference_error": reference_error,
            "slopes": slope_rows,
            "lambda": lambda_rows,
            "split_accuracies": prepared["accuracies"],
            "quick_mode_warning": args.quick,
        },
    )
    report_progress("exp1", "Generating figures", started=started)
    figures = generate_plots(paths.root)
    report_progress("exp1", "Finished", started=started)
    return {
        "output_dir": str(paths.root),
        "figures": [str(path) for path in figures],
        "slopes": slope_rows,
        "lambda": lambda_rows,
        "reference_error": reference_error,
        "total_seconds": total_seconds,
        "checkpoint": str(paths.checkpoints / "base_model.pt"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="run the smoke-size experiment"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output-dir", default="outputs/trajectory_convergence")
    parser.add_argument(
        "--checkpoint", default=None, help="reuse an existing base checkpoint"
    )
    parser.add_argument("--plots-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/rnode-mpl-cache")))
    if args.plots_only:
        figures = generate_plots(args.output_dir)
        print(
            json.dumps(
                {"plots_only": True, "figures": [str(p) for p in figures]}, indent=2
            )
        )
        return
    result = run_experiment(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
