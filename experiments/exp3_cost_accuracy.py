#!/usr/bin/env python3
"""Calibrate and test the theoretical cost--accuracy trade-off."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rnode-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rnode-xdg-cache")

from rnode.batches import (
    make_bernoulli,
    make_fixed_disjoint_partition,
    make_full_batch,
    make_uniform_fixed_size,
    sample_batch_sequence,
)
from rnode.integrators import integrate_fixed_grid

try:
    from experiments._paper_common import (
        ArtifactPaths,
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
    from experiments.exp3_cost_accuracy_plots import generate_plots
except ModuleNotFoundError:
    from _paper_common import (
        ArtifactPaths,
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
    from exp3_cost_accuracy_plots import generate_plots


def experiment_configuration(quick: bool) -> dict:
    return {
        "gamma": 0.25,
        "h_values": (
            [0.5, 0.25, 0.125, 0.0625]
            if quick
            else [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        ),
        "reference_dt": 1 / (1024 if quick else 4096),
        "fine_schedule_steps_per_h": 32,
        "test_schedules": 6 if quick else 100,
        "calibration_schedules": 4 if quick else 40,
        "wall_warmup": 1 if quick else 3,
        "wall_repetitions": 3 if quick else 10,
        "uniform_r": [4, 8, 12],
        "partition_r": 8,
        "bernoulli_q": 1 / 3,
        "n_tolerances": 8 if quick else 16,
        "integrator": "explicit_euler",
        "metric": "sqrt(mean_data(max_time(mean_schedule(squared_euclidean_error))))",
        "proxy": "n_steps * expected_active_neurons",
    }


def _load_optimized_partition(path, p):
    payload = json.loads(Path(path).read_text())
    partition = payload["minimization"]["partition"]
    if partition is None or sorted(sum(partition, [])) != list(range(p)):
        raise ValueError("design results do not contain a valid optimized partition")
    return partition, payload["minimization"]


def _make_schemes(p, config, optimized_partition, rng):
    random_partition = np.array_split(rng.permutation(p), p // config["partition_r"])
    schemes = {"full": make_full_batch(p)}
    for r in config["uniform_r"]:
        schemes[f"uniform_r{r}"] = make_uniform_fixed_size(p, r)
    schemes["optimized_disjoint_r8"] = make_fixed_disjoint_partition(
        p, optimized_partition, name="Optimized disjoint, r=8"
    )
    schemes["random_disjoint_r8"] = make_fixed_disjoint_partition(
        p, random_partition, name="Random disjoint, r=8"
    )
    schemes["bernoulli_q1_3"] = make_bernoulli(p, config["bernoulli_q"])
    return schemes, [block.tolist() for block in random_partition]


def _rms(trajectories, reference):
    errors = (trajectories - reference[:, None]).square().sum(dim=-1)
    squared = errors.permute(1, 0, 2).detach().cpu().numpy()
    return float(np.sqrt(ordered_trajectory_statistic(squared))), squared


def _run_schedules(model, features, T, h, dt, scheme, schedules, method):
    trajectories = []
    times = None
    with torch.no_grad():
        for schedule in schedules:
            times, trajectory = integrate_fixed_grid(
                model,
                features,
                T,
                dt,
                h,
                schedule,
                inclusion_probs=scheme.inclusion_probs,
                method=method,
            )
            trajectories.append(trajectory)
    return times, torch.stack(trajectories, dim=1)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _benchmark(call, device, warmup, repetitions):
    for _ in range(warmup):
        call()
    _synchronize(device)
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        call()
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    return float(np.median(samples)), samples


def _hstar(epsilon, stochastic_constant, discretization_constant, gamma):
    if stochastic_constant <= 0:
        return epsilon / (discretization_constant * gamma)
    inner = np.sqrt(
        1 + 4 * discretization_constant * gamma * epsilon / stochastic_constant
    )
    return 4 * epsilon**2 / stochastic_constant / (1 + inner) ** 2


def _crossover_status(tolerances, rbm_costs, full_costs):
    cheaper = np.asarray(rbm_costs) < np.asarray(full_costs)
    changes = np.flatnonzero(cheaper[1:] != cheaper[:-1]) + 1
    return {
        "within_measured_range": bool(len(changes)),
        "tolerance": float(tolerances[changes[0]]) if len(changes) else None,
        "rbm_cheaper_anywhere": bool(cheaper.any()),
        "rbm_cheaper_everywhere": bool(cheaper.all()),
    }


def _default_design_path(quick: bool) -> str:
    """Return the design artifact produced by the matching runner mode."""
    root = "current_paper_quick" if quick else "current_paper_full"
    return f"outputs/{root}/design/data/milp_results.json"


def run_experiment(args):
    paths = ArtifactPaths.create(args.output_dir)
    device = resolve_device(args.device)
    config = experiment_configuration(args.quick)
    started = time.perf_counter()
    mode = "quick" if args.quick else "full"
    report_progress("exp3", f"Starting cost--accuracy analysis ({mode} mode)", started=started)
    report_progress("exp3", "Loading the shared checkpoint and partition design", started=started)
    prepared = prepare_base_model(
        paths,
        quick=args.quick,
        seed=args.seed,
        dtype_name=args.dtype,
        device=device,
        checkpoint=args.checkpoint,
    )
    model = prepared["model"]
    T = prepared["base_config"]["model"]["T"]
    p = model.hidden_dim
    design_path = args.design_results or _default_design_path(args.quick)
    optimized, optimizer_record = _load_optimized_partition(design_path, p)
    scheme_rng = np.random.default_rng(prepared["seeds"]["partition_generation"] + 31)
    schemes, random_partition = _make_schemes(p, config, optimized, scheme_rng)
    labels = {
        "full": "Full model",
        "uniform_r4": "Uniform fixed-size, r=4",
        "uniform_r8": "Uniform fixed-size, r=8",
        "uniform_r12": "Uniform fixed-size, r=12",
        "optimized_disjoint_r8": "Optimized disjoint, r=8",
        "random_disjoint_r8": "Random disjoint, r=8",
        "bernoulli_q1_3": "Bernoulli, retention 1/3",
    }
    X_calibration = prepared["datasets"]["calibration"][0]
    X_test = prepared["datasets"]["test"][0]
    calibration_reference_times, calibration_reference = reference_trajectory(
        model, X_calibration, T, config["reference_dt"]
    )
    test_reference_times, test_reference = reference_trajectory(
        model, X_test, T, config["reference_dt"]
    )

    seed_rng = np.random.default_rng(prepared["seeds"]["schedule_generation"] + 31)
    calibration_rows = []
    constants = {}
    fine_h_values = sorted(config["h_values"])[:2]
    calibration_total = (len(schemes) - 1) * len(fine_h_values)
    calibration_done = 0
    for scheme_name, scheme in schemes.items():
        if scheme_name == "full":
            continue
        mse_values, c_values = [], []
        for h in fine_h_values:
            report_progress(
                "exp3",
                f"Calibration {calibration_done + 1}/{calibration_total}: "
                f"scheme={scheme_name}, h={h:g}",
                started=started,
            )
            seeds = seed_rng.integers(
                0, 2**32 - 1, config["calibration_schedules"], dtype=np.uint32
            )
            schedules = [
                sample_batch_sequence(
                    scheme, round(T / h), np.random.default_rng(int(seed))
                )
                for seed in seeds
            ]
            fine_dt = h / config["fine_schedule_steps_per_h"]
            fine_times, fine_trajectories = _run_schedules(
                model, X_calibration, T, h, fine_dt, scheme, schedules, "rk4"
            )
            full_at_fine = trajectory_at_times(
                calibration_reference_times, calibration_reference, fine_times
            )
            stochastic_rms, _ = _rms(fine_trajectories, full_at_fine)
            coarse_dt = config["gamma"] * h
            coarse_times, coarse_trajectories = _run_schedules(
                model, X_calibration, T, h, coarse_dt, scheme, schedules, "euler"
            )
            fine_at_coarse = torch.stack(
                [
                    trajectory_at_times(
                        fine_times, fine_trajectories[:, k], coarse_times
                    )
                    for k in range(len(schedules))
                ],
                dim=1,
            )
            # The paired statistic must compare schedule k with the same schedule k.
            paired = (
                (coarse_trajectories - fine_at_coarse)
                .square()
                .sum(-1)
                .permute(1, 0, 2)
                .cpu()
                .numpy()
            )
            discretization_rms = float(np.sqrt(ordered_trajectory_statistic(paired)))
            mse_values.append(stochastic_rms**2)
            c_values.append(discretization_rms / coarse_dt)
            calibration_rows.append(
                {
                    "scheme": scheme_name,
                    "h": h,
                    "stochastic_mse_fine_schedule": stochastic_rms**2,
                    "mse_over_h": stochastic_rms**2 / h,
                    "euler_vs_fine_schedule_rms": discretization_rms,
                    "c_D_sample": discretization_rms / coarse_dt,
                    "n_schedules": len(schedules),
                }
            )
            calibration_done += 1
        h_array = np.asarray(fine_h_values)
        stochastic_constant = float(
            np.dot(h_array, mse_values) / np.dot(h_array, h_array)
        )
        constants[scheme_name] = {
            "S_D": stochastic_constant,
            "c_D": float(np.median(c_values)),
        }

    h_fine = min(config["h_values"])
    full_dt = config["gamma"] * h_fine
    with torch.no_grad():
        full_times, full_euler = integrate_fixed_grid(
            model, X_calibration, T, full_dt, h_fine, method="euler"
        )
    full_reference_coarse = trajectory_at_times(
        calibration_reference_times, calibration_reference, full_times
    )
    full_rms, _ = _rms(full_euler[:, None], full_reference_coarse)
    c_F = full_rms / full_dt
    constants["full"] = {"S_D": 0.0, "c_D": c_F}
    calibration_rows.append(
        {
            "scheme": "full",
            "h": h_fine,
            "stochastic_mse_fine_schedule": 0.0,
            "mse_over_h": 0.0,
            "euler_vs_fine_schedule_rms": full_rms,
            "c_D_sample": c_F,
            "n_schedules": 1,
        }
    )
    write_csv(paths.data / "calibration.csv", calibration_rows)
    write_json(
        paths.data / "calibrated_constants.json", {"schemes": constants, "c_F": c_F}
    )

    report_progress("exp3", "Calibration finished; evaluating test cost and error", started=started)
    rows, seed_manifest, wall_samples = [], {}, {}
    test_total = len(schemes) * len(config["h_values"])
    test_done = 0
    for scheme_name, scheme in schemes.items():
        seed_manifest[scheme_name], wall_samples[scheme_name] = {}, {}
        for h in config["h_values"]:
            report_progress(
                "exp3",
                f"Test case {test_done + 1}/{test_total}: scheme={scheme_name}, h={h:g}",
                started=started,
            )
            dt = config["gamma"] * h
            n_schedules = 1 if scheme_name == "full" else config["test_schedules"]
            seeds = seed_rng.integers(0, 2**32 - 1, n_schedules, dtype=np.uint32)
            schedules = (
                [None]
                if scheme_name == "full"
                else [
                    sample_batch_sequence(
                        scheme, round(T / h), np.random.default_rng(int(seed))
                    )
                    for seed in seeds
                ]
            )
            if scheme_name == "full":
                with torch.no_grad():
                    times, trajectory = integrate_fixed_grid(
                        model, X_test, T, dt, h, method="euler"
                    )
                trajectories = trajectory[:, None]
            else:
                times, trajectories = _run_schedules(
                    model, X_test, T, h, dt, scheme, schedules, "euler"
                )
            reference = trajectory_at_times(test_reference_times, test_reference, times)
            rms, squared = _rms(trajectories, reference)
            n_steps = round(T / dt)
            proxy = n_steps * scheme.expected_batch_size
            bound = (
                c_F * dt
                if scheme_name == "full"
                else np.sqrt(constants[scheme_name]["S_D"] * h)
                + constants[scheme_name]["c_D"] * dt
            )
            benchmark_schedule = schedules[0]

            def timed_call():
                with torch.no_grad():
                    integrate_fixed_grid(
                        model,
                        X_test,
                        T,
                        dt,
                        h,
                        benchmark_schedule,
                        inclusion_probs=(
                            None if scheme_name == "full" else scheme.inclusion_probs
                        ),
                        method="euler",
                    )

            wall_median, samples = _benchmark(
                timed_call, device, config["wall_warmup"], config["wall_repetitions"]
            )
            rows.append(
                {
                    "scheme": scheme_name,
                    "scheme_label": labels[scheme_name],
                    "h": h,
                    "dt": dt,
                    "rms_error": rms,
                    "calibrated_bound": bound,
                    "n_steps": n_steps,
                    "expected_active_neurons": scheme.expected_batch_size,
                    "proxy_cost": proxy,
                    "wall_clock_median_seconds": wall_median,
                    "n_test": len(X_test),
                    "n_schedules": n_schedules,
                }
            )
            key = str(h)
            seed_manifest[scheme_name][key] = seeds.tolist()
            wall_samples[scheme_name][key] = samples
            np.savez_compressed(
                paths.data / f"raw_{scheme_name}_h_{str(h).replace('.', 'p')}.npz",
                squared_errors=squared,
            )
            test_done += 1
    write_csv(paths.data / "cost_accuracy.csv", rows)
    write_json(paths.data / "schedule_seeds.json", seed_manifest)
    write_json(paths.data / "wall_clock_samples.json", wall_samples)

    report_progress("exp3", "Comparing predicted and observed tolerances", started=started)
    positive_errors = np.array(
        [row["rms_error"] for row in rows if row["rms_error"] > 0]
    )
    tolerances = np.geomspace(
        positive_errors.min() * 1.05,
        positive_errors.max() * 1.05,
        config["n_tolerances"],
    )
    tolerance_rows = []
    for epsilon in tolerances:
        per_tolerance = []
        for scheme_name, scheme in schemes.items():
            candidates = [
                row
                for row in rows
                if row["scheme"] == scheme_name and row["rms_error"] <= epsilon
            ]
            observed = max(candidates, key=lambda row: row["h"]) if candidates else None
            constant = constants[scheme_name]
            predicted_h = _hstar(
                epsilon, constant["S_D"], constant["c_D"], config["gamma"]
            )
            cost_h = min(predicted_h, T)
            predicted_cost = T / (config["gamma"] * cost_h) * scheme.expected_batch_size
            record = {
                "tolerance": epsilon,
                "scheme": scheme_name,
                "scheme_label": labels[scheme_name],
                "predicted_h": predicted_h,
                "predicted_h_capped_at_T": cost_h,
                "observed_h": observed["h"] if observed else np.nan,
                "predicted_proxy_cost": predicted_cost,
                "observed_min_proxy_cost": (
                    observed["proxy_cost"] if observed else np.nan
                ),
                "empirical_global_optimum": False,
            }
            per_tolerance.append(record)
        feasible = [
            row for row in per_tolerance if np.isfinite(row["observed_min_proxy_cost"])
        ]
        if feasible:
            min(feasible, key=lambda row: row["observed_min_proxy_cost"])[
                "empirical_global_optimum"
            ] = True
        tolerance_rows.extend(per_tolerance)
    write_csv(paths.data / "tolerance_comparison.csv", tolerance_rows)

    grouped = [
        [row for row in tolerance_rows if row["tolerance"] == epsilon]
        for epsilon in tolerances
    ]
    observed_rbm = [
        min(
            [
                row["observed_min_proxy_cost"]
                for row in group
                if row["scheme"] != "full"
                and np.isfinite(row["observed_min_proxy_cost"])
            ],
            default=np.nan,
        )
        for group in grouped
    ]
    observed_full = [
        next(row["observed_min_proxy_cost"] for row in group if row["scheme"] == "full")
        for group in grouped
    ]
    predicted_rbm = [
        min(row["predicted_proxy_cost"] for row in group if row["scheme"] != "full")
        for group in grouped
    ]
    predicted_full = [
        next(row["predicted_proxy_cost"] for row in group if row["scheme"] == "full")
        for group in grouped
    ]
    observed_valid = np.isfinite(observed_rbm) & np.isfinite(observed_full)
    crossover = {
        "observed": (
            _crossover_status(
                tolerances[observed_valid],
                np.asarray(observed_rbm)[observed_valid],
                np.asarray(observed_full)[observed_valid],
            )
            if observed_valid.any()
            else {
                "within_measured_range": False,
                "tolerance": None,
                "rbm_cheaper_anywhere": False,
                "rbm_cheaper_everywhere": False,
            }
        ),
        "predicted": _crossover_status(tolerances, predicted_rbm, predicted_full),
    }
    write_json(paths.data / "crossover.json", crossover)

    elapsed = time.perf_counter() - started
    hardware = version_information(args.device, device, args.dtype)
    hardware.update(
        {"processor": platform.processor(), "torch_threads": torch.get_num_threads()}
    )
    if device.type == "cuda":
        hardware["cuda_device_name"] = torch.cuda.get_device_name(device)
    write_json(paths.config / "hardware.json", hardware)
    write_json(paths.config / "seeds.json", prepared["seeds"])
    complete_config = {
        "script": "exp3_cost_accuracy.py",
        "cli": vars(args),
        "experiment": config,
        "base": prepared["base_config"],
        "design_results": str(Path(design_path).resolve()),
        "optimized_partition_solver_record": optimizer_record,
        "partitions": {"optimized": optimized, "random": random_partition},
        "checkpoint_source": prepared["checkpoint_source"],
        "elapsed_seconds": elapsed,
        "data_usage": {
            "calibration": "S_D, c_D and c_F only",
            "test": "h-star and cost comparison only",
        },
    }
    write_json(paths.config / "config.json", complete_config)
    write_manifest(
        paths.root,
        configuration=complete_config,
        seed=prepared["seeds"],
        versions=hardware,
    )
    report_progress("exp3", "Generating figures", started=started)
    figures = generate_plots(paths.root)
    report_progress("exp3", "Finished", started=started)
    return {
        "output_dir": str(paths.root),
        "figures": [str(path) for path in figures],
        "constants": constants,
        "crossover": crossover,
        "total_seconds": elapsed,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output-dir", default="outputs/cost_accuracy")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--design-results", default=None)
    parser.add_argument("--plots-only", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.plots_only:
        print(
            json.dumps(
                {"figures": [str(path) for path in generate_plots(args.output_dir)]},
                indent=2,
            )
        )
    else:
        print(json.dumps(run_experiment(args), indent=2))


if __name__ == "__main__":
    main()
