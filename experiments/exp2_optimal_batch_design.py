#!/usr/bin/env python3
"""Reproducible MILP design and evaluation of balanced neuron partitions."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rnode-mpl-cache")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rnode-xdg-cache")

from rnode.design import (
    neuron_contributions_along_trajectory,
    optimize_balanced_partition,
    partition_to_one_based,
    partition_objective,
    random_balanced_partition,
    weighted_gram_matrix,
)
from rnode.integrators import integrate_masked_ensemble

try:
    from experiments._paper_common import (
        ArtifactPaths,
        bootstrap_ordered_statistic,
        make_seed_manifest,
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
    from experiments.exp2_optimal_batch_design_plots import generate_plots
except ModuleNotFoundError:
    from _paper_common import (
        ArtifactPaths,
        bootstrap_ordered_statistic,
        make_seed_manifest,
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
    from exp2_optimal_batch_design_plots import generate_plots


def experiment_configuration(quick: bool) -> dict:
    return {
        "p": 24,
        "r": 8,
        "n_random_partitions": 200,
        "gram_dt": 1.0 / (64 if quick else 256),
        "reference_dt": 1.0 / (512 if quick else 2048),
        "scatter_h": 0.25,
        "scatter_schedules": 4 if quick else 20,
        "scatter_partition_chunk": 32 if quick else 8,
        "curve_h_values": (
            [0.5, 0.25, 0.125, 0.0625]
            if quick
            else [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        ),
        "curve_schedules": 20 if quick else 200,
        "rk_steps_per_switch": 8,
        "bootstrap": 500 if quick else 2000,
        "milp_time_limit_seconds": 12.0 if quick else 900.0,
        "milp_relative_gap_target": 0.02 if quick else 0.0,
    }


def _unique_random_partitions(p, r, count, rng):
    partitions = []
    seen = set()
    while len(partitions) < count:
        partition = random_balanced_partition(p, r, rng)
        if partition not in seen:
            seen.add(partition)
            partitions.append(partition)
    return partitions


def _schedule_masks(partitions, n_schedules, h, p, rng):
    n_intervals = int(round(1.0 / h))
    n_blocks = len(partitions[0])
    masks = np.zeros((len(partitions) * n_schedules, n_intervals, p), dtype=np.float64)
    schedule_seeds = rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=(len(partitions), n_schedules),
        dtype=np.uint32,
    )
    for partition_index, partition in enumerate(partitions):
        for schedule_index in range(n_schedules):
            local_rng = np.random.default_rng(
                int(schedule_seeds[partition_index, schedule_index])
            )
            ensemble_index = partition_index * n_schedules + schedule_index
            chosen_blocks = local_rng.integers(0, n_blocks, size=n_intervals)
            for interval, block_index in enumerate(chosen_blocks):
                masks[ensemble_index, interval, list(partition[block_index])] = 1.0
    return masks, schedule_seeds


def _evaluate_partition_errors(
    model,
    features,
    reference_times,
    reference_values,
    partitions,
    *,
    h,
    n_schedules,
    steps_per_switch,
    seed_rng,
    chunk_size,
    keep_raw,
    progress_label=None,
    progress_started=None,
):
    p = model.hidden_dim
    n_blocks = len(partitions[0])
    pi = np.full(p, 1.0 / n_blocks)
    all_masks, schedule_seeds = _schedule_masks(partitions, n_schedules, h, p, seed_rng)
    estimates = np.empty(len(partitions))
    raw_chunks = []
    dt = h / steps_per_switch
    for start in range(0, len(partitions), chunk_size):
        stop = min(start + chunk_size, len(partitions))
        if progress_label is not None:
            report_progress(
                "exp2",
                f"{progress_label}: partitions {start + 1}-{stop}/{len(partitions)}",
                started=progress_started,
            )
        masks = all_masks[start * n_schedules : stop * n_schedules]
        with torch.no_grad():
            times, trajectories = integrate_masked_ensemble(
                model,
                features,
                1.0,
                dt,
                h,
                masks,
                pi,
                method="rk4",
            )
        reference = trajectory_at_times(reference_times, reference_values, times)
        errors = (
            (trajectories - reference[:, None])
            .square()
            .sum(dim=-1)
            .permute(1, 0, 2)
            .cpu()
            .numpy()
        )
        errors = errors.reshape(
            stop - start, n_schedules, len(times), features.shape[0]
        )
        estimates[start:stop] = errors.mean(axis=1).max(axis=1).mean(axis=1)
        if keep_raw:
            raw_chunks.append(errors)
    raw = np.concatenate(raw_chunks, axis=0) if keep_raw else None
    return estimates, raw, all_masks, schedule_seeds


def _fallback_partition(result, random_partitions, G, minimize):
    if result.partition is not None:
        return result.partition, "milp_incumbent"
    values = np.array(
        [partition_objective(G, partition) for partition in random_partitions]
    )
    index = int(values.argmin() if minimize else values.argmax())
    return random_partitions[index], "random_fallback_no_milp_incumbent"


def run_experiment(args) -> dict:
    paths = ArtifactPaths.create(args.output_dir)
    device = resolve_device(args.device)
    experiment = experiment_configuration(args.quick)
    design_seeds = make_seed_manifest(args.seed)
    started = time.perf_counter()
    mode = "quick" if args.quick else "full"
    report_progress("exp2", f"Starting optimal batch design ({mode} mode)", started=started)
    report_progress("exp2", "Loading the shared checkpoint", started=started)
    prepared = prepare_base_model(
        paths,
        quick=args.quick,
        seed=args.seed,
        dtype_name=args.dtype,
        device=device,
        checkpoint=args.checkpoint,
    )
    model = prepared["model"]
    if model.hidden_dim != experiment["p"]:
        raise ValueError("checkpoint hidden dimension does not match p=24")
    X_calibration, _, _ = prepared["datasets"]["calibration"]
    X_test, _, _ = prepared["datasets"]["test"]

    report_progress("exp2", "Building calibration and test Gram matrices", started=started)
    calibration_times, calibration_trajectory = reference_trajectory(
        model, X_calibration, 1.0, experiment["gram_dt"]
    )
    test_times, test_trajectory = reference_trajectory(
        model, X_test, 1.0, experiment["gram_dt"]
    )
    with torch.no_grad():
        calibration_contributions = neuron_contributions_along_trajectory(
            model, calibration_times, calibration_trajectory
        )
        test_contributions = neuron_contributions_along_trajectory(
            model, test_times, test_trajectory
        )
    G_calibration = weighted_gram_matrix(
        calibration_contributions, calibration_times
    ).cpu()
    G_test = weighted_gram_matrix(test_contributions, test_times).cpu()
    np.savez_compressed(
        paths.data / "gram_matrices.npz",
        G_calibration=G_calibration.numpy(),
        G_test=G_test.numpy(),
        calibration_times=calibration_times.cpu().numpy(),
        test_times=test_times.cpu().numpy(),
    )

    partition_rng = np.random.default_rng(design_seeds["partition_generation"])
    random_partitions = _unique_random_partitions(
        experiment["p"],
        experiment["r"],
        experiment["n_random_partitions"],
        partition_rng,
    )
    report_progress("exp2", "MILP 1/2: searching for the minimizing partition", started=started)
    minimum_result = optimize_balanced_partition(
        G_calibration,
        experiment["r"],
        time_limit=experiment["milp_time_limit_seconds"],
        mip_rel_gap=experiment["milp_relative_gap_target"],
    )
    report_progress("exp2", "MILP 2/2: searching for the maximizing partition", started=started)
    maximum_result = optimize_balanced_partition(
        G_calibration,
        experiment["r"],
        maximize=True,
        time_limit=experiment["milp_time_limit_seconds"],
        mip_rel_gap=experiment["milp_relative_gap_target"],
    )
    optimized_partition, optimized_source = _fallback_partition(
        minimum_result, random_partitions, G_calibration, True
    )
    adversarial_partition, adversarial_source = _fallback_partition(
        maximum_result, random_partitions, G_calibration, False
    )
    baseline_partition = tuple(
        tuple(range(start, start + experiment["r"]))
        for start in range(0, experiment["p"], experiment["r"])
    )

    partition_rows = []
    for index, partition in enumerate(random_partitions):
        partition_rows.append(
            {
                "partition_id": f"random_{index:03d}",
                "category": "random",
                "batches": [list(block) for block in partition],
                "batches_one_based": [
                    list(block) for block in partition_to_one_based(partition)
                ],
                "objective_calibration": partition_objective(G_calibration, partition),
                "objective_test": partition_objective(G_test, partition),
                "lambda_test": len(partition) * partition_objective(G_test, partition),
                "solver_status": "not_applicable",
                "solver_gap": "",
                "certified_optimal": False,
            }
        )
    special = [
        ("optimized", optimized_partition, minimum_result, optimized_source),
        ("adversarial", adversarial_partition, maximum_result, adversarial_source),
        ("baseline", baseline_partition, None, "fixed_by_index"),
    ]
    for category, partition, solver_result, source in special:
        partition_rows.append(
            {
                "partition_id": category,
                "category": category,
                "batches": [list(block) for block in partition],
                "batches_one_based": [
                    list(block) for block in partition_to_one_based(partition)
                ],
                "objective_calibration": partition_objective(G_calibration, partition),
                "objective_test": partition_objective(G_test, partition),
                "lambda_test": len(partition) * partition_objective(G_test, partition),
                "solver_status": (
                    solver_result.status if solver_result else "not_applicable"
                ),
                "solver_gap": solver_result.mip_gap if solver_result else "",
                "certified_optimal": (
                    solver_result.certified_optimal if solver_result else False
                ),
                "selection_source": source,
                "scientific_label": (
                    solver_result.label if solver_result else "fixed baseline"
                ),
            }
        )
    write_csv(paths.data / "partitions.csv", partition_rows)
    write_json(
        paths.data / "milp_results.json",
        {
            "minimization": minimum_result.to_dict(),
            "maximization": maximum_result.to_dict(),
            "optimized_selection_source": optimized_source,
            "adversarial_selection_source": adversarial_source,
        },
    )

    fine_reference_times, fine_reference = reference_trajectory(
        model, X_test, 1.0, experiment["reference_dt"]
    )
    scatter_partitions = random_partitions + [
        baseline_partition,
        optimized_partition,
        adversarial_partition,
    ]
    scatter_categories = ["random"] * len(random_partitions) + [
        "baseline",
        "optimized",
        "adversarial",
    ]
    scatter_ids = [f"random_{index:03d}" for index in range(len(random_partitions))] + [
        "baseline",
        "optimized",
        "adversarial",
    ]
    schedule_rng = np.random.default_rng(design_seeds["schedule_generation"])
    report_progress(
        "exp2",
        f"Evaluating {len(scatter_partitions)} partitions on held-out data",
        started=started,
    )
    scatter_estimates, scatter_raw, scatter_masks, scatter_schedule_seeds = (
        _evaluate_partition_errors(
            model,
            X_test,
            fine_reference_times,
            fine_reference,
            scatter_partitions,
            h=experiment["scatter_h"],
            n_schedules=experiment["scatter_schedules"],
            steps_per_switch=experiment["rk_steps_per_switch"],
            seed_rng=schedule_rng,
            chunk_size=experiment["scatter_partition_chunk"],
            keep_raw=args.quick,
            progress_label="Held-out scatter",
            progress_started=started,
        )
    )
    scatter_rows = []
    for partition_id, category, partition, estimate in zip(
        scatter_ids, scatter_categories, scatter_partitions, scatter_estimates
    ):
        lambda_test = len(partition) * partition_objective(G_test, partition)
        scatter_rows.append(
            {
                "partition_id": partition_id,
                "category": category,
                "lambda_test": lambda_test,
                "h": experiment["scatter_h"],
                "mean_error": estimate,
                "error_over_h": estimate / experiment["scatter_h"],
                "n_schedules": experiment["scatter_schedules"],
            }
        )
    write_csv(paths.data / "lambda_vs_error.csv", scatter_rows)
    np.savez_compressed(
        paths.data / "scatter_raw.npz",
        masks=scatter_masks.astype(np.uint8),
        schedule_seeds=scatter_schedule_seeds,
        squared_errors=(scatter_raw if scatter_raw is not None else np.empty(0)),
    )
    random_scatter = [row for row in scatter_rows if row["category"] == "random"]
    lambda_values = np.array([row["lambda_test"] for row in random_scatter])
    error_values = np.array([row["error_over_h"] for row in random_scatter])
    pearson = pearsonr(lambda_values, error_values)
    spearman = spearmanr(lambda_values, error_values)
    correlations = {
        "population": "random_partitions_only",
        "n_partitions": len(random_scatter),
        "pearson_r": float(pearson.statistic),
        "pearson_pvalue": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "h": experiment["scatter_h"],
        "n_schedules": experiment["scatter_schedules"],
        "quick_mode_warning": bool(args.quick),
    }
    write_json(paths.data / "correlations.json", correlations)

    random_objectives = np.array(
        [row["lambda_test"] for row in partition_rows if row["category"] == "random"]
    )
    median_random_index = int(
        np.argmin(np.abs(random_objectives - np.median(random_objectives)))
    )
    median_random_partition = random_partitions[median_random_index]
    curve_partitions = [
        optimized_partition,
        median_random_partition,
        adversarial_partition,
    ]
    curve_categories = ["optimized", "random_median", "adversarial"]
    curve_rows = []
    bootstrap_rng = np.random.default_rng(design_seeds["bootstrap"])
    for curve_index, h in enumerate(experiment["curve_h_values"], start=1):
        report_progress(
            "exp2",
            f"Design curve {curve_index}/{len(experiment['curve_h_values'])}: h={h:g}",
            started=started,
        )
        estimates, raw, masks, seeds = _evaluate_partition_errors(
            model,
            X_test,
            fine_reference_times,
            fine_reference,
            curve_partitions,
            h=h,
            n_schedules=experiment["curve_schedules"],
            steps_per_switch=experiment["rk_steps_per_switch"],
            seed_rng=schedule_rng,
            chunk_size=3 if args.quick else 1,
            keep_raw=True,
            progress_started=started,
        )
        np.savez_compressed(
            paths.data / f"raw_design_curve_h_{str(h).replace('.', 'p')}.npz",
            squared_errors=raw,
            masks=masks.astype(np.uint8),
            schedule_seeds=seeds,
        )
        for partition_index, category in enumerate(curve_categories):
            bootstrap = bootstrap_ordered_statistic(
                raw[partition_index], bootstrap_rng, experiment["bootstrap"]
            )
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
            curve_rows.append(
                {
                    "category": category,
                    "h": h,
                    "dt": h / experiment["rk_steps_per_switch"],
                    "mean_error": estimates[partition_index],
                    "standard_deviation": float(bootstrap.std(ddof=1)),
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "n_schedules": experiment["curve_schedules"],
                }
            )
    write_csv(paths.data / "design_error_curves.csv", curve_rows)

    scatter_lookup = {row["partition_id"]: row for row in scatter_rows}
    table_rows = []
    for category, partition, solver_result, source in special:
        scatter_row = scatter_lookup[category]
        table_rows.append(
            {
                "category": category,
                "scientific_label": (
                    solver_result.label if solver_result else "fixed baseline"
                ),
                "batches": [list(block) for block in partition],
                "batches_one_based": [
                    list(block) for block in partition_to_one_based(partition)
                ],
                "objective_calibration": partition_objective(G_calibration, partition),
                "objective_test": partition_objective(G_test, partition),
                "lambda_test": len(partition) * partition_objective(G_test, partition),
                "solver_gap": solver_result.mip_gap if solver_result else "",
                "solver_status": (
                    solver_result.status if solver_result else "not_applicable"
                ),
                "empirical_error_h": experiment["scatter_h"],
                "empirical_error": scatter_row["mean_error"],
                "selection_source": source,
            }
        )
    write_csv(paths.data / "partition_table.csv", table_rows)
    write_json(paths.data / "partition_table.json", table_rows)

    total_seconds = time.perf_counter() - started
    complete_config = {
        "script": "exp2_optimal_batch_design.py",
        "cli": vars(args),
        "base": prepared["base_config"],
        "experiment": experiment,
        "checkpoint_source": prepared["checkpoint_source"],
        "split_accuracies": prepared["accuracies"],
        "data_usage": {
            "calibration": "Gram matrix and MILP objective only",
            "test": "post-design Lambda and trajectory-error evaluation only",
            "test_used_for_design": False,
        },
        "timing_seconds": {
            "training": prepared["training_seconds"],
            "milp_min": minimum_result.elapsed_seconds,
            "milp_max": maximum_result.elapsed_seconds,
            "total": total_seconds,
        },
    }
    write_json(paths.config / "config.json", complete_config)
    write_json(
        paths.config / "seeds.json",
        {"base_model_and_data": prepared["seeds"], "design": design_seeds},
    )
    versions = version_information(args.device, device, args.dtype)
    write_json(paths.config / "versions.json", versions)
    write_manifest(
        paths.root,
        configuration=complete_config,
        seed={"base_model_and_data": prepared["seeds"], "design": design_seeds},
        versions=versions,
    )
    write_json(
        paths.data / "summary.json",
        {
            "milp": {
                "min": minimum_result.to_dict(),
                "max": maximum_result.to_dict(),
            },
            "correlations": correlations,
            "table": table_rows,
            "split_accuracies": prepared["accuracies"],
            "quick_mode_warning": bool(args.quick),
        },
    )
    report_progress("exp2", "Generating figures", started=started)
    figures = generate_plots(paths.root)
    report_progress("exp2", "Finished", started=started)
    return {
        "output_dir": str(paths.root),
        "figures": [str(path) for path in figures],
        "milp_min": minimum_result.to_dict(),
        "milp_max": maximum_result.to_dict(),
        "correlations": correlations,
        "table": table_rows,
        "total_seconds": total_seconds,
        "checkpoint": str(paths.checkpoints / "base_model.pt"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output-dir", default="outputs/optimal_batch_design")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="base-model checkpoint; use experiment 1's checkpoint to share parameters",
    )
    parser.add_argument("--plots-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
