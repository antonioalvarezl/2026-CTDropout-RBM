#!/usr/bin/env python3
"""Consistency and expected-objective training for one shared control."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time

import numpy as np
from scipy.optimize import nnls
from scipy.stats import t as student_t
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rnode-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rnode-xdg-cache")

from rnode.batches import make_uniform_fixed_size, sample_batch_sequence
from rnode.integrators import integrate_fixed_grid
from rnode.objectives import nearest_target_accuracy, paper_objective
from rnode.training import train_expected_objective, train_full_objective

try:
    from experiments._paper_common import (
        ArtifactPaths,
        construct_model,
        make_seed_manifest,
        prepare_base_model,
        report_progress,
        resolve_device,
        version_information,
        write_csv,
        write_json,
        write_manifest,
    )
    from experiments.exp5_training_consistency_plots import generate_plots
except ModuleNotFoundError:
    from _paper_common import (
        ArtifactPaths,
        construct_model,
        make_seed_manifest,
        prepare_base_model,
        report_progress,
        resolve_device,
        version_information,
        write_csv,
        write_json,
        write_manifest,
    )
    from exp5_training_consistency_plots import generate_plots


def experiment_configuration(quick: bool) -> dict:
    return {
        "scheme": "uniform_fixed_size_r8",
        "h_values": (
            [0.5, 0.25, 0.125, 0.0625]
            if quick
            else [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        ),
        "fixed_control_schedules": 50 if quick else 1000,
        "rk_steps_per_switch": 8,
        "fixed_reference_dt": 1 / (256 if quick else 1024),
        "bootstrap_repetitions": 400 if quick else 2000,
        "ensemble_sizes": [1, 2, 4, 8, 16, 32, 64],
        "ensemble_groups": 500 if quick else 2000,
        "training": {
            "epochs": 8 if quick else 300,
            "trials": 1 if quick else 5,
            "h_values": [0.25, 0.125] if quick else [0.25, 0.125, 0.0625],
            "m_train_values": [1, 4] if quick else [1, 4, 16],
            "steps_per_switch": 4 if quick else 8,
            "full_dt": 1 / (32 if quick else 64),
            "evaluation_reference_dt": 1 / (128 if quick else 512),
            "evaluation_schedules": 12 if quick else 200,
        },
        "functional_normalization": "objective divided by number of data points",
        "weak_fit_rule": "fit only h whose signed-mean 95% CI excludes zero",
    }


def _control_times(features, T, dt):
    steps = int(round(T / dt))
    return torch.linspace(
        0.0, T, steps + 1, dtype=features.dtype, device=features.device
    )


def _full_metrics(model, features, labels, targets, T, dt):
    with torch.no_grad():
        times, trajectory = integrate_fixed_grid(
            model, features, T, dt, T, method="rk4"
        )
        terms = paper_objective(
            trajectory,
            times,
            targets,
            model,
            alpha=model._paper_alpha,
            beta=model._paper_beta,
            control_times=times,
        )
    return float(terms.total.cpu() / len(features)), nearest_target_accuracy(
        trajectory[-1], labels
    )


def _random_metrics(
    model,
    features,
    labels,
    targets,
    scheme,
    h,
    T,
    steps_per_switch,
    schedule_seeds,
    control_dt,
):
    values, accuracies, schedules = [], [], []
    dt = h / steps_per_switch
    control_times = _control_times(features, T, control_dt)
    with torch.no_grad():
        for schedule_seed in schedule_seeds:
            schedule = sample_batch_sequence(
                scheme,
                round(T / h),
                np.random.default_rng(int(schedule_seed)),
            )
            times, trajectory = integrate_fixed_grid(
                model,
                features,
                T,
                dt,
                h,
                schedule,
                inclusion_probs=scheme.inclusion_probs,
                method="rk4",
            )
            terms = paper_objective(
                trajectory,
                times,
                targets,
                model,
                alpha=model._paper_alpha,
                beta=model._paper_beta,
                control_times=control_times,
            )
            values.append(float(terms.total.cpu() / len(features)))
            accuracies.append(nearest_target_accuracy(trajectory[-1], labels))
            schedules.append(schedule)
    return np.asarray(values), np.asarray(accuracies), schedules


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    half_width = float(
        student_t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return mean - half_width, mean + half_width


def _absolute_interval(lower, upper):
    if lower <= 0 <= upper:
        return 0.0, max(abs(lower), abs(upper))
    return min(abs(lower), abs(upper)), max(abs(lower), abs(upper))


def _slope_row(metric, h, estimates, bootstraps, mask, reason):
    if mask.sum() < 3:
        return {
            "metric": metric,
            "fit_performed": False,
            "slope": np.nan,
            "slope_ci95_lower": np.nan,
            "slope_ci95_upper": np.nan,
            "h_min": np.nan,
            "h_max": np.nan,
            "n_points": int(mask.sum()),
            "reason": reason,
        }
    slope, intercept = np.polyfit(np.log(h[mask]), np.log(estimates[mask]), 1)
    slope_samples = []
    for sample in bootstraps:
        if np.all(sample[mask] > 0):
            slope_samples.append(
                np.polyfit(np.log(h[mask]), np.log(sample[mask]), 1)[0]
            )
    lower, upper = np.quantile(slope_samples, [0.025, 0.975])
    return {
        "metric": metric,
        "fit_performed": True,
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_ci95_lower": float(lower),
        "slope_ci95_upper": float(upper),
        "h_min": float(h[mask].min()),
        "h_max": float(h[mask].max()),
        "n_points": int(mask.sum()),
        "reason": reason,
    }


def _peak_rss_megabytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bytes_used = value if sys.platform == "darwin" else value * 1024
    return float(bytes_used / 1024**2)


def _condition_label(condition):
    if condition["kind"] == "full":
        return "Full objective"
    return f"Expected h={condition['h']:g}, M={condition['m_train']}"


def _atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _training_result_payload(result):
    return {key: value for key, value in result.items() if key != "model"}


def _save_training_checkpoint(
    path, model, condition, trial, initialization_seed, result
):
    _atomic_torch_save(
        {
            "format_version": 2,
            "model_class": "TimeDepODE",
            "condition": condition,
            "trial": trial,
            "initialization_seed": initialization_seed,
            "epochs": result["optimizer_steps"],
            "training_complete": True,
            "training_result": _training_result_payload(result),
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        path,
    )


def _coerce_csv_value(value):
    if value == "":
        return ""
    if value in {"True", "False"}:
        return value == "True"
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer() and not any(marker in value.lower() for marker in (".", "e")):
        return int(number)
    return number


def _read_csv(path):
    with path.open(newline="") as handle:
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _condition_payload_matches(
    payload, condition, trial, initialization_seed, expected_epochs
):
    return (
        payload.get("condition") == condition
        and payload.get("trial") == trial
        and payload.get("initialization_seed") == initialization_seed
        and payload.get("epochs") == expected_epochs
    )


def run_experiment(args):
    paths = ArtifactPaths.create(args.output_dir)
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # It can only be set before the first inter-op task in a process.
            pass
    config = experiment_configuration(args.quick)
    started = time.perf_counter()
    mode = "quick" if args.quick else "full"
    report_progress("exp5", f"Starting training consistency ({mode} mode)", started=started)
    report_progress("exp5", "Loading the shared checkpoint", started=started)
    prepared = prepare_base_model(
        paths,
        quick=args.quick,
        seed=args.seed,
        dtype_name=args.dtype,
        device=device,
        checkpoint=args.checkpoint,
    )
    fixed_model = prepared["model"]
    base = prepared["base_config"]
    fixed_model._paper_alpha = base["training"]["alpha"]
    fixed_model._paper_beta = base["training"]["beta"]
    fixed_model.requires_grad_(False)
    T, p = base["model"]["T"], fixed_model.hidden_dim
    scheme = make_uniform_fixed_size(p, 8)
    fixed_features, fixed_labels, fixed_targets = prepared["datasets"]["test"]
    J_fixed, fixed_accuracy = _full_metrics(
        fixed_model,
        fixed_features,
        fixed_labels,
        fixed_targets,
        T,
        config["fixed_reference_dt"],
    )

    experiment_seeds = make_seed_manifest(args.seed + 5000)
    fixed_stage_files = [
        paths.data / "fixed_control_objectives.csv",
        paths.data / "fixed_control_consistency.csv",
        paths.data / "slope_fits.csv",
        paths.data / "ensemble_averaging.csv",
        paths.data / "ensemble_group_means.npz",
        paths.data / "ensemble_fit.json",
    ]
    resume_fixed_stage = args.resume and all(path.exists() for path in fixed_stage_files)
    if resume_fixed_stage:
        report_progress(
            "exp5",
            "Resume: reusing completed fixed-control and ensemble stages",
            started=started,
        )
        consistency_rows = _read_csv(paths.data / "fixed_control_consistency.csv")
        slope_rows = _read_csv(paths.data / "slope_fits.csv")
        strong_row = next(row for row in slope_rows if row["metric"] == "strong")
        weak_row = next(row for row in slope_rows if row["metric"] == "weak")
        ensemble_fit = json.loads((paths.data / "ensemble_fit.json").read_text())
    else:
        schedule_rng = np.random.default_rng(experiment_seeds["schedule_generation"])
        bootstrap_rng = np.random.default_rng(experiment_seeds["bootstrap"])
        raw_rows, consistency_rows, raw_by_h = [], [], {}
        strong_boot_columns, weak_boot_columns = [], []
        for h_index, h in enumerate(config["h_values"], start=1):
            report_progress(
                "exp5",
                f"Fixed-control consistency {h_index}/{len(config['h_values'])}: h={h:g}",
                started=started,
            )
            schedule_seeds = schedule_rng.integers(
                0,
                2**32 - 1,
                config["fixed_control_schedules"],
                dtype=np.uint32,
            )
            values, _, _ = _random_metrics(
                fixed_model,
                fixed_features,
                fixed_labels,
                fixed_targets,
                scheme,
                h,
                T,
                config["rk_steps_per_switch"],
                schedule_seeds,
                config["fixed_reference_dt"],
            )
            delta = values - J_fixed
            raw_by_h[h] = values
            for index, (schedule_seed, value) in enumerate(zip(schedule_seeds, values)):
                raw_rows.append(
                    {
                        "h": h,
                        "schedule_index": index,
                        "schedule_seed": int(schedule_seed),
                        "J_full": J_fixed,
                        "J_hat": value,
                        "signed_difference": value - J_fixed,
                        "squared_difference": (value - J_fixed) ** 2,
                    }
                )
            strong_samples = np.empty(config["bootstrap_repetitions"])
            weak_samples = np.empty(config["bootstrap_repetitions"])
            for bootstrap_index in range(config["bootstrap_repetitions"]):
                sample = delta[bootstrap_rng.integers(0, len(delta), size=len(delta))]
                strong_samples[bootstrap_index] = np.mean(sample**2)
                weak_samples[bootstrap_index] = abs(sample.mean())
            strong_boot_columns.append(strong_samples)
            weak_boot_columns.append(weak_samples)
            signed_lower, signed_upper = _mean_ci(delta)
            abs_lower, abs_upper = _absolute_interval(signed_lower, signed_upper)
            strong_lower, strong_upper = np.quantile(strong_samples, [0.025, 0.975])
            consistency_rows.append(
                {
                    "h": h,
                    "J_full": J_fixed,
                    "mean_J_hat": float(values.mean()),
                    "strong_mse": float(np.mean(delta**2)),
                    "strong_ci95_lower": float(strong_lower),
                    "strong_ci95_upper": float(strong_upper),
                    "signed_weak_bias": float(delta.mean()),
                    "weak_bias": float(abs(delta.mean())),
                    "weak_signed_ci95_lower": signed_lower,
                    "weak_signed_ci95_upper": signed_upper,
                    "weak_abs_ci95_lower": abs_lower,
                    "weak_abs_ci95_upper": abs_upper,
                    "weak_distinguishable_from_zero": bool(
                        signed_lower > 0 or signed_upper < 0
                    ),
                    "sample_variance_J_hat": float(values.var(ddof=1)),
                    "K": len(values),
                }
            )
        write_csv(paths.data / "fixed_control_objectives.csv", raw_rows)
        write_csv(paths.data / "fixed_control_consistency.csv", consistency_rows)

        h_array = np.asarray(config["h_values"], dtype=float)
        strong_estimates = np.asarray([row["strong_mse"] for row in consistency_rows])
        weak_estimates = np.asarray([row["weak_bias"] for row in consistency_rows])
        strong_boot = np.stack(strong_boot_columns, axis=1)
        weak_boot = np.stack(weak_boot_columns, axis=1)
        strong_row = _slope_row(
            "strong",
            h_array,
            strong_estimates,
            strong_boot,
            np.ones(len(h_array), dtype=bool),
            "all measured h values",
        )
        weak_mask = np.asarray(
            [row["weak_distinguishable_from_zero"] for row in consistency_rows]
        )
        weak_row = _slope_row(
            "weak",
            h_array,
            weak_estimates,
            weak_boot,
            weak_mask,
            (
                "only h values whose signed-mean 95% CI excludes zero"
                if weak_mask.sum() >= 3
                else "fewer than three statistically distinguishable weak biases"
            ),
        )
        write_csv(paths.data / "slope_fits.csv", [strong_row, weak_row])

        report_progress("exp5", "Computing ensemble variance reduction", started=started)
        ensemble_rows, ensemble_arrays = [], {}
        fit_design, fit_response = [], []
        for h in config["h_values"]:
            values = raw_by_h[h]
            for M in config["ensemble_sizes"]:
                samples = bootstrap_rng.choice(
                    values,
                    size=(config["ensemble_groups"], M),
                    replace=True,
                ).mean(axis=1)
                errors = samples - J_fixed
                row = {
                    "h": h,
                    "M": M,
                    "mse": float(np.mean(errors**2)),
                    "variance": float(samples.var(ddof=1)),
                    "bias_squared": float((samples.mean() - J_fixed) ** 2),
                    "groups": len(samples),
                }
                ensemble_rows.append(row)
                fit_design.append([h / M, h**2])
                fit_response.append(row["mse"])
                ensemble_arrays[f"h_{str(h).replace('.', 'p')}__M_{M}"] = samples
        coefficients, residual = nnls(np.asarray(fit_design), np.asarray(fit_response))
        for row, design in zip(ensemble_rows, fit_design):
            row["fitted_mse"] = float(np.dot(coefficients, design))
        write_csv(paths.data / "ensemble_averaging.csv", ensemble_rows)
        np.savez_compressed(paths.data / "ensemble_group_means.npz", **ensemble_arrays)
        ensemble_fit = {
            "a": float(coefficients[0]),
            "b": float(coefficients[1]),
            "residual_norm": float(residual),
            "model": "a*h/M + b*h^2",
        }
        write_json(paths.data / "ensemble_fit.json", ensemble_fit)

    train_config = config["training"]
    conditions = [{"kind": "full", "h": None, "m_train": 1}]
    conditions.extend(
        {"kind": "expected", "h": h, "m_train": m_train}
        for h in train_config["h_values"]
        for m_train in train_config["m_train_values"]
    )
    X_train, labels_train, targets_train = prepared["datasets"]["train"]
    trial_sequence = np.random.SeedSequence(experiment_seeds["model_initialization"])
    trial_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in trial_sequence.spawn(train_config["trials"])
    ]
    training_rows, history_rows, training_schedule_manifest = [], [], {}
    evaluation_rows, evaluation_summary = [], []
    condition_results_dir = paths.data / "condition_results"
    condition_results_dir.mkdir(parents=True, exist_ok=True)

    total_trainings = len(trial_seeds) * len(conditions)
    for trial, initialization_seed in enumerate(trial_seeds):
        for condition_index, condition in enumerate(conditions):
            condition_name = (
                "full"
                if condition["kind"] == "full"
                else f"h_{str(condition['h']).replace('.', 'p')}__M_{condition['m_train']}"
            )
            label = _condition_label(condition)
            training_index = trial * len(conditions) + condition_index + 1
            result_key = f"trial_{trial}__{condition_name}"
            condition_result_path = condition_results_dir / f"{result_key}.pt"
            checkpoint_path = paths.checkpoints / f"{result_key}.pt"
            if args.resume and condition_result_path.exists():
                completed = torch.load(
                    condition_result_path, map_location="cpu", weights_only=False
                )
                if not _condition_payload_matches(
                    completed,
                    condition,
                    trial,
                    initialization_seed,
                    train_config["epochs"],
                ):
                    raise RuntimeError(
                        f"resume data does not match current condition: {result_key}"
                    )
                training_rows.extend(completed["training_rows"])
                history_rows.extend(completed["history_rows"])
                training_schedule_manifest[result_key] = completed[
                    "schedule_seeds"
                ]
                evaluation_rows.extend(completed["evaluation_rows"])
                evaluation_summary.extend(completed["evaluation_summary"])
                report_progress(
                    "exp5",
                    f"Resume: condition {training_index}/{total_trainings} already complete",
                    started=started,
                )
                continue

            condition_started = time.perf_counter()
            report_progress(
                "exp5",
                f"Training {training_index}/{total_trainings}: trial={trial + 1}, {label}",
                started=started,
            )

            def training_progress(epoch, epochs, record):
                report_progress(
                    "exp5",
                    f"  training {training_index}/{total_trainings}, epoch "
                    f"{epoch}/{epochs}, objective="
                    f"{record['expected_objective_estimate']:.5g}",
                    started=condition_started,
                )

            training_checkpoint = None
            if args.resume and checkpoint_path.exists():
                candidate = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                if (
                    candidate.get("training_complete")
                    and "training_result" in candidate
                    and _condition_payload_matches(
                        candidate,
                        condition,
                        trial,
                        initialization_seed,
                        train_config["epochs"],
                    )
                ):
                    training_checkpoint = candidate

            model = construct_model(
                base["model"], initialization_seed, prepared["dtype"], device
            )
            model._paper_alpha = base["training"]["alpha"]
            model._paper_beta = base["training"]["beta"]
            if training_checkpoint is not None:
                model.load_state_dict(training_checkpoint["state_dict"])
                result = dict(training_checkpoint["training_result"])
                result["model"] = model
                report_progress(
                    "exp5",
                    f"Resume: training for condition {training_index}/{total_trainings} "
                    "is complete; continuing with evaluation",
                    started=started,
                )
            elif condition["kind"] == "full":
                result = train_full_objective(
                    model,
                    X_train,
                    targets_train,
                    dt=train_config["full_dt"],
                    T=T,
                    epochs=train_config["epochs"],
                    learning_rate=base["training"]["learning_rate"],
                    alpha=model._paper_alpha,
                    beta=model._paper_beta,
                    control_dt=train_config["full_dt"],
                    progress_callback=training_progress,
                )
            else:
                result = train_expected_objective(
                    model,
                    X_train,
                    targets_train,
                    scheme=scheme,
                    h=condition["h"],
                    dt=condition["h"] / train_config["steps_per_switch"],
                    T=T,
                    epochs=train_config["epochs"],
                    m_train=condition["m_train"],
                    learning_rate=base["training"]["learning_rate"],
                    alpha=model._paper_alpha,
                    beta=model._paper_beta,
                    seed=experiment_seeds["train_data"]
                    + 1000 * trial
                    + condition_index,
                    control_dt=train_config["full_dt"],
                    progress_callback=training_progress,
                )
            model = result["model"].eval()
            if training_checkpoint is None:
                _save_training_checkpoint(
                    checkpoint_path,
                    model,
                    condition,
                    trial,
                    initialization_seed,
                    result,
                )

            condition_training_rows = [
                {
                    "trial": trial,
                    "condition": condition_name,
                    "condition_label": label,
                    "kind": condition["kind"],
                    "h": condition["h"] if condition["h"] is not None else "",
                    "M_train": condition["m_train"],
                    "initialization_seed": initialization_seed,
                    "epochs": train_config["epochs"],
                    "optimizer_steps": result["optimizer_steps"],
                    "backward_calls": result["backward_calls"],
                    "wall_seconds": result["wall_seconds"],
                    "neuron_evaluations": result["neuron_evaluations"],
                    "neuron_data_evaluations": result["neuron_data_evaluations"],
                    "final_training_loss_estimate": result["history"][-1][
                        "expected_objective_estimate"
                    ],
                }
            ]
            condition_history_rows = []
            for history in result["history"]:
                condition_history_rows.append(
                    {
                        "trial": trial,
                        "condition": condition_name,
                        **history,
                    }
                )

            report_progress(
                "exp5",
                f"Evaluating trained model {training_index}/{total_trainings}",
                started=started,
            )
            evaluation_h_values = (
                train_config["h_values"]
                if condition["kind"] == "full"
                else [condition["h"]]
            )
            condition_evaluation_rows = []
            condition_evaluation_summary = []
            evaluation_rng = np.random.default_rng(
                experiment_seeds["miscellaneous"]
                + 1000 * trial
                + condition_index
            )
            for split, (features, labels, targets) in prepared["datasets"].items():
                if split == "calibration":
                    continue
                full_value, full_split_accuracy = _full_metrics(
                    model,
                    features,
                    labels,
                    targets,
                    T,
                    train_config["evaluation_reference_dt"],
                )
                for evaluation_h in evaluation_h_values:
                    report_progress(
                        "exp5",
                        f"  evaluation split={split}, h={evaluation_h:g}",
                        started=started,
                    )
                    fresh_seeds = evaluation_rng.integers(
                        0,
                        2**32 - 1,
                        train_config["evaluation_schedules"],
                        dtype=np.uint32,
                    )
                    values, accuracies, schedules = _random_metrics(
                        model,
                        features,
                        labels,
                        targets,
                        scheme,
                        evaluation_h,
                        T,
                        config["rk_steps_per_switch"],
                        fresh_seeds,
                        train_config["evaluation_reference_dt"],
                    )
                    for schedule_index, (seed_value, value, accuracy) in enumerate(
                        zip(fresh_seeds, values, accuracies)
                    ):
                        condition_evaluation_rows.append(
                            {
                                "trial": trial,
                                "condition": condition_name,
                                "split": split,
                                "evaluation_h": evaluation_h,
                                "schedule_index": schedule_index,
                                "schedule_seed": int(seed_value),
                                "J_full": full_value,
                                "J_hat": value,
                                "squared_difference": (value - full_value) ** 2,
                                "random_accuracy": accuracy,
                            }
                        )
                    steps_per_interval = config["rk_steps_per_switch"]
                    neuron_evaluations = sum(
                        4 * steps_per_interval * sum(len(batch) for batch in schedule)
                        for schedule in schedules
                    )
                    condition_evaluation_summary.append(
                        {
                            "trial": trial,
                            "condition": condition_name,
                            "condition_label": label,
                            "split": split,
                            "evaluation_h": evaluation_h,
                            "primary_evaluation": bool(
                                condition["kind"] != "full"
                                or evaluation_h == min(train_config["h_values"])
                            ),
                            "full_objective": full_value,
                            "mean_random_objective": float(values.mean()),
                            "strong_mse": float(np.mean((values - full_value) ** 2)),
                            "full_accuracy": full_split_accuracy,
                            "mean_random_accuracy": float(accuracies.mean()),
                            "random_objective_std": float(values.std(ddof=1)),
                            "fresh_schedules": len(values),
                            "evaluation_neuron_evaluations": neuron_evaluations,
                        }
                    )

            completed_payload = {
                "format_version": 1,
                "condition": condition,
                "trial": trial,
                "initialization_seed": initialization_seed,
                "epochs": train_config["epochs"],
                "training_rows": condition_training_rows,
                "history_rows": condition_history_rows,
                "schedule_seeds": result["schedule_seeds"],
                "evaluation_rows": condition_evaluation_rows,
                "evaluation_summary": condition_evaluation_summary,
            }
            _atomic_torch_save(completed_payload, condition_result_path)
            training_rows.extend(condition_training_rows)
            history_rows.extend(condition_history_rows)
            training_schedule_manifest[result_key] = result["schedule_seeds"]
            evaluation_rows.extend(condition_evaluation_rows)
            evaluation_summary.extend(condition_evaluation_summary)
            report_progress(
                "exp5",
                f"Saved resumable condition {training_index}/{total_trainings}",
                started=started,
            )

    baseline = {
        (row["trial"], row["split"]): row["full_objective"]
        for row in evaluation_summary
        if row["condition"] == "full" and row["primary_evaluation"]
    }
    for row in evaluation_summary:
        row["near_minimizer_full_objective_gap"] = (
            row["full_objective"] - baseline[(row["trial"], row["split"])]
        )
    write_csv(paths.data / "training_runs.csv", training_rows)
    write_csv(paths.data / "training_history.csv", history_rows)
    write_json(paths.data / "training_schedule_seeds.json", training_schedule_manifest)
    write_csv(paths.data / "fresh_schedule_evaluations.csv", evaluation_rows)
    write_csv(paths.data / "training_evaluation_summary.csv", evaluation_summary)

    total_seconds = time.perf_counter() - started
    peak_memory = _peak_rss_megabytes()
    complete_config = {
        "script": "exp5_training_consistency.py",
        "cli": vars(args),
        "base": base,
        "experiment": config,
        "checkpoint_source": prepared["checkpoint_source"],
        "fixed_control": {
            "J": J_fixed,
            "test_accuracy": fixed_accuracy,
            "frozen_after_full_training": True,
        },
        "training_semantics": {
            "models_per_condition_and_trial": 1,
            "fresh_schedules_each_optimizer_step": True,
            "backward_calls_per_step": 1,
            "parameter_projection_after_step": True,
        },
        "timing_seconds": {
            "base_training": prepared["training_seconds"],
            "total": total_seconds,
        },
        "peak_process_memory_mb": peak_memory,
    }
    versions = version_information(args.device, device, args.dtype)
    all_seeds = {"base": prepared["seeds"], "experiment": experiment_seeds}
    write_json(paths.config / "config.json", complete_config)
    write_json(paths.config / "seeds.json", all_seeds)
    write_json(paths.config / "versions.json", versions)
    write_manifest(
        paths.root,
        configuration=complete_config,
        seed=all_seeds,
        versions=versions,
    )
    summary = {
        "fixed_control_J": J_fixed,
        "fixed_control_accuracy": fixed_accuracy,
        "consistency": consistency_rows,
        "slopes": [strong_row, weak_row],
        "ensemble_fit": ensemble_fit,
        "training_runs": training_rows,
        "training_evaluation": evaluation_summary,
        "total_seconds": total_seconds,
        "peak_process_memory_mb": peak_memory,
        "quick_mode_warning": args.quick,
    }
    write_json(paths.data / "summary.json", summary)
    report_progress("exp5", "Generating figures", started=started)
    figures = generate_plots(paths.root)
    report_progress("exp5", "Finished", started=started)
    return {
        "output_dir": str(paths.root),
        "figures": [str(path) for path in figures],
        "slopes": [strong_row, weak_row],
        "ensemble_fit": ensemble_fit,
        "total_seconds": total_seconds,
        "peak_process_memory_mb": peak_memory,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output-dir", default="outputs/training_consistency")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete fixed stages, trainings, and evaluated conditions",
    )
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
