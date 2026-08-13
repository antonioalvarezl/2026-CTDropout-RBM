#!/usr/bin/env python3
"""Flow-matching and random-batch transport of a compact initial measure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rnode-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rnode-xdg-cache")

from rnode.batches import (
    make_bernoulli,
    make_fixed_disjoint_partition,
    make_uniform_fixed_size,
    sample_batch_sequence,
)
from rnode.data import (
    TARGET_MIXTURE_CENTERS,
    TARGET_MIXTURE_COVARIANCES,
    initial_density,
    initial_density_quadrature,
    sample_initial_compact,
    target_density,
)
from rnode.flow import Flow
from rnode.transport import (
    average_densities,
    l1_density_error,
    normalize_density,
    particle_coupling_error,
    transport_particles,
    weighted_kde,
    weighted_kde_with_diagnostics,
)

SCHEME_LABELS = {
    "uniform_r16": "Uniform fixed-size, r=16",
    "fixed_disjoint_r16": "Random disjoint, r=16",
    "bernoulli_q1_3": "Bernoulli, retention 1/3",
}

try:
    from experiments._paper_common import (
        ArtifactPaths,
        make_seed_manifest,
        report_progress,
        resolve_device,
        resolve_dtype,
        version_information,
        write_csv,
        write_json,
        write_manifest,
    )
    from experiments.exp4_measure_transport_plots import generate_plots
except ModuleNotFoundError:
    from _paper_common import (
        ArtifactPaths,
        make_seed_manifest,
        report_progress,
        resolve_device,
        resolve_dtype,
        version_information,
        write_csv,
        write_json,
        write_manifest,
    )
    from exp4_measure_transport_plots import generate_plots


def experiment_configuration(quick: bool) -> dict:
    return {
        "T": 1.0,
        "hidden": 48,
        "h_values": (
            [0.5, 0.25, 0.125, 0.0625]
            if quick
            else [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        ),
        "rk_steps_per_switch": 8,
        "reference_dt": 1 / (1024 if quick else 4096),
        "n_schedules": 5 if quick else 80,
        "quadrature_points_per_axis": 23 if quick else 61,
        "density_grid_size": 64 if quick else 160,
        "density_bounds": {"x": [-2.5, 7.5], "y": [-2.5, 4.5]},
        "bandwidth": 0.22,
        "refinement_grid_sizes": [48, 64, 96] if quick else [96, 160, 224],
        "refinement_bandwidths": [0.16, 0.22, 0.30],
        "flow_matching": {
            "epochs": 250 if quick else 30000,
            "batch_size": 384 if quick else 2048,
            "learning_rate": 3e-3 if quick else 1e-3,
            "validation_size": 768 if quick else 8192,
            "validation_interval": 25 if quick else 100,
            "scheduler_patience_checks": 4 if quick else 12,
            "scheduler_factor": 0.5,
            "minimum_learning_rate": 2e-5,
        },
        "flow_quality": {
            "target_kde_benchmark_repetitions": 2 if quick else 5,
            "max_l1_ratio_to_kde_benchmark": 2.0,
            "abort_before_rbm_on_failure": not quick,
        },
        "kde_randomness": "none; deterministic weighted Gaussian KDE",
        "particle_masses": "fixed normalized initial quadrature masses",
        "divergence_estimator": "not used (analytic Flow.divergence available)",
    }


def _balanced_target(n_samples, rng):
    counts = np.full(3, n_samples // 3, dtype=int)
    counts[: n_samples % 3] += 1
    blocks = [
        rng.multivariate_normal(center, covariance, int(count))
        for center, covariance, count in zip(
            TARGET_MIXTURE_CENTERS, TARGET_MIXTURE_COVARIANCES, counts
        )
    ]
    samples = np.concatenate(blocks)
    return samples[rng.permutation(n_samples)]


def _new_flow(config, seed, dtype, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Flow(dim=2, hidden=config["hidden"])
    return model.to(dtype=dtype, device=device)


def _train_flow(config, seeds, dtype, device):
    model = _new_flow(config, seeds["model_initialization"], dtype, device)
    train = config["flow_matching"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train["scheduler_factor"],
        patience=train["scheduler_patience_checks"],
        threshold=1e-3,
        threshold_mode="rel",
        min_lr=train["minimum_learning_rate"],
    )
    rng = np.random.default_rng(seeds["train_data"])
    validation_rng = np.random.default_rng(seeds["calibration_data"])
    validation_x0 = torch.tensor(
        sample_initial_compact(train["validation_size"], rng=validation_rng),
        dtype=dtype,
        device=device,
    )
    validation_x1 = torch.tensor(
        _balanced_target(train["validation_size"], validation_rng),
        dtype=dtype,
        device=device,
    )
    validation_t = torch.tensor(
        validation_rng.random(train["validation_size"]), dtype=dtype, device=device
    )
    validation_interpolation = (
        (1 - validation_t[:, None]) * validation_x0
        + validation_t[:, None] * validation_x1
    )
    validation_velocity = validation_x1 - validation_x0
    history = []
    best_validation_mse = float("inf")
    best_epoch = 0
    best_state = None
    started = time.perf_counter()
    for epoch in range(train["epochs"]):
        x0 = torch.tensor(
            sample_initial_compact(train["batch_size"], rng=rng),
            dtype=dtype,
            device=device,
        )
        x1 = torch.tensor(
            _balanced_target(train["batch_size"], rng), dtype=dtype, device=device
        )
        t = torch.tensor(rng.random(train["batch_size"]), dtype=dtype, device=device)
        interpolation = (1 - t[:, None]) * x0 + t[:, None] * x1
        target_velocity = x1 - x0
        loss = (model(t, interpolation) - target_velocity).square().sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        validation_mse = None
        if (epoch + 1) % train["validation_interval"] == 0 or epoch == 0:
            with torch.no_grad():
                validation_mse = float(
                    (
                        model(validation_t, validation_interpolation)
                        - validation_velocity
                    )
                    .square()
                    .sum(dim=1)
                    .mean()
                    .cpu()
                )
            scheduler.step(validation_mse)
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        history.append(
            {
                "epoch": epoch + 1,
                "flow_matching_mse": float(loss.detach().cpu()),
                "validation_mse": validation_mse,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        report_every = max(1, train["epochs"] // 20)
        if (epoch + 1) % report_every == 0 or epoch + 1 == train["epochs"]:
            report_progress(
                "exp4",
                f"Flow matching epoch {epoch + 1}/{train['epochs']} "
                f"(MSE={history[-1]['flow_matching_mse']:.4g})",
                started=started,
            )
    if best_state is None:
        raise RuntimeError("flow matching did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    trailing_window = min(500, max(20, len(history) // 5))
    diagnostics = {
        "selection_rule": "lowest MSE on one fixed independent validation batch",
        "selected_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
        "selected_training_mse": history[best_epoch - 1]["flow_matching_mse"],
        "final_training_mse": history[-1]["flow_matching_mse"],
        "trailing_training_mse_mean": float(
            np.mean([row["flow_matching_mse"] for row in history[-trailing_window:]])
        ),
        "epochs_run": len(history),
        "validation_size": train["validation_size"],
    }
    report_progress(
        "exp4",
        f"Selected epoch {best_epoch}/{len(history)} with validation "
        f"MSE={best_validation_mse:.5g}",
        started=started,
    )
    return model.eval(), history, time.perf_counter() - started, diagnostics


def _save_flow_checkpoint(
    path, model, config, seeds, dtype_name, history, training_diagnostics
):
    torch.save(
        {
            "format_version": 1,
            "model_class": "Flow",
            "flow_formula": "W0 tanh(A0 x + b0 + b1 t)",
            "experiment_config": config,
            "seeds": seeds,
            "dtype": dtype_name,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "training_history": history,
            "training_diagnostics": training_diagnostics,
        },
        path,
    )


def _load_or_train(args, paths, config, seeds, dtype, device):
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = _new_flow(config, seeds["model_initialization"], dtype, device)
        model.load_state_dict(payload["state_dict"])
        history, training_seconds, source = (
            payload.get("training_history", []),
            0.0,
            str(Path(args.checkpoint).resolve()),
        )
        training_diagnostics = payload.get(
            "training_diagnostics",
            {"selection_rule": "not recorded in supplied checkpoint"},
        )
    else:
        model, history, training_seconds, training_diagnostics = _train_flow(
            config, seeds, dtype, device
        )
        source = "trained"
    _save_flow_checkpoint(
        paths.checkpoints / "flow.pt",
        model,
        config,
        seeds,
        args.dtype,
        history,
        training_diagnostics,
    )
    if history:
        write_csv(paths.data / "training_history.csv", history)
    return model, history, training_seconds, source, training_diagnostics


def assess_flow_quality(full_target_l1, benchmark_l1_values, max_ratio):
    """Compare flow error with the finite-particle/KDE resolution benchmark."""
    values = np.asarray(benchmark_l1_values, dtype=float)
    if (
        not np.isfinite(full_target_l1)
        or full_target_l1 < 0
        or values.ndim != 1
        or len(values) == 0
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
        or not np.isfinite(max_ratio)
        or max_ratio <= 0
    ):
        raise ValueError("flow-quality inputs must be positive and finite")
    benchmark_mean = float(values.mean())
    ratio = float(full_target_l1 / benchmark_mean)
    return {
        "full_terminal_to_target_l1": float(full_target_l1),
        "target_kde_benchmark_l1_values": values.tolist(),
        "target_kde_benchmark_l1_mean": benchmark_mean,
        "l1_ratio_to_kde_benchmark": ratio,
        "maximum_accepted_ratio": float(max_ratio),
        "passed": bool(ratio <= max_ratio),
    }


def _density_grid(config, size=None):
    size = config["density_grid_size"] if size is None else size
    x = np.linspace(*config["density_bounds"]["x"], size)
    y = np.linspace(*config["density_bounds"]["y"], size)
    return x, y


def _schemes(p, rng):
    partition = np.array_split(rng.permutation(p), 3)
    return {
        "uniform_r16": make_uniform_fixed_size(p, p // 3),
        "fixed_disjoint_r16": make_fixed_disjoint_partition(
            p, partition, name="Random disjoint, r=16"
        ),
        "bernoulli_q1_3": make_bernoulli(p, 1 / 3),
    }, [block.tolist() for block in partition]


def _interpolate_density(source_x, source_y, density, target_x, target_y):
    interpolator = RegularGridInterpolator(
        (source_y, source_x), density, bounds_error=False, fill_value=0.0
    )
    xx, yy = np.meshgrid(target_x, target_y)
    values = interpolator(np.stack([yy.ravel(), xx.ravel()], axis=1)).reshape(xx.shape)
    return normalize_density(np.maximum(values, 0.0), target_x, target_y)


def run_experiment(args):
    paths = ArtifactPaths.create(args.output_dir)
    device, dtype = resolve_device(args.device), resolve_dtype(args.dtype)
    config, seeds = experiment_configuration(args.quick), make_seed_manifest(args.seed)
    started = time.perf_counter()
    mode = "quick" if args.quick else "full"
    report_progress("exp4", f"Starting measure transport ({mode} mode)", started=started)
    report_progress("exp4", "Loading or training the flow-matching model", started=started)
    (
        model,
        history,
        training_seconds,
        checkpoint_source,
        training_diagnostics,
    ) = _load_or_train(
        args, paths, config, seeds, dtype, device
    )
    report_progress("exp4", "Flow model ready; transporting the full reference", started=started)
    points_cpu, masses_cpu = initial_density_quadrature(
        config["quadrature_points_per_axis"]
    )
    points = points_cpu.to(dtype=dtype, device=device)
    masses = masses_cpu.numpy()
    np.savez_compressed(
        paths.data / "initial_particles.npz", points=points_cpu.numpy(), masses=masses
    )
    x_grid, y_grid = _density_grid(config)

    with torch.no_grad():
        full_times, full_trajectory, unchanged_masses = transport_particles(
            model,
            points,
            masses_cpu,
            config["T"],
            config["reference_dt"],
            config["T"],
            method="rk4",
        )
    if not np.array_equal(unchanged_masses.numpy(), masses):
        raise RuntimeError("particle masses changed during full transport")
    full_terminal = full_trajectory[-1].cpu().numpy()
    full_density, full_kde_diagnostics = weighted_kde_with_diagnostics(
        full_terminal, masses, x_grid, y_grid, config["bandwidth"]
    )
    snapshot_times = np.array([0.0, 0.5, 1.0])
    snapshot_indices = np.rint(snapshot_times / config["reference_dt"]).astype(int)
    snapshot_densities = np.stack(
        [
            weighted_kde(
                full_trajectory[index].cpu().numpy(),
                masses,
                x_grid,
                y_grid,
                config["bandwidth"],
            )
            for index in snapshot_indices
        ]
    )
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    initial_grid_density = normalize_density(
        initial_density(grid_points).reshape(xx.shape), x_grid, y_grid
    )
    target_grid_density = normalize_density(
        target_density(grid_points).reshape(xx.shape), x_grid, y_grid
    )
    full_target_l1 = l1_density_error(full_density, target_grid_density, x_grid, y_grid)
    np.savez_compressed(
        paths.data / "distributions.npz",
        x_grid=x_grid,
        y_grid=y_grid,
        initial_density=initial_grid_density,
        target_density=target_grid_density,
        full_terminal_density=full_density,
        snapshot_times=snapshot_times,
        full_snapshot_positions=full_trajectory[snapshot_indices].cpu().numpy(),
        full_snapshot_densities=snapshot_densities,
    )

    quality_config = config["flow_quality"]
    benchmark_rng = np.random.default_rng(seeds["miscellaneous"] + 401)
    benchmark_masses = np.full(len(masses), 1.0 / len(masses))
    benchmark_l1_values = []
    benchmark_kde_diagnostics = []
    for benchmark_index in range(
        quality_config["target_kde_benchmark_repetitions"]
    ):
        report_progress(
            "exp4",
            f"Flow-quality KDE benchmark {benchmark_index + 1}/"
            f"{quality_config['target_kde_benchmark_repetitions']}",
            started=started,
        )
        benchmark_points = _balanced_target(len(masses), benchmark_rng)
        benchmark_density, benchmark_kde_diagnostic = weighted_kde_with_diagnostics(
            benchmark_points,
            benchmark_masses,
            x_grid,
            y_grid,
            config["bandwidth"],
        )
        benchmark_kde_diagnostics.append(benchmark_kde_diagnostic)
        benchmark_l1_values.append(
            l1_density_error(
                benchmark_density, target_grid_density, x_grid, y_grid
            )
        )
    flow_quality = assess_flow_quality(
        full_target_l1,
        benchmark_l1_values,
        quality_config["max_l1_ratio_to_kde_benchmark"],
    )
    flow_quality.update(
        {
            "benchmark_role": (
                "separate seeded diagnostic for finite-particle/KDE resolution; "
                "not included in the RBM error expectation"
            ),
            "benchmark_seed": seeds["miscellaneous"] + 401,
            "benchmark_particles": len(masses),
            "full_kde_domain": full_kde_diagnostics,
            "benchmark_kde_domain": benchmark_kde_diagnostics,
            "training_diagnostics": training_diagnostics,
        }
    )
    write_json(paths.data / "flow_quality.json", flow_quality)
    quality_message = (
        f"full-to-target L1={full_target_l1:.4g}; KDE benchmark="
        f"{flow_quality['target_kde_benchmark_l1_mean']:.4g}; ratio="
        f"{flow_quality['l1_ratio_to_kde_benchmark']:.3g}/"
        f"{flow_quality['maximum_accepted_ratio']:.3g}"
    )
    if flow_quality["passed"]:
        report_progress("exp4", f"Flow quality passed: {quality_message}", started=started)
    else:
        report_progress("exp4", f"WARNING: flow quality failed: {quality_message}", started=started)
        if quality_config["abort_before_rbm_on_failure"] and not args.allow_poor_flow:
            raise RuntimeError(
                "Flow-quality check failed; RBM transport was not started. "
                "Inspect data/flow_quality.json. Use --allow-poor-flow only to "
                "run an explicitly diagnostic, non-reference simulation."
            )
    if args.quality_only:
        quality_only_result = {
            "output_dir": str(paths.root),
            "quality_only": True,
            "checkpoint": str(paths.checkpoints / "flow.pt"),
            "training_diagnostics": training_diagnostics,
            "flow_quality": flow_quality,
            "total_seconds": time.perf_counter() - started,
        }
        write_json(paths.data / "quality_only_summary.json", quality_only_result)
        report_progress(
            "exp4", "Quality-only run finished; RBM transport was skipped", started=started
        )
        return quality_only_result

    schemes, partition = _schemes(
        model.hidden_dim, np.random.default_rng(seeds["partition_generation"])
    )
    schedule_seed_rng = np.random.default_rng(seeds["schedule_generation"])
    rows, average_arrays, schedule_manifest = [], {}, {}
    total_cases = len(schemes) * len(config["h_values"])
    completed_cases = 0
    for scheme_name, scheme in schemes.items():
        schedule_manifest[scheme_name] = {}
        for h in config["h_values"]:
            report_progress(
                "exp4",
                f"Transport case {completed_cases + 1}/{total_cases}: "
                f"scheme={scheme_name}, h={h:g}",
                started=started,
            )
            seeds_for_case = schedule_seed_rng.integers(
                0, 2**32 - 1, config["n_schedules"], dtype=np.uint32
            )
            densities, terminal_positions, l1_values, coupling_values = [], [], [], []
            kde_diagnostics = []
            report_every = max(1, len(seeds_for_case) // 5)
            for schedule_index, schedule_seed in enumerate(seeds_for_case, start=1):
                schedule = sample_batch_sequence(
                    scheme,
                    round(config["T"] / h),
                    np.random.default_rng(int(schedule_seed)),
                )
                with torch.no_grad():
                    _, trajectory, returned_masses = transport_particles(
                        model,
                        points,
                        masses_cpu,
                        config["T"],
                        h / config["rk_steps_per_switch"],
                        h,
                        schedule,
                        inclusion_probs=scheme.inclusion_probs,
                        method="rk4",
                    )
                if not np.array_equal(returned_masses.numpy(), masses):
                    raise RuntimeError("particle masses changed during RBM transport")
                terminal = trajectory[-1].cpu().numpy()
                density, kde_diagnostic = weighted_kde_with_diagnostics(
                    terminal, masses, x_grid, y_grid, config["bandwidth"]
                )
                kde_diagnostics.append(kde_diagnostic)
                terminal_positions.append(terminal)
                densities.append(density)
                l1_values.append(
                    l1_density_error(full_density, density, x_grid, y_grid)
                )
                coupling_values.append(
                    particle_coupling_error(full_terminal, terminal, masses)
                )
                if schedule_index % report_every == 0 or schedule_index == len(seeds_for_case):
                    report_progress(
                        "exp4",
                        f"  schedules {schedule_index}/{len(seeds_for_case)} "
                        f"for {scheme_name}, h={h:g}",
                        started=started,
                    )
            densities = np.stack(densities)
            averaged = average_densities(densities, x_grid, y_grid)
            key = f"{scheme_name}__h_{str(h).replace('.', 'p')}"
            average_arrays[key] = averaged
            np.savez_compressed(
                paths.data / f"realizations_{key}.npz",
                densities=densities,
                terminal_positions=np.stack(terminal_positions),
                schedule_seeds=seeds_for_case,
                l1_errors_T=np.asarray(l1_values),
                particle_coupling_errors_T=np.asarray(coupling_values),
                fixed_masses=masses,
            )
            rows.append(
                {
                    "scheme": scheme_name,
                    "scheme_label": SCHEME_LABELS[scheme_name],
                    "h": h,
                    "dt": h / config["rk_steps_per_switch"],
                    "expected_l1_error_T": float(np.mean(l1_values)),
                    "standard_error_l1_T": float(
                        np.std(l1_values, ddof=1) / np.sqrt(len(l1_values))
                    ),
                    "l1_error_of_mean_density_T": l1_density_error(
                        full_density, averaged, x_grid, y_grid
                    ),
                    "mean_particle_coupling_error_T": float(np.mean(coupling_values)),
                    "n_schedules": len(l1_values),
                    "n_particles": len(masses),
                    "density_mass": float(
                        averaged.sum()
                        * (x_grid[1] - x_grid[0])
                        * (y_grid[1] - y_grid[0])
                    ),
                    "kde_mass_before_normalization_mean": float(
                        np.mean(
                            [item["mass_before_normalization"] for item in kde_diagnostics]
                        )
                    ),
                    "kde_mass_before_normalization_min": float(
                        np.min(
                            [item["mass_before_normalization"] for item in kde_diagnostics]
                        )
                    ),
                    "kde_renormalization_factor_mean": float(
                        np.mean(
                            [item["renormalization_factor"] for item in kde_diagnostics]
                        )
                    ),
                    "kde_estimated_truncation_loss_max": float(
                        np.max(
                            [item["estimated_truncation_loss"] for item in kde_diagnostics]
                        )
                    ),
                }
            )
            schedule_manifest[scheme_name][str(h)] = seeds_for_case.tolist()
            completed_cases += 1
    write_csv(paths.data / "terminal_errors.csv", rows)
    np.savez_compressed(paths.data / "averaged_densities.npz", **average_arrays)
    write_json(paths.data / "schedule_seeds.json", schedule_manifest)
    representative_scheme, representative_h = "fixed_disjoint_r16", min(
        config["h_values"]
    )
    write_json(
        paths.data / "representative.json",
        {
            "scheme": representative_scheme,
            "scheme_label": SCHEME_LABELS[representative_scheme],
            "h": representative_h,
            "density_key": f"{representative_scheme}__h_{str(representative_h).replace('.', 'p')}",
        },
    )

    refinement_arrays, refinement_rows = {}, []
    finest_size = max(config["refinement_grid_sizes"])
    finest_x, finest_y = _density_grid(config, finest_size)
    finest_by_bandwidth = {}
    for bandwidth in config["refinement_bandwidths"]:
        finest_by_bandwidth[bandwidth] = weighted_kde_with_diagnostics(
            full_terminal, masses, finest_x, finest_y, bandwidth
        )[0]
    baseline_finest = finest_by_bandwidth[config["bandwidth"]]
    refinement_total = len(config["refinement_bandwidths"]) * len(
        config["refinement_grid_sizes"]
    )
    refinement_done = 0
    for bandwidth in config["refinement_bandwidths"]:
        for grid_size in config["refinement_grid_sizes"]:
            report_progress(
                "exp4",
                f"KDE refinement {refinement_done + 1}/{refinement_total}: "
                f"grid={grid_size}, bandwidth={bandwidth:g}",
                started=started,
            )
            grid_x, grid_y = _density_grid(config, grid_size)
            density, kde_diagnostic = weighted_kde_with_diagnostics(
                full_terminal, masses, grid_x, grid_y, bandwidth
            )
            refinement_arrays[f"n{grid_size}_bw{str(bandwidth).replace('.', 'p')}"] = (
                density
            )
            on_finest = (
                density
                if grid_size == finest_size
                else _interpolate_density(grid_x, grid_y, density, finest_x, finest_y)
            )
            refinement_rows.append(
                {
                    "grid_size": grid_size,
                    "bandwidth": bandwidth,
                    "numerical_mass": float(
                        density.sum()
                        * (grid_x[1] - grid_x[0])
                        * (grid_y[1] - grid_y[0])
                    ),
                    **kde_diagnostic,
                    "l1_to_finest_same_bandwidth": l1_density_error(
                        on_finest, finest_by_bandwidth[bandwidth], finest_x, finest_y
                    ),
                    "l1_to_baseline_bandwidth": l1_density_error(
                        on_finest, baseline_finest, finest_x, finest_y
                    ),
                }
            )
            refinement_done += 1
    np.savez_compressed(paths.data / "refinement_densities.npz", **refinement_arrays)
    write_csv(paths.data / "refinement.csv", refinement_rows)

    elapsed = time.perf_counter() - started
    complete_config = {
        "script": "exp4_measure_transport.py",
        "cli": vars(args),
        "experiment": config,
        "flow_formula": "F(x,t)=W0 tanh(A0 x+b0+b1 t)",
        "initial_distribution": "compact paper density 2/pi*(1-|x-(-1,-1)|^2)_+",
        "target_distribution": "exactly balanced three-Gaussian training batches",
        "checkpoint_source": checkpoint_source,
        "training_seconds": training_seconds,
        "training_diagnostics": training_diagnostics,
        "flow_quality": flow_quality,
        "total_seconds": elapsed,
        "partition": partition,
        "mass_handling": "Dirac masses remain constant; no exp(-integral div F) factor",
        "kde_domain_diagnostic": full_kde_diagnostics,
        "error_separation": "schedule expectation only; KDE has no random component",
    }
    write_json(paths.config / "config.json", complete_config)
    write_json(paths.config / "seeds.json", seeds)
    versions = version_information(args.device, device, args.dtype)
    write_json(paths.config / "versions.json", versions)
    write_manifest(
        paths.root,
        configuration=complete_config,
        seed=seeds,
        versions=versions,
    )
    write_json(
        paths.data / "summary.json",
        {
            "terminal_errors": rows,
            "refinement": refinement_rows,
            "full_terminal_to_target_l1": full_target_l1,
            "training_selected_mse": training_diagnostics.get(
                "selected_training_mse",
                history[-1]["flow_matching_mse"] if history else None,
            ),
            "training_final_mse": history[-1]["flow_matching_mse"] if history else None,
            "flow_quality": flow_quality,
            "quick_mode_warning": args.quick,
        },
    )
    report_progress("exp4", "Generating figures", started=started)
    figures = generate_plots(paths.root)
    report_progress("exp4", "Finished", started=started)
    return {
        "output_dir": str(paths.root),
        "figures": [str(path) for path in figures],
        "training_final_mse": history[-1]["flow_matching_mse"] if history else None,
        "training_selected_mse": training_diagnostics.get(
            "selected_training_mse",
            history[-1]["flow_matching_mse"] if history else None,
        ),
        "full_terminal_to_target_l1": full_target_l1,
        "flow_quality": flow_quality,
        "terminal_errors": rows,
        "total_seconds": elapsed,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output-dir", default="outputs/measure_transport")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--allow-poor-flow",
        action="store_true",
        help="continue RBM diagnostics even when the full flow fails quality checks",
    )
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="train and validate the full flow, then stop before RBM transport",
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
