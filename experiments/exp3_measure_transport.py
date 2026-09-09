#!/usr/bin/env python3
"""Characteristic-based validation of random-batch measure transport."""

from __future__ import annotations

import argparse
import json
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

from rnode.batches import make_uniform_fixed_size, sample_batch_sequence
from rnode.data import (
    TARGET_MIXTURE_CENTERS,
    TARGET_MIXTURE_COVARIANCES,
    initial_density,
    initial_density_quadrature,
    sample_initial_compact,
)
from rnode.flow import Flow
from rnode.transport import (
    coupling_squared,
    integrate_characteristics,
    l1_from_full_change_of_variables,
    terminal_log_density_from_backward,
    terminal_log_density_from_forward,
)

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
    from experiments.exp3_measure_transport_plots import (
        generate_classification_trajectories,
        generate_plots,
        generate_qualitative,
        generate_transport_density_evolution,
    )
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
    from exp3_measure_transport_plots import (
        generate_classification_trajectories,
        generate_plots,
        generate_qualitative,
        generate_transport_density_evolution,
    )


PROBES = np.array(
    [[-1.0, -1.0], [-0.6, -1.0], [-1.0, -0.4], [-1.8, -1.0], [-1.0, -1.9]],
    dtype=float,
)


def experiment_configuration(quick: bool) -> dict:
    if quick:
        return {
            "T": 1.0,
            "hidden": 48,
            "h_powers": [3, 4, 5],
            "fit_h_powers": [3, 4, 5],
            "steps_per_switch": 4,
            "reference_dt": 2.0**-8,
            "reference_check_dt": 2.0**-7,
            "quadrature_rule": "polar",
            "coupling_quadrature": 6,
            "density_quadrature": 8,
            "n_schedules": 6,
            "l1_schedules": 3,
            "dt_refinement_schedules": 2,
            "quadrature_refinement": [5, 8, 11],
            "quadrature_refinement_schedules": 1,
            "flow_epochs": 120,
            "flow_batch": 256,
            "flow_validation": 512,
            "flow_validation_interval": 20,
        }
    return {
        "T": 1.0,
        "hidden": 48,
        "h_powers": [4, 5, 6, 7, 8, 9],
        "fit_h_powers": [6, 7, 8, 9],
        "steps_per_switch": 8,
        "reference_dt": 2.0**-12,
        "reference_check_dt": 2.0**-11,
        # Polar radial-node counts; the cloud holds 4 n^2 points.
        # 16 -> 1024 for the smooth coupling integrand, 26 -> 2704 for the
        # L1 integrand, whose positive part has a kink and so needs more.
        "quadrature_rule": "polar",
        "coupling_quadrature": 16,
        "density_quadrature": 26,
        "n_schedules": 40,
        "l1_schedules": 12,
        "dt_refinement_schedules": 6,
        # Brackets the production resolution so the 26 -> 37 gap measures the
        # quadrature floor at the resolution actually used.
        "quadrature_refinement": [13, 26, 37],
        "quadrature_refinement_schedules": 3,
        "flow_epochs": 12000,
        "flow_batch": 1024,
        "flow_validation": 4096,
        "flow_validation_interval": 100,
    }


def _balanced_target(n, rng):
    counts = np.full(3, n // 3, dtype=int)
    counts[: n % 3] += 1
    blocks = [
        rng.multivariate_normal(c, cov, int(k))
        for c, cov, k in zip(TARGET_MIXTURE_CENTERS, TARGET_MIXTURE_COVARIANCES, counts)
    ]
    values = np.concatenate(blocks)
    return values[rng.permutation(n)]


def _new_flow(config, seed, dtype, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = Flow(dim=2, hidden=config["hidden"])
    return model.to(dtype=dtype, device=device)


def _train_flow(config, seeds, dtype, device, started):
    model = _new_flow(config, seeds["model_initialization"], dtype, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, threshold=1e-3, min_lr=2e-5
    )
    rng = np.random.default_rng(seeds["train_data"])
    vrng = np.random.default_rng(seeds["calibration_data"])
    n_val = config["flow_validation"]
    x0v = torch.as_tensor(sample_initial_compact(n_val, rng=vrng), dtype=dtype, device=device)
    x1v = torch.as_tensor(_balanced_target(n_val, vrng), dtype=dtype, device=device)
    tv = torch.as_tensor(vrng.random(n_val), dtype=dtype, device=device)
    xv = (1 - tv[:, None]) * x0v + tv[:, None] * x1v
    vv = x1v - x0v

    best = {"mse": float("inf"), "epoch": 0, "state": None}
    validation_rows = []
    for epoch in range(1, config["flow_epochs"] + 1):
        n = config["flow_batch"]
        x0 = torch.as_tensor(sample_initial_compact(n, rng=rng), dtype=dtype, device=device)
        x1 = torch.as_tensor(_balanced_target(n, rng), dtype=dtype, device=device)
        t = torch.as_tensor(rng.random(n), dtype=dtype, device=device)
        x = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = (model(t, x) - (x1 - x0)).square().sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % config["flow_validation_interval"] == 0:
            with torch.no_grad():
                val = float((model(tv, xv) - vv).square().sum(1).mean().cpu())
            scheduler.step(val)
            validation_rows.append(
                {"epoch": epoch, "training_mse": float(loss.detach().cpu()), "validation_mse": val}
            )
            if val < best["mse"]:
                best = {
                    "mse": val,
                    "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }
        if epoch % max(1, config["flow_epochs"] // 20) == 0:
            report_progress("exp3", f"Flow matching {epoch}/{config['flow_epochs']}", started=started)

    if best["state"] is None:
        raise RuntimeError("flow matching produced no validation checkpoint")
    model.load_state_dict(best["state"])
    return model.eval(), validation_rows, {
        "selection_rule": "minimum MSE on a fixed independent validation sample",
        "selected_epoch": best["epoch"],
        "best_validation_mse": best["mse"],
    }


def _load_or_train(args, paths, config, seeds, dtype, device, started):
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = _new_flow(config, seeds["model_initialization"], dtype, device)
        model.load_state_dict(payload["state_dict"])
        rows = payload.get("validation_history", [])
        diagnostics = payload.get("training_diagnostics", {"selection_rule": "supplied checkpoint"})
        source = str(Path(args.checkpoint).expanduser().resolve())
    else:
        model, rows, diagnostics = _train_flow(config, seeds, dtype, device, started)
        source = "trained"
    torch.save(
        {
            "format_version": 2,
            "model_class": "Flow",
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "validation_history": rows,
            "training_diagnostics": diagnostics,
        },
        paths.checkpoints / "flow.pt",
    )
    if rows:
        write_csv(paths.data / "flow_validation.csv", rows)
    return model, source, diagnostics


def _quadrature(n, dtype, device, rule="polar"):
    points, masses = initial_density_quadrature(n, rule=rule, dtype=torch.float64)
    return points.to(dtype=dtype, device=device), masses.numpy()


def _full_density(model, points, config):
    terminal, increment = integrate_characteristics(
        model, points, config["T"], config["reference_dt"], config["T"],
        direction="forward", track_log_density=True,
    )
    log_density = terminal_log_density_from_forward(
        initial_density, points.cpu().numpy(), increment.cpu().numpy()
    )
    return terminal, log_density


def _schedule(seed, scheme, T, h):
    return sample_batch_sequence(
        scheme, int(round(T / h)), np.random.default_rng(int(seed))
    )


def _density_refinement(model, config, schedules_by_h, dtype, device):
    """Paired spatial/temporal checks per h; never subtract ensemble means."""
    scheme = make_uniform_fixed_size(model.hidden_dim, model.hidden_dim // 3)
    spatial, temporal, floors = [], [], {}
    probe_terminal, probe_log = _full_density(model, torch.as_tensor(PROBES, dtype=dtype, device=device), config)
    probe_y, probe_q = integrate_characteristics(model, probe_terminal, config["T"],
        config["reference_dt"]/2, config["T"], direction="backward", track_log_density=True)
    probe_full_fine = np.exp(terminal_log_density_from_backward(
        initial_density, probe_y.cpu().numpy(), probe_q.cpu().numpy()))
    for h, schedules in schedules_by_h.items():
        report_progress("exp3-check", f"Density refinement h={h:g}")
        schedules = schedules[:config["quadrature_refinement_schedules"]]
        values_by_resolution = {}
        for resolution in config["quadrature_refinement"]:
            points, masses = _quadrature(resolution, dtype, device, config["quadrature_rule"])
            terminal, full_log = _full_density(model, points, config)
            values = []
            production = resolution == config["density_quadrature"]
            if production:
                finer_config = {**config, "reference_dt": config["reference_dt"]/2}
                fine_terminal, fine_full_log = _full_density(model, points, finer_config)
            for k, schedule in enumerate(schedules):
                dt = h/config["steps_per_switch"]
                def evaluate(at, log_full, step):
                    y, q = integrate_characteristics(model, at, config["T"], step, h,
                        schedule, inclusion_probs=scheme.inclusion_probs,
                        direction="backward", track_log_density=True)
                    log_random = terminal_log_density_from_backward(initial_density, y.cpu().numpy(), q.cpu().numpy())
                    return l1_from_full_change_of_variables(log_full, log_random, masses)
                value = evaluate(terminal, full_log, dt)
                values.append(value)
                spatial.append({"h": h, "schedule_index": k, "radial_nodes": resolution,
                                "n_points": len(masses), "l1": value, "production": production})
                if production:
                    random_fine = evaluate(terminal, full_log, dt/2)
                    both_fine = evaluate(fine_terminal, fine_full_log, dt/2)
                    probe_rho = []
                    for step in (dt, dt/2):
                        y, q = integrate_characteristics(model, probe_terminal, config["T"], step, h,
                            schedule, inclusion_probs=scheme.inclusion_probs,
                            direction="backward", track_log_density=True)
                        probe_rho.append(np.exp(terminal_log_density_from_backward(
                            initial_density, y.cpu().numpy(), q.cpu().numpy())))
                    temporal.append({"h": h, "schedule_index": k, "dt": dt,
                                     "l1": value, "l1_random_dt_half": random_fine,
                                     "l1_both_dt_half": both_fine,
                                     "random_dt_difference": value-random_fine,
                                     "reference_dt_difference": random_fine-both_fine,
                                     "paired_total_difference": value-both_fine,
                                     "pointwise_density_dt_difference": (probe_rho[0]-probe_rho[1]).tolist(),
                                     "pointwise_mse": ((probe_rho[0]-np.exp(probe_log))**2).tolist(),
                                     "pointwise_mse_refined": ((probe_rho[1]-probe_full_fine)**2).tolist()})
            values_by_resolution[resolution] = np.asarray(values)
        resolutions = sorted(values_by_resolution)
        i = resolutions.index(config["density_quadrature"])
        if i+1 == len(resolutions):
            raise ValueError("spatial refinement must include a resolution finer than production")
        paired = values_by_resolution[resolutions[i]] - values_by_resolution[resolutions[i+1]]
        time_rows = [r for r in temporal if r["h"] == h]
        floors[h] = {"quadrature_floor": float(np.abs(paired).max()),
                     "quadrature_mean_absolute_difference": float(np.abs(paired).mean()),
                     "quadrature_signed_mean_difference": float(paired.mean()),
                     "temporal_floor": max(abs(r["random_dt_difference"])+abs(r["reference_dt_difference"]) for r in time_rows),
                     "n_refinement_schedules": len(schedules),
                     "quadrature_floor_rule": "maximum absolute paired production-to-finer change at this h; diagnostic, not a certified bound"}
    return spatial, temporal, floors


def run_diagnostics(args):
    source = ArtifactPaths.latest_run(args.output_dir)
    config = json.loads((source / "config/config.json").read_text())["experiment"]
    config = {**config, "quadrature_refinement_schedules": min(3, config["l1_schedules"])}
    paths = ArtifactPaths.new_run(source, prefix="diagnostics")
    device, dtype = resolve_device(args.device), resolve_dtype(args.dtype)
    if device.type == "cpu":
        torch.set_num_threads(1)
    model = _new_flow(config, 0, dtype, device)
    payload = torch.load(source / "checkpoints/flow.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval().requires_grad_(False)
    scheme = make_uniform_fixed_size(model.hidden_dim, model.hidden_dim//3)
    seeds = json.loads((source / "data/schedule_seeds.json").read_text())
    schedules = {float(h): [_schedule(s, scheme, config["T"], float(h))
                           for s in ss[:config["quadrature_refinement_schedules"]]] for h, ss in seeds.items()}
    started = time.perf_counter()
    spatial, temporal, floors = _density_refinement(model, config, schedules, dtype, device)
    write_csv(paths.data / "density_spatial_pairs.csv", spatial)
    write_csv(paths.data / "density_temporal_pairs.csv", temporal)
    write_csv(paths.data / "density_resolution.csv", [{"h": h, **v} for h, v in floors.items()])
    write_json(paths.config / "config.json", {"source_run": str(source), "configuration": config,
               "scope": "same saved schedules and learned checkpoint; L1 spatial and temporal refinement only",
               "seconds": time.perf_counter()-started})
    return {"output_dir": str(paths.root), "seconds": time.perf_counter()-started}


def _mean_se(values):
    values = np.asarray(values, dtype=float)
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(se)


def _fit(h, values, powers, fit_powers):
    mask = np.array([p in set(fit_powers) for p in powers])
    y = np.asarray(values, dtype=float)[mask]
    if mask.sum() < 3 or np.any(y <= 0):
        return {"fit_performed": False, "slope": np.nan, "n_points": int(mask.sum())}
    x = np.log(np.asarray(h, dtype=float)[mask])
    slope, intercept = np.polyfit(x, np.log(y), 1)
    return {
        "fit_performed": True,
        "slope": float(slope),
        "intercept": float(intercept),
        "n_points": int(mask.sum()),
        "h_min": float(np.exp(x.min())),
        "h_max": float(np.exp(x.max())),
    }


def _reference_diagnostics(model, config, dtype, device):
    points = torch.as_tensor(PROBES, dtype=dtype, device=device)
    fine, fine_q = integrate_characteristics(
        model, points, config["T"], config["reference_dt"], config["T"],
        direction="forward", track_log_density=True,
    )
    coarse, coarse_q = integrate_characteristics(
        model, points, config["T"], config["reference_check_dt"], config["T"],
        direction="forward", track_log_density=True,
    )
    recovered, backward_q = integrate_characteristics(
        model, fine, config["T"], config["reference_dt"], config["T"],
        direction="backward", track_log_density=True,
    )
    return {
        "reference_vs_check_rms_position": float((fine - coarse).square().sum(1).mean().sqrt().cpu()),
        "reference_vs_check_max_log_increment": float((fine_q - coarse_q).abs().max().cpu()),
        "full_roundtrip_max_position": float((recovered - points).norm(dim=1).max().cpu()),
        "full_roundtrip_max_log_increment": float((backward_q - fine_q).abs().max().cpu()),
    }


def run_experiment(args):
    paths = ArtifactPaths.new_run(args.output_dir)
    device, dtype = resolve_device(args.device), resolve_dtype(args.dtype)
    if device.type == "cpu":
        torch.set_num_threads(1)
    config = experiment_configuration(args.quick)
    config["h_values"] = [2.0**-p for p in config["h_powers"]]
    seeds = make_seed_manifest(args.seed)
    started = time.perf_counter()
    report_progress("exp3", f"Starting transport ({'quick' if args.quick else 'full'} mode)")

    model, checkpoint_source, training_diagnostics = _load_or_train(
        args, paths, config, seeds, dtype, device, started
    )
    scheme = make_uniform_fixed_size(model.hidden_dim, model.hidden_dim // 3)
    reference = _reference_diagnostics(model, config, dtype, device)

    coupling_points, coupling_masses = _quadrature(config["coupling_quadrature"], dtype, device, config["quadrature_rule"])
    full_coupling, _ = integrate_characteristics(
        model, coupling_points, config["T"], config["reference_dt"], config["T"]
    )
    full_coupling = full_coupling.cpu().numpy()

    density_points, density_masses = _quadrature(config["density_quadrature"], dtype, device, config["quadrature_rule"])
    full_density_points, full_log_density = _full_density(model, density_points, config)
    full_density_np = full_density_points.cpu().numpy()

    probe_initial = torch.as_tensor(PROBES, dtype=dtype, device=device)
    probe_terminal, probe_q = integrate_characteristics(
        model, probe_initial, config["T"], config["reference_dt"], config["T"],
        direction="forward", track_log_density=True,
    )
    probe_terminal_np = probe_terminal.cpu().numpy()
    probe_full_log = terminal_log_density_from_forward(initial_density, PROBES, probe_q.cpu().numpy())
    probe_full_density = np.exp(probe_full_log)

    coupling_rows, l1_rows, point_rows = [], [], []
    scalar_samples, schedule_manifest, schedules_by_h = {}, {}, {}
    random_roundtrip = []

    for power, h in zip(config["h_powers"], config["h_values"]):
        case_rng = np.random.default_rng(np.random.SeedSequence([seeds["schedule_generation"], power]))
        case_seeds = case_rng.integers(0, 2**32 - 1, config["n_schedules"], dtype=np.uint32)
        schedule_manifest[str(h)] = case_seeds.tolist()
        schedules = [_schedule(seed, scheme, config["T"], h) for seed in case_seeds]
        schedules_by_h[h] = schedules
        sq_values, rms_values, l1_values = [], [], []
        point_samples = [[] for _ in PROBES]
        report_progress("exp3", f"h=2^-{power}", started=started)

        for k, schedule in enumerate(schedules):
            random_terminal, _ = integrate_characteristics(
                model, coupling_points, config["T"], h / config["steps_per_switch"], h,
                schedule, inclusion_probs=scheme.inclusion_probs,
            )
            sq = coupling_squared(full_coupling, random_terminal.cpu().numpy(), coupling_masses)
            sq_values.append(sq)
            rms_values.append(np.sqrt(sq))

            terminal_eval = probe_terminal_np
            if k < config["l1_schedules"]:
                terminal_eval = np.concatenate([full_density_np, probe_terminal_np])
            preimage, random_q = integrate_characteristics(
                model, torch.as_tensor(terminal_eval, dtype=dtype, device=device),
                config["T"], h / config["steps_per_switch"], h, schedule,
                inclusion_probs=scheme.inclusion_probs, direction="backward", track_log_density=True,
            )
            random_log = terminal_log_density_from_backward(
                initial_density, preimage.cpu().numpy(), random_q.cpu().numpy()
            )
            if k < config["l1_schedules"]:
                n = len(full_density_np)
                l1_values.append(
                    l1_from_full_change_of_variables(full_log_density, random_log[:n], density_masses)
                )
                random_probe_log = random_log[n:]
            else:
                random_probe_log = random_log
            random_probe_density = np.exp(np.clip(random_probe_log, -745.0, 700.0))
            for j, value in enumerate(np.square(random_probe_density - probe_full_density)):
                point_samples[j].append(float(value))

            if power == config["h_powers"][-1] and k < 2:
                x, qf = integrate_characteristics(
                    model, probe_initial, config["T"], h / config["steps_per_switch"], h,
                    schedule, inclusion_probs=scheme.inclusion_probs,
                    direction="forward", track_log_density=True,
                )
                y, qb = integrate_characteristics(
                    model, x, config["T"], h / config["steps_per_switch"], h,
                    schedule, inclusion_probs=scheme.inclusion_probs,
                    direction="backward", track_log_density=True,
                )
                random_roundtrip.append(
                    {
                        "schedule_index": k,
                        "max_position_error": float((y - probe_initial).norm(dim=1).max().cpu()),
                        "max_log_increment_error": float((qb - qf).abs().max().cpu()),
                    }
                )

        mean_sq, se_sq = _mean_se(sq_values)
        mean_rms, se_rms = _mean_se(rms_values)
        coupling_rows.append(
            {
                "h_power": power, "h": h,
                "mean_squared_coupling": mean_sq, "se_squared_coupling": se_sq,
                "mean_rms_coupling": mean_rms, "se_rms_coupling": se_rms,
                "n_schedules": len(schedules),
            }
        )
        mean_l1, se_l1 = _mean_se(l1_values)
        l1_rows.append(
            {
                "h_power": power, "h": h, "expected_l1_error": mean_l1,
                "se_l1_error": se_l1, "n_schedules": len(l1_values),
            }
        )
        for j, samples in enumerate(point_samples):
            mean, se = _mean_se(samples)
            point_rows.append(
                {
                    "h_power": power, "h": h, "point_index": j,
                    "initial_x1": PROBES[j, 0], "initial_x2": PROBES[j, 1],
                    "terminal_x1": probe_terminal_np[j, 0], "terminal_x2": probe_terminal_np[j, 1],
                    "full_density": probe_full_density[j],
                    "density_mse": mean, "se_density_mse": se, "n_schedules": len(samples),
                }
            )
        scalar_samples[f"coupling_sq__p{power}"] = np.asarray(sq_values)
        scalar_samples[f"l1__p{power}"] = np.asarray(l1_values)

    write_csv(paths.data / "coupling.csv", coupling_rows)
    write_csv(paths.data / "density_l1.csv", l1_rows)
    write_csv(paths.data / "density_pointwise.csv", point_rows)
    np.savez_compressed(paths.data / "scalar_samples.npz", **scalar_samples)
    write_json(paths.data / "schedule_seeds.json", schedule_manifest)

    slope_rows = [
        {"metric": "squared_coupling", "bound_exponent": 1.0, **_fit(
            [r["h"] for r in coupling_rows], [r["mean_squared_coupling"] for r in coupling_rows],
            [r["h_power"] for r in coupling_rows], config["fit_h_powers"]
        )},
        {"metric": "rms_coupling", "bound_exponent": 0.5, **_fit(
            [r["h"] for r in coupling_rows], [r["mean_rms_coupling"] for r in coupling_rows],
            [r["h_power"] for r in coupling_rows], config["fit_h_powers"]
        )},
    ]
    for j in range(len(PROBES)):
        selected = [r for r in point_rows if r["point_index"] == j]
        slope_rows.append(
            {"metric": "pointwise_density_mse", "point_index": j, "bound_exponent": 1.0, **_fit(
                [r["h"] for r in selected], [r["density_mse"] for r in selected],
                [r["h_power"] for r in selected], config["fit_h_powers"]
            )}
        )
    # slope_fits.csv is written after the quadrature refinement, which decides
    # which h values carry a resolved L1 signal.

    finest_h = config["h_values"][-1]
    dt_rows = []
    for k, schedule in enumerate(schedules_by_h[finest_h][: config["dt_refinement_schedules"]]):
        coarse, _ = integrate_characteristics(
            model, coupling_points, config["T"], finest_h / config["steps_per_switch"], finest_h,
            schedule, inclusion_probs=scheme.inclusion_probs,
        )
        fine, _ = integrate_characteristics(
            model, coupling_points, config["T"], finest_h / (2 * config["steps_per_switch"]), finest_h,
            schedule, inclusion_probs=scheme.inclusion_probs,
        )
        dt_rows.append(
            {
                "schedule_index": k,
                "coarse_squared_coupling": coupling_squared(full_coupling, coarse.cpu().numpy(), coupling_masses),
                "refined_squared_coupling": coupling_squared(full_coupling, fine.cpu().numpy(), coupling_masses),
                "coarse_refined_rms": np.sqrt(coupling_squared(coarse.cpu().numpy(), fine.cpu().numpy(), coupling_masses)),
            }
        )
    write_csv(paths.data / "dt_refinement.csv", dt_rows)

    quadrature_rows, density_time_rows, floors = _density_refinement(
        model, config, schedules_by_h, dtype, device
    )
    write_csv(paths.data / "quadrature_refinement.csv", quadrature_rows)
    write_csv(paths.data / "density_dt_refinement.csv", density_time_rows)
    for row in l1_rows:
        diagnostic = floors[row["h"]]
        row.update(diagnostic)
        scale = max(diagnostic["quadrature_floor"], diagnostic["temporal_floor"])
        row["signal_over_quadrature_floor"] = (
            row["expected_l1_error"] / diagnostic["quadrature_floor"]
            if diagnostic["quadrature_floor"] > 0 else np.inf
        )
        row["l1_resolved_above_quadrature_floor"] = bool(row["expected_l1_error"] > 10*scale)
    write_csv(paths.data / "density_l1.csv", l1_rows)

    # The L1 slope is fitted only where the signal clears the quadrature floor.
    resolved_powers = [
        row["h_power"]
        for row in l1_rows
        if row["l1_resolved_above_quadrature_floor"] and row["h_power"] in config["fit_h_powers"]
    ]
    slope_rows.append({
        "metric": "l1_density",
        "bound_exponent": 0.5,
        "fit_restriction": "predeclared h range, further restricted to L1 above "
        "ten times the paired spatial and temporal diagnostics at the same h",
        "n_excluded_by_quadrature_floor": len(
            [p for p in config["fit_h_powers"] if p not in resolved_powers]
        ),
        **_fit(
            [r["h"] for r in l1_rows],
            [r["expected_l1_error"] for r in l1_rows],
            [r["h_power"] for r in l1_rows],
            resolved_powers,
        ),
    })
    write_csv(paths.data / "slope_fits.csv", slope_rows)

    diagnostics = {"full_reference": reference, "random_roundtrip": random_roundtrip}
    write_json(paths.data / "characteristic_diagnostics.json", diagnostics)
    elapsed = time.perf_counter() - started
    complete = {
        "script": "exp3_measure_transport.py",
        "cli": vars(args),
        "experiment": config,
        "flow_checkpoint_source": checkpoint_source,
        "flow_training_diagnostics": training_diagnostics,
        "sampling": "uniform fixed-size, r=16",
        "density_method": "Liouville formula along characteristics; no KDE",
        "l1_method": "2 E_rho0[(1-rho_hat/rho)_+] through the full flow; equal exact unit masses",
        "target_role": "used to train a nontrivial full field, not as the reference density in RBM errors",
        "same_schedule_for_all_points": True,
        "backward_reuses_forward_schedule": True,
        "probe_initial_points": PROBES,
        "diagnostics": diagnostics,
        "total_seconds": elapsed,
    }
    versions = version_information(args.device, device, args.dtype)
    write_json(paths.config / "config.json", complete)
    write_json(paths.config / "seeds.json", seeds)
    write_json(paths.config / "versions.json", versions)
    write_manifest(paths.root, configuration=complete, seed=seeds, versions=versions)
    write_json(
        paths.data / "summary.json",
        {
            "coupling": coupling_rows,
            "density_l1": l1_rows,
            "density_pointwise": point_rows,
            "slope_fits": slope_rows,
            "dt_refinement": dt_rows,
            "quadrature_refinement": quadrature_rows,
            "diagnostics": diagnostics,
            "quick_mode_warning": bool(args.quick),
        },
    )
    figures = generate_plots(paths.root)
    report_progress("exp3", "Finished", started=started)
    return {"output_dir": str(paths.root), "figures": [str(p) for p in figures], "total_seconds": elapsed}


def run_qualitative(args):
    """Regenerate the frozen-flow illustration panels into the exp3 run folder.

    These panels reuse the trained classifier from the most recent ``exp1`` run
    and this experiment's flow; nothing is retrained.  The three groups (static
    illustration, real classification trajectories, and the transported-density
    evolution) are rewritten in place.
    """
    transport_run = ArtifactPaths.latest_run(args.output_dir)
    classification_run = ArtifactPaths.latest_run(args.classification_dir)
    figures = transport_run / "figures"
    # Each panel is written beside the run that produced the model it shows:
    # the classification panels with the classifier, the transport ones here.
    classification_figures = classification_run / "figures"
    report_progress("exp3-qual", f"classifier {classification_run.name}, flow {transport_run.name}")
    outputs = list(generate_qualitative(
        classification_run, transport_run, figures,
        classification_figure_dir=classification_figures, overwrite=True,
    ))
    report_progress("exp3-qual", "illustration panels done")
    outputs += generate_classification_trajectories(
        classification_run, classification_figures,
        source_illustration=figures, overwrite=True,
    )
    report_progress("exp3-qual", "classification trajectories done")
    outputs += generate_transport_density_evolution(
        transport_run, figures, source_illustration=figures, overwrite=True,
    )
    report_progress("exp3-qual", "density evolution done")
    return {"output_dir": str(transport_run), "figures": [str(p) for p in outputs]}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output-dir", default="results/exp3",
                        help="Parent directory for run folders; --plots-only reuses the most recent one")
    parser.add_argument("--classification-dir", default="results/exp1",
                        help="Parent directory of the exp1 runs whose classifier feeds --qualitative")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--qualitative", action="store_true",
                        help="Only rebuild the frozen-flow illustration panels in the latest run")
    parser.add_argument("--diagnostics-only", action="store_true",
                        help="Refine densities from a saved checkpoint and schedules without training")
    return parser


def main():
    args = build_parser().parse_args()
    if args.diagnostics_only:
        print(json.dumps(run_diagnostics(args), indent=2))
        return
    if args.qualitative:
        print(json.dumps(run_qualitative(args), indent=2))
        return
    result = (
        {"figures": [str(p) for p in generate_plots(ArtifactPaths.latest_run(args.output_dir))]}
        if args.plots_only else run_experiment(args)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
