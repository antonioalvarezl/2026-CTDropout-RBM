#!/usr/bin/env python3
"""Small work--accuracy illustration for the frozen classification flow."""

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
from rnode.integrators import integrate_fixed_grid

try:
    from experiments._paper_common import (
        ArtifactPaths,
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
    from experiments.exp4_work_accuracy_plots import generate_plots
except ModuleNotFoundError:
    from _paper_common import (
        ArtifactPaths,
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
    from exp4_work_accuracy_plots import generate_plots


def experiment_configuration(quick: bool) -> dict:
    """Numerical choices for the compact work--accuracy experiment."""
    if quick:
        return {
            "gammas": [0.5, 0.25],
            "h_powers": [3, 4, 5, 6],
            "batch_sizes": [4, 8, 12],
            "n_schedules": 8,
            "full_dt_powers": [2, 3, 4, 5, 6, 7],
            "reference_dt": 2.0**-11,
            "reference_check_dt": 2.0**-10,
        }
    return {
        "gammas": [0.5, 0.25],
        "h_powers": [4, 5, 6, 7, 8, 9],
        "batch_sizes": [4, 8, 12],
        "n_schedules": 80,
        "full_dt_powers": [3, 4, 5, 6, 7, 8, 9, 10],
        "reference_dt": 2.0**-16,
        "reference_check_dt": 2.0**-15,
    }


def _schedule_seeds(base_seed: int, h_power: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(
        np.random.SeedSequence([int(base_seed), 4107, int(h_power)])
    )
    return rng.integers(0, np.iinfo(np.uint32).max, size=n, dtype=np.uint32)


def _deterministic_rms(trajectory: torch.Tensor, reference: torch.Tensor) -> float:
    """sqrt(mean_data(max_time squared error)) for one deterministic trajectory."""
    squared = (trajectory - reference).square().sum(dim=-1)
    return float(torch.sqrt(squared.max(dim=0).values.mean()).cpu())


def _test_loss(terminal: torch.Tensor, targets: torch.Tensor) -> float:
    """mean_data ||x_T-y||^2 for one realized terminal state."""
    return float((terminal - targets).square().sum(dim=-1).mean().cpu())


ACCURACY_NOTE = (
    "Accuracy is the terminal RMS error. Every method carries t=T on its own "
    "grid, so this is one functional evaluated identically for all of them. A "
    "maximum over time is not used as the primary metric here: the tested "
    "steps span 2^-3 to 2^-11, a coarse method would maximise over a few "
    "nodes and a fine one over hundreds, and forcing a common grid would make "
    "it coarse enough to miss the maximum entirely. The maximum over each "
    "method's own grid is reported alongside as a resolution-dependent "
    "diagnostic and must not be compared across different steps."
)

PREDICTIVE_NOTE = (
    "test_loss is mean_data ||x_T-y||^2 on the held-out test split, with the "
    "squared error of each realized terminal state averaged over schedules. "
    "The loss of the schedule-averaged prediction is deliberately NOT "
    "reported: that would evaluate a different, ensemble, predictor. The "
    "control is frozen and no configuration is selected using these labels."
)


@torch.no_grad()
def _random_accuracy(
    model,
    features: torch.Tensor,
    targets: torch.Tensor,
    reference_times: torch.Tensor,
    reference_values: torch.Tensor,
    *,
    T: float,
    h: float,
    dt: float,
    scheme,
    schedule_seeds: np.ndarray,
) -> dict:
    """RMS against the full flow and test loss against the labels, per schedule.

    Only the running sums and one scalar loss per schedule are kept, so the
    realizations themselves are never stored.
    """
    n_steps = int(round(T / dt))
    requested_times = torch.linspace(
        0.0,
        T,
        n_steps + 1,
        dtype=features.dtype,
        device=features.device,
    )
    reference = trajectory_at_times(
        reference_times, reference_values, requested_times
    )
    error_sum = torch.zeros(
        (n_steps + 1, features.shape[0]),
        dtype=features.dtype,
        device=features.device,
    )

    losses = []
    n_intervals = int(round(T / h))
    for seed in schedule_seeds:
        schedule = sample_batch_sequence(
            scheme,
            n_intervals,
            np.random.default_rng(int(seed)),
        )
        times, trajectory = integrate_fixed_grid(
            model,
            features,
            T,
            dt,
            h,
            schedule,
            inclusion_probs=scheme.inclusion_probs,
            method="euler",
        )
        if not torch.allclose(times, requested_times):
            raise RuntimeError("unexpected integration grid")
        error_sum += (trajectory - reference).square().sum(dim=-1)
        # One loss per realization: the schedule average is taken over these
        # scalars, not over the predictions, which would be another predictor.
        losses.append(_test_loss(trajectory[-1], targets))

    mean_error = error_sum / len(schedule_seeds)
    losses = np.asarray(losses, dtype=float)
    return {
        "rms_error": float(torch.sqrt(mean_error[-1].mean()).cpu()),
        "rms_error_max_own_grid": float(torch.sqrt(mean_error.max(dim=0).values.mean()).cpu()),
        "own_grid_nodes": int(mean_error.shape[0]),
        "test_loss": float(losses.mean()),
        "se_test_loss": float(losses.std(ddof=1) / np.sqrt(losses.size))
        if losses.size > 1 else 0.0,
    }


def _reference_check(model, features, T, fine_dt, check_dt):
    check_times, check = reference_trajectory(model, features, T, check_dt)
    fine_times, fine = reference_trajectory(model, features, T, fine_dt)
    fine_at_check = trajectory_at_times(fine_times, fine, check_times)
    return fine_times, fine, _deterministic_rms(check, fine_at_check)


def _default_checkpoint(base_run_dir) -> Path | None:
    """Adopt exp1's most recent trained model if one exists.

    exp1 and exp4 use the same classification model and, with matching
    seeds, training it again would be numerically close but not guaranteed
    bit-identical (thread/BLAS reduction order). Reusing the checkpoint
    keeps every experiment's work--accuracy figures about the one model
    exp1 and exp2 already characterised, and skips ~5 minutes of retraining.
    Falls back to training a fresh model, as before, if none is found.
    """
    parent = Path(base_run_dir).expanduser()
    if not parent.is_dir():
        return None
    runs = sorted(
        run for run in parent.glob("*")
        if (run / "checkpoints" / "base_model.pt").is_file()
    )
    return runs[-1] / "checkpoints" / "base_model.pt" if runs else None


def run_experiment(args):
    paths = ArtifactPaths.new_run(args.output_dir)
    device = resolve_device(args.device)
    config = experiment_configuration(args.quick)
    started = time.perf_counter()
    mode = "quick" if args.quick else "full"

    checkpoint = args.checkpoint or _default_checkpoint(args.base_run_dir)
    if checkpoint is None:
        report_progress("exp4", f"No trained run found under {args.base_run_dir}; training a fresh model", started=started)
    else:
        report_progress("exp4", f"Reusing trained model: {checkpoint}", started=started)

    report_progress("exp4", f"Starting work--accuracy illustration ({mode} mode)", started=started)
    prepared = prepare_base_model(
        paths,
        quick=args.quick,
        seed=args.seed,
        dtype_name=args.dtype,
        device=device,
        checkpoint=str(checkpoint) if checkpoint else None,
    )
    model = prepared["model"].eval()
    model.requires_grad_(False)
    base = prepared["base_config"]
    T = float(base["model"]["T"])
    p = int(model.hidden_dim)
    # Evaluation is on the held-out test split only; the control stays frozen.
    features, _, targets = prepared["datasets"]["test"]

    report_progress("exp4", "Computing accurate full reference", started=started)
    reference_times, reference, reference_check_rms = _reference_check(
        model,
        features,
        T,
        config["reference_dt"],
        config["reference_check_dt"],
    )
    # The fine RK4 flow is both the accuracy reference and the predictive
    # baseline: the test loss the frozen control attains when integrated exactly.
    reference_test_loss = _test_loss(reference[-1], targets)
    write_csv(
        paths.data / "reference_check.csv",
        [{
            "reference_dt": config["reference_dt"],
            "check_dt": config["reference_check_dt"],
            "rms_difference": reference_check_rms,
            "test_loss": reference_test_loss,
        }],
    )

    rows = []
    schedule_manifest = {}

    # Full model: its numerical step is varied independently of any switching h.
    report_progress("exp4", "Evaluating full-model Euler curve", started=started)
    for power in config["full_dt_powers"]:
        dt = 2.0 ** (-power)
        with torch.no_grad():
            times, trajectory = integrate_fixed_grid(
                model, features, T, dt, T, method="euler"
            )
        reference_on_grid = trajectory_at_times(reference_times, reference, times)
        squared = (trajectory - reference_on_grid).square().sum(dim=-1)
        n_steps = int(round(T / dt))
        rows.append({
            "scheme": "full",
            "scheme_label": "Full model",
            "batch_size": p,
            "h": "",
            "gamma": "",
            "dt": dt,
            "rms_error": float(torch.sqrt(squared[-1].mean()).cpu()),
            "rms_error_max_own_grid": _deterministic_rms(trajectory, reference_on_grid),
            "own_grid_nodes": int(squared.shape[0]),
            "test_loss": _test_loss(trajectory[-1], targets),
            "se_test_loss": 0.0,
            "n_steps": n_steps,
            # Euler takes one stage per step, so each step evaluates the
            # control once and p neuron components once.
            "component_evaluations": n_steps * p,
            "control_evaluations": n_steps,
            "work_units": n_steps * p,
            "n_schedules": 1,
        })

    # RBM: uniform fixed-size batches, dt = gamma h, as in the first-order work model.
    # The two gammas are the same experiment at two Euler steps: the schedules
    # are drawn per h alone, so a given (h, r) pair is integrated from the same
    # realizations at dt=h/2 and dt=h/4 and the comparison stays paired.
    total_cases = (len(config["gammas"]) * len(config["batch_sizes"])
                   * len(config["h_powers"]))
    completed = 0
    for h_power in config["h_powers"]:
        h = 2.0 ** (-h_power)
        seeds = _schedule_seeds(
            prepared["seeds"]["schedule_generation"],
            h_power,
            config["n_schedules"],
        )
        schedule_manifest[str(h)] = seeds.tolist()

        for gamma in config["gammas"]:
            dt = gamma * h
            for r in config["batch_sizes"]:
                completed += 1
                report_progress(
                    "exp4",
                    f"RBM case {completed}/{total_cases}: r={r}, h=2^-{h_power}, "
                    f"dt=h/{round(1 / gamma)}",
                    started=started,
                )
                scheme = make_uniform_fixed_size(p, r)
                accuracy = _random_accuracy(
                    model,
                    features,
                    targets,
                    reference_times,
                    reference,
                    T=T,
                    h=h,
                    dt=dt,
                    scheme=scheme,
                    schedule_seeds=seeds,
                )
                n_steps = int(round(T / dt))
                rows.append({
                    "scheme": f"uniform_r{r}_g{round(1 / gamma)}",
                    "scheme_label": f"Uniform fixed-size, r={r}, dt=h/{round(1 / gamma)}",
                    "batch_size": r,
                    "h": h,
                    "gamma": gamma,
                    "dt": dt,
                    **accuracy,
                    "n_steps": n_steps,
                    # Per input trajectory and per schedule, so the abscissa is
                    # the cost of one evaluation, not of the Monte Carlo sweep.
                    # Only the component count shrinks with r: the control is
                    # generated once per step whatever the batch size, so it is
                    # reported separately rather than folded into the proxy.
                    "component_evaluations": n_steps * r,
                    "control_evaluations": n_steps,
                    "work_units": n_steps * r,
                    "n_schedules": len(seeds),
                })

    write_csv(paths.data / "work_accuracy.csv", rows)
    write_json(paths.data / "schedule_seeds.json", schedule_manifest)

    elapsed = time.perf_counter() - started
    complete_config = {
        "script": "exp4_work_accuracy.py",
        "cli": vars(args),
        "base": base,
        "experiment": config,
        "checkpoint_source": prepared["checkpoint_source"],
        "integrator": "explicit Euler",
        "reference_integrator": "RK4",
        "accuracy_metric": ACCURACY_NOTE,
        "predictive_metric": PREDICTIVE_NOTE,
        "work_proxy": (
            "work_units counts evaluated neuron components per input "
            "trajectory: for RBM W=(T/dt)r=T r/(gamma h), for full W=(T/dt_F)p. "
            "Both are per input trajectory and per schedule, so they measure "
            "one evaluation rather than the cost of the Monte Carlo repetitions. "
            "control_evaluations counts hyper-network evaluations, which are "
            "one per step regardless of r and are therefore NOT reduced by "
            "random batching; the paper's proxy omits them, so work_units "
            "overstates the advantage of the random model by a margin that "
            "grows as r shrinks"
        ),
        "interpretation": (
            "empirical illustration of work versus accuracy; not an exact "
            "validation of a crossover formula. Wall-clock time is not "
            "reported because the integrators here evaluate the control and "
            "mask components rather than skipping them, so measured time is "
            "not evidence of speedup"
        ),
        "reference_check_rms": reference_check_rms,
        "reference_test_loss": reference_test_loss,
        "training_seconds": prepared["training_seconds"],
        "total_seconds": elapsed,
    }
    versions = version_information(args.device, device, args.dtype)
    write_json(paths.config / "config.json", complete_config)
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
            "reference_check_rms": reference_check_rms,
            "reference_test_loss": reference_test_loss,
            "rows": rows,
            "total_seconds": elapsed,
            "quick_mode_warning": bool(args.quick),
        },
    )

    report_progress("exp4", "Generating figure", started=started)
    figures = generate_plots(paths.root)
    report_progress("exp4", "Finished", started=started)
    return {
        "output_dir": str(paths.root),
        "figures": [str(path) for path in figures],
        "reference_check_rms": reference_check_rms,
        "reference_test_loss": reference_test_loss,
        "total_seconds": elapsed,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--output-dir",
        default="results/exp4",
        help="Parent directory for run folders; --plots-only reuses the most recent one",
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Trained classification model; by default the most recent run under --base-run-dir, else a freshly trained one")
    parser.add_argument("--base-run-dir", default="results/exp1",
                        help="Where to look for a trained classification run when --checkpoint is omitted")
    parser.add_argument("--plots-only", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.plots_only:
        result = {"figures": [str(path) for path in generate_plots(ArtifactPaths.latest_run(args.output_dir))]}
    else:
        result = run_experiment(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
