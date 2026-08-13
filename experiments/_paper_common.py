"""Shared reproducibility, training, and serialization helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import scipy
import torch
from torch import Tensor

from rnode.data import make_circles_with_targets
from rnode.integrators import integrate_fixed_grid
from rnode.models import TimeDepODE
from rnode.objectives import nearest_target_accuracy, paper_objective


def report_progress(label: str, message: str, *, started: float | None = None) -> None:
    """Print a timestamped progress line immediately, even through a runner."""
    elapsed = ""
    if started is not None:
        seconds = max(0, round(time.perf_counter() - started))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed = f" +{hours:02d}:{minutes:02d}:{seconds:02d}"
    print(f"[{label}{elapsed}] {message}", flush=True)


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    config: Path
    data: Path
    checkpoints: Path
    figures: Path

    @classmethod
    def create(cls, root: str | Path) -> "ArtifactPaths":
        root = Path(root).expanduser().resolve()
        result = cls(
            root=root,
            config=root / "config",
            data=root / "data",
            checkpoints=root / "checkpoints",
            figures=root / "figures",
        )
        for directory in (
            result.root,
            result.config,
            result.data,
            result.checkpoints,
            result.figures,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return result


def base_configuration(quick: bool) -> dict[str, Any]:
    """Return the common model, data, and training configuration."""
    return {
        "quick": bool(quick),
        "model": {
            "input_dim": 2,
            "hidden_dim": 24,
            "net_hidden": 16,
            "T": 1.0,
            "activation": "GeLU",
            "parameter_box": [-5.0, 5.0],
        },
        "data": {
            "n_train": 64 if quick else 384,
            "n_calibration": 24 if quick else 128,
            "n_test": 32 if quick else 256,
            "noise": 0.05,
            "factor": 0.5,
            "independent_splits": True,
        },
        "training": {
            "epochs": 60 if quick else 800,
            "dt": 1.0 / (16 if quick else 64),
            "learning_rate": 5e-3 if quick else 2e-3,
            "alpha": 1e-3,
            "beta": 0.1,
            "optimizer": "Adam",
            "projection_after_each_step": True,
        },
    }


def make_seed_manifest(seed: int) -> dict[str, int]:
    """Derive named, independently reusable seeds from one CLI seed."""
    sequence = np.random.SeedSequence(int(seed))
    children = sequence.spawn(8)
    values = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    names = [
        "model_initialization",
        "train_data",
        "calibration_data",
        "test_data",
        "schedule_generation",
        "bootstrap",
        "partition_generation",
        "miscellaneous",
    ]
    return {"root": int(seed), **dict(zip(names, values))}


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[name]
    except KeyError as error:
        raise ValueError("dtype must be float32 or float64") from error


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(json_safe(value))
                        if isinstance(value, (list, tuple, dict, np.ndarray))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def write_manifest(
    root: str | Path,
    *,
    configuration: dict[str, Any],
    seed: Any,
    versions: dict[str, Any],
) -> dict[str, Any]:
    """Write the required top-level reproducibility manifest."""
    manifest = {
        "configuration": configuration,
        "seed": seed,
        "commit": versions.get("git_commit"),
        "git_dirty": versions.get("git_dirty"),
        "device": versions.get("resolved_device"),
        "requested_device": versions.get("requested_device"),
        "dtype": versions.get("dtype"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(root) / "manifest.json", manifest)
    return manifest


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def version_information(
    requested_device: str,
    resolved_device: torch.device,
    dtype_name: str,
) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "requested_device": requested_device,
        "resolved_device": str(resolved_device),
        "dtype": dtype_name,
        "torch_default_dtype": str(torch.get_default_dtype()),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": commit,
        "git_dirty": dirty,
    }


def generate_datasets(
    data_config: dict[str, Any],
    seeds: dict[str, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
    datasets = {}
    for split, count_key, seed_key in (
        ("train", "n_train", "train_data"),
        ("calibration", "n_calibration", "calibration_data"),
        ("test", "n_test", "test_data"),
    ):
        X, labels, targets = make_circles_with_targets(
            data_config[count_key],
            noise=data_config["noise"],
            factor=data_config["factor"],
            seed=None,
            rng=np.random.default_rng(seeds[seed_key]),
        )
        datasets[split] = (
            X.to(device=device, dtype=dtype),
            labels.to(device=device, dtype=dtype),
            targets.to(device=device, dtype=dtype),
        )
    return datasets


def save_datasets(path: str | Path, datasets) -> None:
    arrays = {}
    for split, (features, labels, targets) in datasets.items():
        arrays[f"{split}_X"] = features.detach().cpu().numpy()
        arrays[f"{split}_labels"] = labels.detach().cpu().numpy()
        arrays[f"{split}_targets"] = targets.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)


def construct_model(model_config, seed: int, dtype: torch.dtype, device: torch.device):
    # fork_rng makes initialization reproducible without leaving global RNG state changed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TimeDepODE(
            hidden_dim=model_config["hidden_dim"],
            input_dim=model_config["input_dim"],
            net_hidden=model_config["net_hidden"],
            parameter_box=tuple(model_config["parameter_box"]),
        )
    return model.to(device=device, dtype=dtype)


def _portable_state_dict(model) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def train_base_model(
    config: dict[str, Any],
    seeds: dict[str, int],
    datasets,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[TimeDepODE, list[dict[str, Any]], float]:
    model = construct_model(
        config["model"], seeds["model_initialization"], dtype, device
    )
    training = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    X_train, labels_train, targets_train = datasets["train"]
    history = []
    started = time.perf_counter()
    for epoch in range(training["epochs"]):
        optimizer.zero_grad(set_to_none=True)
        times, trajectory = integrate_fixed_grid(
            model,
            X_train,
            T=config["model"]["T"],
            dt=training["dt"],
            h=config["model"]["T"],
            method="rk4",
        )
        terms = paper_objective(
            trajectory,
            times,
            targets_train,
            model,
            alpha=training["alpha"],
            beta=training["beta"],
        )
        # A positive global scaling does not change the objective minimizer.
        (terms.total / X_train.shape[0]).backward()
        optimizer.step()
        model.project_parameters_()
        history.append(
            {
                "epoch": epoch + 1,
                "terminal": float(terms.terminal.detach().cpu()),
                "running": float(terms.running.detach().cpu()),
                "control": float(terms.control.detach().cpu()),
                "total": float(terms.total.detach().cpu()),
                "train_accuracy": nearest_target_accuracy(
                    trajectory[-1].detach(), labels_train
                ),
            }
        )
        report_every = max(1, training["epochs"] // 20)
        if (epoch + 1) % report_every == 0 or epoch + 1 == training["epochs"]:
            report_progress(
                "base-model",
                f"Training epoch {epoch + 1}/{training['epochs']} "
                f"(objective={history[-1]['total'] / X_train.shape[0]:.5g})",
                started=started,
            )
    elapsed = time.perf_counter() - started
    return model, history, elapsed


def evaluate_split_accuracies(model, datasets, T: float, dt: float) -> dict[str, float]:
    accuracies = {}
    with torch.no_grad():
        for split, (features, labels, _) in datasets.items():
            _, trajectory = integrate_fixed_grid(
                model, features, T, dt, T, method="rk4"
            )
            accuracies[split] = nearest_target_accuracy(trajectory[-1], labels)
    return accuracies


def save_checkpoint(
    path: str | Path,
    model: TimeDepODE,
    config: dict[str, Any],
    seeds: dict[str, int],
    history: list[dict[str, Any]],
    dtype_name: str,
) -> dict[str, Any]:
    payload = {
        "format_version": 1,
        "model_class": "TimeDepODE",
        "model_config": config["model"],
        "data_config": config["data"],
        "training_config": config["training"],
        "quick": config["quick"],
        "seeds": seeds,
        "dtype": dtype_name,
        "state_dict": _portable_state_dict(model),
        "training_history": history,
    }
    torch.save(payload, path)
    return payload


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    requested_dtype: torch.dtype | None = None,
) -> tuple[TimeDepODE, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_dtype = resolve_dtype(payload.get("dtype", "float64"))
    dtype = checkpoint_dtype if requested_dtype is None else requested_dtype
    model = construct_model(
        payload["model_config"],
        payload["seeds"]["model_initialization"],
        dtype,
        device,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def prepare_base_model(
    paths: ArtifactPaths,
    *,
    quick: bool,
    seed: int,
    dtype_name: str,
    device: torch.device,
    checkpoint: str | Path | None = None,
):
    dtype = resolve_dtype(dtype_name)
    if checkpoint is None:
        config = base_configuration(quick)
        seeds = make_seed_manifest(seed)
        datasets = generate_datasets(config["data"], seeds, dtype=dtype, device=device)
        model, history, training_seconds = train_base_model(
            config, seeds, datasets, dtype=dtype, device=device
        )
        payload = save_checkpoint(
            paths.checkpoints / "base_model.pt",
            model,
            config,
            seeds,
            history,
            dtype_name,
        )
        source = "trained"
    else:
        model, payload = load_checkpoint(
            checkpoint, device=device, requested_dtype=dtype
        )
        config = {
            "quick": payload["quick"],
            "model": payload["model_config"],
            "data": payload["data_config"],
            "training": payload["training_config"],
        }
        seeds = payload["seeds"]
        datasets = generate_datasets(config["data"], seeds, dtype=dtype, device=device)
        history = payload.get("training_history", [])
        training_seconds = 0.0
        save_checkpoint(
            paths.checkpoints / "base_model.pt",
            model,
            config,
            seeds,
            history,
            dtype_name,
        )
        source = str(Path(checkpoint).resolve())

    save_datasets(paths.data / "dataset_splits.npz", datasets)
    if history:
        write_csv(paths.data / "training_history.csv", history)
    accuracies = evaluate_split_accuracies(
        model, datasets, config["model"]["T"], config["training"]["dt"]
    )
    return {
        "model": model,
        "datasets": datasets,
        "base_config": config,
        "seeds": seeds,
        "checkpoint_payload": payload,
        "checkpoint_source": source,
        "training_seconds": training_seconds,
        "accuracies": accuracies,
        "dtype": dtype,
    }


def reference_trajectory(model, features: Tensor, T: float, dt: float):
    with torch.no_grad():
        return integrate_fixed_grid(model, features, T, dt, T, method="rk4")


def trajectory_at_times(
    reference_times: Tensor,
    reference_trajectory_values: Tensor,
    requested_times: Tensor,
    *,
    tolerance: float = 1e-8,
) -> Tensor:
    """Select aligned values from a finer reference grid."""
    dt = reference_times[1] - reference_times[0]
    indices = torch.round((requested_times - reference_times[0]) / dt).long()
    selected_times = reference_times[indices]
    if not torch.allclose(
        selected_times, requested_times, rtol=tolerance, atol=tolerance
    ):
        raise ValueError("requested times are not aligned with the reference grid")
    return reference_trajectory_values[indices]


def ordered_trajectory_statistic(squared_errors: np.ndarray) -> float:
    r"""Compute ``mean_m max_t mean_k error[k,t,m]`` in theorem order."""
    errors = np.asarray(squared_errors, dtype=float)
    if errors.ndim != 3:
        raise ValueError("squared_errors must have shape [schedule, time, data]")
    return float(errors.mean(axis=0).max(axis=0).mean())


def bootstrap_ordered_statistic(
    squared_errors: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> np.ndarray:
    errors = np.asarray(squared_errors, dtype=float)
    n_schedules = errors.shape[0]
    values = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        resample = rng.integers(0, n_schedules, size=n_schedules)
        values[index] = ordered_trajectory_statistic(errors[resample])
    return values


def fit_loglog_range(
    h_values: Sequence[float],
    errors: Sequence[float],
    reference_error: float,
    *,
    minimum_points: int = 3,
) -> dict[str, Any]:
    """Fit above the measured numerical plateau, using at least three points."""
    h = np.asarray(h_values, dtype=float)
    error = np.asarray(errors, dtype=float)
    order = np.argsort(h)
    h, error = h[order], error[order]
    floor = max(5.0 * reference_error, np.finfo(float).eps * 100)
    eligible = np.flatnonzero(error > floor)
    if len(eligible) < minimum_points:
        eligible = np.argsort(error)[-minimum_points:]
    eligible = np.sort(eligible)
    x = np.log(h[eligible])
    y = np.log(error[eligible])
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    residual = np.sum((y - prediction) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "h_min": float(h[eligible].min()),
        "h_max": float(h[eligible].max()),
        "n_points": int(len(eligible)),
        "indices_sorted": eligible.tolist(),
        "plateau_threshold": float(floor),
    }


def elapsed_record(started: float) -> float:
    return time.perf_counter() - started
