"""Shared model, reproducibility, and serialization helpers for paper experiments."""

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

import numpy as np
import torch
from torch import Tensor

from rnode.data import make_circles_with_targets
from rnode.integrators import integrate_fixed_grid
from rnode.models import TimeDepODE
from rnode.objectives import paper_objective


def report_progress(label, message, *, started=None):
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
    def create(cls, root):
        root = Path(root).expanduser().resolve()
        result = cls(
            root, root / "config", root / "data", root / "checkpoints", root / "figures"
        )
        for path in (result.root, result.config, result.data, result.checkpoints, result.figures):
            path.mkdir(parents=True, exist_ok=True)
        return result

    @classmethod
    def latest_run(cls, path):
        """Resolve a run directory, accepting the parent or the run itself.

        Figures are regenerated far more often than results are computed, so
        ``--plots-only`` should not require hunting for a timestamp.
        """
        path = Path(path).expanduser().resolve()
        if (path / "data").is_dir():
            return path
        runs = sorted(
            (child for child in path.glob("*-*") if (child / "data").is_dir()),
            key=lambda child: child.name,
        )
        if not runs:
            raise FileNotFoundError(
                f"no run directory with results under {path}; "
                "run the experiment first, or pass a specific run directory"
            )
        return runs[-1]

    @classmethod
    def new_run(cls, parent, prefix="run"):
        """Create a fresh timestamped run directory under ``parent``.

        Every writer uses fixed file names, so each execution must own a
        directory of its own; otherwise a rerun silently destroys the results
        it is being compared against.
        """
        parent = Path(parent).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = parent / f"{prefix}-{stamp}"
        suffix = 0
        while candidate.exists():
            suffix += 1
            candidate = parent / f"{prefix}-{stamp}-{suffix:02d}"
        return cls.create(candidate)


def base_configuration(quick: bool) -> dict:
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
            # Sample coordinates are stored at this precision before being cast
            # to the run dtype.  Historical checkpoints were produced from
            # float32-rounded data, so a checkpoint that omits this key is
            # regenerated as float32 to keep verification exact.
            "sample_dtype": "float64",
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


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(_json_safe(value)) if isinstance(value, (list, tuple, dict, np.ndarray)) else value
                for key, value in row.items()
            })


def write_manifest(root, *, configuration, seed, versions):
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


def version_information(requested_device, resolved_device, dtype_name):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "requested_device": requested_device,
        "resolved_device": str(resolved_device),
        "dtype": dtype_name,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": commit,
        "git_dirty": dirty,
    }


def generate_datasets(config, seeds, *, dtype, device):
    # A checkpoint written before this key existed came from float32 data.
    sample_dtype = resolve_dtype(config.get("sample_dtype", "float32"))
    datasets = {}
    for split, count_key, seed_key in (
        ("train", "n_train", "train_data"),
        ("calibration", "n_calibration", "calibration_data"),
        ("test", "n_test", "test_data"),
    ):
        X, labels, targets = make_circles_with_targets(
            config[count_key],
            noise=config["noise"],
            factor=config["factor"],
            rng=np.random.default_rng(seeds[seed_key]),
            dtype=sample_dtype,
        )
        datasets[split] = (
            X.to(device=device, dtype=dtype),
            labels.to(device=device, dtype=dtype),
            targets.to(device=device, dtype=dtype),
        )
    return datasets


def save_datasets(path, datasets):
    arrays = {}
    for split, (X, labels, targets) in datasets.items():
        arrays[f"{split}_X"] = X.detach().cpu().numpy()
        arrays[f"{split}_labels"] = labels.detach().cpu().numpy()
        arrays[f"{split}_targets"] = targets.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)


def construct_model(config, seed, dtype, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TimeDepODE(
            hidden_dim=config["hidden_dim"],
            input_dim=config["input_dim"],
            net_hidden=config["net_hidden"],
            parameter_box=tuple(config["parameter_box"]),
        )
    return model.to(device=device, dtype=dtype)


def train_base_model(config, seeds, datasets, *, dtype, device):
    model = construct_model(config["model"], seeds["model_initialization"], dtype, device)
    train = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train["learning_rate"])
    X, _, targets = datasets["train"]
    started = time.perf_counter()

    for epoch in range(train["epochs"]):
        optimizer.zero_grad(set_to_none=True)
        times, trajectory = integrate_fixed_grid(
            model, X, config["model"]["T"], train["dt"], config["model"]["T"], method="rk4"
        )
        objective = paper_objective(
            trajectory, times, targets, model, alpha=train["alpha"], beta=train["beta"]
        ).total / len(X)
        objective.backward()
        optimizer.step()
        model.project_parameters_()
        every = max(1, train["epochs"] // 20)
        if (epoch + 1) % every == 0 or epoch + 1 == train["epochs"]:
            report_progress(
                "base-model",
                f"Training epoch {epoch + 1}/{train['epochs']} (objective={float(objective.detach()):.5g})",
                started=started,
            )
    return model, time.perf_counter() - started


def _portable_state_dict(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def save_checkpoint(path, model, config, seeds, dtype_name):
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
    }
    torch.save(payload, path)
    return payload


def load_checkpoint(path, *, device, requested_dtype=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    dtype = requested_dtype or resolve_dtype(payload.get("dtype", "float64"))
    model = construct_model(
        payload["model_config"], payload["seeds"]["model_initialization"], dtype, device
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def prepare_base_model(paths, *, quick, seed, dtype_name, device, checkpoint=None):
    dtype = resolve_dtype(dtype_name)
    if checkpoint is None:
        config = base_configuration(quick)
        seeds = make_seed_manifest(seed)
        datasets = generate_datasets(config["data"], seeds, dtype=dtype, device=device)
        model, training_seconds = train_base_model(
            config, seeds, datasets, dtype=dtype, device=device
        )
        payload = save_checkpoint(
            paths.checkpoints / "base_model.pt", model, config, seeds, dtype_name
        )
        source = "trained"
    else:
        model, payload = load_checkpoint(checkpoint, device=device, requested_dtype=dtype)
        config = {
            "quick": payload.get("quick", False),
            "model": payload["model_config"],
            "data": payload["data_config"],
            "training": payload["training_config"],
        }
        seeds = payload["seeds"]
        datasets = generate_datasets(config["data"], seeds, dtype=dtype, device=device)
        training_seconds = 0.0
        save_checkpoint(paths.checkpoints / "base_model.pt", model, config, seeds, dtype_name)
        source = str(Path(checkpoint).resolve())

    save_datasets(paths.data / "dataset_splits.npz", datasets)
    return {
        "model": model,
        "datasets": datasets,
        "base_config": config,
        "seeds": seeds,
        "checkpoint_source": source,
        "training_seconds": training_seconds,
        "dtype": dtype,
    }


def reference_trajectory(model, features, T, dt):
    with torch.no_grad():
        return integrate_fixed_grid(model, features, T, dt, T, method="rk4")


def trajectory_at_times(reference_times, reference_values, requested_times, tolerance=1e-8):
    dt = reference_times[1] - reference_times[0]
    indices = torch.round((requested_times - reference_times[0]) / dt).long()
    selected = reference_times[indices]
    if not torch.allclose(selected, requested_times, rtol=tolerance, atol=tolerance):
        raise ValueError("requested times are not aligned with the reference grid")
    return reference_values[indices]
