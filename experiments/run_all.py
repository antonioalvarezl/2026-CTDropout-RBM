#!/usr/bin/env python3
"""Safe orchestration for tests, individual experiments, and plot regeneration."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


EXPERIMENTS = {
    "exp1": ("experiments.exp1_trajectory_convergence", "trajectory"),
    "exp2": ("experiments.exp2_optimal_batch_design", "design"),
    "exp3": ("experiments.exp3_cost_accuracy", "cost_accuracy"),
    "exp4": ("experiments.exp4_measure_transport", "measure_transport"),
    "exp5": ("experiments.exp5_training_consistency", "training_consistency"),
}


def _paths(output_root):
    root = Path(output_root).expanduser().resolve()
    return {name: root / directory for name, (_, directory) in EXPERIMENTS.items()}


def _command(experiment, paths, args, *, plots_only=False):
    module, _ = EXPERIMENTS[experiment]
    command = [
        sys.executable,
        "-m",
        module,
        "--output-dir",
        str(paths[experiment]),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
    ]
    if plots_only:
        return command + ["--plots-only"]
    if not args.full:
        command.append("--quick")
    checkpoint = args.checkpoint or str(paths["exp1"] / "checkpoints" / "base_model.pt")
    design = args.design_results or str(paths["exp2"] / "data" / "milp_results.json")
    if experiment in {"exp2", "exp3", "exp5"}:
        command.extend(["--checkpoint", checkpoint])
    if experiment == "exp3":
        command.extend(["--design-results", design])
    if experiment == "exp5" and getattr(args, "resume", False):
        command.append("--resume")
    return command


def _selected(name):
    return list(EXPERIMENTS) if name == "all" else [name]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("test", help="run the smoke test suite only")

    run = subparsers.add_parser("run", help="run quick mode unless full is explicit")
    run.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    run.add_argument("--full", action="store_true")
    run.add_argument(
        "--confirm-full",
        action="store_true",
        help="second guard required together with --full",
    )
    run.add_argument("--output-root", default=None)
    run.add_argument("--checkpoint", default=None)
    run.add_argument("--design-results", default=None)
    run.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete exp5 stages and conditions from the output directory",
    )
    run.add_argument("--seed", type=int, default=2026)
    run.add_argument("--device", default="cpu")
    run.add_argument("--dtype", choices=("float32", "float64"), default="float64")

    plots = subparsers.add_parser("plots", help="regenerate plots from saved data")
    plots.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    plots.add_argument("--output-root", default="outputs/current_paper_quick")
    plots.add_argument("--seed", type=int, default=2026)
    plots.add_argument("--device", default="cpu")
    plots.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    plots.set_defaults(full=False, checkpoint=None, design_results=None, resume=False)
    return parser


def main():
    args = build_parser().parse_args()
    if args.action == "test":
        subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
        return
    if args.action == "run" and args.full and not args.confirm_full:
        raise SystemExit("Full mode requires both --full and --confirm-full")
    output_root = args.output_root or (
        "outputs/current_paper_full" if args.full else "outputs/current_paper_quick"
    )
    paths = _paths(output_root)
    selected = _selected(args.experiment)
    overall_started = time.perf_counter()
    mode = "plots" if args.action == "plots" else ("full" if args.full else "quick")
    print(
        f"[run_all] Starting {len(selected)} experiment(s) in {mode} mode; "
        f"output root: {Path(output_root).expanduser().resolve()}",
        flush=True,
    )
    for index, experiment in enumerate(selected, start=1):
        command = _command(
            experiment,
            paths,
            args,
            plots_only=args.action == "plots",
        )
        experiment_started = time.perf_counter()
        print(
            f"[run_all] [{index}/{len(selected)}] Starting {experiment}",
            flush=True,
        )
        subprocess.run(command, check=True)
        elapsed = time.perf_counter() - experiment_started
        print(
            f"[run_all] [{index}/{len(selected)}] Finished {experiment} "
            f"in {elapsed / 60:.1f} min",
            flush=True,
        )
    total = time.perf_counter() - overall_started
    print(f"[run_all] All requested work finished in {total / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
