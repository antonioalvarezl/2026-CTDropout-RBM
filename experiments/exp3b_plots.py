"""
Training and figure-generation for Experiment 3b — Fixed inner weights (§5.3b).

Only the outer layer W_2(t) is trainable. Inner weights (W_1, b_1) are
randomly initialised and frozen. Includes a scaling analysis (accuracy vs
hidden dimension p).
"""

import os, time, json, random
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rnode.models import FixedInnerODE, FixedInnerRBM
from rnode.data import make_circles_data, make_grid
from rnode.utils import (compute_accuracy, compute_loss,
                         count_trainable_params, count_total_params,
                         split_neurons, generate_batch_sequence)

# ── Style ──────────────────────────────────────────────────────────────────

DPI = 300
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

SEED = 42


def _rc():
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "font.family": "serif",
        "mathtext.fontset": "cm",
    })


def save(fig, path):
    fig.savefig(f"{path}.pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"{path}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}.{{pdf,png}}")


# ── Training ───────────────────────────────────────────────────────────────

def train_model(model, X, y, t_span, cfg, verbose=True):
    """Train (only trainable params). Returns (losses, elapsed)."""
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
    dt = float(t_span[1] - t_span[0])
    losses = []
    t0 = time.time()
    for epoch in range(cfg["n_epochs"]):
        opt.zero_grad()
        y_pred = odeint(model, X, t_span, method="rk4")
        loss = compute_loss(y_pred, y, dt, model, cfg["alpha"], cfg["beta"])
        loss.backward(); opt.step(); losses.append(loss.item())
        if verbose and (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{cfg['n_epochs']}  loss={loss.item():.4f}")
    return losses, time.time() - t0


def train_rbm_ensemble(X, y, t_train, t_eval, cfg,
                       rep=1, inner_seed=42, verbose=True):
    """Train K FixedInnerRBM realisations with shared inner weights.

    Returns dict with models, losses, averaged trajectory, timing.
    """
    p = cfg["hidden_dim"]
    n_batches = cfg["n_batches"]
    K = cfg["n_realizations"]
    pi = 1.0 / n_batches
    predefined = split_neurons(p, n_batches, seed=1)
    dt = float(t_train[1] - t_train[0])
    step_eval = float(t_eval[-1] - t_eval[0]) / (len(t_eval) - 1)

    models, all_losses = [], []
    total_time = 0.0

    for r in range(K):
        if verbose:
            print(f"  Realisation {r+1}/{K}...", end=" ")

        random.seed(SEED + r)
        sched = generate_batch_sequence(predefined, len(t_train), rep)

        torch.manual_seed(SEED + r)
        model = FixedInnerRBM(p, pi, sched, t_train, seed=inner_seed)
        opt = torch.optim.Adam(
            filter(lambda pp: pp.requires_grad, model.parameters()),
            lr=cfg["lr"])

        t0 = time.time()
        losses = []
        for epoch in range(cfg["n_epochs"]):
            opt.zero_grad()
            y_pred = odeint(model, X, t_train, method="rk4")
            loss = compute_loss(y_pred, y, dt, model, cfg["alpha"], cfg["beta"])
            loss.backward(); opt.step(); losses.append(loss.item())
        elapsed = time.time() - t0
        total_time += elapsed

        if verbose:
            print(f"loss={losses[-1]:.4f}  ({elapsed:.1f}s)")

        models.append(model); all_losses.append(losses)

    results = []
    with torch.no_grad():
        for m in models:
            results.append(odeint(m, X, t_eval, method="rk4",
                                  options={"step_size": step_eval}))
    y_avg = torch.mean(torch.stack(results), dim=0)

    return {"models": models, "losses": all_losses, "y_avg": y_avg,
            "total_time": total_time, "avg_time": total_time / K}


# ── Scaling analysis ───────────────────────────────────────────────────────

def run_scaling_analysis(X_train, y_train, X_test, y_test,
                         t_train, t_eval, cfg, out):
    """Accuracy vs hidden dimension p (§5.3b)."""
    print("\n[Scaling] Accuracy vs hidden dimension...")
    _rc()
    step_eval = float(t_eval[-1] - t_eval[0]) / (len(t_eval) - 1)
    dt = float(t_train[1] - t_train[0])
    hidden_dims = cfg["hidden_dims_scaling"]
    n_trials = cfg["n_trials_scaling"]

    results = {}
    for p in hidden_dims:
        print(f"\n  p = {p}...")
        trials = []
        for trial in range(n_trials):
            torch.manual_seed(SEED + trial)
            model = FixedInnerODE(hidden_dim=p, seed=42 + trial)
            opt = torch.optim.Adam(
                filter(lambda pp: pp.requires_grad, model.parameters()),
                lr=cfg["lr"])
            t0 = time.time()
            for epoch in range(cfg["n_epochs"]):
                opt.zero_grad()
                yp = odeint(model, X_train, t_train, method="rk4")
                loss = compute_loss(yp, y_train, dt, model,
                                    cfg["alpha"], cfg["beta"])
                loss.backward(); opt.step()
            elapsed = time.time() - t0

            with torch.no_grad():
                y_tr = odeint(model, X_train, t_eval, method="rk4",
                              options={"step_size": step_eval})
                y_te = odeint(model, X_test, t_eval, method="rk4",
                              options={"step_size": step_eval})
            trials.append({
                "time": elapsed, "loss": loss.item(),
                "train_acc": compute_accuracy(y_tr[-1], y_train),
                "test_acc": compute_accuracy(y_te[-1], y_test),
                "trainable_params": count_trainable_params(model),
            })

        results[p] = {
            "time_mean": np.mean([t["time"] for t in trials]),
            "time_std": np.std([t["time"] for t in trials]),
            "train_acc_mean": np.mean([t["train_acc"] for t in trials]),
            "train_acc_std": np.std([t["train_acc"] for t in trials]),
            "test_acc_mean": np.mean([t["test_acc"] for t in trials]),
            "test_acc_std": np.std([t["test_acc"] for t in trials]),
            "trainable_params": trials[0]["trainable_params"],
        }
        print(f"    train={results[p]['train_acc_mean']:.2%}  "
              f"test={results[p]['test_acc_mean']:.2%}  "
              f"time={results[p]['time_mean']:.1f}s  "
              f"params={results[p]['trainable_params']}")

    # Plot
    ps = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.errorbar(ps, [results[p]["train_acc_mean"] for p in ps],
                yerr=[results[p]["train_acc_std"] for p in ps],
                marker="o", capsize=3, color=COLORS[0])
    ax.errorbar(ps, [results[p]["test_acc_mean"] for p in ps],
                yerr=[results[p]["test_acc_std"] for p in ps],
                marker="s", capsize=3, color=COLORS[1])
    ax.set_xlabel("Hidden dimension $p$"); ax.set_ylabel("Accuracy")
    ax.set_xscale("log", base=2); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(ps, [results[p]["time_mean"] for p in ps],
                yerr=[results[p]["time_std"] for p in ps],
                marker="o", capsize=3, color=COLORS[2])
    ax.set_xlabel("Hidden dimension $p$"); ax.set_ylabel("Training time (s)")
    ax.set_xscale("log", base=2); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(ps, [results[p]["trainable_params"] for p in ps],
            "o-", color=COLORS[3])
    ax.set_xlabel("Hidden dimension $p$"); ax.set_ylabel("Trainable parameters")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, os.path.join(out, "ex3b_scaling"))

    path = os.path.join(out, "ex3b_scaling.json")
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"  -> {path}")
    return results


# ── Decision boundaries ────────────────────────────────────────────────────

def compute_decision_boundaries(func_ref, X_train, y_train, X_test, y_test,
                                t_train, t_eval, cfg, out):
    """Compute and plot boundaries for NODE + rNODE at h = 1,2,3 Δt."""
    print("\n[Decision boundaries]...")
    _rc()
    step_eval = float(t_eval[-1] - t_eval[0]) / (len(t_eval) - 1)
    n_grid = cfg.get("n_grid", 150)
    xx, yy, grid = make_grid(n_points=n_grid)

    boundaries, accuracies = {}, {}

    # Reference NODE
    print("  Computing NODE boundary...")
    with torch.no_grad():
        y_grid = odeint(func_ref, grid, t_eval, method="rk4",
                        options={"step_size": step_eval})
    boundaries["NODE"] = y_grid[-1, :, 0].numpy()
    with torch.no_grad():
        y_te = odeint(func_ref, X_test, t_eval, method="rk4",
                      options={"step_size": step_eval})
    accuracies["NODE"] = compute_accuracy(y_te[-1], y_test)

    # rNODE at different h
    h_values = cfg.get("h_reps", [1, 2, 3])
    K_bnd = cfg.get("n_real_boundary", 20)
    cfg_ens = {**cfg, "n_realizations": K_bnd}
    for rep in h_values:
        print(f"  Computing rNODE boundary (h={rep}*dt)...")
        rbm_h = train_rbm_ensemble(
            X_train, y_train, t_train, t_eval, cfg_ens,
            rep=rep, inner_seed=42, verbose=False)

        grid_preds = []
        test_preds = []
        with torch.no_grad():
            for m in rbm_h["models"]:
                y_g = odeint(m, grid, t_eval, method="rk4",
                             options={"step_size": step_eval})
                grid_preds.append(y_g[-1, :, 0])
                y_te = odeint(m, X_test, t_eval, method="rk4",
                              options={"step_size": step_eval})
                test_preds.append(y_te[-1])

        boundaries[f"h={rep}"] = torch.mean(
            torch.stack(grid_preds), dim=0).numpy()
        y_avg = torch.mean(torch.stack(test_preds), dim=0)
        accuracies[f"h={rep}"] = compute_accuracy(y_avg, y_test)
        print(f"    acc = {accuracies[f'h={rep}']:.2%}")

    # Plot
    keys = ["NODE"] + [f"h={r}" for r in h_values]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5))
    Xtn = X_test.numpy(); ytn = y_test.squeeze().numpy()
    for ax, key in zip(axes, keys):
        bnd = boundaries[key].reshape(xx.shape)
        ax.contourf(xx, yy, -bnd, levels=80, cmap="RdBu",
                    alpha=0.8, vmin=-1, vmax=1)
        ax.scatter(Xtn[:, 0], Xtn[:, 1], c=ytn, edgecolor="k",
                   s=30, cmap="jet")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    save(fig, os.path.join(out, "ex3b_decision_boundaries"))
    return boundaries, accuracies


# ── Benchmark ──────────────────────────────────────────────────────────────

def run_benchmarks(X_train, y_train, X_test, y_test,
                   t_train, t_eval, cfg, out):
    """Benchmark NODE vs rNODE with different batch counts."""
    print("\n[Benchmarks]...")
    step_eval = float(t_eval[-1] - t_eval[0]) / (len(t_eval) - 1)
    dt = float(t_train[1] - t_train[0])
    batch_configs = cfg.get("batch_configs", [2, 3, 4, 6])
    n_trials = cfg.get("n_benchmark_trials", 5)
    K = cfg.get("n_realizations_bench", 10)
    p = cfg["hidden_dim"]

    results = {}

    # Full NODE
    print("  Benchmarking NODE (fixed inner)...")
    trials = []
    for trial in range(n_trials):
        torch.manual_seed(SEED + trial)
        model = FixedInnerODE(hidden_dim=p, seed=42 + trial)
        opt = torch.optim.Adam(
            filter(lambda pp: pp.requires_grad, model.parameters()),
            lr=cfg["lr"])
        t0 = time.time()
        for epoch in range(cfg["n_epochs"]):
            opt.zero_grad()
            yp = odeint(model, X_train, t_train, method="rk4")
            loss = compute_loss(yp, y_train, dt, model,
                                cfg["alpha"], cfg["beta"])
            loss.backward(); opt.step()
        elapsed = time.time() - t0
        with torch.no_grad():
            y_tr = odeint(model, X_train, t_eval, method="rk4",
                          options={"step_size": step_eval})
            y_te = odeint(model, X_test, t_eval, method="rk4",
                          options={"step_size": step_eval})
        trials.append({"time": elapsed, "loss": loss.item(),
                        "train_acc": compute_accuracy(y_tr[-1], y_train),
                        "test_acc": compute_accuracy(y_te[-1], y_test)})
    results["NODE"] = _agg(trials)
    print(f"    test={results['NODE']['test_acc_mean']:.2%}")

    # RBM per batch count
    for nb in batch_configs:
        print(f"  Benchmarking RBM ({nb} batches)...")
        pi = 1.0 / nb
        predefined = split_neurons(p, nb, seed=1)
        trials = []
        for trial in range(n_trials):
            models_ens, ens_times = [], []
            for r in range(K):
                random.seed(SEED + trial * 100 + r)
                torch.manual_seed(SEED + trial * 100 + r)
                sched = generate_batch_sequence(predefined, len(t_train))
                model = FixedInnerRBM(p, pi, sched, t_train, seed=42)
                opt = torch.optim.Adam(
                    filter(lambda pp: pp.requires_grad, model.parameters()),
                    lr=cfg["lr"])
                t0 = time.time()
                for epoch in range(cfg["n_epochs"]):
                    opt.zero_grad()
                    yp = odeint(model, X_train, t_train, method="rk4")
                    loss = compute_loss(yp, y_train, dt, model,
                                        cfg["alpha"], cfg["beta"])
                    loss.backward(); opt.step()
                ens_times.append(time.time() - t0)
                models_ens.append(model)
            with torch.no_grad():
                tr_p, te_p = [], []
                for m in models_ens:
                    y_tr = odeint(m, X_train, t_eval, method="rk4",
                                  options={"step_size": step_eval})
                    y_te = odeint(m, X_test, t_eval, method="rk4",
                                  options={"step_size": step_eval})
                    tr_p.append(y_tr[-1]); te_p.append(y_te[-1])
            y_tr_avg = torch.mean(torch.stack(tr_p), dim=0)
            y_te_avg = torch.mean(torch.stack(te_p), dim=0)
            trials.append({
                "time": np.mean(ens_times), "loss": loss.item(),
                "train_acc": compute_accuracy(y_tr_avg, y_train),
                "test_acc": compute_accuracy(y_te_avg, y_test),
            })
        results[f"{nb}_batches"] = _agg(trials)
        print(f"    test={results[f'{nb}_batches']['test_acc_mean']:.2%}")

    # Plot
    _rc()
    labels = ["NODE"] + [f"{nb}B" for nb in batch_configs]
    tr = [results["NODE"]["train_acc_mean"]] + \
         [results[f"{nb}_batches"]["train_acc_mean"] for nb in batch_configs]
    te = [results["NODE"]["test_acc_mean"]] + \
         [results[f"{nb}_batches"]["test_acc_mean"] for nb in batch_configs]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w / 2, tr, w, color=COLORS[0], edgecolor="k")
    ax.bar(x + w / 2, te, w, color=COLORS[1], edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
    save(fig, os.path.join(out, "ex3b_benchmark"))

    path = os.path.join(out, "ex3b_benchmark.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  -> {path}")
    return results


def _agg(trials):
    return {
        "time_mean": np.mean([t["time"] for t in trials]),
        "time_std": np.std([t["time"] for t in trials]),
        "loss_mean": np.mean([t["loss"] for t in trials]),
        "loss_std": np.std([t["loss"] for t in trials]),
        "train_acc_mean": np.mean([t["train_acc"] for t in trials]),
        "train_acc_std": np.std([t["train_acc"] for t in trials]),
        "test_acc_mean": np.mean([t["test_acc"] for t in trials]),
        "test_acc_std": np.std([t["test_acc"] for t in trials]),
    }
