"""
Training and figure-generation for Experiment 3a — Full training (§5.3a).

Both layers are trainable. Each rNODE realisation uses a FIXED batch
schedule throughout training (structured time-dependent pruning). Final
predictions are ensemble-averaged over K realisations.
"""

import os, time, json, random
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rnode.models import TimeDepODE_ELU, RBMTrainableODE
from rnode.data import make_circles_data, make_grid
from rnode.utils import (compute_accuracy, compute_loss,
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
    """Train a full model (all parameters). Returns (losses, elapsed)."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
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
                       rep=1, verbose=True):
    """Train an ensemble of K RBM-NODE realisations.

    Each realisation gets a different FIXED batch schedule, but uses the
    same architecture and hyperparameters.

    Returns dict with models, per-realisation losses, averaged trajectory,
    and timing information.
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
        model = RBMTrainableODE(p, pi, sched, t_train)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

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

    # Ensemble-average trajectory on eval grid
    results = []
    with torch.no_grad():
        for m in models:
            results.append(odeint(m, X, t_eval, method="rk4",
                                  options={"step_size": step_eval}))
    y_avg = torch.mean(torch.stack(results), dim=0)

    return {"models": models, "losses": all_losses, "y_avg": y_avg,
            "total_time": total_time, "avg_time": total_time / K}


# ── Evaluation helpers ─────────────────────────────────────────────────────

def _eval_ensemble(models, X, t_eval, step_eval):
    """Average forward pass of an ensemble."""
    with torch.no_grad():
        preds = [odeint(m, X, t_eval, method="rk4",
                        options={"step_size": step_eval})
                 for m in models]
    return torch.mean(torch.stack(preds), dim=0)


# ── Figures ────────────────────────────────────────────────────────────────

def plot_dynamics(traj, y_labels, title_tag, out, fname):
    """Trajectory plot (lines + final markers)."""
    _rc()
    fig, ax = plt.subplots(figsize=(8, 8))
    y_np = traj.numpy()
    labels = y_labels.squeeze().numpy()
    cmap = plt.get_cmap("bwr")
    for i in range(traj.shape[1]):
        ax.plot(y_np[:, i, 0], y_np[:, i, 1], "-",
                color=cmap(0.0 if labels[i] < 0 else 1.0), alpha=0.3, lw=0.5)
    ax.scatter(y_np[-1, :, 0], y_np[-1, :, 1], c=labels, cmap="bwr",
               edgecolor="k", s=50, marker="s", zorder=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, os.path.join(out, fname))


def plot_decision_boundaries(boundaries, accuracies, xx, yy,
                             X_test, y_test, out, fname):
    _rc()
    keys = list(boundaries.keys())
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    Xtn = X_test.numpy(); ytn = y_test.squeeze().numpy()
    for ax, key in zip(axes, keys):
        bnd = boundaries[key].reshape(xx.shape)
        ax.contourf(xx, yy, -bnd, levels=80, cmap="RdBu",
                    alpha=0.8, vmin=-1, vmax=1)
        ax.scatter(Xtn[:, 0], Xtn[:, 1], c=ytn, edgecolor="k",
                   s=30, cmap="jet")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    save(fig, os.path.join(out, fname))


def plot_benchmark_table(benchmark_results, batch_configs, out, fname):
    """Bar chart: accuracy by config."""
    _rc()
    labels = ["NODE"] + [f"{nb}B" for nb in batch_configs]
    tr = [benchmark_results["NODE"]["train_acc_mean"]] + \
         [benchmark_results[f"{nb}_batches"]["train_acc_mean"] for nb in batch_configs]
    te = [benchmark_results["NODE"]["test_acc_mean"]] + \
         [benchmark_results[f"{nb}_batches"]["test_acc_mean"] for nb in batch_configs]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w / 2, tr, w, color=COLORS[0], edgecolor="k")
    ax.bar(x + w / 2, te, w, color=COLORS[1], edgecolor="k")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, os.path.join(out, fname))

    # JSON summary
    rows = []
    for lbl, t_a, te_a in zip(labels, tr, te):
        rows.append({"config": lbl, "train_acc": t_a, "test_acc": te_a})
    path = os.path.join(out, fname + ".json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  -> {path}")
    return rows
