"""
Plotting and analysis routines for Experiment 2 (flow matching, §5.2).

Each public function generates one figure. All functions receive the trained
flow model and configuration as arguments so that the companion notebook
stays short and readable.
"""

import os, time, json, random as _random
import numpy as np
import scipy.stats as sp_stats
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rnode.flow import Flow, FlowRBM, compute_kde
from rnode.data import sample_initial_density, sample_target_density, sample_mesh_particles
from rnode.batches import (
    make_balanced_disjoint, make_drop_one, make_bernoulli, make_interleaved,
    sample_batch_sequence,
)

# ── Style ──────────────────────────────────────────────────────────────────

DPI = 300
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
CMAP_DENSITY = "Oranges"


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


# ── Helpers ────────────────────────────────────────────────────────────────

def integrate_manual(flow_or_rbm, particles, weights, n_steps, dt, T):
    """Step-by-step integration returning (final_x, final_w, snapshots)."""
    x = particles.clone()
    log_det = torch.zeros(len(particles))
    snaps = {}
    with torch.no_grad():
        for step in range(n_steps):
            ts = torch.tensor(step * dt)
            te = torch.tensor((step + 1) * dt)
            x, log_det = flow_or_rbm.midpoint_step(x, log_det, ts, te)
            cur_t = (step + 1) * dt
            if abs(cur_t - 0.5) < dt / 2:
                snaps[0.5] = (x.clone(), weights * torch.exp(log_det.clone()))
    return x, weights * torch.exp(log_det), snaps


def build_rbm(flow, scheme, n_steps, dt, p, T):
    t_span = torch.linspace(0, T, n_steps + 1)
    sched = sample_batch_sequence(scheme, n_steps + 1, p=p)
    return FlowRBM(flow, t_span, sched, scheme.pi_min)


def build_rbm_rep(flow, scheme, n_steps, rep, p, T):
    t_span = torch.linspace(0, T, n_steps + 1)
    sched = sample_batch_sequence(scheme, n_steps + 1, p=p, rep=rep)
    return FlowRBM(flow, t_span, sched, scheme.pi_min)


def l1_hist_error(p1, w1, p2, w2, x_edges, y_edges, cell_area):
    def _h(p, w):
        pn = p.detach().cpu().numpy()
        wn = w.detach().cpu().numpy()
        h, _, _ = np.histogram2d(pn[:, 0], pn[:, 1],
                                 bins=[x_edges, y_edges], weights=wn)
        return h
    return float(np.abs(_h(p1, w1) - _h(p2, w2)).sum() * cell_area)


def compute_trajectory(seed_pt, flow_model, dt, T):
    x = seed_pt.clone()
    log_det = torch.zeros(1)
    traj = [x.detach().cpu().numpy().flatten()]
    t = 0.0
    while t < T - dt / 2:
        ts = torch.tensor(t)
        te = torch.tensor(min(t + dt, T))
        x, log_det = flow_model.midpoint_step(x, log_det, ts, te)
        traj.append(x.detach().cpu().numpy().flatten())
        t += dt
    return np.array(traj)


def timeit(fn, n):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return np.mean(ts), np.std(ts)


def four_schemes(p):
    return {
        "Balanced-disjoint ($r$=32)": make_balanced_disjoint(p, 32, seed=0),
        "Balanced-disjoint ($r$=16)": make_balanced_disjoint(p, 16, seed=0),
        "Drop-one":                   make_drop_one(p),
        "Bernoulli ($q_B$=1/3)":      make_bernoulli(p, q_B=1 / 3),
    }


# ── Figure 1 — Distributions ──────────────────────────────────────────────

def fig1_data(cfg, out):
    print("\n[Fig 1] Distributions...")
    _rc()
    X0 = sample_initial_density(400, seed=0)
    X1 = sample_target_density(200, seed=0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(X0[:, 0], X0[:, 1], alpha=0.7, c=COLORS[0], edgecolor="k", s=25)
    ax.scatter(X1[:, 0], X1[:, 1], alpha=0.7, c=COLORS[1], edgecolor="k", s=25)
    ax.set_xlim(*cfg["extent"][:2]); ax.set_ylim(*cfg["extent"][2:])
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, os.path.join(out, "fig_flow1_data"))


# ── Figure 2 — Full-model density + trajectories ──────────────────────────

def fig2_density(flow, cfg, out):
    print("\n[Fig 2] Full-model density + trajectories...")
    _rc()
    xg = np.linspace(*cfg["extent"][:2], cfg["grid_res"])
    yg = np.linspace(*cfg["extent"][2:], cfg["grid_res"])
    particles, weights = sample_mesh_particles(cfg["n_mesh"])

    x_f, w_f, snaps = integrate_manual(flow, particles, weights,
                                        cfg["n_steps"], cfg["dt"], cfg["T"])
    d_ini = compute_kde(particles, weights, xg, yg)
    d_mid = compute_kde(snaps[0.5][0], snaps[0.5][1], xg, yg)
    d_fin = compute_kde(x_f, w_f, xg, yg)

    thr = 1e-4
    lvl_i = np.linspace(thr, np.max(d_ini), 10)
    lvl_f = np.linspace(thr, np.max(d_fin), 10)
    lvl_m = np.linspace(np.max(d_mid) * 0.1, np.max(d_mid) * 0.8, 5)

    # Panel 1: density snapshots
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.contourf(xg, yg, d_ini, levels=lvl_i, cmap=CMAP_DENSITY,
                extend="max", zorder=2)
    ax.contour(xg, yg, d_ini, levels=lvl_i, colors="k", linewidths=0.8)
    ax.contourf(xg, yg, d_fin, levels=lvl_f, cmap=CMAP_DENSITY, extend="max")
    ax.contour(xg, yg, d_fin, levels=lvl_f, colors="k", linewidths=0.8)
    ax.contour(xg, yg, d_mid, levels=lvl_m, colors="k", linewidths=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    save(fig, os.path.join(out, "fig_flow2_density"))

    # Panel 2: with trajectories
    angles = np.linspace(0, 2 * np.pi, cfg["n_seeds"], endpoint=False)
    seeds = torch.tensor([[-1 + 0.9 * np.cos(a), -1 + 0.9 * np.sin(a)]
                          for a in angles], dtype=torch.float32)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.contourf(xg, yg, d_ini, levels=lvl_i, cmap=CMAP_DENSITY,
                extend="max", zorder=2)
    ax.contour(xg, yg, d_ini, levels=lvl_i, colors="k", linewidths=0.8)
    ax.contourf(xg, yg, d_fin, levels=lvl_f, cmap=CMAP_DENSITY, extend="max")
    ax.contour(xg, yg, d_fin, levels=lvl_f, colors="k", linewidths=0.8)
    with torch.no_grad():
        for s in seeds:
            traj = compute_trajectory(s.unsqueeze(0), flow, cfg["dt"], cfg["T"])
            ax.plot(traj[:, 0], traj[:, 1], "k-", alpha=0.4, lw=0.5, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    save(fig, os.path.join(out, "fig_flow2_trajectories"))
    return d_ini, d_fin, lvl_i, lvl_f


# ── Figure 3 — Full vs rNODE ──────────────────────────────────────────────

def fig3_comparison(flow, d_ini, d_fin, lvl_i, lvl_f, cfg, out):
    print("\n[Fig 3] Full vs rNODE density comparison...")
    _rc()
    p = cfg["hidden"]; K = cfg["n_real_viz"]
    scheme = make_interleaved(p, cfg["n_batches_default"])
    particles, weights = sample_mesh_particles(cfg["n_mesh"])

    X_all, w_all = [], []
    for _ in range(K):
        rbm = build_rbm(flow, scheme, cfg["n_steps"], cfg["dt"], p, cfg["T"])
        x, w, _ = integrate_manual(rbm, particles, weights,
                                    cfg["n_steps"], cfg["dt"], cfg["T"])
        X_all.append(x); w_all.append(w)
    X_avg = torch.stack(X_all).mean(0)
    w_avg = torch.stack(w_all).mean(0)

    xg = np.linspace(*cfg["extent"][:2], cfg["grid_res"])
    yg = np.linspace(*cfg["extent"][2:], cfg["grid_res"])
    d_rbm = compute_kde(X_avg, w_avg, xg, yg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, density in [(axes[0], d_fin), (axes[1], d_rbm)]:
        ax.contourf(xg, yg, d_ini, levels=lvl_i, cmap=CMAP_DENSITY,
                    extend="max", zorder=2)
        ax.contour(xg, yg, d_ini, levels=lvl_i, colors="k", linewidths=0.8)
        dm = np.where(density > 1e-4, density, np.nan)
        ax.contourf(xg, yg, dm, levels=lvl_f, cmap=CMAP_DENSITY, extend="max")
        ax.contour(xg, yg, dm, levels=lvl_f, colors="k", linewidths=0.8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    save(fig, os.path.join(out, "fig_flow3_comparison"))


# ── Figure 4 — L1 convergence ─────────────────────────────────────────────

def fig4_convergence(flow, cfg, out):
    print("\n[Fig 4] L1 convergence...")
    _rc()
    p = cfg["hidden"]; dt_c = cfg["dt_conv"]
    n_steps_c = cfg["n_steps_conv"]; K = cfg["n_real_conv"]
    scheme = make_interleaved(p, cfg["n_batches_default"])

    hr = cfg["hist_res"]
    x_e = np.linspace(*cfg["extent"][:2], hr + 1)
    y_e = np.linspace(*cfg["extent"][2:], hr + 1)
    ca = (x_e[1] - x_e[0]) * (y_e[1] - y_e[0])

    print("  Computing reference (full model)...")
    particles, weights = sample_mesh_particles(cfg["n_mesh"])
    x_ref, w_ref, _ = integrate_manual(flow, particles, weights,
                                        n_steps_c, dt_c, cfg["T"])

    rep_values = np.unique(np.logspace(0, 2.2, 10).astype(int))
    h_values = rep_values * dt_c

    means, stds = [], []
    for rep in rep_values:
        errs = []
        for _ in range(K):
            rbm = build_rbm_rep(flow, scheme, n_steps_c, int(rep), p, cfg["T"])
            x_r, w_r, _ = integrate_manual(rbm, particles, weights,
                                            n_steps_c, dt_c, cfg["T"])
            errs.append(l1_hist_error(x_r, w_r, x_ref, w_ref, x_e, y_e, ca))
        means.append(np.mean(errs)); stds.append(np.std(errs))
        print(f"  rep={rep:>3d}  h={rep*dt_c:.4f}  err={means[-1]:.4e}")

    means = np.array(means); stds = np.array(stds)
    n_fit = len(h_values) - 2
    sl, ic, rv, *_ = sp_stats.linregress(
        np.log(h_values[:n_fit]), np.log(means[:n_fit]))
    print(f"  Fitted slope: {sl:.3f}  (theory: 0.5)")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ref_c = means[0] / (h_values[0] ** 0.5) * 1.5
    ax.loglog(h_values, ref_c * h_values ** 0.5, "+-", color=COLORS[3], lw=1.2)
    ci = 1.96 * stds / np.sqrt(K)
    ax.errorbar(h_values, means, yerr=ci, fmt="ko", capsize=3, ms=5)
    ax.loglog(h_values, np.exp(ic) * h_values ** sl, "--",
              color=COLORS[0], lw=1.5)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\mathbb{E}[\|\rho_T-\hat{\rho}_T\|_{L^1}]$")
    ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
    save(fig, os.path.join(out, "fig_flow4_convergence"))
    return {"slope": sl, "h": h_values.tolist(), "errors": means.tolist()}


# ── Figure 5 — Scheme convergence comparison ──────────────────────────────

def fig5_scheme_convergence(flow, cfg, out):
    print("\n[Fig 5] Scheme convergence comparison...")
    _rc()
    p = cfg["hidden"]; dt_c = cfg["dt_conv"]
    n_steps_c = cfg["n_steps_conv"]; K = cfg["n_real_scheme"]

    hr = cfg["hist_res"]
    x_e = np.linspace(*cfg["extent"][:2], hr + 1)
    y_e = np.linspace(*cfg["extent"][2:], hr + 1)
    ca = (x_e[1] - x_e[0]) * (y_e[1] - y_e[0])

    particles, weights = sample_mesh_particles(cfg["n_mesh"])
    x_ref, w_ref, _ = integrate_manual(flow, particles, weights,
                                        n_steps_c, dt_c, cfg["T"])

    rep_values = np.unique(np.logspace(0, 2.0, 8).astype(int))
    h_values = rep_values * dt_c
    schemes = four_schemes(p)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    all_data = {}
    for i, (name, sch) in enumerate(schemes.items()):
        ms = []
        for rep in rep_values:
            errs = []
            for _ in range(K):
                rbm = build_rbm_rep(flow, sch, n_steps_c, int(rep), p, cfg["T"])
                x_r, w_r, _ = integrate_manual(rbm, particles, weights,
                                                n_steps_c, dt_c, cfg["T"])
                errs.append(l1_hist_error(x_r, w_r, x_ref, w_ref, x_e, y_e, ca))
            ms.append(np.mean(errs))
        ms = np.array(ms)
        sl, *_ = sp_stats.linregress(np.log(h_values), np.log(ms))
        ax.loglog(h_values, ms, MARKERS[i] + "-", color=COLORS[i],
                  ms=6, lw=1.2)
        all_data[name] = {"slope": sl}
        print(f"  [{name}] slope = {sl:.3f}")

    ref = 0.15 / h_values[0] ** 0.5
    ax.loglog(h_values, ref * h_values ** 0.5, "k--", lw=1, alpha=0.5)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\mathbb{E}[\|\rho_T-\hat{\rho}_T\|_{L^1}]$")
    ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
    save(fig, os.path.join(out, "fig_flow5_scheme_conv"))
    return all_data


# ── Figure 6 — Variance by scheme ─────────────────────────────────────────

def fig6_variance(flow, cfg, out):
    print("\n[Fig 6] Variance by scheme...")
    _rc()
    p = cfg["hidden"]; dt_c = cfg["dt_conv"]
    n_steps_c = cfg["n_steps_conv"]; K = cfg["n_real_scheme"]

    hr = cfg["hist_res"]
    x_e = np.linspace(*cfg["extent"][:2], hr + 1)
    y_e = np.linspace(*cfg["extent"][2:], hr + 1)
    ca = (x_e[1] - x_e[0]) * (y_e[1] - y_e[0])

    particles, weights = sample_mesh_particles(cfg["n_mesh"])
    x_ref, w_ref, _ = integrate_manual(flow, particles, weights,
                                        n_steps_c, dt_c, cfg["T"])

    rep_values = np.unique(np.logspace(0, 2.0, 8).astype(int))
    h_values = rep_values * dt_c
    schemes = four_schemes(p)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (name, sch) in enumerate(schemes.items()):
        vs = []
        for rep in rep_values:
            errs = []
            for _ in range(K):
                rbm = build_rbm_rep(flow, sch, n_steps_c, int(rep), p, cfg["T"])
                x_r, w_r, _ = integrate_manual(rbm, particles, weights,
                                                n_steps_c, dt_c, cfg["T"])
                errs.append(l1_hist_error(x_r, w_r, x_ref, w_ref, x_e, y_e, ca))
            vs.append(np.var(errs))
        ax.loglog(h_values, vs, MARKERS[i] + "-", color=COLORS[i],
                  ms=6, lw=1.2)
        print(f"  [{name}] var range: [{min(vs):.2e}, {max(vs):.2e}]")

    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\mathrm{Var}_\omega[\|\rho_T-\hat{\rho}_T\|_{L^1}]$")
    ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
    save(fig, os.path.join(out, "fig_flow6_variance"))


# ── Figure 7 — Benchmark bar chart ────────────────────────────────────────

def fig7_benchmark(flow, cfg, out):
    print("\n[Fig 7] Benchmark time vs batch count...")
    _rc()
    p = cfg["hidden"]; nt = cfg["n_bench_trials"]
    n_steps = cfg["n_steps"]
    particles, weights = sample_mesh_particles(cfg["n_mesh"])

    def _full():
        integrate_manual(flow, particles, weights, n_steps, cfg["dt"], cfg["T"])
    t_full, t_full_s = timeit(_full, nt)

    batch_configs = [2, 3, 4, 5, 8]
    times_m, times_s = [t_full], [t_full_s]
    labels = ["Full"]

    for nb in batch_configs:
        sch = make_interleaved(p, nb)
        def _rbm(sch_=sch):
            rbm = build_rbm(flow, sch_, n_steps, cfg["dt"], p, cfg["T"])
            integrate_manual(rbm, particles, weights, n_steps, cfg["dt"], cfg["T"])
        tm, ts = timeit(_rbm, nt)
        times_m.append(tm); times_s.append(ts)
        labels.append(f"{nb}")
        print(f"  {nb} batches: {tm*1000:.1f} ms")
    print(f"  Full: {t_full*1000:.1f} ms")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x_pos = np.arange(len(labels))
    cb = [COLORS[3]] + [COLORS[0]] * (len(labels) - 1)
    ax.bar(x_pos, [t * 1000 for t in times_m],
           yerr=[t * 1000 for t in times_s],
           color=cb, edgecolor="k", capsize=4)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels)
    ax.set_ylabel("Forward-pass time (ms)")
    ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
    save(fig, os.path.join(out, "fig_flow7_benchmark"))
    return dict(zip(labels, times_m))
