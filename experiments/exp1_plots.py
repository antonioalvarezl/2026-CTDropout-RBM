"""
Figure-generation functions for Experiment 1 — Forward pass (§5.1).

All imports from rnode.*. Each public function produces one figure/table.
"""

import os, time, json
import numpy as np
import scipy.stats as sp_stats
import torch
import torch.nn as nn
from torchdiffeq import odeint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rnode.models import ConstantODE, TimeDepODE, RBMConstant, RBMTimeDep
from rnode.batches import (
    make_balanced_disjoint, make_drop_one, make_pick_one, make_bernoulli,
    sample_batch_sequence, sample_batch_sequence_from_h,
)
from rnode.data import make_circles_data, make_grid
from rnode.utils import compute_accuracy, timeit

# ── Style ──────────────────────────────────────────────────────────────────

DPI = 300
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]


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


# ── Integration helpers ────────────────────────────────────────────────────

def _dt(t_span):
    return float(t_span[-1] - t_span[0]) / (len(t_span) - 1)


def _rbm_cls(model):
    return RBMConstant if isinstance(model, ConstantODE) else RBMTimeDep


def fwd(model, X, t_span):
    with torch.no_grad():
        return odeint(model, X, t_span, method="rk4",
                      options={"step_size": _dt(t_span)})


def rbm_fwd(model, X, t_span, scheme, rep, p):
    sched = sample_batch_sequence(scheme, len(t_span), p=p, rep=rep)
    rbm = _rbm_cls(model)(model, t_span, sched, scheme.pi_min)
    with torch.no_grad():
        return odeint(rbm, X, t_span, method="rk4",
                      options={"step_size": _dt(t_span)})


def rbm_fwd_h(model, X, t_span, scheme, h, p):
    sched = sample_batch_sequence_from_h(scheme, t_span, h, p=p)
    rbm = _rbm_cls(model)(model, t_span, sched, scheme.pi_min)
    with torch.no_grad():
        return odeint(rbm, X, t_span, method="rk4",
                      options={"step_size": _dt(t_span)})


def traj_err(y_ref, y_rbm):
    return (y_ref - y_rbm).pow(2).sum(-1).mean(-1).max().item()


def _default_scheme(p):
    return make_balanced_disjoint(p, 8, seed=1)


def _five_schemes(p):
    return {
        "Balanced ($r$=8)":  make_balanced_disjoint(p, 8, seed=0),
        "Balanced ($r$=12)": make_balanced_disjoint(p, 12, seed=0),
        "Drop-one":          make_drop_one(p),
        "Pick-one":          make_pick_one(p),
        "Bernoulli ($q_B$=1/3)": make_bernoulli(p, q_B=1 / 3),
    }


def _four_schemes(p):
    return {
        "Balanced ($r$=8)":  make_balanced_disjoint(p, 8, seed=0),
        "Balanced ($r$=12)": make_balanced_disjoint(p, 12, seed=0),
        "Drop-one":          make_drop_one(p),
        "Bernoulli ($q_B$=1/3)": make_bernoulli(p, q_B=1 / 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1 — Trajectories
# ═══════════════════════════════════════════════════════════════════════════

def fig1_trajectories(mc, mt, Xc, yc, Xt, yt, cfg, out):
    print("\n[Fig 1] Trajectories...")
    _rc()
    p, K = cfg["p"], cfg["n_real_db"]
    sch = _default_scheme(p)
    t_fine = torch.linspace(0, cfg["T"], 500)
    t_rbm = torch.linspace(0, cfg["T"], cfg["N_conv"])

    tc = fwd(mc, Xc, t_fine); tt = fwd(mt, Xt, t_fine)
    rc = torch.stack([rbm_fwd(mc, Xc, t_rbm, sch, 5, p)
                      for _ in range(K)]).mean(0)
    rt = torch.stack([rbm_fwd(mt, Xt, t_rbm, sch, 5, p)
                      for _ in range(K)]).mean(0)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))

    def _pt(ax, traj, y):
        lb = y.squeeze().numpy(); cm = plt.get_cmap("bwr")
        for i in range(traj.shape[1]):
            ax.plot(traj[:, i, 0].numpy(), traj[:, i, 1].numpy(),
                    color=cm(lb[i]), alpha=0.08)
        ax.scatter(traj[-1, :, 0].numpy(), traj[-1, :, 1].numpy(),
                   c=lb, cmap="bwr", edgecolor="k", s=40, zorder=100)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_facecolor("#f5f5f5"); ax.grid(True, alpha=0.3)
        ax.set_xticks([]); ax.set_yticks([])

    ax0 = axes[0]
    ax0.scatter(Xc[:, 0].numpy(), Xc[:, 1].numpy(),
                c=yc.squeeze().numpy(), cmap="bwr", edgecolor="k", s=40)
    ax0.set_facecolor("#f5f5f5"); ax0.grid(True, alpha=0.3)
    ax0.set_xlim(-1.5, 1.5); ax0.set_ylim(-1.5, 1.5)
    ax0.set_xticks([]); ax0.set_yticks([])

    _pt(axes[1], tc, yc); _pt(axes[2], rc, yc)
    _pt(axes[3], tt, yt); _pt(axes[4], rt, yt)
    fig.tight_layout()
    save(fig, os.path.join(out, "fig1_trajectories"))


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2 — Convergence log-log
# ═══════════════════════════════════════════════════════════════════════════

def fig2_convergence(mc, mt, Xc, Xt, cfg, out):
    print("\n[Fig 2] Convergence...")
    _rc()
    p, N, T, K = cfg["p"], cfg["N_conv"], cfg["T"], cfg["n_real_conv"]
    sch = _default_scheme(p)
    reps = np.unique(np.logspace(0, 1.3, 12).astype(int))

    def _conv(model, X, tag):
        ts = torch.linspace(0, T, N); yr = fwd(model, X, ts)
        hs, ms, ss = [], [], []
        for rep in reps:
            h = rep * T / N; hs.append(h)
            errs = [traj_err(yr, rbm_fwd(model, X, ts, sch, rep, p))
                    for _ in range(K)]
            ms.append(np.mean(errs)); ss.append(np.std(errs))
            print(f"  [{tag}] rep={rep:2d}  h={h:.6f}  err={ms[-1]:.2e}")
        return np.array(hs), np.array(ms), np.array(ss)

    hc, mc_, sc = _conv(mc, Xc, "const")
    sl_c, ic_c, *_ = sp_stats.linregress(np.log(hc), np.log(mc_))
    ht, mt_, st = _conv(mt, Xt, "tdep")
    sl_t, ic_t, *_ = sp_stats.linregress(np.log(ht), np.log(mt_))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, hv, mv, sv, sl, ic in [(axes[0], ht, mt_, st, sl_t, ic_t),
                                    (axes[1], hc, mc_, sc, sl_c, ic_c)]:
        ref = (mv[0] / hv[0]) * 1.5 * hv
        ax.loglog(hv, ref, "+-", color=COLORS[3], lw=1.2)
        ax.errorbar(hv, mv, yerr=1.96 * sv / np.sqrt(K),
                    fmt="ko", capsize=3, ms=5)
        ax.loglog(hv, np.exp(ic) * hv ** sl, "--", color=COLORS[0], lw=1.5)
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(r"$\max_t\,\mathbb{E}_\omega[\|x_t-\hat{x}_t\|^2]$")
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save(fig, os.path.join(out, "fig2_convergence"))
    print(f"  Slopes — const: {sl_c:.3f}, tdep: {sl_t:.3f}")
    return {"const": sl_c, "tdep": sl_t}


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 — Decision boundaries
# ═══════════════════════════════════════════════════════════════════════════

def fig3_decision(mc, mt, cfg, out):
    print("\n[Fig 3] Decision boundaries...")
    _rc()
    p, K, N = cfg["p"], cfg["n_real_db"], cfg["N_boundary"]
    sch = _default_scheme(p)
    ts = torch.linspace(0, cfg["T"], N)
    xx, yy, grid = make_grid(n_points=cfg["n_grid"])
    reps = [0] + cfg["h_reps_db"]
    Xte, yte = make_circles_data(n_samples=200, seed=123)

    for label, model in [("const", mc), ("tdep", mt)]:
        bnds = []
        for rep in reps:
            if rep == 0:
                b = fwd(model, grid, ts)[-1, :, 0].numpy()
            else:
                b = np.mean([rbm_fwd(model, grid, ts, sch, rep, p
                                     )[-1, :, 0].numpy()
                             for _ in range(K)], axis=0)
            bnds.append(b)
        fig, axes = plt.subplots(1, len(reps), figsize=(4.8 * len(reps), 4.2))
        for ax, bnd in zip(axes, bnds):
            ax.contourf(xx, yy, -bnd.reshape(xx.shape),
                        levels=80, cmap="RdBu", alpha=0.8, vmin=-1, vmax=1)
            ax.scatter(Xte[:, 0].numpy(), Xte[:, 1].numpy(),
                       c=yte.squeeze().numpy(), edgecolor="k", s=35, cmap="jet")
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        save(fig, os.path.join(out, f"fig3_decision_{label}"))


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4 — Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def fig4_benchmarks(mc, mt, cfg, out):
    print("\n[Fig 4] Benchmarks...")
    _rc()
    p, N, nt = cfg["p"], cfg["N_bench"], cfg["n_trials_bench"]
    sch = _default_scheme(p)
    ts = torch.linspace(0, cfg["T"], N)

    for label, model in [("const", mc), ("tdep", mt)]:
        ntm, nts, rtm, rts = [], [], [], []
        for ns in cfg["data_sizes"]:
            Xb, _ = make_circles_data(n_samples=ns, seed=99)
            tm, ts_ = timeit(lambda x0=Xb: fwd(model, x0, ts), nt)
            ntm.append(tm); nts.append(ts_)
            tr, trs = timeit(lambda x0=Xb: rbm_fwd(model, x0, ts, sch, 1, p), nt)
            rtm.append(tr); rts.append(trs)
            print(f"  [{label}] n={ns:>6d}  NODE={tm:.3f}s  rNODE={tr:.3f}s")
        ds = cfg["data_sizes"]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(ds, ntm, "o-", color=COLORS[3], lw=1.5)
        ax.fill_between(ds, np.array(ntm)-np.array(nts),
                        np.array(ntm)+np.array(nts), color=COLORS[3], alpha=0.2)
        ax.plot(ds, rtm, "s-", color=COLORS[0], lw=1.5)
        ax.fill_between(ds, np.array(rtm)-np.array(rts),
                        np.array(rtm)+np.array(rts), color=COLORS[0], alpha=0.2)
        ax.set_xlabel("Dataset size"); ax.set_ylabel("Forward-pass time (s)")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        save(fig, os.path.join(out, f"fig4_benchmark_{label}"))


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5 — Cost vs Error scatter
# ═══════════════════════════════════════════════════════════════════════════

def fig5_cost_vs_error(model, X, cfg, out):
    print("\n[Fig 5] Cost vs Error scatter...")
    _rc()
    p, N = cfg["p"], cfg["N_conv"]
    ts = torch.linspace(0, cfg["T"], N); yr = fwd(model, X, ts)
    K, nt = cfg["n_real_scatter"], cfg["n_time_repeats"]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, (name, sch) in enumerate(_five_schemes(p).items()):
        for h in cfg["h_scatter"]:
            errs = [traj_err(yr, rbm_fwd_h(model, X, ts, sch, h, p))
                    for _ in range(K)]
            me = np.mean(errs)
            mt_, st_ = timeit(lambda: rbm_fwd_h(model, X, ts, sch, h, p), nt)
            ax.errorbar(mt_, me, xerr=st_, fmt=MARKERS[i], color=COLORS[i],
                        ms=9, capsize=3, lw=1.2)
            ax.annotate(f"$h$={h}", (mt_, me), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=COLORS[i])
            print(f"  [{name}] h={h}  err={me:.2e}  t={mt_:.4f}s")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel(r"$\max_t\,\mathbb{E}_\omega[\|x_t-\hat{x}_t\|^2]$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
    save(fig, os.path.join(out, "fig5_cost_vs_error"))


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6 — Scheme convergence
# ═══════════════════════════════════════════════════════════════════════════

def fig6_scheme_convergence(model, X, cfg, out):
    print("\n[Fig 6] Scheme convergence comparison...")
    _rc()
    p, N, T, K = cfg["p"], cfg["N_conv"], cfg["T"], cfg["n_real_scheme"]
    ts = torch.linspace(0, T, N); yr = fwd(model, X, ts)
    reps = np.unique(np.logspace(0, 1.3, 10).astype(int))
    hvs = reps * T / N

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (name, sch) in enumerate(_four_schemes(p).items()):
        ms = np.array([np.mean([traj_err(yr, rbm_fwd(model, X, ts, sch,
                                                      int(r), p))
                                for _ in range(K)]) for r in reps])
        sl, *_ = sp_stats.linregress(np.log(hvs), np.log(ms))
        ax.loglog(hvs, ms, MARKERS[i]+"-", color=COLORS[i], ms=6, lw=1.2)
        print(f"  [{name}] slope = {sl:.3f}")
    ref = 0.3 / hvs[0]; ax.loglog(hvs, ref * hvs, "k--", lw=1, alpha=0.5)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\max_t\,\mathbb{E}_\omega[\|x_t-\hat{x}_t\|^2]$")
    ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
    save(fig, os.path.join(out, "fig6_scheme_convergence"))


# ═══════════════════════════════════════════════════════════════════════════
# Fig 7 — Variance by scheme
# ═══════════════════════════════════════════════════════════════════════════

def fig7_variance(model, X, cfg, out):
    print("\n[Fig 7] Empirical variance by scheme...")
    _rc()
    p, N, T, K = cfg["p"], cfg["N_conv"], cfg["T"], cfg["n_real_scheme"]
    ts = torch.linspace(0, T, N); yr = fwd(model, X, ts)
    reps = np.unique(np.logspace(0, 1.3, 10).astype(int))
    hvs = reps * T / N

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (name, sch) in enumerate(_four_schemes(p).items()):
        vs = [np.var([traj_err(yr, rbm_fwd(model, X, ts, sch, int(r), p))
                      for _ in range(K)]) for r in reps]
        ax.loglog(hvs, vs, MARKERS[i]+"-", color=COLORS[i], ms=6, lw=1.2)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\mathrm{Var}_\omega[\max_t \|x_t-\hat{x}_t\|^2]$")
    ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
    save(fig, os.path.join(out, "fig7_variance_by_scheme"))


# ═══════════════════════════════════════════════════════════════════════════
# Fig 8 — Pareto front
# ═══════════════════════════════════════════════════════════════════════════

def fig8_pareto(model, X, cfg, out):
    print("\n[Fig 8] Pareto front...")
    _rc()
    p, N, K = cfg["p"], cfg["N_conv"], cfg["n_real_pareto"]
    h = cfg["h_pareto"]
    ts = torch.linspace(0, cfg["T"], N); yr = fwd(model, X, ts)

    r_vals = sorted([r for r in range(1, p+1) if p % r == 0])
    errs = []
    for r in r_vals:
        sch = make_balanced_disjoint(p, r, seed=0)
        e = np.mean([traj_err(yr, rbm_fwd_h(model, X, ts, sch, h, p))
                     for _ in range(K)])
        errs.append(e); print(f"  r={r:>2d}  err={e:.2e}")
    errs = np.array(errs); costs = np.array(r_vals)

    par = [i for i in range(len(costs))
           if not any(costs[j] <= costs[i] and errs[j] < errs[i]
                      for j in range(len(costs)) if j != i)]
    par = sorted(par, key=lambda i: costs[i])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(costs, errs, "o", color=COLORS[0], ms=8, zorder=5)
    ax.plot(costs[par], errs[par], "--", color=COLORS[3], lw=1.5)
    ax.set_xlabel(r"Batch size $r$")
    ax.set_ylabel(r"$\max_t\,\mathbb{E}_\omega[\|x_t-\hat{x}_t\|^2]$")
    ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save(fig, os.path.join(out, "fig8_pareto"))


# ═══════════════════════════════════════════════════════════════════════════
# Fig 9 — Optimal h*
# ═══════════════════════════════════════════════════════════════════════════

def fig9_optimal_h(model, X, cfg, out):
    print("\n[Fig 9] Optimal h*...")
    _rc()
    p, N, T, K = cfg["p"], cfg["N_conv"], cfg["T"], cfg["n_real_opth"]
    ts = torch.linspace(0, T, N); yr = fwd(model, X, ts)
    sch = _default_scheme(p)

    h_sw = np.logspace(-3.5, -0.5, 25)
    emp = np.array([np.mean([traj_err(yr, rbm_fwd_h(model, X, ts, sch, h, p))
                             for _ in range(K)]) for h in h_sw])
    mask = h_sw < 0.05
    sl, ic, *_ = sp_stats.linregress(np.log(h_sw[mask]), np.log(emp[mask]))
    S = np.exp(ic)
    print(f"  S = {S:.4f}, slope = {sl:.3f}")

    eps = np.logspace(np.log10(emp.min()*1.5), np.log10(emp.max()*0.8), 12)
    h_emp = np.interp(eps, emp[::-1], h_sw[::-1])
    h_th = eps**2 / S

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(h_sw, emp, "o-", color=COLORS[0], ms=5, lw=1.2)
    hf = np.logspace(-3.5, -0.5, 100)
    axes[0].loglog(hf, S*hf**sl, "--", color=COLORS[3], lw=1.5)
    axes[0].set_xlabel(r"$h$"); axes[0].set_ylabel(r"$\mathcal{E}(h)$")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].loglog(h_th, h_emp, "o", color=COLORS[0], ms=7)
    lims = [min(h_th.min(), h_emp.min())*0.5, max(h_th.max(), h_emp.max())*2]
    axes[1].loglog(lims, lims, "k--", lw=1, alpha=0.5)
    axes[1].set_xlabel(r"$h^*_{\mathrm{theory}}$")
    axes[1].set_ylabel(r"$h^*_{\mathrm{empirical}}$")
    axes[1].set_xlim(lims); axes[1].set_ylim(lims); axes[1].set_aspect("equal")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save(fig, os.path.join(out, "fig9_optimal_h"))


# ═══════════════════════════════════════════════════════════════════════════
# Table 1 — Batch count performance
# ═══════════════════════════════════════════════════════════════════════════

def table1_batch_counts(model, X, y_raw, cfg, out):
    print("\n[Table 1] Performance across batch counts...")
    p, N, K = cfg["p"], cfg["N_boundary"], cfg["n_real_db"]
    ts = torch.linspace(0, cfg["T"], N)
    Xte, yte = make_circles_data(n_samples=200, seed=123)
    yte_oh = torch.zeros(200, 2)
    yte_oh[yte.squeeze()==-1, 0] = -1; yte_oh[yte.squeeze()==1, 1] = 1

    yf = fwd(model, Xte, ts)
    rows = [{"config": "Full NODE",
             "test_loss": nn.MSELoss()(yf[-1], yte_oh).item(),
             "test_acc": compute_accuracy(yf[-1], yte),
             "time": timeit(lambda: fwd(model, Xte, ts), 5)[0]}]
    for nb in [2, 3, 4, 6, 8]:
        r = p // nb
        if p % nb: continue
        sch = make_balanced_disjoint(p, r, seed=0)
        ls, ac, ti = [], [], []
        for _ in range(K):
            t0 = time.perf_counter()
            yr = rbm_fwd(model, Xte, ts, sch, 1, p)
            ti.append(time.perf_counter()-t0)
            ls.append(nn.MSELoss()(yr[-1], yte_oh).item())
            ac.append(compute_accuracy(yr[-1], yte))
        rows.append({"config": f"rNODE {nb} batches (r={r})",
                      "test_loss": np.mean(ls), "test_acc": np.mean(ac),
                      "time": np.mean(ti)})
        print(f"  {nb} batches: acc={np.mean(ac):.2%}")
    path = os.path.join(out, "table1_batch_counts.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  -> {path}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Fig 10 — Error constant S vs π_min
# ═══════════════════════════════════════════════════════════════════════════

def fig10_error_constant_vs_pimin(model, X, cfg, out):
    """Estimate S = E[err^2] / h for each scheme and plot vs π_min."""
    print("\n[Fig 10] Error constant S vs π_min...")
    _rc()
    p, N, T, K = cfg["p"], cfg["N_conv"], cfg["T"], cfg["n_real_scheme"]
    ts = torch.linspace(0, T, N); yr = fwd(model, X, ts)

    # Use a small h for best S estimate
    h_ref = cfg.get("h_S_estimate", 0.01)
    rep_ref = max(1, int(round(h_ref * N / T)))

    # Collect all schemes with varying pi_min
    all_schemes = {}
    all_schemes["Drop-one"] = make_drop_one(p)
    all_schemes["Pick-one"] = make_pick_one(p)
    for r in [2, 3, 4, 6, 8, 12]:
        if p % (p // r) == 0 and r < p:
            all_schemes[f"Balanced $r$={r}"] = make_balanced_disjoint(p, r, seed=0)
    for q in [0.25, 0.5, 0.75]:
        all_schemes[f"Bernoulli $q_B$={q}"] = make_bernoulli(p, q_B=q)

    pimins, S_vals, labels = [], [], []
    for name, sch in all_schemes.items():
        errs = [traj_err(yr, rbm_fwd(model, X, ts, sch, rep_ref, p))
                for _ in range(K)]
        S = np.mean(errs) / h_ref
        pimins.append(sch.pi_min)
        S_vals.append(S)
        labels.append(name)
        print(f"  {name:30s}  π_min={sch.pi_min:.3f}  S={S:.4f}")

    pimins = np.array(pimins); S_vals = np.array(S_vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pimins, S_vals, s=100, c=COLORS[0], edgecolor="k", zorder=5)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (pimins[i], S_vals[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Theoretical curve: S ~ C / pi_min (or similar)
    pi_fit = np.linspace(pimins.min() * 0.8, pimins.max() * 1.1, 100)
    mask = pimins > 0.05  # avoid pick-one outlier for fit
    if mask.sum() >= 3:
        sl, ic, *_ = sp_stats.linregress(np.log(pimins[mask]),
                                          np.log(S_vals[mask]))
        ax.plot(pi_fit, np.exp(ic) * pi_fit ** sl, "--", color=COLORS[3],
                lw=1.5, alpha=0.7)
        print(f"  Fitted S ~ pi_min^{sl:.2f}")

    ax.set_xlabel(r"$\pi_{\min}$")
    ax.set_ylabel(r"Error constant $S = \mathcal{E}/h$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
    save(fig, os.path.join(out, "fig10_error_constant_vs_pimin"))
    return dict(zip(labels, zip(pimins.tolist(), S_vals.tolist())))


# ═══════════════════════════════════════════════════════════════════════════
# Fig 11 — Speedup analysis
# ═══════════════════════════════════════════════════════════════════════════

def fig11_speedup(model, X, cfg, out):
    """Speedup of rNODE vs full NODE for different batch sizes r."""
    print("\n[Fig 11] Speedup analysis...")
    _rc()
    p, N, nt = cfg["p"], cfg["N_bench"], cfg["n_trials_bench"]
    ts = torch.linspace(0, cfg["T"], N)

    t_full, _ = timeit(lambda: fwd(model, X, ts), nt)

    r_vals = sorted([r for r in [2, 3, 4, 6, 8, 12] if r < p and p % r == 0])
    speedups, r_list, t_rbms = [], [], []
    for r in r_vals:
        sch = make_balanced_disjoint(p, r, seed=0)
        t_r, _ = timeit(lambda: rbm_fwd(model, X, ts, sch, 1, p), nt)
        sp = t_full / t_r
        speedups.append(sp); r_list.append(r); t_rbms.append(t_r)
        print(f"  r={r:>2d}  time={t_r*1000:.1f}ms  speedup={sp:.2f}x")
    print(f"  Full  time={t_full*1000:.1f}ms")

    fig, ax = plt.subplots(figsize=(6, 4))
    pi_vals = np.array(r_list) / p
    ax.plot(pi_vals, speedups, "o-", color=COLORS[0], ms=8, lw=1.5)
    ax.axhline(1.0, ls="--", color=COLORS[7], lw=1)
    ax.set_xlabel(r"$\pi = r/p$")
    ax.set_ylabel("Speedup (full / rNODE)")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    save(fig, os.path.join(out, "fig11_speedup"))

    rows = [{"r": r, "pi": r/p, "time_ms": t*1000, "speedup": sp}
            for r, t, sp in zip(r_list, t_rbms, speedups)]
    rows.append({"r": p, "pi": 1.0, "time_ms": t_full*1000, "speedup": 1.0})
    path = os.path.join(out, "fig11_speedup.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  -> {path}")
    return rows
