"""
Generate the README header GIF with trained parameters.

Goal of the trained dynamics:
- red points -> (-1, -1)
- blue points -> (1, 1)

The two panels show:
- Neural ODE (full model)
- Dropout-trained Neural ODE (simulated without dropout)
"""

import os
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Exact colors from your simulator HTML.
# BAND_COLORS (dark, for background regions)
BG_RED = "#801D1A"
BG_BLUE = "#2B56A1"
# CLASS_COLORS for M=2 (light, for point interiors)
POINT_RED = "#EF8BA4"
POINT_BLUE = "#78A4F8"

# Dataset sizes
TRAIN_POINTS = 140
GIF_POINTS = 700

# Training lengths
EPOCHS_DEFAULT = 1500
EPOCHS_CIRCLES = 2600

# Vector-field overlay (semi-transparent background arrows)
VF_GRID = 15
VF_ALPHA = 0.22
VF_COLOR_FULL = "#101010"
VF_COLOR_DROPOUT = "#2A2A2A"
VF_NB = 3
VF_REP = 1


# ── Data ───────────────────────────────────────────────────────────────────

def make_circles(n=120, noise=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n_half = n // 2
    t1 = rng.uniform(0, 2 * np.pi, n_half)
    t2 = rng.uniform(0, 2 * np.pi, n - n_half)
    X_outer = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    X_inner = 0.5 * np.stack([np.cos(t2), np.sin(t2)], axis=1)
    X = np.concatenate([X_outer, X_inner], axis=0)
    X += rng.randn(*X.shape) * noise
    y = np.concatenate([np.ones(n_half), -np.ones(n - n_half)], axis=0)
    return X.astype(np.float64), y.astype(np.float64)


def make_pinwheel(n=120, radial_std=0.25, tangential_std=0.08, rate=0.35, seed=42):
    """Two-class pinwheel dataset in 2D."""
    rng = np.random.RandomState(seed)
    n_classes = 2

    base_count = n // n_classes
    extra = n % n_classes
    counts = [base_count + (1 if c < extra else 0) for c in range(n_classes)]
    labels = np.concatenate([np.full(cnt, c, dtype=np.int64) for c, cnt in enumerate(counts)])

    feats = rng.randn(n, 2) * np.array([radial_std, tangential_std], dtype=np.float64)
    feats[:, 0] += 1.0

    base_angles = labels * (2 * np.pi / n_classes)
    angles = base_angles + rate * np.exp(feats[:, 0])
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    X = np.empty_like(feats)
    X[:, 0] = cos_a * feats[:, 0] - sin_a * feats[:, 1]
    X[:, 1] = sin_a * feats[:, 0] + cos_a * feats[:, 1]
    X *= 0.95

    perm = rng.permutation(n)
    X = X[perm]
    labels = labels[perm]
    # Keep same binary convention used by the rest of the script.
    y = np.where(labels == 0, 1.0, -1.0)
    return X.astype(np.float64), y.astype(np.float64)


def make_complex_spiral(n=120, seed=42):
    """
    Harder two-class dataset than pinwheel:
    intertwined spirals + nonlinear warping + heteroscedastic noise.
    """
    rng = np.random.RandomState(seed)
    n_half = n // 2
    rem = n - n_half

    t0 = rng.uniform(0.08, 1.0, n_half)
    t1 = rng.uniform(0.08, 1.0, rem)

    # Spiral core with moderate modulation.
    theta0 = 4.8 * np.pi * t0 + 0.35 * np.sin(9 * np.pi * t0)
    theta1 = theta0[:rem] + np.pi + 0.22 * np.cos(7 * np.pi * t1)
    r0 = 0.20 + 1.62 * t0 + 0.09 * np.sin(12 * np.pi * t0)
    r1 = 0.22 + 1.55 * t1 + 0.10 * np.cos(10 * np.pi * t1)

    x0 = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)
    x1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)

    # Class-dependent anisotropic noise (reduced vs previous version).
    n0 = np.stack(
        [
            rng.normal(scale=0.030 + 0.015 * t0),
            rng.normal(scale=0.017 + 0.010 * np.sqrt(t0)),
        ],
        axis=1,
    )
    n1 = np.stack(
        [
            rng.normal(scale=0.029 + 0.014 * t1),
            rng.normal(scale=0.018 + 0.010 * np.sqrt(t1)),
        ],
        axis=1,
    )
    x0 += n0
    x1 += n1

    X = np.concatenate([x0, x1], axis=0)
    # Nonlinear warp (kept, but softened to avoid excessive clutter).
    Xw = X.copy()
    Xw[:, 0] = X[:, 0] + 0.12 * np.sin(1.7 * X[:, 1]) + 0.04 * np.cos(2.4 * X[:, 0])
    Xw[:, 1] = X[:, 1] + 0.10 * np.sin(1.9 * X[:, 0]) - 0.04 * np.cos(2.1 * X[:, 1])

    # Keep in roughly [-2, 2]^2
    mx = np.max(np.abs(Xw))
    Xw = 1.75 * Xw / max(mx, 1e-12)

    y = np.concatenate([np.ones(n_half), -np.ones(rem)], axis=0)
    perm = rng.permutation(n)
    return Xw[perm].astype(np.float64), y[perm].astype(np.float64)


def make_dataset(kind, n, seed):
    if kind == "circles":
        return make_circles(n=n, noise=0.05, seed=seed)
    if kind == "pinwheel":
        return make_pinwheel(n=n, seed=seed)
    if kind == "complex_spiral":
        return make_complex_spiral(n=n, seed=seed)
    raise ValueError(f"Unknown dataset kind: {kind}")


def build_targets(y):
    tgt = np.zeros((len(y), 2), dtype=np.float64)
    tgt[y > 0] = np.array([-1.0, -1.0], dtype=np.float64)  # red
    tgt[y < 0] = np.array([1.0, 1.0], dtype=np.float64)   # blue
    return tgt


# ── Model and Integration ──────────────────────────────────────────────────

def init_params(p=40, d=2, seed=42):
    rng = np.random.RandomState(seed)
    return {
        "W1": rng.randn(p, d) * 0.42,
        "b1": rng.randn(p) * 0.23,
        "W2": rng.randn(d, p) * 0.20,
    }


def make_batches(p, nb=3, seed=123):
    rng = np.random.RandomState(seed)
    idx = np.arange(p)
    rng.shuffle(idx)
    batches = [idx[i::nb].copy() for i in range(nb)]
    pi = min(len(b) for b in batches) / p
    return batches, float(pi)


def make_schedule(N, nb, rep=3, seed=7):
    rng = np.random.RandomState(seed)
    sched = np.zeros(N, dtype=np.int64)
    b = rng.randint(nb)
    for s in range(N):
        if s % rep == 0:
            b = rng.randint(nb)
        sched[s] = b
    return sched


def forward_step(x, params, active=None, pi=1.0):
    if active is None:
        W1 = params["W1"]
        b1 = params["b1"]
        W2 = params["W2"]
        z = x @ W1.T + b1
        h = np.maximum(0.0, z)
        dx = h @ W2.T
        cache = (x, z, h, None, 1.0)
        return dx, cache

    W1a = params["W1"][active]
    b1a = params["b1"][active]
    W2a = params["W2"][:, active]
    z = x @ W1a.T + b1a
    h = np.maximum(0.0, z)
    dx = (h @ W2a.T) / pi
    cache = (x, z, h, active, pi)
    return dx, cache


def integrate(params, X0, N, dt, batches=None, schedule=None, pi=1.0):
    x = X0.copy()
    traj = [x.copy()]
    caches = []
    for s in range(N):
        if batches is None:
            dx, cache = forward_step(x, params, active=None, pi=1.0)
        else:
            active = batches[schedule[s]]
            dx, cache = forward_step(x, params, active=active, pi=pi)
        x = x + dt * dx
        caches.append(cache)
        traj.append(x.copy())
    return np.stack(traj, axis=0), caches


def endpoint_loss_and_grad(x_end, y, target):
    n, d = x_end.shape
    diff = x_end - target
    loss_point = np.mean(diff * diff)
    grad = 2.0 * diff / (n * d)

    red_mask = y > 0
    blue_mask = y < 0
    red_mean = x_end[red_mask].mean(axis=0)
    blue_mean = x_end[blue_mask].mean(axis=0)

    red_tgt = np.array([-1.0, -1.0], dtype=np.float64)
    blue_tgt = np.array([1.0, 1.0], dtype=np.float64)
    d_red = red_mean - red_tgt
    d_blue = blue_mean - blue_tgt
    loss_mean = 0.5 * (np.sum(d_red * d_red) + np.sum(d_blue * d_blue))

    grad_mean = np.zeros_like(x_end)
    grad_mean[red_mask] += d_red / max(1, red_mask.sum())
    grad_mean[blue_mask] += d_blue / max(1, blue_mask.sum())

    loss = loss_point + 0.55 * loss_mean
    grad_total = grad + 0.55 * grad_mean
    stats = {
        "loss_point": float(loss_point),
        "loss_mean": float(loss_mean),
        "red_mean": red_mean.copy(),
        "blue_mean": blue_mean.copy(),
    }
    return float(loss), grad_total, stats


def backward_through_euler(params, caches, grad_x_end, dt):
    W1 = params["W1"]
    W2 = params["W2"]
    gW1 = np.zeros_like(W1)
    gb1 = np.zeros_like(params["b1"])
    gW2 = np.zeros_like(W2)

    gx = grad_x_end.copy()
    for cache in reversed(caches):
        x, z, h, active, pi = cache
        if active is None:
            gdx = gx
            gW2 += gdx.T @ h * dt
            gh = (gdx @ W2) * dt
            gz = gh * (z > 0.0)
            gW1 += gz.T @ x
            gb1 += gz.sum(axis=0)
            gx = gx + gz @ W1
        else:
            W1a = W1[active]
            W2a = W2[:, active]
            gdx = gx
            gscaled = (gdx * dt) / pi
            gW2[:, active] += gscaled.T @ h
            gh = gscaled @ W2a
            gz = gh * (z > 0.0)
            gW1[active] += gz.T @ x
            gb1[active] += gz.sum(axis=0)
            gx = gx + gz @ W1a

    return {"W1": gW1, "b1": gb1, "W2": gW2}


# ── Optimizer ──────────────────────────────────────────────────────────────

def adam_init(params):
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    return m, v


def adam_step(params, grads, m, v, t, lr=0.03, b1=0.9, b2=0.999, eps=1e-8):
    for k in params:
        m[k] = b1 * m[k] + (1.0 - b1) * grads[k]
        v[k] = b2 * v[k] + (1.0 - b2) * (grads[k] * grads[k])
        m_hat = m[k] / (1.0 - b1 ** t)
        v_hat = v[k] / (1.0 - b2 ** t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ── Training ───────────────────────────────────────────────────────────────

def _summarize_model(params, X, y, target, N, dt, label):
    traj_end, _ = integrate(params, X, N, dt)
    x_end = traj_end[-1]
    red_mean = x_end[y > 0].mean(axis=0)
    blue_mean = x_end[y < 0].mean(axis=0)
    mse = np.mean((x_end - target) ** 2)
    print(
        f"{label} final means:"
        f" red -> ({red_mean[0]:.3f}, {red_mean[1]:.3f}),"
        f" blue -> ({blue_mean[0]:.3f}, {blue_mean[1]:.3f}),"
        f" endpoint MSE={mse:.5f}"
    )


def _train_single_model(X, y, target, seed, use_dropout_train=False, epochs=EPOCHS_DEFAULT):
    params = init_params(p=40, d=2, seed=seed)
    m, v = adam_init(params)
    best_params = {k: val.copy() for k, val in params.items()}
    best_score = float("inf")

    N = 64
    dt = 1.0 / N
    nb = 3
    rep = 1
    batches, pi = make_batches(params["W1"].shape[0], nb=nb, seed=seed + 11)

    mode = "dropout-train" if use_dropout_train else "full-train"
    print(f"Training {mode} parameters (numpy BPTT)...")

    for ep in range(1, epochs + 1):
        if use_dropout_train:
            # Train under random batch/dropout dynamics.
            sched = make_schedule(N, nb=nb, rep=rep, seed=ep + 100)
            traj, cache = integrate(
                params, X, N, dt, batches=batches, schedule=sched, pi=pi
            )
            loss, g_end, stats = endpoint_loss_and_grad(traj[-1], y, target)
            grads = backward_through_euler(params, cache, g_end, dt)
            score = loss
        else:
            # Train standard deterministic Neural ODE.
            traj, cache = integrate(params, X, N, dt)
            loss, g_end, stats = endpoint_loss_and_grad(traj[-1], y, target)
            grads = backward_through_euler(params, cache, g_end, dt)
            score = loss

        lam_reg = 2e-4
        for k in params:
            grads[k] += lam_reg * params[k]

        gnorm = 0.0
        for k in grads:
            gnorm += np.sum(grads[k] * grads[k])
        gnorm = np.sqrt(gnorm)
        if gnorm > 5.0:
            scale = 5.0 / (gnorm + 1e-12)
            for k in grads:
                grads[k] *= scale

        lr = 0.026 if ep <= 900 else 0.014
        adam_step(params, grads, m, v, ep, lr=lr)

        if score < best_score:
            best_score = score
            best_params = {k: val.copy() for k, val in params.items()}

        if ep % 200 == 0:
            rm = stats["red_mean"]
            bm = stats["blue_mean"]
            print(
                f"  [{mode}] epoch {ep:4d} | loss={loss:.5f} "
                f"| red=({rm[0]:.3f},{rm[1]:.3f}) "
                f"| blue=({bm[0]:.3f},{bm[1]:.3f})"
            )

    return best_params


def train_parameters(seed=42, dataset_kind="circles", epochs=EPOCHS_DEFAULT):
    random.seed(seed)
    np.random.seed(seed)

    X, y = make_dataset(dataset_kind, n=TRAIN_POINTS, seed=seed)
    target = build_targets(y)
    N_eval = 64
    dt_eval = 1.0 / N_eval

    params_full = _train_single_model(
        X, y, target, seed=seed, use_dropout_train=False, epochs=epochs
    )
    params_dropout = _train_single_model(
        X, y, target, seed=seed + 1, use_dropout_train=True, epochs=epochs
    )

    _summarize_model(params_full, X, y, target, N_eval, dt_eval, "Neural ODE")
    _summarize_model(
        params_dropout, X, y, target, N_eval, dt_eval, "Dropout-trained Neural ODE"
    )

    return params_full, params_dropout


# ── Rendering ──────────────────────────────────────────────────────────────

def draw_background(ax):
    xmin, xmax = -2.0, 2.0
    ymin, ymax = -2.0, 2.0

    # Split by the secondary diagonal y = -x.
    # Region y > -x (upper-right side): blue region.
    ax.fill([xmin, xmax, xmax], [ymax, ymax, ymin],
            color=BG_BLUE, alpha=1.0, zorder=0,
            edgecolor="none", linewidth=0.0, antialiased=False)
    # Region y < -x (lower-left side): red region.
    ax.fill([xmin, xmin, xmax], [ymax, ymin, ymin],
            color=BG_RED, alpha=1.0, zorder=0,
            edgecolor="none", linewidth=0.0, antialiased=False)


def _draw_vector_field(ax, params, x_grid, y_grid, color, active=None, pi=1.0):
    pts = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    vel, _ = forward_step(pts, params, active=active, pi=pi)
    U = vel[:, 0].reshape(x_grid.shape)
    V = vel[:, 1].reshape(y_grid.shape)

    # Clamp very large vectors to keep arrows legible.
    mag = np.sqrt(U * U + V * V)
    vmax = 1.6
    fac = np.minimum(1.0, vmax / (mag + 1e-12))
    U = U * fac
    V = V * fac

    ax.quiver(
        x_grid,
        y_grid,
        U,
        V,
        angles="xy",
        scale_units="xy",
        scale=8.0,
        color=color,
        alpha=VF_ALPHA,
        width=0.0026,
        headwidth=3.0,
        headlength=4.0,
        headaxislength=3.5,
        pivot="mid",
        zorder=1.3,
    )


def render_frames(params_full, params_dropout, traj_full, traj_rbm, y, dt):
    # Point interiors use the same light palette as in your HTML CLASS_COLORS.
    point_blue, point_red = POINT_BLUE, POINT_RED
    point_colors = np.array([point_blue if yi < 0 else point_red for yi in y])

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "figure.dpi": 220,
    })

    frames = []
    n_frames = traj_full.shape[0]

    # Precompute vector-field grid and a time-varying dropout schedule for right panel.
    grid_axis = np.linspace(-1.85, 1.85, VF_GRID)
    xg, yg = np.meshgrid(grid_axis, grid_axis)
    batches_vf, pi_vf = make_batches(params_dropout["W1"].shape[0], nb=VF_NB, seed=911)
    sched_vf = make_schedule(n_frames, nb=VF_NB, rep=VF_REP, seed=2718)

    print(f"Rendering {n_frames} frames...")
    for fr in range(n_frames):
        t_cur = fr * dt
        fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.7),
                                 gridspec_kw={"wspace": 0.08})
        for ai, (ax, tr, tag) in enumerate(
            zip(
                axes,
                [traj_full, traj_rbm],
                ["Neural ODE", "Dropout - Neural ODE"],
            )
        ):
            draw_background(ax)
            if ai == 0:
                _draw_vector_field(ax, params_full, xg, yg, color=VF_COLOR_FULL)
            else:
                active = batches_vf[sched_vf[fr]]
                _draw_vector_field(
                    ax,
                    params_dropout,
                    xg,
                    yg,
                    color=VF_COLOR_DROPOUT,
                    active=active,
                    pi=pi_vf,
                )

            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.20, lw=0.6)

            s0 = max(0, fr - 9)
            for i in range(tr.shape[1]):
                ax.plot(
                    tr[s0:fr + 1, i, 0],
                    tr[s0:fr + 1, i, 1],
                    "-",
                    color=point_colors[i],
                    alpha=0.30,
                    lw=0.85,
                    zorder=2,
                )
            ax.scatter(
                tr[fr, :, 0], tr[fr, :, 1],
                c=point_colors, s=44,
                edgecolors="#121212",
                linewidths=0.52,
                zorder=3
            )

            ax.set_title(tag, fontsize=16, pad=10)
            for sp in ax.spines.values():
                sp.set_edgecolor("#000000")
                sp.set_linewidth(1.8)

        fig.suptitle(f"t = {t_cur:.2f}", fontsize=20, y=0.97)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.08)
        fig.canvas.draw()
        img = Image.frombuffer(
            "RGBA",
            fig.canvas.get_width_height(),
            fig.canvas.buffer_rgba()
        ).convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        frames.append(img)
        plt.close(fig)

        if (fr + 1) % 12 == 0:
            print(f"  frame {fr + 1}/{n_frames}")
    return frames


def _render_one_gif(params_full, params_dropout, X_viz, y_viz, out_path, dataset_kind, split_name):
    # Playback settings.
    N = 72
    dt = 1.0 / N

    print(
        f"Integrating deterministic trajectories (no dropout at inference) "
        f"for dataset='{dataset_kind}', split='{split_name}'..."
    )
    traj_full, _ = integrate(params_full, X_viz, N, dt)
    traj_rbm, _ = integrate(params_dropout, X_viz, N, dt)

    frames = render_frames(params_full, params_dropout, traj_full, traj_rbm, y_viz, dt)
    for _ in range(4):
        frames.append(frames[-1].copy())

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=45,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"Saved {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")


def _generate_circles_outputs(split="both", out_prefix=None):
    """Generate circles GIFs: train/test plus legacy main header."""
    print(
        f"\nGenerating circles GIFs (train/test + main) with {EPOCHS_CIRCLES} epochs..."
    )
    params_full, params_dropout = train_parameters(
        seed=42, dataset_kind="circles", epochs=EPOCHS_CIRCLES
    )
    prefix = out_prefix or "assets/rnode_header_circles"
    split_seed = {"train": 123, "test": 2026}
    splits = ["train", "test"] if split == "both" else [split]
    for sp in splits:
        X_viz, y_viz = make_dataset("circles", n=GIF_POINTS, seed=split_seed[sp])
        _render_one_gif(
            params_full,
            params_dropout,
            X_viz,
            y_viz,
            out_path=f"{prefix}_{sp}.gif",
            dataset_kind="circles",
            split_name=sp,
        )

    # Keep backward-compatible main header path.
    X_main, y_main = make_dataset("circles", n=GIF_POINTS, seed=123)
    _render_one_gif(
        params_full,
        params_dropout,
        X_main,
        y_main,
        out_path="assets/rnode_header.gif",
        dataset_kind="circles",
        split_name="main",
    )


def generate_gifs(dataset_kind="complex_spiral", split="both", out_prefix=None):
    if dataset_kind == "circles":
        _generate_circles_outputs(split=split, out_prefix=out_prefix)
        return

    epochs = EPOCHS_DEFAULT
    params_full, params_dropout = train_parameters(
        seed=42, dataset_kind=dataset_kind, epochs=epochs
    )

    def _default_prefix(kind):
        return f"assets/rnode_header_{kind}"

    prefix = out_prefix or _default_prefix(dataset_kind)

    split_seed = {"train": 123, "test": 2026}
    splits = ["train", "test"] if split == "both" else [split]
    for sp in splits:
        X_viz, y_viz = make_dataset(dataset_kind, n=GIF_POINTS, seed=split_seed[sp])
        out_path = f"{prefix}_{sp}.gif"
        _render_one_gif(
            params_full, params_dropout, X_viz, y_viz, out_path, dataset_kind, sp
        )

    # Keep generating the classic two-circles header in the same run.
    _generate_circles_outputs(split="both", out_prefix="assets/rnode_header_circles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate README GIF for Neural ODE comparison.")
    parser.add_argument(
        "--dataset",
        choices=["circles", "pinwheel", "complex_spiral"],
        default="complex_spiral",
        help="Dataset used for training and visualization.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split GIF(s) to generate.",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix path. Files are saved as '<prefix>_train.gif' and/or '<prefix>_test.gif'.",
    )
    args = parser.parse_args()

    generate_gifs(dataset_kind=args.dataset, split=args.split, out_prefix=args.out_prefix)
