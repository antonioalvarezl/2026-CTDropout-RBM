"""Transport summaries and one predeclared illustration from frozen checkpoints."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import numpy as np

try:
    from experiments._style import (
        MUTED, annotate, color, dyadic_ticks, grid, headroom, reference_line,
        save, stack_labels, use_paper_style,
    )
except ModuleNotFoundError:
    from _style import (
        MUTED, annotate, color, dyadic_ticks, grid, headroom, reference_line,
        save, stack_labels, use_paper_style,
    )
import matplotlib.pyplot as plt


def _rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def _band(axis, h, value, se, tint, label=None):
    # Bounds below zero are clipped only for log rendering, not for inference.
    low, high = np.maximum(value - 1.96 * se, value * 1e-3), value + 1.96 * se
    axis.fill_between(h, low, high, color=tint, alpha=.13, lw=0)
    axis.plot(h, value, '-o', color=tint, markeredgecolor='white', label=label)


def _gaussian_blur(sigma_pt):
    """An ``agg_filter`` that softens an artist by a Gaussian of ``sigma_pt`` points.

    Used on the trajectory strokes so they read as a diffuse haze of transport
    rather than a web of hairlines; falls back to identity without SciPy.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ModuleNotFoundError:
        return None

    def _filter(image, dpi):
        sigma = sigma_pt * dpi / 72.0
        return gaussian_filter(image.astype(float), (sigma, sigma, 0)), 0, 0

    return _filter


def _decade_ticks(axis):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    axis.yaxis.set_minor_formatter(NullFormatter())


def _fit_slope(h, value):
    """Least-squares log-log slope, for an unlabelled trend guide only."""
    finite = np.isfinite(value) & (value > 0)
    return float(np.polyfit(np.log(h[finite]), np.log(value[finite]), 1)[0])


def _coupling_figure(root, figure_dir):
    """Squared coupling and its root, the discrete-``W_2`` upper bound, together."""
    rows = sorted(_rows(root / 'data/coupling.csv'), key=lambda r: float(r['h']))
    h = np.array([float(r['h']) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for axis, key, se_key, ylabel, label, index in (
        (axes[0], 'mean_squared_coupling', 'se_squared_coupling',
         r'$\mathbb{E}\,C_2(\omega)$', 'squared coupling', 0),
        (axes[1], 'mean_rms_coupling', 'se_rms_coupling',
         r'$\mathbb{E}\,\sqrt{C_2(\omega)}$', r'$W_2$ upper bound', 2),
    ):
        value = np.array([float(r[key]) for r in rows])
        se = np.array([float(r[se_key]) for r in rows])
        _band(axis, h, value, se, color(index))
        reference_line(axis, h, value[-1], h[-1], _fit_slope(h, value))
        axis.set(xscale='log', yscale='log', xlabel='Switching interval $h$', ylabel=ylabel)
        dyadic_ticks(axis, h); grid(axis, which='major'); _decade_ticks(axis)
        headroom(axis, right=.5, top=.12)
        annotate(axis, h[-1], value[-1], label, color=color(index))
    return save(fig, figure_dir, 'transport_coupling')


def _density_figure(root, figure_dir):
    """Pointwise density error at the predeclared probes and the global ``L^1`` error."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    points = _rows(root / 'data/density_pointwise.csv')
    labels = []
    for index in sorted({int(r['point_index']) for r in points}):
        selected = sorted((r for r in points if int(r['point_index']) == index),
                          key=lambda r: float(r['h']))
        h = np.array([float(r['h']) for r in selected])
        value = np.array([float(r['density_mse']) for r in selected])
        se = np.array([float(r['se_density_mse']) for r in selected])
        _band(axes[0], h, value, se, color(index))
        distance = np.hypot(float(selected[0]['initial_x1']) + 1.0,
                            float(selected[0]['initial_x2']) + 1.0)
        labels.append((value[-1], rf'$\|y-c\|={distance:.1f}$', color(index)))
    axes[0].set(xscale='log', yscale='log', xlabel='Switching interval $h$',
                ylabel=r'$\mathbb{E}\,|\rho_T(x)-\hat\rho_T(x)|^2$')
    dyadic_ticks(axes[0], h); grid(axes[0], which='major'); _decade_ticks(axes[0])
    headroom(axes[0], right=.5, top=.1)
    stack_labels(axes[0], labels, h[-1])

    rows = sorted(_rows(root / 'data/density_l1.csv'), key=lambda r: float(r['h']))
    hl = np.array([float(r['h']) for r in rows])
    value = np.array([float(r['expected_l1_error']) for r in rows])
    se = np.array([float(r['se_l1_error']) for r in rows])
    _band(axes[1], hl, value, se, color(2))
    unresolved = np.array([str(r.get('l1_resolved_above_quadrature_floor', 'true')).lower() != 'true'
                           for r in rows])
    axes[1].plot(hl[unresolved], value[unresolved], 'o', mfc='white', mec=color(2))
    reference_line(axes[1], hl, value[-1], hl[-1], _fit_slope(hl, value))
    floors = [float(r['quadrature_floor']) for r in rows if r.get('quadrature_floor')]
    if floors:
        # The refinement floor varies with h; the flat guide marks its smallest
        # value, i.e. the most optimistic resolution these estimates could claim.
        axes[1].axhline(min(floors), ls=':', color=MUTED, lw=.8)
    axes[1].set(xscale='log', yscale='log', xlabel='Switching interval $h$',
                ylabel=r'$\mathbb{E}\,\|\rho_T-\hat\rho_T\|_{L^1}$')
    dyadic_ticks(axes[1], hl); grid(axes[1], which='major'); _decade_ticks(axes[1])
    headroom(axes[1], right=.42, top=.1)
    annotate(axes[1], hl[-1], value[-1], r'$L^1$ error', color=color(2))
    return save(fig, figure_dir, 'transport_density')


def generate_plots(output_dir: str | Path, *, figure_dir=None) -> list[Path]:
    use_paper_style()
    root = Path(output_dir).expanduser().resolve()
    figure_dir = Path(figure_dir) if figure_dir else root / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    return _coupling_figure(root, figure_dir) + _density_figure(root, figure_dir)


def _disk_mesh(nr, na):
    """Connected polar mesh for density rendering, including its zero boundary."""
    angles = 2 * np.pi * np.arange(na) / na
    rings = np.arange(1, nr + 1) / nr
    points = np.vstack(([[-1., -1.]],
                        (-1 + rings[:, None, None] * np.stack((np.cos(angles), np.sin(angles)), axis=1)).reshape(-1, 2)))
    triangles = [(0, 1+j, 1+(j+1) % na) for j in range(na)]
    for ring in range(nr-1):
        a, b = 1 + ring*na, 1 + (ring+1)*na
        for j in range(na):
            k = (j+1) % na
            triangles.extend(((a+j, b+j, b+k), (a+j, b+k, a+k)))
    return points, np.asarray(triangles)


def generate_qualitative(classification_run, transport_run, figure_dir, *,
                         classification_figure_dir=None,
                         source_illustration=None, h=2**-8, overwrite=False,
                         classification_seed=20260908, transport_seed=20260909,
                         sample_seed=20260910, n_transport=1024,
                         density_radial_nodes=48, density_angles=192):
    """Separate panels, using all saved classification splits and a density mesh.

    Only new points are integrated when source_illustration is supplied. The
    old transport sample and both old schedules are reused without selection.
    """
    import time
    import torch
    from experiments._paper_common import load_checkpoint
    from rnode.batches import make_uniform_fixed_size, sample_batch_sequence
    from rnode.data import initial_density, sample_initial_compact
    from rnode.flow import Flow
    from rnode.transport import integrate_characteristics, terminal_log_density_from_forward

    if density_radial_nodes < 2 or density_angles < 4 or density_radial_nodes % 2 or density_angles % 2:
        raise ValueError('Use an even density mesh with at least two radial rings and four angles')
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    # The classification cloud belongs with the run that trained the classifier.
    classification_figure_dir = (
        Path(classification_figure_dir) if classification_figure_dir else figure_dir
    )
    classification_figure_dir.mkdir(parents=True, exist_ok=True)
    config_path = figure_dir / 'qualitative_config.json'
    if config_path.exists() and not overwrite:
        raise FileExistsError(f'Choose a new directory; {config_path} already exists')
    cr, tr = Path(classification_run).resolve(), Path(transport_run).resolve()
    cc, tc = [json.loads((r / 'config/config.json').read_text()) for r in (cr, tr)]
    checkpoints = [cr / 'checkpoints/base_model.pt', tr / 'checkpoints/flow.pt']
    sources = checkpoints + [cr / 'data/dataset_splits.npz', cr / 'config/config.json', tr / 'config/config.json']
    previous = {}
    if source_illustration is not None:
        source = Path(source_illustration)
        old = json.loads((source / 'qualitative_config.json').read_text())
        for key, value in (('h', h), ('classification_schedule_seed', classification_seed),
                           ('transport_schedule_seed', transport_seed), ('transport_sample_seed', sample_seed),
                           ('transport_samples', n_transport)):
            if old[key] != value:
                raise ValueError(f'The saved illustration has a different {key}')
        for p in checkpoints:
            if old['provenance'][str(p)] != hashlib.sha256(p.read_bytes()).hexdigest():
                raise ValueError(f'Illustration checkpoint mismatch: {p}')
        with np.load(source / 'qualitative_samples.npz') as a:
            previous = {k: a[k].copy() for k in a.files}
        sources += [source / 'qualitative_samples.npz', source / 'qualitative_config.json']
    cfgs = [cc['experiment'], tc['experiment']]
    numerics = [dict(T=1., r=r, full_dt=c['reference_dt'], random_dt=h/s)
                for r, c, s in zip((8, 16), cfgs,
                    (cfgs[0]['rk_steps_per_switch'], cfgs[1]['steps_per_switch']))]
    config = dict(h=h, classification_schedule_seed=classification_seed,
                  transport_schedule_seed=transport_seed, transport_sample_seed=sample_seed,
                  transport_samples=n_transport, classification_run=str(cr), transport_run=str(tr),
                  classification_points='all saved splits, test then train then calibration; illustrative only',
                  density_method='initial density and learned full-flow density on a mapped mesh; no KDE or renormalization',
                  density_mesh=dict(radial_nodes=density_radial_nodes, angles=density_angles),
                  numerics=numerics, selection='unchanged predeclared schedules; no selection by appearance',
                  provenance={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    started = time.perf_counter()
    model, _ = load_checkpoint(checkpoints[0], device=torch.device('cpu'), requested_dtype=torch.float64)
    flow = Flow(dim=2, hidden=tc['experiment']['hidden']).double().eval()
    flow.load_state_dict(torch.load(checkpoints[1], map_location='cpu', weights_only=False)['state_dict'])
    with np.load(cr / 'data/dataset_splits.npz') as ds:
        xclass = np.concatenate([ds[s+'_X'] for s in ('test', 'train', 'calibration')])
        labels = np.concatenate([ds[s+'_labels'].reshape(-1) for s in ('test', 'train', 'calibration')])
    xtransport = previous.get('transport_initial')
    if xtransport is None:
        xtransport = sample_initial_compact(n_transport, seed=sample_seed)
    arrays = dict(classification_labels=labels, classification_targets=np.array([[-1., 0.], [0., 1.]]))
    diagnostics = []
    with torch.no_grad():
        for name, field, points, p, r, seed, cfg in zip(
            ('classification', 'transport'), (model, flow), (xclass, xtransport),
            (24, tc['experiment']['hidden']), (8, 16),
            (classification_seed, transport_seed), numerics):
            scheme = make_uniform_fixed_size(p, r)
            schedule = sample_batch_sequence(scheme, round(1/h), np.random.default_rng(seed))
            n_old = len(previous[name+'_initial']) if name+'_initial' in previous else 0
            if n_old:
                if not np.array_equal(previous[name+'_initial'], points[:n_old]):
                    raise ValueError(f'Existing points differ: {name}')
                if not np.array_equal(previous[name+'_schedule'], np.asarray(schedule)):
                    raise ValueError(f'Existing schedule differs: {name}')
            full, random = [previous.get(name+'_'+stage, np.empty((0, 2))) for stage in ('full', 'random')]
            if n_old < len(points):
                x = torch.as_tensor(points[n_old:], dtype=torch.float64)
                terminal, _ = integrate_characteristics(field, x, 1., cfg['full_dt'], 1.)
                rb, _ = integrate_characteristics(field, x, 1., cfg['random_dt'], h, schedule,
                                                  inclusion_probs=scheme.inclusion_probs)
                full = np.concatenate((full, terminal.numpy()))
                random = np.concatenate((random, rb.numpy()))
                refined, _ = integrate_characteristics(field, x, 1., cfg['random_dt']/2, h, schedule,
                                                        inclusion_probs=scheme.inclusion_probs)
                diagnostics.append(dict(row=name, new_points=len(x), random_dt_halving_max_displacement=float(
                    torch.linalg.vector_norm(rb-refined, dim=1).max())))
            arrays.update({name+'_initial': points, name+'_full': full,
                           name+'_random': random, name+'_schedule': np.asarray(schedule)})
            print(f'Illustration: {name}, {len(points)} points, {n_old} reused', flush=True)
        mesh, triangles = _disk_mesh(density_radial_nodes, density_angles)
        terminal, inc = integrate_characteristics(flow, torch.as_tensor(mesh), 1., numerics[1]['full_dt'], 1., track_log_density=True)
        rho = np.exp(terminal_log_density_from_forward(initial_density, mesh, inc.numpy()))
        arrays.update(density_initial_mesh=mesh, density_full_mesh=terminal.numpy(), density_triangles=triangles,
                      density_initial_values=initial_density(mesh), density_full_values=rho)
        # Compare a nested coarse mesh using the SAME characteristic evaluations.
        # This isolates interpolation/mesh sensitivity, not integration error.
        nr, na = density_radial_nodes, density_angles
        if nr % 2 or na % 2:
            raise ValueError('Density mesh sizes must be even for the nested check')
        coarse_indices = np.r_[0, (1 + np.arange(1, nr, 2)[:, None]*na + np.arange(0, na, 2)).ravel()]
        _, coarse_triangles = _disk_mesh(nr//2, na//2)
        for stage in ('initial', 'full'):
            xy, values = arrays['density_'+stage+'_mesh'], arrays['density_'+stage+'_values']
            mass = []
            for indices, tri in ((np.arange(len(xy)), triangles), (coarse_indices, coarse_triangles)):
                v = xy[indices][tri]
                u, w = v[:, 1]-v[:, 0], v[:, 2]-v[:, 0]
                area = .5*(u[:, 0]*w[:, 1]-u[:, 1]*w[:, 0])
                if np.any(area <= 0):
                    raise ValueError('Folded density rendering mesh; refine before plotting')
                mass.append(float(np.sum(area * values[indices][tri].mean(axis=1))))
            diagnostics.append(dict(density=stage, fine_mesh_mass=mass[0], coarse_mesh_mass=mass[1],
                                    mass_refinement_difference=mass[0]-mass[1]))
    np.savez_compressed(figure_dir/'qualitative_samples.npz', **arrays)
    config.update(classification_samples=len(xclass), diagnostics=diagnostics,
                  elapsed_seconds=time.perf_counter()-started,
                  density_color_limits=[0., float(max(arrays['density_initial_values'].max(), rho.max()))])
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    return _qualitative_figure(arrays, figure_dir, classification_figure_dir)


def _qualitative_figure(arrays, figure_dir, classification_figure_dir=None):
    """One PDF per panel; titles and color/marker explanations stay in captions."""
    import matplotlib.tri as mtri
    from matplotlib.ticker import MaxNLocator

    use_paper_style()
    outputs = []
    stages = ('initial', 'full', 'random')

    # Only the initial classification cloud is drawn here: the flow itself is
    # shown as trajectories, which carry the same points and more.
    xy = arrays['classification_initial']
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    center, radius = (lo + hi) / 2, .58 * max(hi - lo)
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    ax.scatter(*xy.T, s=15, alpha=.95, zorder=3, edgecolors='#23262c', linewidths=.7,
               c=np.where(arrays['classification_labels'].reshape(-1) > 0, color(1), color(0)))
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
           aspect='equal', xlabel='$x_1$', ylabel='$x_2$')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    outputs += save(fig, classification_figure_dir or figure_dir, 'classification_initial')
    if 'transport_initial' not in arrays:
        return outputs
    points = np.vstack([arrays['transport_'+s] for s in stages] +
                       [arrays['density_initial_mesh'], arrays['density_full_mesh']])
    lo, hi = points.min(axis=0), points.max(axis=0)
    padding = .055*(hi-lo)
    limits = dict(xlim=(lo[0]-padding[0], hi[0]+padding[0]), ylim=(lo[1]-padding[1], hi[1]+padding[1]),
                  aspect='equal', xlabel='$x_1$', ylabel='$x_2$')
    for stage in ('densities', 'full', 'random'):
        fig, ax = plt.subplots(figsize=(3.7, 3.0))
        if stage == 'densities':
            # Filled level sets with thin black isolines. The initial and
            # terminal densities share one positive-floor level scale, so the
            # terminal blob reads as lighter because its mass has spread, not
            # because it was rescaled.
            vmax = max(arrays['density_initial_values'].max(), arrays['density_full_values'].max())
            levels = np.linspace(.05 * vmax, vmax, 11)
            ax.set_facecolor(plt.get_cmap('Oranges')(.05))
            for name in ('initial', 'full'):
                tri = mtri.Triangulation(*arrays['density_'+name+'_mesh'].T, arrays['density_triangles'])
                ax.tricontourf(tri, arrays['density_'+name+'_values'], levels=levels,
                               cmap='Oranges', extend='max')
                ax.tricontour(tri, arrays['density_'+name+'_values'], levels=levels,
                              colors='#1a1a1a', linewidths=.35, alpha=.8)
        else:
            for name, tint in (('initial', color(0)), (stage, color(1))):
                ax.scatter(*arrays['transport_'+name].T, c=tint, s=4, alpha=.45, linewidths=0)
        ax.set(**limits)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        outputs += save(fig, figure_dir, 'transport_'+('densities' if stage=='densities' else stage+'_samples'))
    return outputs


def generate_classification_trajectories(
    classification_run, figure_dir, *, h=2**-11, schedule_seed=20260908,
    source_illustration=None, overwrite=False,
):
    """Render real full/RBM trajectories at the smallest classification h.

    The saved states are the actual RK4 states at switching boundaries; no
    straight-line interpolation is used.  The model and points are frozen.
    """
    import torch
    from experiments._paper_common import load_checkpoint
    from rnode.batches import make_uniform_fixed_size, sample_batch_sequence
    from rnode.integrators import integrate_fixed_grid

    root = Path(classification_run).resolve()
    out = Path(figure_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "classification_trajectory_config.json"
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"Choose a new directory; {config_path} already exists")
    if source_illustration is None:
        raise ValueError("source_illustration is required for the fixed sample cloud")
    source = Path(source_illustration).resolve()
    with np.load(source / "qualitative_samples.npz") as saved:
        points = saved["classification_initial"].copy()
        labels = saved["classification_labels"].reshape(-1).copy()
    model, payload = load_checkpoint(root / "checkpoints/base_model.pt",
                                     device=torch.device("cpu"), requested_dtype=torch.float64)
    model.eval()
    scheme = make_uniform_fixed_size(24, 8)
    n_intervals = round(1.0 / h)
    schedule = sample_batch_sequence(scheme, n_intervals,
                                     np.random.default_rng(schedule_seed))
    x0 = torch.as_tensor(points, dtype=torch.float64)
    dt = h / 8
    with torch.no_grad():
        times, full = integrate_fixed_grid(model, x0, 1.0, dt, h, method="rk4")
        _, random = integrate_fixed_grid(
            model, x0, 1.0, dt, h, schedule,
            inclusion_probs=scheme.inclusion_probs, method="rk4",
        )
    # Keep the initial point and each switching boundary: these are genuine
    # integrator states and show the schedule-induced path without 8193 frames.
    stride = 8
    arrays = {
        "times": times[::stride].numpy(),
        "full": full[::stride].numpy(),
        "random": random[::stride].numpy(),
        "initial": points,
        "labels": labels,
        "targets": np.array([[-1.0, 0.0], [0.0, 1.0]]),
        "schedule": np.asarray(schedule),
    }
    np.savez_compressed(out / "classification_trajectories.npz", **arrays)
    config = {
        "h": h, "dt": dt, "steps_per_switch": 8,
        "schedule_seed": schedule_seed, "sampling": "uniform fixed size r=8",
        "n_points": len(points), "n_switching_intervals": n_intervals,
        "states_saved": "initial plus every switching boundary; all are RK4 states",
        "checkpoint": str(root / "checkpoints/base_model.pt"),
        "source_points": str(source / "qualitative_samples.npz"),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    use_paper_style()
    colors = np.where(labels > 0, color(1), color(0))
    # Targets are stored for provenance but deliberately not drawn, and are
    # excluded from the view so they cannot stretch it.
    all_points = np.concatenate((arrays["full"].reshape(-1, 2), arrays["random"].reshape(-1, 2)))
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    center, radius = (lo + hi) / 2, .54 * max(hi - lo)
    outputs = []
    from matplotlib.collections import LineCollection
    for key in ("full", "random"):
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        path = arrays[key]
        # True trajectories as a soft blurred bundle, colored by class;
        # endpoints remain opaque on top.
        for cls, tint in ((-1, color(0)), (1, color(1))):
            selected = np.flatnonzero(labels == cls)
            bundle = LineCollection([path[:, j, :] for j in selected], colors=[tint],
                                    linewidths=.8, alpha=.10, capstyle="round",
                                    zorder=1, rasterized=True)
            blur_filter = _gaussian_blur(2.0)
            if blur_filter is not None:
                bundle.set_agg_filter(blur_filter)
            ax.add_collection(bundle)
        # A dark rim keeps individual endpoints separable over the paths.
        ax.scatter(*path[-1].T, c=colors, s=15, alpha=.95,
                   edgecolors="#23262c", linewidths=.7, zorder=3)
        ax.set(xlim=(center[0] - radius, center[0] + radius),
               ylim=(center[1] - radius, center[1] + radius), aspect="equal",
               xlabel="$x_1$", ylabel="$x_2$")
        outputs += save(fig, out, "classification_trajectory_" + key)
    return outputs


def generate_transport_density_evolution(
    transport_run, figure_dir, *, source_illustration, h=2**-8, overwrite=False,
    schedule_seed=20260909, sample_seed=20260909, n_kde_times=4,
    n_samples=20000, grid_size=260,
):
    """Overlay the transported density at several times as Oranges level sets.

    All times are drawn on one axes. The time-zero field is the analytic
    compact density of the paper, and each later time is a Gaussian KDE of the
    same empirical sample propagated by the flow. A single level set is shared
    by every time, so the visible lightening of the intermediate blobs is the
    physical spreading of mass, not a per-panel rescaling.
    """
    import torch
    from experiments._paper_common import load_checkpoint
    from rnode.batches import make_uniform_fixed_size, sample_batch_sequence
    from rnode.data import initial_density, sample_initial_compact
    from rnode.flow import Flow
    from rnode.integrators import integrate_fixed_grid

    source = Path(source_illustration).resolve()
    out = Path(figure_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "transport_density_evolution_config.json"
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"Choose a new directory; {config_path} already exists")
    with np.load(source / "qualitative_samples.npz") as saved:
        schedule = saved["transport_schedule"].copy()
    run = Path(transport_run).resolve()
    cfg = json.loads((run / "config/config.json").read_text())["experiment"]
    if len(schedule) != round(1 / h):
        raise ValueError("The saved schedule does not match the requested h")
    model_payload = torch.load(run / "checkpoints/flow.pt", map_location="cpu", weights_only=False)
    model = Flow(dim=2, hidden=cfg["hidden"]).double().eval()
    model.load_state_dict(model_payload["state_dict"])
    scheme = make_uniform_fixed_size(cfg["hidden"], 16)
    # Verify that the saved schedule is the predeclared seed sequence.
    expected = sample_batch_sequence(scheme, round(1 / h), np.random.default_rng(schedule_seed))
    if not np.array_equal(np.asarray(expected), schedule):
        raise ValueError("Saved transport schedule does not match its declared seed")

    # A KDE needs many more particles than the 1024-point illustration cloud,
    # so we draw a fresh, larger sample from the exact initial density.
    points = sample_initial_compact(n_samples, rng=np.random.default_rng(sample_seed))
    x0 = torch.as_tensor(points, dtype=torch.float64)
    dt = h / cfg["steps_per_switch"]
    with torch.no_grad():
        times, full = integrate_fixed_grid(model, x0, 1., dt, h, method="rk4")
        _, random = integrate_fixed_grid(model, x0, 1., dt, h, schedule,
                                         inclusion_probs=scheme.inclusion_probs, method="rk4")
    # Time zero is drawn from the analytic density, so the KDE times are the
    # n_kde_times equispaced instants that end at T.
    selected = np.linspace(0, len(times) - 1, n_kde_times + 1, dtype=int)[1:]
    times_np = times.numpy()[selected]
    full_np, random_np = full.numpy()[selected], random.numpy()[selected]

    all_points = np.concatenate((points, full_np.reshape(-1, 2), random_np.reshape(-1, 2)))
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    pad = .06 * np.maximum(hi - lo, 1e-8)
    xgrid = np.linspace(lo[0] - pad[0], hi[0] + pad[0], grid_size)
    ygrid = np.linspace(lo[1] - pad[1], hi[1] + pad[1], grid_size)
    xx, yy = np.meshgrid(xgrid, ygrid)
    query_points = np.vstack((xx.ravel(), yy.ravel())).T

    def kde(state):
        """Isotropic Gaussian KDE with Silverman-type bandwidth, in chunks."""
        bandwidth = max(float(np.mean(np.std(state, axis=0, ddof=1)) * len(state) ** (-1 / 6)), 1e-6)
        estimate = np.zeros(len(query_points), dtype=float)
        for start in range(0, len(query_points), 1024):
            block = query_points[start:start + 1024]
            delta = block[:, :, None] - state.T[None, :, :]
            estimate[start:start + 1024] = np.exp(
                -.5 * np.sum(delta * delta, axis=1) / bandwidth**2).mean(axis=1)
        return (estimate / (2 * np.pi * bandwidth**2)).reshape(grid_size, grid_size), bandwidth

    initial_field = initial_density(np.stack((xx, yy), axis=-1))
    fields, bandwidths = {}, {}
    for name, trajectories in (("full", full_np), ("random", random_np)):
        values, widths = [], []
        for state in trajectories:
            field, bandwidth = kde(state)
            values.append(field)
            widths.append(bandwidth)
        fields[name] = np.asarray(values)
        bandwidths[name] = widths

    payload = dict(times=times_np, full=full_np, random=random_np, initial_sample=points,
                   xgrid=xgrid, ygrid=ygrid, initial_density=initial_field,
                   full_density=fields["full"], random_density=fields["random"],
                   schedule=schedule)
    np.savez_compressed(out / "transport_density_evolution.npz", **payload)
    config_path.write_text(json.dumps({
        "h": h, "dt": dt, "schedule_seed": schedule_seed, "sample_seed": sample_seed,
        "n_samples": int(n_samples), "n_kde_times": int(n_kde_times),
        "kde_times": times_np.tolist(), "grid_size": grid_size,
        "kde_bandwidths": bandwidths,
        "density_initial": "analytic compact density from the paper, not a KDE",
        "density_later_times": "isotropic Gaussian KDE of propagated equal-weight samples",
        "levels": "one level set shared by every time and by both models",
        "same_initial_sample": True, "same_schedule_for_all_points": True,
        "transport_run": str(run), "source_illustration": str(source),
    }, indent=2) + "\n")

    use_paper_style()
    # One level set for every time and both models, so intensity is comparable.
    # The lowest level is strictly positive: with levels starting at zero every
    # contourf would repaint the whole axes in its lightest band and erase the
    # fill of the times drawn before it, leaving only their contour lines.
    vmax = float(max(initial_field.max(), fields["full"].max(), fields["random"].max()))
    floor = .04 * vmax
    levels = np.linspace(floor, vmax, 12)
    background = plt.get_cmap("Oranges")(.05)

    def draw(axis, field):
        axis.contourf(xx, yy, field, levels=levels, cmap="Oranges", extend="max")
        axis.contour(xx, yy, field, levels=levels, colors="#1a1a1a",
                     linewidths=.4, alpha=.85)

    outputs = []
    for name, stem in (("full", "transport_density_full_evolution"),
                       ("random", "transport_density_random_evolution")):
        fig, axis = plt.subplots(figsize=(6.4, 4.4))
        draw(axis, initial_field)
        for field in fields[name]:
            draw(axis, field)
        axis.set(xlim=(xgrid[0], xgrid[-1]), ylim=(ygrid[0], ygrid[-1]),
                 aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
        axis.set_facecolor(background)
        outputs += save(fig, out, stem)

    # Initial and terminal full density on the same axes and the same levels.
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    draw(axis, initial_field)
    draw(axis, fields["full"][-1])
    axis.set(xlim=(xgrid[0], xgrid[-1]), ylim=(ygrid[0], ygrid[-1]),
             aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
    axis.set_facecolor(background)
    outputs += save(fig, out, "transport_density_initial_final")
    return outputs
