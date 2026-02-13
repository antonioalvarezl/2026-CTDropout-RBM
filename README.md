# Random Batch Neural ODEs (rNODE)

<p align="center">
  <img src="assets/rnode_header.gif" alt="Neural ODE vs rNODE (dropout)" width="700">
</p>

Companion code for

> A. Álvarez-López and M. Hernández, *Convergence, design and training of continuous-time dropout as a random batch method*, arXiv preprint [arXiv:2510.13134](https://arxiv.org/abs/2510.13134), 2025.

## What is this?

**Neural ODEs** model depth as continuous integration of a learned vector field.  **Dropout** — randomly silencing neurons during forward passes — is a standard regulariser in discrete networks, but applying it in continuous time raises new challenges: naïve masking can break the ODE solver's convergence guarantees.

This work frames **continuous-time dropout as a random batch method (RBM)**.  At each time interval of length *h* a random subset of neurons is sampled and the output is rescaled to keep the estimator unbiased (Horvitz–Thompson correction).  We prove:

| Result | Rate | Reference |
|--------|------|-----------|
| Trajectory error (expected uniform) | *O*(*h*) | Theorem 1 |
| Measure-transport error (total variation) | *O*(√*h*) | Corollary 3 |
| Training cost / control deviation | bounded by *h* | §3.3, Pontryagin adjoint |
| Optimal switching step *h** | closed-form | §4.2 |

Standard **Bernoulli dropout** is recovered as a special case of the RBM framework.  We compare seven canonical batch sampling schemes and derive explicit dependence of the error constants on the minimum inclusion probability π_min.

The experiments validate the theory on a single-layer Neural ODE applied to **classification** (two-circles) and **flow matching** (Gaussian → tri-modal density transport).

## Repository structure

```
rnode/                        Core library
├── models.py                 All model architectures (see table below)
├── batches.py                7 canonical batch sampling schemes (§4.1)
├── flow.py                   Flow matching: Flow, FlowRBM, integration, KDE
├── data.py                   Dataset generation (two-circles, flow densities, mesh)
└── utils.py                  Metrics, timing, batch generation helpers

experiments/                  Notebooks + figure-generation modules
├── exp1_forward.ipynb        §5.1 — Forward-pass convergence & scheme comparison
├── exp1_plots.py
├── exp2_flow.ipynb           §5.2 — Flow matching / measure transport
├── exp2_plots.py
├── exp3a_training.ipynb      §5.3 — Training (all params trainable)
├── exp3a_plots.py
├── exp3b_training.ipynb      §5.3 — Training (fixed inner weights)
└── exp3b_plots.py

assets/                       README header animation
generate_readme_gif.py        Script to regenerate the GIF
outputs/                      Generated figures and data (created on run)
captions.tex                  LaTeX figure captions for the paper
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tested with Python 3.10+.

### Quick troubleshooting

- `A module that was compiled using NumPy 1.x cannot be run in NumPy 2...`
  means your current PyTorch build is not compatible with NumPy 2. Use:
  `python -m pip install "numpy<2"`.
- `ModuleNotFoundError: No module named 'rnode'` means the local package is
  not installed in the active environment. Run:
  `python -m pip install -e .`.

To regenerate the header GIF (does not require PyTorch):

```bash
python generate_readme_gif.py
```

## Experiments

Each experiment lives in a notebook that sets the configuration, trains the
model, and calls figure functions from the companion `*_plots.py` module.
Set `QUICK = True` in the configuration cell for a fast test run.

All figures are saved as PDF + PNG to `outputs/`.  Figures have no titles or
legends; the corresponding LaTeX captions are in `captions.tex`.

### Experiment 1 — Forward pass (§5.1)

Validates Theorem 1 (*O*(*h*) trajectory convergence), compares batch
sampling schemes, and benchmarks runtime.

| Output | Description |
|--------|-------------|
| `fig1_trajectories` | Dataset + NODE/rNODE dynamics (time-indep & time-dep) |
| `fig2_convergence` | Log-log convergence: error vs h (validates Theorem 1) |
| `fig3_decision_{const,tdep}` | Decision boundaries at h ∈ {1, 2, 3}Δt |
| `fig4_benchmark_{const,tdep}` | Wall-clock time vs dataset size |
| `fig5_cost_vs_error` | Scatter: cost vs error across 5 schemes, h ∈ {0.001, 0.01, 0.1} |
| `fig6_scheme_convergence` | Convergence rate comparison across batch schemes |
| `fig7_variance_by_scheme` | Empirical variance by scheme (validates Λ_t) |
| `fig8_pareto` | Pareto front: error vs batch size r at fixed h |
| `fig9_optimal_h` | Theoretical vs empirical optimal h* (validates §4.2) |
| `fig10_error_constant_vs_pimin` | Error constant S vs π_min across all schemes |
| `fig11_speedup` | Wall-clock speedup vs π = r/p |
| `table1_batch_counts.json` | Accuracy/loss/time for different batch counts |

### Experiment 2 — Flow matching (§5.2)

Validates Corollary 3 (*O*(√*h*) L¹ convergence) for the continuity equation.

| Output | Description |
|--------|-------------|
| `fig_flow1_data` | Initial and target distributions |
| `fig_flow2_density` | Full-model density evolution (t = 0, 0.5, 1) |
| `fig_flow2_trajectories` | Flow lines from initial to final |
| `fig_flow3_comparison` | Full model vs rNODE averaged density |
| `fig_flow4_convergence` | L¹ convergence log-log (theoretical slope 0.5) |
| `fig_flow5_scheme_conv` | Scheme convergence comparison (L¹ metric) |
| `fig_flow6_variance` | Empirical variance of L¹ error by batch count |
| `fig_flow7_benchmark` | Forward-pass time vs number of batches |

### Experiment 3a — Training, all parameters (§5.3)

Both layers trainable.  Each rNODE realisation uses a **fixed** batch schedule
(structured pruning).  Ensemble averaging over K realisations.

| Output | Description |
|--------|-------------|
| `ex3a_node_dynamics` | Full NODE trajectories |
| `ex3a_rnode_dynamics` | rNODE ensemble-averaged trajectories |
| `ex3a_decision_boundaries` | NODE + rNODE at h ∈ {1, 2, 3}Δt |
| `ex3a_benchmark` | Train/test accuracy: NODE vs rNODE (2, 3, 4, 6 batches) |

### Experiment 3b — Training, fixed inner weights (§5.3)

Only W₂(t) is trainable; W₁, b₁ are frozen random features.

| Output | Description |
|--------|-------------|
| `ex3b_scaling` | Accuracy / time / params vs hidden dimension p |
| `ex3b_decision_boundaries` | NODE + rNODE at h ∈ {1, 2, 3}Δt |
| `ex3b_benchmark` | Train/test accuracy: NODE vs rNODE (2, 3, 4, 6 batches) |

## Model architectures (`rnode/models.py`)

| Class | Description | Experiment |
|-------|-------------|------------|
| `ConstantODE` | Time-independent NODE | §5.1 |
| `TimeDepODE` | Time-dependent W(t), b(t) with ReLU | §5.1 |
| `RBMConstant` | RBM inference for ConstantODE | §5.1 |
| `RBMTimeDep` | RBM inference for TimeDepODE | §5.1, §5.3 |
| `TimeDepODE_ELU` | Time-dependent with ELU (trainable) | §5.3a |
| `RBMTrainableODE` | Trainable RBM with fixed schedule | §5.3a |
| `FixedInnerODE` | Frozen W₁, b₁ + trainable W₂(t) | §5.3b |
| `FixedInnerRBM` | RBM variant of FixedInnerODE | §5.3b |

## Batch sampling schemes (`rnode/batches.py`)

| # | Scheme | Constructor | π_min |
|---|--------|-------------|-------|
| 1 | Single-batch | `make_single_batch(p)` | 1 |
| 2 | Drop-one | `make_drop_one(p)` | (p−1)/p |
| 3 | Pick-one | `make_pick_one(p)` | 1/p |
| 4 | Balanced subsets | `make_balanced_subsets(p, r)` | r/p |
| 5 | Balanced disjoint | `make_balanced_disjoint(p, r)` | r/p |
| 5' | Interleaved | `make_interleaved(p, n_batches)` | ⌊p/n⌋/p |
| 7 | Bernoulli dropout | `make_bernoulli(p, q_B)` | q_B |

## Citation

```bibtex
@article{alvarez2025dropout,
  title     = {Convergence, design and training of continuous-time dropout
               as a random batch method},
  author    = {{\'A}lvarez-L{\'o}pez, Antonio and Hern{\'a}ndez, Mart{\'\i}n},
  journal   = {arXiv preprint arXiv:2510.13134},
  year      = {2025},
  url       = {https://arxiv.org/abs/2510.13134}
}
```

## License

MIT License
