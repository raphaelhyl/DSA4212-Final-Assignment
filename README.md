# BO vs CMA-ES: Ambient Dimension, Effective Dimension and Active-Subspace Alignment

A computational comparison of Gaussian-process Bayesian optimisation (BO) and CMA-ES on synthetic black-box
minimisation problems. The study varies the ambient dimension D, the effective dimension k, and the orientation of a
low-dimensional active subspace, and measures evaluation efficiency (best-so-far regret) and optimiser runtime.

- Report: `REPORT.md` (results and methods summary)
- Author: `AUTHOR.md`

## Repository structure

| Path | Contents |
|---|---|
| `benchmarks.py` | Benchmark functions: shifted active-subspace quadratic and embedded Rosenbrock; aligned, random and partial rotations |
| `objective.py` | `BudgetedObjective` wrapper: counts evaluations, enforces the budget, checks feasibility, logs every call |
| `optimizers/` | BO-ARD, BO-iso, CMA-ES and random-search implementations |
| `config.py` | Experiment definitions, budgets, seeds and algorithm settings |
| `run.py` | Runs the experiments and writes raw per-run results |
| `make_figures.py` | Builds all summary tables and figures from the raw results |
| `analysis/` | Statistics (paired tests, bootstrap CIs, ERT, ECDF) and diagnostics |
| `report/` | Supplementary scripts (e.g. `supp_gp_noise_check.py`) |
| `tests/` | Validation tests for the benchmarks and budget wrapper |
| `requirements.txt`, `pyproject.toml` | Dependencies |

## Environment and dependencies

Results in the report were produced with:

| Component | Version |
|---|---|
| Python | 3.13.13 |
| PyTorch | 2.15.0 |
| BoTorch | 0.18.1 |
| GPyTorch | 1.15.2 |
| pycma (`cma`) | 4.5.0 |
| NumPy | (see `requirements.txt`) |

Hardware: Apple M5 Pro (18 cores), one thread per process, 15–20 processes run concurrently. Runtimes will differ
on other machines; regret results do not depend on hardware beyond floating-point differences.

Set up a virtual environment and install the dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data preparation

No external data are needed. All benchmark instances are synthetic and generated deterministically from seeds:

- The optimum x\* is drawn from Uniform[−0.5, 0.5]^D and depends only on (seed, D).
- The active-subspace basis A is axis-aligned, a seeded Gaussian QR rotation, or a partial rotation by angle θ.
- Each seed has separate instance and algorithm random streams (`numpy.random.SeedSequence`), so every algorithm
  sees the identical instance for a given seed.

Seed ranges: final runs 0–19, pilot 1000–1004, sensitivity 2000–2019.

Before running experiments, check that instances are generated correctly:

```bash
pytest tests/
```

The tests check orthonormality of A, f ≥ 0, f(x\*) = 0, invariance along null(Aᵀ), principal angles of the partial
rotation, and seed reproducibility.

## Reproducing the main results

**1. Run the experiments.** This produces the raw per-run logs in `results/raw/`:

```bash
python run.py
```

Use `python run.py --help` to see the options for running a single experiment or seed range. The full study is
1,220 runs (540 of them BO). BO runs are slow: about 1.5–8 minutes each at D = 20, and the 500-evaluation extended
runs are longer still. CMA-ES and random search finish in well under a second.

| Experiment | What it varies | Runs |
|---|---|---|
| 1 | Ambient dimension D ∈ {2, 5, 10, 20}, k = D | 240 |
| 1-ext | D = k = 20, extended budget (BO 500; CMA-ES, random 1000) | 60 |
| 2 | Effective dimension k ∈ {2, 5, 10} at D = 20, aligned | 180 |
| 3 | Aligned vs random rotation, D = 20, k = 2 (adds BO-iso) | 100 |
| 4 | Partial rotation θ ∈ {22.5°, 45°, 67.5°, 90°} | 240 |
| 5 | Embedded Rosenbrock, aligned vs rotated, D = 20, k = 2 | 160 |
| Sensitivity | CMA-ES σ0, λ; BO initial design size | 240 |

**2. Build tables and figures.** All numbers in the report are regenerated from `results/raw/`:

```bash
python make_figures.py
```

Outputs:
- `results/summaries/all_tables.md`, with a CSV per table alongside it.
- `results/figures/`, containing figures `fig1`–`fig8`. Each figure is described in `REPORT.md` §6.

**3. Supplementary BO noise-floor check (optional):**

```bash
python report/supp_gp_noise_check.py
```

Output: `results/summaries/supp_gp_noise_check.txt`.

## Notes on reproducibility

- Every objective call goes through `BudgetedObjective`, so each run uses exactly B evaluations.
- Algorithms use library defaults with no tuning on the final seeds.
- The GP is refit from scratch at every BO iteration in float64. Small numerical differences across platforms or
  library versions can change individual BO runs slightly, but should not change the paired statistics.
- Timings were measured under shared load, so compare them across methods rather than as absolute values.
