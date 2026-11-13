# BO vs CMA-ES: Ambient Dimension, Effective Dimension and Active-Subspace Alignment — Results and Methods

This document summarises the setup, results and hypothesis checks of the study. All numbers regenerate from
`results/raw/` via `python make_figures.py` (see `README.md`). Full tables are in
`results/summaries/all_tables.md`.

**Notation.** D = ambient dimension, k = effective dimension, A = D×k orthonormal basis of the active subspace,
x* = optimum, B = evaluation budget, r_t = best-so-far simple regret after t evaluations (f* = 0),
L = log10(r_B + 1e-12). Differences in L are in **decades (orders of magnitude)**; negative = first method better.
"Median [95% CI]" = median across 20 paired seeds with percentile-bootstrap CI (10,000 resamples).

---

## 1. Research question and hypotheses

How do D, k and the orientation of a low-dimensional active subspace affect the evaluation efficiency and runtime of
GP-based BO and CMA-ES, and why?

| ID | Prediction | Falsified if |
|---|---|---|
| H1 | At B = 200, BO's regret grows more slowly with D than CMA-ES's; gap narrows/reverses at the extended budget. | CMA-ES matches/beats BO at B = 200 for D ≥ 10, or gap does not narrow at extended budget. |
| H2 | At D = 20, BO-ARD benefits more than CMA-ES from smaller axis-aligned k (learns long length scales on inactive coordinates). | CMA-ES's improvement k = 20 → 2 is as large as BO's, or BO's length scales do not separate active/inactive coordinates. |
| H3 | Rotating the active subspace hurts BO-ARD much more than CMA-ES. | Interaction Δ 95% CI includes 0, or CMA-ES clearly affected by rotation. |
| H3-mech | (a) BO-iso shows little aligned-vs-rotated gap; (b) BO-ARD degrades smoothly with rotation angle θ; (c) CMA-ES covariance aligns with span(A) at similar rates whatever the orientation. | — (assessed descriptively) |

---

## 2. Methods

### 2.1 Benchmarks (domain X = [−1, 1]^D, minimisation, deterministic, f* = 0)

- **Shifted subspace quadratic:** f(x) = ‖Aᵀ(x − x*)‖² / k.
- **Embedded Rosenbrock (Exp 5 only):** u = 1 + 2Aᵀ(x − x*), f = Σ_{i=1}^{k−1} [100(u_{i+1} − u_i²)² + (1 − u_i)²].
- x* ~ Uniform[−0.5, 0.5]^D (seeded; not at the domain centre). The minimiser set {x : Aᵀ(x − x*) = 0} lies partly
  inside the box.
- **A:** axis-aligned = [e_1 … e_k]; random rotation = QR of a seeded D×k Gaussian (sign-fixed);
  partial rotation A(θ) = cos θ · E + sin θ · Q⊥ with Q⊥ ⟂ span(E), so all principal angles to span(E) equal θ
  (θ = 0 → axis-aligned exactly; θ = π/2 → random subspace of the inactive coordinates).
- **Seeding:** each seed gives independent instance and algorithm streams (`numpy.random.SeedSequence`). x* depends
  only on (seed, D), so aligned/rotated/partial instances and Exp 1 D = 20 / Exp 2 k = 20 share x*. All algorithms see
  the same instance per seed.
- Validation tests (all pass): ‖AᵀA − I‖ < 1e-10; f ≥ 0 on 10⁴ points; f(x*) = 0; invariance along null(Aᵀ); partial
  rotation principal angles equal θ; same seed → identical instance.

### 2.2 Algorithms (all see only a black-box f, bounds, budget, seed)

| | BO-ARD (primary) | BO-iso (ablation) | CMA-ES | Random |
|---|---|---|---|---|
| Library | BoTorch 0.18.1 / GPyTorch 1.15.2 | same | pycma 4.5.0 | numpy |
| Model / search | `SingleTaskGP`, inputs in [0,1]^D, `Standardize` outcome, float64; Matérn-5/2 with one length scale per dimension | same, single shared length scale | ask/tell, x0 ~ U[−1,1]^D (independent of x*), σ0 = 0.6 (= 0.3 × range), λ = 4 + ⌊3 ln D⌋ (12 at D = 20), C0 = I | uniform samples |
| Length-scale prior | LogNormal(√2 + 0.5 ln D, √3) (dimension-scaled, Hvarfner et al. 2024), constraint ≥ 0.025, init at prior mode | identical prior | — | — |
| Other | no output scale; default Gaussian likelihood, noise prior LogNormal(−4, 1), noise ≥ 1e-4 (standardised units); refit from scratch every iteration (`fit_gpytorch_mll`) | same | bounds via pycma `BoundTransform`; native termination criteria disabled; the final partial generation is evaluated but not passed to `tell` | — |
| Acquisition | `LogExpectedImprovement`, `optimize_acqf` (10 restarts, 512 raw samples), q = 1 | same | — | — |
| Initial design | scrambled Sobol, n0 = 2D (counted in budget) | same | — | — |

### 2.3 Fair-comparison protocol

- Every objective call goes through a single `BudgetedObjective` wrapper. It counts calls, raises at call B + 1,
  asserts feasibility and logs every evaluation. All 1,220 runs used exactly B evaluations; there were no early
  terminations and 0 GP fit failures.
- Budgets: B = 200 for all methods and conditions. Extended budget (Exp 1, D = k = 20 only): CMA-ES and random
  1000, **BO-ARD 500** (a 1000-evaluation BO run was estimated at 2–4 CPU-h per seed).
- Library-default settings for both methods with no tuning on final seeds. A predeclared sensitivity grid was run on
  separate seeds.
- Seeds: final 0–19 (20 paired seeds per condition); sensitivity 2000–2019.
- Runtime: Apple M5 Pro (18 cores), Python 3.13.13, torch 2.15.0, 1 thread per process, 15–20 concurrent processes.
  Wall time, time inside the objective and optimiser overhead (= wall − objective) are logged per evaluation.
- Statistics: paired by seed on L. Tests: Wilcoxon signed-rank, bootstrap 95% CI of the median paired difference, and
  Holm correction within each experiment table.
  - Interaction Δ_s = (L_BO,rot − L_BO,aligned) − (L_CMA,rot − L_CMA,aligned), reported as median [CI].
    Δ > 0 means rotation hurts BO more than CMA-ES.
  - Targets ε ∈ {1e-1, 1e-2, 1e-3}: success rate, hitting times (right-censored at B) and ERT.
  - COCO-style ECDF over 13 log-spaced targets in [1e-4, 1].

### 2.4 Experiments (20 seeds per cell; reused cells are identical runs selected by condition)

| Exp | Benchmark | D | k | Orientation | Algorithms | New runs |
|---|---|---|---|---|---|---|
| 1 | quad | 2, 5, 10, 20 | = D | aligned (A = I) | BO-ARD, CMA-ES, Random | 240 |
| 1-ext | quad | 20 | 20 | aligned | BO-ARD (500), CMA-ES & Random (1000) | 60 |
| 2 | quad | 20 | 2, 5, 10 (+20 from Exp 1) | aligned | BO-ARD, CMA-ES, Random; oracle = Exp 1 at D = k | 180 |
| 3 | quad | 20 | 2 | aligned (from Exp 2) vs random rotation | BO-ARD, BO-iso, CMA-ES, Random | 100 |
| 4 | quad | 20 | 2 | partial θ ∈ {0, 22.5°, 45°, 67.5°, 90°} (θ = 0 from Exp 2/3) | BO-ARD, BO-iso, CMA-ES | 240 |
| 5 | Rosenbrock | 20 | 2 | aligned vs random rotation | BO-ARD, BO-iso, CMA-ES, Random | 160 |
| Sens | quad | 20 | 2 | aligned & rotated | CMA-ES σ0 ∈ {0.3, 0.6, 1.0}, λ ∈ {12, 24}; BO-ARD n0 ∈ {D, 2D} (one factor at a time) | 240 |

Total: 1,220 runs (540 BO).

---

## 3. Results

### 3.1 Experiment 1: ambient dimension (H1). Quadratic, k = D, aligned

Median regret at B = 200 [IQR]:

| D | BO-ARD | CMA-ES | Random | L(BO) − L(CMA): median [CI], p_Holm |
|---|---|---|---|---|
| 2 | 2.0e-6 [8.6e-7, 3.9e-6] | **1.0e-8** [5.6e-9, 7.9e-8] | 2.3e-3 | **+2.09** [1.58, 2.50], 3.6e-5 (CMA-ES better) |
| 5 | **2.0e-5** [1.7e-5, 2.3e-5] | 1.6e-3 [6.2e-4, 2.6e-3] | 4.1e-2 | −2.00 [−2.18, −1.51], 7.6e-6 |
| 10 | **1.4e-4** [1.0e-4, 1.8e-4] | 3.5e-2 [1.9e-2, 6.3e-2] | 0.105 | −2.50 [−2.64, −2.15], 7.6e-6 |
| 20 | **2.9e-4** [2.1e-4, 4.2e-4] | 0.110 [0.098, 0.119] | 0.178 | −2.57 [−2.72, −2.45], 7.6e-6 |

- Change in median log-regret from D = 2 to D = 20: BO-ARD +2.2 decades; CMA-ES +7.0 decades.
- Success at ε = 1e-3 (B = 200): BO-ARD 100 / 100 / 100 / 95 %; CMA-ES 100 / 40 / 0 / 0 % (D = 2 / 5 / 10 / 20).
- ERT to 1e-2: BO-ARD 8.4 / 18 / 42 / 87; CMA-ES 28 / 145 / 1970 / ∞ evaluations.
- At D = 2, BO-ARD plateaus at ≈ 2e-6 after ≈ 50 evaluations (r50 = 4.5e-6, r100 = 2.0e-6, r200 = 2.0e-6); see §3.8.

**Extended budget (D = k = 20).** Median regret [IQR]:

| t | BO-ARD | CMA-ES | Random | L(BO) − L(CMA) at t: median [CI] |
|---|---|---|---|---|
| 50 | 0.070 | 0.198 | 0.215 | −0.51 [−0.59, −0.37] |
| 100 | 9.1e-4 | 0.140 | 0.203 | −2.18 [−2.24, −2.00] |
| 200 | 2.9e-4 | 0.110 | 0.178 | −2.57 [−2.72, −2.45] |
| 500 | 2.0e-4 [1.6e-4, 2.7e-4] | 2.4e-2 [1.6e-2, 3.1e-2] | 0.146 | −2.10 [−2.25, −1.73] |
| 1000 | (not run) | 3.6e-4 [1.9e-4, 5.6e-4] | 0.138 | — |

- All paired p_Holm = 7.6e-6.
- The gap is largest at t = 200 and narrows by ≈ 0.5 decades by t = 500 (the CIs do not overlap).
- BO-ARD improves by only 0.16 decades between t = 200 and 500.
- CMA-ES at t = 1000 (3.6e-4) is at about the level BO-ARD reached by t = 200 (2.9e-4).
- Whether the order reverses after t = 500 was not tested (BO stopped at 500).
- CMA-ES at 1000: success 100 % at 1e-2, 90 % at 1e-3; ERT to 1e-3 = 969 evaluations.

**H1 check:** BO beats CMA-ES at B = 200 for D = 5, 10 and 20, and the gap narrows at the extended budget, so H1 is
**not falsified**. Exception: at D = 2, CMA-ES is better by 2.1 decades.

Figures: `fig1_exp1_regret_ecdf.png`, `fig1b_exp1_extended.png`, `fig3a_final_vs_D.png`.

### 3.2 Experiment 2: effective dimension (H2). Quadratic, D = 20, aligned

Median regret at B = 200:

| k | BO-ARD | CMA-ES | Random | L(BO) − L(CMA) [CI] | BO-ARD oracle (D = k) | CMA-ES oracle (D = k) |
|---|---|---|---|---|---|---|
| 2 | **4.6e-8** | 8.2e-4 | 2.5e-3 | −3.90 [−4.71, −3.30] | 2.0e-6 | 1.0e-8 |
| 5 | **1.0e-5** | 2.4e-2 | 4.6e-2 | −3.29 [−3.40, −3.07] | 2.0e-5 | 1.6e-3 |
| 10 | **2.2e-4** | 4.8e-2 | 0.110 | −2.31 [−2.53, −2.20] | 1.4e-4 | 3.5e-2 |
| 20 | **2.9e-4** | 0.110 | 0.178 | −2.57 [−2.72, −2.45] | = same runs | = same runs |

All four BO − CMA comparisons have p_Holm = 7.6e-6.

Paired improvement relative to k = 20, L(k) − L(20) (more negative = larger gain):

| k | BO-ARD | CMA-ES | Random | Interaction (BO gain − CMA gain) |
|---|---|---|---|---|
| 2 | −3.82 [−4.05, −3.48] | −2.13 [−2.28, −1.85] | −1.82 [−2.20, −1.53] | **−1.44 [−2.27, −0.68]** (BO gains more) |
| 5 | −1.43 [−1.58, −1.27] | −0.68 [−0.78, −0.56] | −0.55 [−0.70, −0.51] | −0.71 [−1.02, −0.48] (BO gains more) |
| 10 | −0.14 [−0.32, +0.01] | −0.33 [−0.37, −0.26] | −0.22 [−0.28, −0.13] | +0.14 [+0.05, +0.34] (CMA-ES gains more) |

**Length-scale diagnostic.** BO-ARD final fit, n = 200, unit-cube units; medians across seeds:

| k | Active length scale | Inactive length scale | min(active) / min(inactive) | Runs with every active length scale < every inactive one |
|---|---|---|---|---|
| 2 | 0.71 | 105 | 0.022 | 100 % |
| 5 | 1.21 | 53 | 0.025 | 100 % |
| 10 | 1.67 | 43 | 0.044 | 100 % |

- The separation appears early: inactive length scales rise from ≈ 3–15 at n = 40 to > 30 by n ≈ 80
  (`fig6a_exp2_lengthscales.png`).
- Oracle comparisons:
  - BO-ARD at D = 20, k = 2 (4.6e-8) ends *better* than its oracle at D = 2 (2.0e-6; the plateau noted above).
  - CMA-ES at D = 20, k = 2 (8.2e-4) is ≈ 5 decades worse than its oracle (1.0e-8).
  - At k = 10 and 20, both methods are close to their oracles.

**H2 check:** BO-ARD gains 1.44 decades more than CMA-ES from k = 20 → 2 (CI excludes 0), and ARD length scales
separate active from inactive coordinates in 100 % of runs, so H2 is **not falsified**.
- Caveats: at k = 10, BO-ARD gains essentially nothing relative to k = 20, and CMA-ES gains slightly more.
- CMA-ES's gain at k = 2 (−2.13) is only slightly larger than random search's (−1.82).

Figures: `fig2_exp2_regret_ecdf.png`, `fig3b_final_vs_k_with_oracle.png`, `fig6a_exp2_lengthscales.png`.

### 3.3 Experiment 3: orientation (H3). Quadratic, D = 20, k = 2

Median regret at B = 200 [IQR]:

| Algorithm | Aligned | Rotated | L(rot) − L(aligned) [CI], p_Holm |
|---|---|---|---|
| BO-ARD | **4.6e-8** [2.5e-8, 2.5e-7] | 2.4e-4 [9.6e-5, 4.0e-4] | **+3.59 [3.02, 3.87]**, 1.9e-5 |
| BO-iso | 7.6e-5 [3.9e-5, 1.3e-4] | **4.5e-5** [3.2e-5, 1.1e-4] | −0.07 [−0.26, 0.29], 1.0 |
| CMA-ES | 8.2e-4 [4.9e-4, 1.6e-3] | 7.6e-4 [4.3e-4, 1.1e-3] | −0.07 [−0.31, 0.28], 1.0 |
| Random | 2.5e-3 | 2.9e-3 | +0.14 [−0.16, 0.39], 1.0 |

- **Interaction Δ (BO-ARD vs CMA-ES) = +3.39 [3.03, 3.96].**
- **Interaction Δ (BO-iso vs CMA-ES) = +0.09 [−0.27, 0.32].**
- Paired between-method differences (p_Holm):
  - Aligned: BO-ARD − CMA = −3.90 (1.9e-5); BO-iso − CMA = −1.05 (3.1e-5); BO-ARD − BO-iso = −2.93 (2.3e-5).
  - Rotated: BO-ARD − CMA = −0.34 [−1.03, −0.04] (p_Holm 0.077); BO-iso − CMA = −1.00 (4.9e-4);
    **BO-ARD − BO-iso = +0.75 [0.23, 0.92] (0.016)**, so BO-iso beats BO-ARD when the subspace is rotated.
- Success at 1e-3: BO-ARD 100 % in both orientations, but ERT rises from 37 (aligned) to 92 (rotated); CMA-ES 55 % /
  70 %.

**H3 check:** the Δ CI excludes 0, and CMA-ES is unaffected by rotation (CI includes 0), so H3 is **not
falsified**. BO-iso Δ ≈ 0, as predicted.

Figures: `fig4a_exp3_regret_ecdf.png`, `fig4b_exp3_delta.png`.

### 3.4 Experiment 4: gradual rotation (H3-mechanism). Quadratic, D = 20, k = 2

Median regret at B = 200:

| θ | BO-ARD | BO-iso | CMA-ES |
|---|---|---|---|
| 0° | 4.6e-8 | 7.6e-5 | 8.2e-4 |
| 22.5° | 1.1e-4 | 6.6e-5 | 8.9e-4 |
| 45° | 1.6e-4 | 5.6e-5 | 4.8e-4 |
| 67.5° | 2.8e-4 | 6.1e-5 | 3.3e-4 |
| 90° | 1.8e-4 | 8.3e-5 | 7.6e-4 |

Paired change relative to θ = 0, L(θ) − L(0), with CI:

| θ | BO-ARD | BO-iso | CMA-ES | Interaction Δ (BO-ARD vs CMA) | Interaction Δ (BO-iso vs CMA) |
|---|---|---|---|---|---|
| 22.5° | +3.38 [2.62, 3.82] | +0.06 [−0.41, 0.14] | +0.21 [−0.40, 0.37] | +3.14 [2.01, 3.90] | −0.07 [−0.46, 0.37] |
| 45° | +3.37 [2.95, 3.74] | +0.11 [−0.47, 0.36] | −0.20 [−0.60, 0.32] | +3.41 [3.18, 3.83] | +0.07 [−0.41, 0.64] |
| 67.5° | +3.69 [2.87, 4.19] | −0.14 [−0.31, 0.08] | −0.40 [−0.57, −0.04] | +3.80 [2.96, 4.60] | +0.27 [−0.11, 0.54] |
| 90° | +3.34 [2.65, 3.92] | −0.03 [−0.54, 0.24] | −0.23 [−0.30, 0.39] | +3.11 [2.63, 3.90] | −0.25 [−0.55, 0.55] |

**Mechanism diagnostics.** Final-generation / final-fit medians. "Active" coordinates for BO = e1, e2. Chance level
for the angle = median largest principal angle between span(E) and a random 2-D subspace of R^20 = **83.9°**.

| θ | BO-ARD: min ls(e1, e2) / min ls(others) | BO-ARD: median ls(e1, e2) | CMA-ES: largest angle, bottom-2 eigvecs of C vs span(A) | CMA-ES: same, top-2 eigvecs |
|---|---|---|---|---|
| 0° | 0.022 | 0.71 | 42.0° | 88.7° |
| 22.5° | 0.36 | 0.52 | 59.4° | 88.8° |
| 45° | 0.61 | 0.61 | 74.0° | 85.1° |
| 67.5° | 1.67 | 1.09 | 79.4° | 84.2° |
| 90° | 17.8 | 10.4 | 76.7° | 84.1° |

**H3-mechanism check:**
- (a) BO-iso shows no aligned-vs-rotated gap at any θ (all CIs include 0). **Supported.**
- (b) BO-ARD's performance does **not** degrade smoothly: the full ≈ 3.4-decade loss is already present at
  θ = 22.5° and is flat for θ ≥ 22.5°. The length-scale ratio, by contrast, rises smoothly and monotonically with θ.
  **Not supported (step, not dose-response).**
- (c) Within 200 evaluations (≈ 17 CMA-ES generations), CMA-ES alignment is weak and depends on orientation: 42° when
  aligned, 59–79° when θ > 0, against a chance level of 84°. CMA-ES's *regret*, however, does not depend on θ (only
  θ = 67.5° has a CI excluding 0, and there the rotated case is better). **Not supported as stated** (alignment
  differs by orientation; performance does not).

Figures: `fig5_exp4_theta_dose_response.png`, `fig6b_cma_alignment.png`.

### 3.5 CMA-ES state diagnostics. Final generation, medians over 20 seeds

| Condition | Generations | σ (σ0 = 0.6) | cond(C) | Bottom-k angle (chance) | Per-coordinate P(out of bounds): gen 1 → final |
|---|---|---|---|---|---|
| D = k = 2 | 34 | 0.012 | 2.3 | — | 0.18 → 0.00 |
| D = k = 5 | 26 | 0.121 | 3.9 | — | 0.22 → 0.21 |
| D = k = 10 | 21 | 0.293 | 3.3 | — | 0.22 → 0.32 |
| D = k = 20 | 17 | 0.478 | 2.2 | — | 0.23 → 0.34 |
| D = 20, k = 2, aligned | 17 | 0.485 | 2.5 | 42.0° (83.9°) | 0.23 → 0.43 |
| D = 20, k = 5, aligned | 17 | 0.489 | 2.2 | 83.9° (86.2°) | 0.23 → 0.41 |
| D = 20, k = 10, aligned | 17 | 0.419 | 2.1 | 84.7° (86.8°) | 0.23 → 0.35 |
| D = 20, k = 2, rotated | 17 | 0.512 | 2.4 | 80.5° (83.9°) | 0.23 → 0.50 |
| Rosenbrock, D = 20, k = 2, aligned / rotated | 17 | 0.46 / 0.47 | 2.3 / 2.3 | 66.8° / 80.6° (83.9°) | 0.23 → 0.40 / 0.40 |

- Fraction of raw CMA-ES samples with *any* coordinate outside the box: median 0.96 overall in Exp 1, and ≈ 1.0 in
  every D = 20 condition.
- Partial last generation: occurs whenever 200 mod λ ≠ 0 (D = 2, λ = 6; D = 20, λ = 12); it is evaluated but not
  passed to `tell`.
- In a D = 20, k = 2, B = 1000 test run, CMA-ES's mean drifted far outside the box (|coordinate| up to 3.5) in
  inactive coordinates while σ stayed ≈ 0.35–0.6.

### 3.6 Experiment 5: generalisation (embedded Rosenbrock, D = 20, k = 2)

Median regret at B = 200 [IQR]. Typical initial values: median ≈ 250, mean ≈ 1300.

| Algorithm | Aligned | Rotated | L(rot) − L(aligned) [CI], p_Holm |
|---|---|---|---|
| BO-ARD | **1.4e-3** [5.6e-4, 3.6e-3] | 0.15 [0.035, 0.26] | **+1.90 [1.72, 2.30]**, 1.9e-5 |
| BO-iso | 0.36 [0.14, 0.53] | 0.27 [0.12, 0.52] | −0.12 [−0.50, 0.17], 1.0 |
| CMA-ES | 0.14 [0.10, 0.30] | **0.10** [0.04, 0.29] | −0.03 [−0.81, 0.22], 1.0 |
| Random | 0.15 | 0.11 | −0.24 [−0.50, 0.35], 1.0 |

- **Interaction Δ (BO-ARD vs CMA-ES) = +2.26 [1.40, 2.68].** Δ (BO-iso vs CMA-ES) = −0.03 [−0.43, 0.23].
- Paired between-method differences (p_Holm):
  - Aligned: BO-ARD − CMA = −2.03 (1.9e-5); BO-iso − CMA = +0.29 (0.12); BO-ARD − BO-iso = −2.21 (2.3e-5).
  - Rotated: BO-ARD − CMA = +0.11 [−0.30, 0.72] (1.0); BO-iso − CMA = +0.57 (0.31); BO-ARD − BO-iso = −0.22
    [−0.80, 0.00] (0.35).
- Success at 1e-2: BO-ARD aligned 100 %, rotated 5 %; CMA-ES 0 % / 10 %; BO-iso 0 % / 5 %.

The H3 pattern (Δ > 0 for ARD, Δ ≈ 0 for iso, CMA-ES unaffected) **reproduces** on a non-quadratic function.
Unlike the quadratic case, rotated BO-ARD is no better than CMA-ES or random.

Figures: `fig4a_exp5_regret_ecdf.png`, `fig4b_exp5_delta.png`.

### 3.7 Sensitivity (seeds 2000–2019; quadratic D = 20, k = 2)

| BO variant | CMA-ES variant | Δ [CI] | BO − CMA, aligned [CI] | BO − CMA, rotated [CI] (p) |
|---|---|---|---|---|
| BO-ARD (n0 = 2D) | baseline (σ0 = 0.6, λ = 12) | +4.06 [3.70, 4.49] | −4.38 [−5.41, −3.63] | −0.55 [−0.84, −0.39] (0.003) |
| BO-ARD n0 = D | baseline | +3.19 [2.54, 3.57] | −3.92 [−4.21, −3.43] | −0.67 [−1.17, −0.42] (1e-4) |
| BO-ARD | σ0 = 0.3 | +4.00 [3.51, 4.34] | −4.05 [−4.94, −3.34] | −0.19 [−0.88, 0.41] (0.39) |
| BO-ARD | σ0 = 1.0 | +3.69 [3.14, 4.81] | −4.58 [−5.66, −3.91] | −0.81 [−1.17, −0.37] (2e-4) |
| BO-ARD | λ = 24 | +4.34 [3.62, 4.71] | −4.84 [−5.43, −4.44] | −0.70 [−0.94, −0.23] (6e-4) |

- Median final regret: CMA-ES aligned / rotated = 9.7e-4 / 6.4e-4 (baseline), 5.2e-4 / 2.4e-4 (σ0 = 0.3),
  1.8e-3 / 1.1e-3 (σ0 = 1.0), 2.2e-3 / 6.6e-4 (λ = 24).
- BO-ARD aligned / rotated = 4.2e-8 / 2.5e-4 (n0 = 2D) and 8.2e-8 / 1.3e-4 (n0 = D).
- No Exp 1–3 conclusion changes. In every variant, Δ > 0 with its CI excluding 0, and BO-ARD beats CMA-ES when
  aligned.
- Rotated BO-ARD vs CMA-ES is a small effect that varies with settings: BO is better in 4 of 5 variants and not
  significantly different for σ0 = 0.3.

### 3.8 Supplementary check: BO regret floor (5 seeds per condition, final data refit)

Script: `report/supp_gp_noise_check.py` → `results/summaries/supp_gp_noise_check.txt`

| Condition | Median regret | Learned noise variance (standardised) | Implied noise SD in f units | sd(Y) | Fraction of last 100 evaluations with f < 1e-3 |
|---|---|---|---|---|---|
| D = 2 | 2.7e-6 | 1.0e-4 (= lower bound) | 1.7e-3 | 0.17 | 0.94 |
| D = 20, k = 2 | 5.2e-7 | 1.0e-4 (= lower bound) | 2.4e-3 | 0.24 | 0.99 |
| D = 20, k = 20 | 2.7e-4 | 1.0e-4 (= lower bound) | 1.7e-3 | 0.17 | 0.94 |

The learned noise sits at its constraint floor in every case, and the implied noise SD (≈ 2e-3) is 1–3 orders of
magnitude larger than the regret levels where BO stalls.

### 3.9 Runtime (median per run; objective cost ≈ 2–8 ms per run, negligible)

| Method / condition | B | Wall / overhead per run | Overhead per evaluation |
|---|---|---|---|
| BO-ARD, D = 2 | 200 | 37 s | 0.18 s |
| BO-ARD, D = 5 / 10 | 200 | 97 s / 98 s | 0.49 s |
| BO-ARD, D = 20, k = 20 | 200 | 94 s | 0.47 s |
| BO-ARD, D = 20, k = 2 / 5 / 10 (aligned) | 200 | 296 / 315 / 315 s | 1.5–1.6 s |
| BO-ARD, D = 20, k = 2 rotated / partial θ | 200 | 340 s / 499 s | 1.7 / 2.5 s |
| BO-ARD, D = 20, k = 20 | 500 | 642 s | 1.3 s |
| BO-iso, D = 20, k = 2 | 200 | 40–51 s | 0.2–0.26 s |
| CMA-ES, any | 200 | 0.012–0.015 s | 0.06 ms |
| CMA-ES, D = 20 | 1000 | 0.066 s | 0.06 ms |
| Random | 200 | ≈ 0.002 s | 0.007–0.009 ms |

- BO-ARD's overhead per evaluation is ≈ 3 × 10³ – 4 × 10⁴ × CMA-ES's (0.18–2.5 s vs 0.06 ms). BO-ARD's cumulative
  overhead grows super-linearly with t (`fig7_runtime.png`).
- GP-fit timing at D = 20 under load: 0.8 s (n = 200), 2.8 s (n = 500), 6.4 s (n = 800), 318 s (n = 1000 with
  GPyTorch's default iterative solver above 800 points), 36 s (n = 1000 with exact Cholesky).
- Times were measured with 15–20 processes sharing 18 cores, so they are inflated relative to an idle machine but
  comparable across methods.

### 3.10 Initial-value distributions (quadratic, D = 20, k = 2; `fig8_initial_values.png`)

| Source | Aligned: mean / median / IQR | Rotated: mean / median / IQR |
|---|---|---|
| Random search, all 200 × 20 values | 0.43 / 0.35 / [0.16, 0.63] | 0.41 / 0.29 / [0.13, 0.58] |
| BO Sobol design, first 40 | 0.42 / 0.34 / [0.16, 0.64] | 0.41 / 0.28 / [0.12, 0.58] |
| CMA-ES generation 1 (12) | 0.45 / 0.40 / [0.24, 0.60] | 0.54 / 0.42 / [0.21, 0.73] |

Means are similar across orientations, as designed. The rotated case has a more skewed spread (lower median, longer
upper tail).

---

## 4. Hypothesis summary

| Hypothesis | Criterion outcome | Key numbers |
|---|---|---|
| H1 | Not falsified (D ≥ 5); reversed at D = 2 | BO − CMA: +2.09 (D = 2), −2.00, −2.50, −2.57 decades; gap at D = 20: −2.57 (t = 200) → −2.10 (t = 500) |
| H2 | Not falsified (caveat at k = 10) | BO gain −3.82 vs CMA −2.13, interaction −1.44 [−2.27, −0.68]; length-scale ratio 0.022, separation in 100 % of runs |
| H3 | Not falsified | Δ_ARD = +3.39 [3.03, 3.96]; CMA-ES rot − aligned −0.07 [−0.31, 0.28] |
| H3-mech (a) BO-iso no gap | Supported | Δ_iso = +0.09 [−0.27, 0.32]; L(θ) − L(0) CIs all include 0 |
| H3-mech (b) smooth degradation | Not supported (step at θ = 22.5°) | BO-ARD L(θ) − L(0) = 3.38, 3.37, 3.69, 3.34; length-scale ratio smooth: 0.022 → 0.36 → 0.61 → 1.67 → 17.8 |
| H3-mech (c) CMA-ES aligns similarly | Not supported as stated | Final angle 42° (θ = 0) vs 59–80° (θ > 0), chance 84°; regret unaffected by θ |
| Generalisation (Exp 5) | H3 pattern reproduces | Δ_ARD = +2.26 [1.40, 2.68]; Δ_iso = −0.03 |
| Sensitivity | No conclusion changes | Δ ∈ [3.19, 4.34] across 5 variants |

---

## 5. Discussion points

- **BO floor.** BO's regret plateaus (≈ 2e-6 at D = 2; ≈ 2–3e-4 at D = k = 20, with little change from t = 200 to
  500). In every case checked, the GP's learned noise is pinned at its lower bound and corresponds to an SD of
  ≈ 2e-3 in f units.
- **CMA-ES at D = 20, B = 200.** Only 16–17 generations: σ falls from 0.6 to only ≈ 0.48, cond(C) ≈ 2, and alignment
  is near chance level. Learning rates in pycma defaults scale as ~ 1/D². Its improvement from k = 20 → 2 (−2.13)
  is close to random search's (−1.82).
- **Boundary handling.** Per coordinate, 23 % of CMA-ES's raw samples fall outside the box initially and 34–50 % by
  the end at D = 20; nearly every sample vector has at least one coordinate outside. `BoundTransform` is
  coordinate-wise, so it is not rotation-invariant.
- **Information asymmetry.** BO's Sobol design (n0 = 2D = 40 at D = 20) covers the box; CMA-ES starts from one point
  with σ0 = 0.6.
- **Threshold dependence.** Rankings are consistent across ε ∈ {1e-1, 1e-2, 1e-3} and the ECDFs, except at
  D = 2 / ε = 1e-3. There BO reaches 1e-3 faster (ERT 12 vs 49), but CMA-ES ends lower.
- **Limitations.**
  - BO extended budget was 500 rather than 1000, so whether CMA-ES overtakes BO at D = 20 was not tested.
  - Runtimes were measured under shared load.
  - k = 2 is the only k used for the rotation experiments.

---

## 6. Figure index (`results/figures/`)

| File | Content |
|---|---|
| `fig1_exp1_regret_ecdf.png` | Exp 1: median regret (IQR bands) and ECDF per D |
| `fig1b_exp1_extended.png` | Exp 1, D = 20 extended budget: regret and ECDF |
| `fig2_exp2_regret_ecdf.png` | Exp 2: regret and ECDF per k |
| `fig3a_final_vs_D.png` | Final regret vs D |
| `fig3b_final_vs_k_with_oracle.png` | Final regret vs k with oracle (D = k) curves |
| `fig4a_exp3_regret_ecdf.png`, `fig4b_exp3_delta.png` | Exp 3: curves; paired Δ_s strip plot with median and CI |
| `fig4a_exp5_regret_ecdf.png`, `fig4b_exp5_delta.png` | Exp 5: same for Rosenbrock |
| `fig5_exp4_theta_dose_response.png` | Exp 4: regret vs θ; BO-ARD length-scale ratio vs θ; CMA-ES angle trajectories per θ |
| `fig6a_exp2_lengthscales.png` | BO-ARD active vs inactive length scales vs training points |
| `fig6b_cma_alignment.png` | CMA-ES bottom-/top-2 eigenvector angles to span(A), aligned vs rotated |
| `fig7_runtime.png` | Overhead vs D; cumulative overhead vs evaluations (D = 20 extended) |
| `fig8_initial_values.png` | Initial-value distributions by orientation |

## 7. Implementation choices

1. BO extended budget 500 (not 1000), chosen from timing estimates only.
2. Extended budget applied to Exp 1, D = 20 only.
3. CMA-ES alignment uses the *bottom*-k eigenvectors of C, since CMA-ES learns small variance along the curved
   directions span(A); top-k results are reported for comparison.
4. GP: no output-scale kernel; length-scale floor 0.025; exact Cholesky forced above 800 points (no effect at the
   budgets used).
5. pycma: all native termination criteria disabled; the sampler uses a seeded Generator.
6. Sensitivity: BO-iso and random search were not re-run on sensitivity seeds.
