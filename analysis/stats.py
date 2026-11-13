from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

N_BOOT = 10_000
BOOT_SEED = 12345


def bootstrap_median_ci(d: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the median."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    lo, hi = np.quantile(meds, [0.025, 0.975])
    return float(lo), float(hi)


def paired_comparison(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired-by-seed comparison of log10 final regret: d = a - b (negative favours a)."""
    d = np.asarray(a) - np.asarray(b)
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
    lo, hi = bootstrap_median_ci(d)
    return {"n": len(d), "median_diff": float(np.median(d)), "ci_lo": lo, "ci_hi": hi, "p_wilcoxon": p}


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def interaction_delta(bo_rot, bo_aligned, cma_rot, cma_aligned) -> np.ndarray:
    """§7.4: Δ_s = (L_BO,rot − L_BO,aligned) − (L_CMA,rot − L_CMA,aligned), per seed."""
    return (np.asarray(bo_rot) - np.asarray(bo_aligned)) - (np.asarray(cma_rot) - np.asarray(cma_aligned))
