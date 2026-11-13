from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.stats import norm


def cma_alignment(evecs: np.ndarray, A: np.ndarray) -> dict[str, float]:
    """Largest principal angle (deg) between span(A) and the k smallest / k largest eigenvectors of C.

    eigh returns eigenvalues in ascending order. On a subspace quadratic CMA-ES learns C ~ H^+,
    i.e. *small* variance along span(A), so the bottom-k eigenvectors are the ones expected to align.
    """
    k = A.shape[1]
    bottom = np.degrees(subspace_angles(evecs[:, :k], A)).max()
    top = np.degrees(subspace_angles(evecs[:, -k:], A)).max()
    return {"angle_bottom_k": float(bottom), "angle_top_k": float(top)}


def cma_state_table(state: dict, A: np.ndarray) -> pd.DataFrame:
    """One row per generation: evaluations so far, sigma, eigenvalue range and alignment angles."""
    rows = []
    for g, (evecs, eigvals) in enumerate(zip(state["evecs"], state["eigvals"])):
        rows.append(
            {
                "generation": g + 1,
                "evals": int(state["evals"][g]),
                "sigma": float(state["sigma"][g]),
                "cond_C": float(eigvals[-1] / eigvals[0]),
                **cma_alignment(evecs, A),
            }
        )
    return pd.DataFrame(rows)


def lengthscale_summary(ls: pd.DataFrame, active: np.ndarray) -> pd.DataFrame:
    """Per fit: median active / inactive length scale and the ratio of shortest active to shortest inactive."""
    cols = [c for c in ls.columns if c.startswith("ls_")]
    L = ls[cols].to_numpy()
    out = pd.DataFrame({"n_train": ls["n_train"].to_numpy()})
    out["ls_active_median"] = np.median(L[:, active], axis=1)
    if (~active).any():
        out["ls_inactive_median"] = np.median(L[:, ~active], axis=1)
        out["ratio_min_active_to_min_inactive"] = L[:, active].min(axis=1) / L[:, ~active].min(axis=1)
    return out


def random_subspace_angle(D: int, k: int, n: int = 5000, seed: int = 0) -> float:
    """Chance level: median largest principal angle (deg) between span(E) and a random k-dim subspace of R^D."""
    rng = np.random.default_rng(seed)
    E = np.eye(D)[:, :k]
    return float(
        np.median([np.degrees(subspace_angles(E, rng.standard_normal((D, k))).max()) for _ in range(n)])
    )


def cma_coordinate_oob(eig: dict, bounds: tuple[float, float]) -> np.ndarray:
    """Per generation: mean over coordinates of P(raw sample coordinate outside the box) under N(m, sigma^2 C)."""
    lo, hi = bounds
    C_diag = np.einsum("gij,gj->gi", eig["evecs"] ** 2, eig["eigvals"])
    s = eig["sigma"][:, None] * np.sqrt(C_diag)
    m = eig["mean"]
    return (norm.cdf((lo - m) / s) + norm.sf((hi - m) / s)).mean(axis=1)
