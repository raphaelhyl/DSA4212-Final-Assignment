from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

F_STAR = 0.0
FLOOR = 1e-12  # §7.5 plotting / log floor
TARGETS = (1e-1, 1e-2, 1e-3)  # §7.2 fixed targets
ECDF_TARGETS = np.logspace(-4, 0, 13)


def load_runs(raw_root: Path) -> pd.DataFrame:
    """Metadata table of every completed run (one row per JSON)."""
    rows = []
    for p in sorted(raw_root.glob("*/*.json")):
        with open(p) as fh:
            rec = json.load(fh)
        rec["path"] = str(p.with_suffix(""))
        rows.append(rec)
    return pd.DataFrame(rows)


def select(meta: pd.DataFrame, **filters) -> pd.DataFrame:
    """Rows matching all filters (theta compared with tolerance), sorted by seed."""
    mask = np.ones(len(meta), dtype=bool)
    for key, val in filters.items():
        if key == "theta":
            mask &= np.isclose(meta["theta"].to_numpy(float), val)
        elif isinstance(val, (list, tuple, set)):
            mask &= meta[key].isin(val).to_numpy()
        else:
            mask &= (meta[key] == val).to_numpy()
    return meta[mask].sort_values("seed")


def regret_curve(path: str, budget: int | None = None) -> np.ndarray:
    """Best-so-far simple regret r_t = min_{i<=t} f(x_i) - f*, t = 1..B."""
    best = pd.read_csv(f"{path}.csv", usecols=["best_f"])["best_f"].to_numpy()
    r = best - F_STAR
    return r if budget is None else r[:budget]


def regret_matrix(runs: pd.DataFrame, budget: int | None = None) -> np.ndarray:
    """Runs x evaluations matrix of best-so-far regret."""
    return np.vstack([regret_curve(p, budget) for p in runs["path"]])


def log_final(R: np.ndarray) -> np.ndarray:
    """§7.3 test statistic per run: log10(r_B + 1e-12)."""
    return np.log10(R[:, -1] + FLOOR)


def hitting_times(R: np.ndarray, eps: float) -> np.ndarray:
    """First evaluation (1-based) with r_t <= eps; NaN if never reached (right-censored at B)."""
    hit = R <= eps
    t = hit.argmax(axis=1).astype(float) + 1
    t[~hit.any(axis=1)] = np.nan
    return t


def success_rate(R: np.ndarray, eps: float) -> float:
    return float(np.mean(R[:, -1] <= eps))


def ert(R: np.ndarray, eps: float) -> float:
    """Expected running time: total evaluations spent across runs / number of successes."""
    t = hitting_times(R, eps)
    n_success = np.sum(~np.isnan(t))
    if n_success == 0:
        return np.inf
    return float(np.nansum(t) + np.sum(np.isnan(t)) * R.shape[1]) / n_success


def ecdf(R: np.ndarray, targets: np.ndarray = ECDF_TARGETS) -> np.ndarray:
    """COCO-style: fraction of (run, target) pairs solved by evaluation t, for t = 1..B."""
    return (R[:, :, None] <= targets[None, None, :]).mean(axis=(0, 2))


def summarise(R: np.ndarray, checkpoints: tuple[int, ...] = (50, 100, 200, 500, 1000)) -> dict:
    """Median/IQR regret at checkpoints, plus per-target success, median hitting time and ERT."""
    out = {"n_runs": R.shape[0]}
    for t in checkpoints:
        if t <= R.shape[1]:
            q1, med, q3 = np.quantile(R[:, t - 1], [0.25, 0.5, 0.75])
            out[f"r{t}_median"], out[f"r{t}_q1"], out[f"r{t}_q3"] = med, q1, q3
    for eps in TARGETS:
        t = hitting_times(R, eps)
        out[f"succ_{eps:g}"] = success_rate(R, eps)
        out[f"t_hit_median_{eps:g}"] = float(np.median(np.where(np.isnan(t), np.inf, t)))
        out[f"ert_{eps:g}"] = ert(R, eps)
    return out
