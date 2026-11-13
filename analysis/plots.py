from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.metrics import ECDF_TARGETS, FLOOR, ecdf

# Fixed colour per algorithm (identity follows the entity, never its rank); random is a neutral reference.
COLORS = {
    "bo_ard": "#2a78d6",
    "cmaes": "#eb6834",
    "bo_iso": "#1baf7a",
    "random": "#8a8984",
    "oracle": "#6250d6",
}
LABELS = {
    "bo_ard": "BO-ARD",
    "bo_iso": "BO-iso",
    "cmaes": "CMA-ES",
    "random": "Random",
    "oracle": "Oracle (D=k)",
}

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2,
        "font.size": 9,
        "legend.frameon": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def color(algo: str) -> str:
    return COLORS.get(algo.split("_n0")[0] if algo.startswith("bo_ard") else algo, "#52514e")


def regret_band(ax, R: np.ndarray, algo: str, label: str | None = None, ls: str = "-") -> None:
    """Median best-so-far regret with IQR band, log y, floor 1e-12 (§7.5.1)."""
    q1, med, q3 = np.quantile(np.maximum(R, FLOOR), [0.25, 0.5, 0.75], axis=0)
    t = np.arange(1, R.shape[1] + 1)
    c = color(algo)
    ax.plot(t, med, color=c, ls=ls, label=label or LABELS.get(algo, algo))
    ax.fill_between(t, q1, q3, color=c, alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best-so-far regret")


def ecdf_curve(ax, R: np.ndarray, algo: str, ls: str = "-", label: str | None = None) -> None:
    """COCO-style ECDF over log-spaced targets in [1e-4, 1] (§7.5.2)."""
    ax.plot(
        np.arange(1, R.shape[1] + 1),
        ecdf(R, ECDF_TARGETS),
        color=color(algo),
        ls=ls,
        label=label or LABELS.get(algo, algo),
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("fraction of (run, target) pairs solved")


def median_iqr_vs(
    ax, xs, finals: list[np.ndarray], algo: str, ls: str = "-", label: str | None = None
) -> None:
    """Median and IQR of final regret against a condition variable (§7.5.3, §7.5.5)."""
    arr = [np.maximum(f, FLOOR) for f in finals]
    med = [np.median(a) for a in arr]
    lo = [np.median(a) - np.quantile(a, 0.25) for a in arr]
    hi = [np.quantile(a, 0.75) - np.median(a) for a in arr]
    ax.errorbar(
        xs,
        med,
        yerr=[lo, hi],
        color=color(algo),
        ls=ls,
        marker="o",
        ms=5,
        capsize=3,
        label=label or LABELS.get(algo, algo),
    )
    ax.set_yscale("log")


def delta_strip(ax, deltas: dict[str, np.ndarray], cis: dict[str, tuple[float, float]]) -> None:
    """Paired Δ_s per seed with median and bootstrap 95% CI (§7.5.4)."""
    rng = np.random.default_rng(0)  # jitter only
    for i, (name, d) in enumerate(deltas.items()):
        c = color(name)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=18, color=c, alpha=0.6, lw=0)
        lo, hi = cis[name]
        ax.errorbar(
            [i + 0.3],
            [np.median(d)],
            yerr=[[np.median(d) - lo], [hi - np.median(d)]],
            fmt="D",
            color="#0b0b0b",
            ms=5,
            capsize=4,
        )
    ax.axhline(0, color="#52514e", lw=1, ls="--")
    ax.set_xticks(range(len(deltas)), [LABELS.get(n, n) + " vs CMA-ES" for n in deltas])
    ax.set_ylabel("Δ_s  (log10 regret units)")


def median_trajectory(ax, x: np.ndarray, Y: np.ndarray, c: str, label: str, ls: str = "-") -> None:
    """Median across seeds of a per-step diagnostic with IQR band (§7.5.6)."""
    q1, med, q3 = np.nanquantile(Y, [0.25, 0.5, 0.75], axis=0)
    ax.plot(x, med, color=c, ls=ls, label=label)
    ax.fill_between(x, q1, q3, color=c, alpha=0.15, lw=0)


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
