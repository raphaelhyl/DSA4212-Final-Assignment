from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

FEASIBILITY_TOL = 1e-9  # floating-point slack (e.g. unit-cube rescaling); larger violations are errors


class BudgetExhausted(Exception):
    """Raised on evaluation budget + 1; optimisers treat it as the normal stop signal."""


class BudgetedObjective:
    """Wraps f; counts every call, enforces the budget, checks bounds, logs each evaluation."""

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        bounds: tuple[float, float],
        budget: int,
        log_path: Path | None = None,
    ) -> None:
        self.f, self.budget, self.log_path = f, budget, log_path
        self.lower, self.upper = bounds
        self.rows: list[dict] = []
        self.best_f = np.inf
        self.t_obj = 0.0
        self.t_start = time.perf_counter()

    @property
    def n_evals(self) -> int:
        return len(self.rows)

    def __call__(
        self, x: np.ndarray, generation: int | None = None, out_of_bounds: bool | None = None
    ) -> float:
        if self.n_evals >= self.budget:
            raise BudgetExhausted
        x = np.asarray(x, dtype=float)
        if np.any(x < self.lower - FEASIBILITY_TOL) or np.any(x > self.upper + FEASIBILITY_TOL):
            raise ValueError(f"infeasible point (§5.7): min {x.min():.3g}, max {x.max():.3g}")
        x = np.clip(x, self.lower, self.upper)
        t0 = time.perf_counter()
        fx = float(self.f(x))
        self.t_obj += time.perf_counter() - t0
        self.best_f = min(self.best_f, fx)
        row = {"eval_idx": self.n_evals + 1, **{f"x_{i}": xi for i, xi in enumerate(x)}}
        row.update(f=fx, best_f=self.best_f, t_wall=time.perf_counter() - self.t_start, t_obj=self.t_obj)
        row.update(generation=generation, out_of_bounds=out_of_bounds)
        self.rows.append(row)
        return fx

    def history(self) -> pd.DataFrame:
        """Evaluation log with the §9 schema."""
        return pd.DataFrame(self.rows)

    def save(self) -> None:
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.history().to_csv(self.log_path, index=False)
