from __future__ import annotations

import cma
import numpy as np

from objective import BudgetedObjective, BudgetExhausted

# §4.2: budget is the only intended stopping rule, so native termination criteria are switched off.
NO_TERMINATION = {
    "tolfun": 0,
    "tolfunhist": 0,
    "tolfunrel": 0,
    "tolx": 0,
    "tolxstagnation": False,
    "tolstagnation": np.inf,
    "tolflatfitness": np.inf,
    "tolconditioncov": np.inf,
    "tolfacupx": np.inf,
    "tolupsigma": np.inf,
    "maxiter": np.inf,
    "maxfevals": np.inf,
}


def run_cmaes(
    objective: BudgetedObjective,
    bounds: tuple[float, float],
    budget: int,
    seed: int,
    D: int,
    sigma0: float = 0.6,
    popsize: int | None = None,
) -> dict:
    """Run CMA-ES until the budget is used; return metadata and per-generation state."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(*bounds, size=D)  # §4.2: independent of x*
    opts = {
        **NO_TERMINATION,
        "bounds": list(bounds),  # default boundary handler is BoundTransform
        "seed": seed,
        "randn": lambda *shape: rng.standard_normal(shape),  # explicit RNG rather than np.random
        "verbose": -9,
        "verb_log": 0,
        "verb_disp": 0,
    }
    if popsize is not None:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    state = {"sigma": [], "mean": [], "evals": [], "evecs": [], "eigvals": []}
    meta = {"popsize": es.popsize, "termination": "budget", "partial_last_generation": False}
    try:
        while True:
            if es.stop():
                meta["termination"] = "pycma:" + ",".join(es.stop().keys())
                break
            _record_state(es, state, objective.n_evals)
            X = es.ask()
            fs = [_evaluate(objective, es, x) for x in X]
            es.tell(X, fs)
    except BudgetExhausted:
        # §4.2: a generation interrupted by the budget is logged but never passed to tell().
        meta["partial_last_generation"] = objective.n_evals % es.popsize != 0
    meta["unused_budget"] = budget - objective.n_evals
    meta["generations_told"] = int(es.countiter)
    meta["state"] = {k: np.array(v) for k, v in state.items()}
    return meta


def _evaluate(objective: BudgetedObjective, es: cma.CMAEvolutionStrategy, x: np.ndarray) -> float:
    raw = es.sent_solutions[x]["geno"]  # pre-BoundTransform sample
    lo, hi = es.opts["bounds"]
    oob = bool(np.any(raw < lo) or np.any(raw > hi))
    return objective(x, generation=es.countiter + 1, out_of_bounds=oob)


def _record_state(es: cma.CMAEvolutionStrategy, state: dict, n_evals: int) -> None:
    """Store sigma, mean and the eigendecomposition of C at the start of each generation."""
    eigvals, evecs = np.linalg.eigh(es.sm.C)
    state["sigma"].append(es.sigma)
    state["mean"].append(np.array(es.mean))
    state["evals"].append(n_evals)
    state["evecs"].append(evecs)
    state["eigvals"].append(eigvals)
