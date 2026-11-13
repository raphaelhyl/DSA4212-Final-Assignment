import math
from unittest import mock

import numpy as np
import pytest

from benchmarks import make_instance
from objective import BudgetedObjective
from optimizers.bo import make_kernel, run_bo
from optimizers.cmaes import run_cmaes
from optimizers.random_search import run_random

BOUNDS = (-1.0, 1.0)


def _run(fn, D, budget, seed=0, **kw):
    inst = make_instance("quad", D, min(D, 2), "rotated", 3)
    obj = BudgetedObjective(inst.objective, BOUNDS, budget)
    meta = fn(obj, BOUNDS, budget, seed, D, **kw)
    return obj.history(), meta


def _check_basic(hist, budget):
    assert len(hist) == budget
    X = hist.filter(like="x_").to_numpy()
    assert X.min() >= -1 and X.max() <= 1


def test_random_search():
    h1, _ = _run(run_random, 5, 37)
    h2, _ = _run(run_random, 5, 37)
    _check_basic(h1, 37)
    assert h1["f"].equals(h2["f"])


@pytest.mark.parametrize("D,budget", [(2, 50), (20, 101)])
def test_cmaes_budget_partial_generation_and_reproducible(D, budget):
    h1, m1 = _run(run_cmaes, D, budget)
    h2, _ = _run(run_cmaes, D, budget)
    _check_basic(h1, budget)
    assert h1["f"].equals(h2["f"])
    lam = m1["popsize"]
    assert m1["partial_last_generation"] == (budget % lam != 0)
    assert m1["generations_told"] == budget // lam  # partial generation never told (§4.2)
    assert len(m1["state"]["sigma"]) == math.ceil(budget / lam)
    assert m1["state"]["evecs"].shape[1:] == (D, D)


def test_cmaes_no_tell_on_partial_generation():
    import cma

    sizes = []
    real_tell = cma.CMAEvolutionStrategy.tell

    def spy(self, X, fs, *a, **k):
        sizes.append(len(X))
        return real_tell(self, X, fs, *a, **k)

    with mock.patch.object(cma.CMAEvolutionStrategy, "tell", spy):
        _, meta = _run(run_cmaes, 5, 30)
    assert all(s == meta["popsize"] for s in sizes) and len(sizes) == 30 // meta["popsize"]


@pytest.mark.parametrize("D", [2, 20])
@pytest.mark.parametrize("kernel", ["ard", "iso"])
def test_bo_budget_feasible_reproducible(D, kernel):
    budget = 30 if D == 2 else 2 * D + 3
    h1, m1 = _run(run_bo, D, budget, kernel=kernel)
    h2, _ = _run(run_bo, D, budget, kernel=kernel)
    _check_basic(h1, budget)
    assert np.allclose(h1["f"], h2["f"])
    ls = m1["lengthscales"]
    assert ls.shape == (budget - 2 * D + 1, D + 1)  # one fit per acquisition + final model
    if kernel == "iso":
        assert np.allclose(ls[:, 1:], ls[:, 1:2])


@pytest.mark.parametrize("kernel", ["ard", "iso"])
def test_bo_kernel_uses_dimension_scaled_prior(kernel):
    D = 20
    k = make_kernel(D, kernel)
    prior = k.lengthscale_prior
    assert math.isclose(float(prior.loc), math.sqrt(2) + 0.5 * math.log(D), rel_tol=1e-6)
    assert math.isclose(float(prior.scale), math.sqrt(3), rel_tol=1e-6)
    assert k.lengthscale.shape[-1] == (D if kernel == "ard" else 1)
