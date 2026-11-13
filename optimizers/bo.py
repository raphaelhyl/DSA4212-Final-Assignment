from __future__ import annotations

import contextlib
import math
import warnings

import gpytorch
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

from objective import BudgetedObjective, BudgetExhausted

DTYPE = torch.float64
NUM_RESTARTS, RAW_SAMPLES = 10, 512
MIN_LENGTHSCALE = 2.5e-2  # same numerical floor as BoTorch's dimension-scaled default


def make_kernel(D: int, kernel: str) -> MaternKernel:
    """Matern-5/2 with the dimension-scaled LogNormal length-scale prior (§4.1), ARD or isotropic."""
    prior = LogNormalPrior(loc=math.sqrt(2) + 0.5 * math.log(D), scale=math.sqrt(3))
    return MaternKernel(
        nu=2.5,
        ard_num_dims=D if kernel == "ard" else None,
        lengthscale_prior=prior,
        lengthscale_constraint=GreaterThan(MIN_LENGTHSCALE, transform=None, initial_value=prior.mode),
    )


def fit_model(U: torch.Tensor, Y: torch.Tensor, kernel: str) -> tuple[SingleTaskGP, bool]:
    """Fit a fresh GP on unit-cube inputs (no warm start); returns (model, fit_succeeded)."""
    model = SingleTaskGP(
        U, Y, covar_module=make_kernel(U.shape[-1], kernel), outcome_transform=Standardize(m=1)
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)
        return model, True
    except ModelFittingError:
        return model, False  # keep prior-mode hyperparameters; recorded in metadata


# GPyTorch switches from Cholesky to slow iterative CG/Lanczos solves above 800 points by default
# (~25x slower fits at n = 1000 in our timing); keep exact inference at every training-set size.
def exact_gp() -> contextlib.ExitStack:
    stack = contextlib.ExitStack()
    stack.enter_context(
        gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False)
    )
    stack.enter_context(gpytorch.settings.max_cholesky_size(10_000))
    return stack


def run_bo(
    objective: BudgetedObjective,
    bounds: tuple[float, float],
    budget: int,
    seed: int,
    D: int,
    kernel: str = "ard",
    n0: int | None = None,
) -> dict:
    """Sequential BO with LogEI until the budget is used; length scales are recorded after every fit."""
    torch.manual_seed(seed)
    lo, hi = bounds
    n0 = 2 * D if n0 is None else n0

    def evaluate(u: torch.Tensor) -> float:
        return objective(lo + (hi - lo) * u.numpy())  # model works in [0,1]^D (§4.1)

    U, Y, lengthscales, fit_failures = [], [], [], 0
    try:
        sobol = torch.quasirandom.SobolEngine(D, scramble=True, seed=seed).draw(n0).to(DTYPE)
        for u in sobol:
            U.append(u)
            Y.append(evaluate(u))
        unit_bounds = torch.stack([torch.zeros(D, dtype=DTYPE), torch.ones(D, dtype=DTYPE)])
        while True:
            train_U, train_Y = torch.stack(U), torch.tensor(Y, dtype=DTYPE).unsqueeze(-1)
            with exact_gp():
                model, ok = fit_model(train_U, train_Y, kernel)
            fit_failures += not ok
            ls = model.covar_module.lengthscale.detach().reshape(-1).numpy()
            lengthscales.append(np.concatenate([[len(Y)], np.broadcast_to(ls, (D,))]))
            acqf = LogExpectedImprovement(model, best_f=train_Y.min(), maximize=False)
            with warnings.catch_warnings(), exact_gp():
                warnings.simplefilter("ignore")
                cand, _ = optimize_acqf(
                    acqf,
                    unit_bounds,
                    q=1,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,
                    options={"seed": seed + len(Y)},
                )
            u = cand.detach().squeeze(0).clamp(0, 1)
            Y.append(evaluate(u))
            U.append(u)
    except BudgetExhausted:
        pass
    return {
        "termination": "budget",
        "n0": n0,
        "kernel": kernel,
        "fit_failures": fit_failures,
        "lengthscales": np.array(lengthscales),
    }
