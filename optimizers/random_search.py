import numpy as np

from objective import BudgetedObjective, BudgetExhausted


def run_random(
    objective: BudgetedObjective, bounds: tuple[float, float], budget: int, seed: int, D: int
) -> dict:
    """Evaluate uniform samples in the box until the budget is exhausted."""
    rng = np.random.default_rng(seed)
    try:
        while True:
            objective(rng.uniform(*bounds, size=D))
    except BudgetExhausted:
        pass
    return {"termination": "budget"}
