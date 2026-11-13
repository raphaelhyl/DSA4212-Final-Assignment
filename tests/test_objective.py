import numpy as np
import pandas as pd
import pytest

from objective import BudgetedObjective, BudgetExhausted


def sphere(x):
    return float(x @ x)


def test_exact_budget_then_raises(tmp_path):
    obj = BudgetedObjective(sphere, (-1, 1), 5, tmp_path / "r.csv")
    for _ in range(5):
        obj(np.zeros(3))
    assert obj.n_evals == 5
    with pytest.raises(BudgetExhausted):
        obj(np.zeros(3))
    assert obj.n_evals == 5


def test_rejects_infeasible():
    obj = BudgetedObjective(sphere, (-1, 1), 5)
    with pytest.raises(ValueError):
        obj(np.array([0.0, 1.5]))
    assert obj.n_evals == 0
    obj(np.array([1.0 + 1e-12, -1.0]))  # rounding slack is clipped, not rejected
    assert obj.rows[0]["x_0"] == 1.0


def test_csv_schema_and_best_so_far(tmp_path):
    obj = BudgetedObjective(sphere, (-1, 1), 3, tmp_path / "r.csv")
    for v in [0.5, 0.1, 0.9]:
        obj(np.array([v, 0.0]), generation=1, out_of_bounds=False)
    obj.save()
    df = pd.read_csv(tmp_path / "r.csv")
    expected = ["eval_idx", "x_0", "x_1", "f", "best_f", "t_wall", "t_obj", "generation", "out_of_bounds"]
    assert list(df.columns) == expected
    assert df["eval_idx"].tolist() == [1, 2, 3]
    assert np.allclose(df["best_f"], [0.25, 0.01, 0.01])
    assert (df["t_wall"].diff().dropna() >= 0).all()
