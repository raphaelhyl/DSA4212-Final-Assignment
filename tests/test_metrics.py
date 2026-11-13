import numpy as np

from analysis.metrics import ecdf, ert, hitting_times, log_final, success_rate
from analysis.stats import holm, interaction_delta, paired_comparison

# two runs, budget 4; regret already best-so-far
R = np.array([[1.0, 0.5, 0.05, 0.005], [0.8, 0.8, 0.2, 0.2]])


def test_hitting_times_and_censoring():
    assert np.array_equal(hitting_times(R, 0.1), [3.0, np.nan], equal_nan=True)
    assert np.array_equal(hitting_times(R, 0.9), [2.0, 1.0])


def test_success_and_ert():
    assert success_rate(R, 0.1) == 0.5
    # run 1 hits at 3, run 2 censored at 4 -> (3 + 4) / 1
    assert ert(R, 0.1) == 7.0
    assert ert(R, 1e-9) == np.inf


def test_ecdf():
    targets = np.array([0.1, 0.01])
    # t=1: none; t=2: none; t=3: run1 hits 0.1 -> 1/4; t=4: run1 hits both -> 2/4
    assert np.allclose(ecdf(R, targets), [0, 0, 0.25, 0.5])


def test_log_final():
    assert np.allclose(log_final(np.array([[1.0, 0.0]])), [-12.0])


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_paired_and_interaction():
    a, b = np.array([1.0, 2, 3, 4, 5]), np.array([0.0, 0, 0, 0, 0])
    res = paired_comparison(a, b)
    assert res["median_diff"] == 3.0 and res["ci_lo"] <= 3.0 <= res["ci_hi"]
    assert np.allclose(interaction_delta([2, 2], [1, 1], [0.5, 0], [0, 0]), [0.5, 1.0])
