import numpy as np
import pytest
from scipy.linalg import null_space, subspace_angles

from benchmarks import algorithm_seed, make_A, make_instance

SEEDS = [0, 1, 7, 1000]
CASES = [
    ("aligned", 0.0),
    ("rotated", 0.0),
    ("partial", np.pi / 8),
    ("partial", np.pi / 4),
    ("partial", np.pi / 2),
]


@pytest.mark.parametrize("benchmark", ["quad", "rosen"])
@pytest.mark.parametrize("atype,theta", CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_instance_properties(benchmark, atype, theta, seed):
    D, k = 20, 2
    inst = make_instance(benchmark, D, k, atype, seed, theta)
    A = inst.A
    assert np.linalg.norm(A.T @ A - np.eye(k)) < 1e-10
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(10_000, D))
    assert min(inst.objective(x) for x in X) >= 0.0
    assert abs(inst.objective(inst.x_star)) < 1e-12
    # null-space invariance: moving within null(A^T) leaves f unchanged
    N = null_space(A.T)
    for _ in range(20):
        v = rng.standard_normal(N.shape[1]) * 0.1
        assert abs(inst.objective(inst.x_star + N @ v) - inst.objective(inst.x_star)) < 1e-12
    # determinism
    again = make_instance(benchmark, D, k, atype, seed, theta)
    assert np.array_equal(again.A, A) and np.array_equal(again.x_star, inst.x_star)


@pytest.mark.parametrize("theta", [0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
@pytest.mark.parametrize("seed", SEEDS)
def test_partial_rotation_principal_angles(theta, seed):
    D, k = 20, 2
    A = make_A(D, k, "partial", seed, theta)
    angles = subspace_angles(A, np.eye(D)[:, :k])
    assert np.allclose(angles, theta, atol=1e-7)


def test_partial_zero_equals_aligned():
    assert np.array_equal(make_A(20, 2, "partial", 3, 0.0), make_A(20, 2, "aligned", 3))


def test_x_star_shared_and_off_centre():
    a = make_instance("quad", 20, 2, "aligned", 5)
    r = make_instance("quad", 20, 2, "rotated", 5)
    full = make_instance("quad", 20, 20, "aligned", 5)
    assert np.array_equal(a.x_star, r.x_star) and np.array_equal(a.x_star, full.x_star)
    assert np.all(np.abs(a.x_star) <= 0.5) and np.linalg.norm(a.x_star) > 0.1


def test_algorithm_seed_independent_of_instance():
    assert algorithm_seed(0) != algorithm_seed(1)
    assert algorithm_seed(3) == algorithm_seed(3)
