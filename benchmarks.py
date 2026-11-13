from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# §5.6: one SeedSequence per seed, split into independent instance / algorithm streams.
# Within the instance stream, x* and A get their own sub-streams so that x* depends only on
# (seed, D): aligned and rotated instances, and Exp 1 D=20 vs Exp 2 k=20, share x* by construction.
_STREAM_INSTANCE, _STREAM_ALGO = 0, 1
_SUB_XSTAR, _SUB_ROTATION, _SUB_PARTIAL = 0, 1, 2


def _instance_rng(seed: int, sub: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(_STREAM_INSTANCE, sub)))


def algorithm_seed(seed: int) -> int:
    """Integer seed for optimiser randomness, independent of the instance stream (§5.6)."""
    ss = np.random.SeedSequence(seed, spawn_key=(_STREAM_ALGO,))
    return int(ss.generate_state(1)[0] % (2**31 - 1))


def _orthonormal(G: np.ndarray) -> np.ndarray:
    """QR with signs fixed so that diag(R) > 0 (§3.3)."""
    Q, R = np.linalg.qr(G)
    return Q * np.sign(np.diag(R))


def make_A(D: int, k: int, atype: str, seed: int, theta: float = 0.0) -> np.ndarray:
    """D x k matrix with orthonormal columns: 'aligned', 'rotated' or 'partial' (angle theta)."""
    E = np.eye(D)[:, :k]
    if atype == "aligned":
        return E
    if atype == "rotated":
        return _orthonormal(_instance_rng(seed, _SUB_ROTATION).standard_normal((D, k)))
    if atype == "partial":
        G = _instance_rng(seed, _SUB_PARTIAL).standard_normal((D - k, k))
        Q_perp = np.vstack([np.zeros((k, k)), _orthonormal(G)])  # first k rows zero => E^T Q_perp = 0
        return np.cos(theta) * E + np.sin(theta) * Q_perp
    raise ValueError(f"unknown atype {atype!r}")


def make_x_star(D: int, seed: int) -> np.ndarray:
    """Optimum location, Uniform[-0.5, 0.5]^D; deliberately not the domain centre (§3.1)."""
    return _instance_rng(seed, _SUB_XSTAR).uniform(-0.5, 0.5, size=D)


def subspace_quadratic(x: np.ndarray, A: np.ndarray, x_star: np.ndarray) -> float:
    """f(x) = ||A^T (x - x*)||^2 / k (§3.1)."""
    z = A.T @ (x - x_star)
    return float(z @ z / A.shape[1])


def embedded_rosenbrock(x: np.ndarray, A: np.ndarray, x_star: np.ndarray) -> float:
    """Rosenbrock in u = 1 + 2 A^T (x - x*), so f* = 0 at x = x* (§3.2)."""
    u = 1.0 + 2.0 * (A.T @ (x - x_star))
    return float(np.sum(100.0 * (u[1:] - u[:-1] ** 2) ** 2 + (1.0 - u[:-1]) ** 2))


BENCHMARKS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
    "quad": subspace_quadratic,
    "rosen": embedded_rosenbrock,
}


@dataclass(frozen=True)
class Instance:
    """A fully specified problem instance; the optimiser only ever sees `objective`."""

    benchmark: str
    D: int
    k: int
    atype: str
    theta: float
    seed: int
    A: np.ndarray
    x_star: np.ndarray

    @property
    def instance_id(self) -> str:
        return f"{self.benchmark}_D{self.D}_k{self.k}_{self.atype}{theta_tag(self.theta)}_s{self.seed}"

    def objective(self, x: np.ndarray) -> float:
        return BENCHMARKS[self.benchmark](x, self.A, self.x_star)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / f"{self.instance_id}.npz", A=self.A, x_star=self.x_star)


def theta_tag(theta: float) -> str:
    """Compact angle label in units of pi/8 (e.g. 't2' = pi/4); empty for theta = 0."""
    return "" if theta == 0 else f"t{round(theta / (np.pi / 8))}"


def make_instance(benchmark: str, D: int, k: int, atype: str, seed: int, theta: float = 0.0) -> Instance:
    """Build the (A, x*) instance for one condition and seed."""
    A = make_A(D, k, atype, seed, theta)
    return Instance(benchmark, D, k, atype, theta, seed, A, make_x_star(D, seed))
