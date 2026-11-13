from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

from benchmarks import theta_tag

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
BOUNDS = (-1.0, 1.0)

FINAL_SEEDS = range(20)
PILOT_SEEDS = range(1000, 1005)
SENS_SEEDS = range(2000, 2020)

BUDGET = 200
PILOT_BUDGET = 100
# §5.3 extended budget, frozen after the pilot from timing only (see README decision log).
BUDGET_EXT = {"bo_ard": 500, "cmaes": 1000, "random": 1000}

# algorithm label -> (optimiser, settings). Variants are used only by the sensitivity study.
ALGOS: dict[str, tuple[str, dict]] = {
    "bo_ard": ("bo", {"kernel": "ard"}),
    "bo_iso": ("bo", {"kernel": "iso"}),
    "cmaes": ("cmaes", {}),
    "random": ("random", {}),
    "cmaes_sig0.3": ("cmaes", {"sigma0": 0.3}),
    "cmaes_sig1.0": ("cmaes", {"sigma0": 1.0}),
    "cmaes_pop2x": ("cmaes", {"popsize_factor": 2}),
    "bo_ard_n0D": ("bo", {"kernel": "ard", "n0_factor": 1}),
}
CORE = ["bo_ard", "cmaes", "random"]
WITH_ISO = ["bo_ard", "bo_iso", "cmaes", "random"]
THETAS = [0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    benchmark: str
    algo: str
    D: int
    k: int
    atype: str
    seed: int
    budget: int
    theta: float = field(default=0.0)

    @property
    def run_id(self) -> str:
        """§9 run_id format."""
        tag = f"{self.atype}{theta_tag(self.theta)}"
        return f"{self.experiment}_{self.benchmark}_{self.algo}_D{self.D}_k{self.k}_{tag}_s{self.seed}"

    @property
    def raw_dir(self) -> Path:
        return RESULTS / "raw" / self.experiment


def _grid(experiment, benchmark, algos, Ds, ks, atypes, seeds, budget, thetas=(0.0,)) -> list[RunSpec]:
    specs = []
    for algo, D, k, atype, theta, seed in product(algos, Ds, ks, atypes, thetas, seeds):
        if k <= D:
            b = budget[algo] if isinstance(budget, dict) else budget
            specs.append(RunSpec(experiment, benchmark, algo, D, k, atype, seed, b, theta))
    return specs


def experiment_specs(name: str) -> list[RunSpec]:
    """Runs *owned* by each experiment; reused conditions (§6) are selected at analysis time."""
    S = FINAL_SEEDS
    if name == "exp1":
        return [
            s for D in (2, 5, 10, 20) for s in _grid("exp1", "quad", CORE, [D], [D], ["aligned"], S, BUDGET)
        ]
    if name == "exp1ext":
        return _grid("exp1ext", "quad", CORE, [20], [20], ["aligned"], S, BUDGET_EXT)
    if name == "exp2":  # k = 20 is exp1 D = 20
        return _grid("exp2", "quad", CORE, [20], [2, 5, 10], ["aligned"], S, BUDGET)
    if name == "exp3":  # aligned BO-ARD/CMA-ES/random are exp2 k = 2
        return _grid("exp3", "quad", WITH_ISO, [20], [2], ["rotated"], S, BUDGET) + _grid(
            "exp3", "quad", ["bo_iso"], [20], [2], ["aligned"], S, BUDGET
        )
    if name == "exp4":  # theta = 0 is the aligned instance (A(0) == E exactly)
        return _grid(
            "exp4", "quad", ["bo_ard", "bo_iso", "cmaes"], [20], [2], ["partial"], S, BUDGET, THETAS[1:]
        )
    if name == "exp5":
        return _grid("exp5", "rosen", WITH_ISO, [20], [2], ["aligned", "rotated"], S, BUDGET)
    if name == "sens":
        algos = ["bo_ard", "bo_ard_n0D", "cmaes", "cmaes_sig0.3", "cmaes_sig1.0", "cmaes_pop2x"]
        return _grid("sens", "quad", algos, [20], [2], ["aligned", "rotated"], SENS_SEEDS, BUDGET)
    if name == "pilot":
        P, B = PILOT_SEEDS, PILOT_BUDGET
        return (
            [s for D in (2, 10, 20) for s in _grid("pilot", "quad", CORE, [D], [D], ["aligned"], P, B)]
            + _grid("pilot", "quad", CORE, [20], [2], ["aligned"], P, B)
            + _grid("pilot", "quad", WITH_ISO, [20], [2], ["rotated"], P, B)
            + _grid("pilot", "quad", ["bo_ard", "bo_iso", "cmaes"], [20], [2], ["partial"], P, B, [np.pi / 4])
            + [RunSpec("pilotext", "quad", "bo_ard", 20, 20, "aligned", 1000, 1000)]  # extended-budget timing
        )
    raise ValueError(f"unknown experiment {name!r}")


EXPERIMENTS = ["exp1", "exp1ext", "exp2", "exp3", "exp4", "exp5", "sens"]
