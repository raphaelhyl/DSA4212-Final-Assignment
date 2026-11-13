from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")  # §5.8: one thread per process

import argparse
import json
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.diagnostics import cma_state_table
from benchmarks import algorithm_seed, make_instance
from config import ALGOS, BOUNDS, EXPERIMENTS, RESULTS, RunSpec, experiment_specs
from objective import BudgetedObjective

PACKAGES = ["numpy", "scipy", "pandas", "torch", "botorch", "gpytorch", "cma"]


def machine_info() -> dict:
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], check=False, capture_output=True, text=True
        ).stdout
    except OSError:
        cpu = platform.processor()
    return {
        "cpu": cpu.strip(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "threads_per_process": 1,
        "cpu_count": os.cpu_count(),
    }


def run_optimizer(spec: RunSpec, objective: BudgetedObjective, seed: int) -> tuple[dict, dict]:
    """Dispatch to the optimiser; returns (metadata, resolved settings)."""
    name, settings = ALGOS[spec.algo]
    common = {"objective": objective, "bounds": BOUNDS, "budget": spec.budget, "seed": seed, "D": spec.D}
    if name == "bo":
        from optimizers.bo import run_bo

        resolved = {"kernel": settings["kernel"], "n0": settings.get("n0_factor", 2) * spec.D}
        return run_bo(**common, **resolved), resolved
    if name == "cmaes":
        from optimizers.cmaes import run_cmaes

        lam = 4 + int(3 * np.log(spec.D))  # pycma default, made explicit for the 2x variant
        resolved = {"sigma0": settings.get("sigma0", 0.6), "popsize": settings.get("popsize_factor", 1) * lam}
        return run_cmaes(**common, **resolved), resolved
    from optimizers.random_search import run_random

    return run_random(**common), {}


def execute(spec: RunSpec) -> str:
    """Run one (condition, algorithm, seed) and write CSV, diagnostics and JSON (written last)."""
    import torch

    torch.set_num_threads(1)
    inst = make_instance(spec.benchmark, spec.D, spec.k, spec.atype, spec.seed, spec.theta)
    inst.save(RESULTS / "instances")
    out = spec.raw_dir / spec.run_id
    objective = BudgetedObjective(inst.objective, BOUNDS, spec.budget, Path(f"{out}.csv"))
    seed = algorithm_seed(spec.seed)
    meta, settings = run_optimizer(spec, objective, seed)
    wall = time.perf_counter() - objective.t_start
    objective.save()
    if "lengthscales" in meta:
        ls = meta.pop("lengthscales")
        cols = ["n_train"] + [f"ls_{i}" for i in range(spec.D)]
        pd.DataFrame(ls, columns=cols).to_csv(f"{out}_lengthscales.csv", index=False)
    if "state" in meta:
        state = meta.pop("state")
        np.savez_compressed(f"{out}_cma_eig.npz", **state)
        # §9: principal angles are computed after the run; the optimiser never sees A.
        cma_state_table(state, inst.A).to_csv(f"{out}_cma_state.csv", index=False)
    hist = objective.history()
    record = {
        "run_id": spec.run_id,
        "experiment": spec.experiment,
        "benchmark": spec.benchmark,
        "algo": spec.algo,
        "optimizer": ALGOS[spec.algo][0],
        "settings": settings,
        "D": spec.D,
        "k": spec.k,
        "atype": spec.atype,
        "theta": spec.theta,
        "seed": spec.seed,
        "algorithm_seed": seed,
        "instance_id": inst.instance_id,
        "budget": spec.budget,
        "evals_used": objective.n_evals,
        "wall_time": wall,
        "objective_time": objective.t_obj,
        "overhead_time": wall - objective.t_obj,
        "final_best_f": float(hist["best_f"].iloc[-1]),
        "frac_out_of_bounds": float(hist["out_of_bounds"].mean()) if meta.get("popsize") else None,
        **meta,
        "packages": {p: version(p) for p in PACKAGES},
        "machine": machine_info(),
    }
    with open(f"{out}.json", "w") as fh:
        json.dump(record, fh, indent=1, default=float)
    return spec.run_id


def pending(specs: list[RunSpec]) -> list[RunSpec]:
    """§9: a run is complete iff its JSON exists; complete runs are skipped."""
    todo = [s for s in specs if not (s.raw_dir / f"{s.run_id}.json").exists()]
    # expensive runs first for better load balancing
    return sorted(todo, key=lambda s: (ALGOS[s.algo][0] != "bo", -s.budget, -s.D))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", default=None, help=f"comma-separated subset of {EXPERIMENTS}, or 'all'"
    )
    parser.add_argument("--pilot", action="store_true", help="run the §8 pilot")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    names = (
        ["pilot"] if args.pilot else (EXPERIMENTS if args.experiment == "all" else args.experiment.split(","))
    )
    todo = pending([s for n in names for s in experiment_specs(n)])
    print(f"{len(todo)} runs to do", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, s) for s in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(todo)} {time.time() - t0:7.0f}s] {fut.result()}", flush=True)


if __name__ == "__main__":
    main()
