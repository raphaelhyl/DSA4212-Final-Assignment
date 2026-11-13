from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator

from analysis.diagnostics import cma_coordinate_oob, lengthscale_summary, random_subspace_angle
from analysis.metrics import FLOOR, load_runs, log_final, regret_matrix, select, summarise
from analysis.plots import (
    COLORS,
    LABELS,
    delta_strip,
    ecdf_curve,
    median_iqr_vs,
    median_trajectory,
    regret_band,
    save,
)
from analysis.stats import bootstrap_median_ci, holm, interaction_delta, paired_comparison
from config import BOUNDS, BUDGET, BUDGET_EXT, PILOT_BUDGET, RESULTS, THETAS

FIG, SUM = RESULTS / "figures", RESULTS / "summaries"
CORE, WITH_ISO = ["bo_ard", "cmaes", "random"], ["bo_ard", "bo_iso", "cmaes", "random"]
TABLES: list[tuple[str, pd.DataFrame]] = []


class Data:
    """Selects paired run sets by condition, regardless of which experiment owns them (§6 reuse)."""

    def __init__(self, meta: pd.DataFrame, budget: int, experiments: list[str]):
        self.meta, self.budget = meta[meta["experiment"].isin(experiments)], budget

    def runs(self, algo: str, budget: int | None = None, **cond) -> pd.DataFrame:
        cond.setdefault("benchmark", "quad")
        if cond.get("atype") == "partial" and np.isclose(cond.get("theta", 0), 0):
            cond["atype"] = "aligned"  # A(0) == E exactly
        if cond.get("atype") == "aligned":
            cond.pop("theta", None)
        return select(self.meta, algo=algo, budget=budget or self.budget, **cond)

    def R(self, algo: str, budget: int | None = None, **cond) -> np.ndarray | None:
        runs = self.runs(algo, budget, **cond)
        return regret_matrix(runs) if len(runs) else None

    def seeds(self, algo: str, **cond) -> list[int]:
        return self.runs(algo, **cond)["seed"].tolist()


def log_x(ax, ticks: list[int]) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(ticks, [str(t) for t in ticks])


def table(name: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    TABLES.append((name, df))
    df.to_csv(SUM / f"{name}.csv", index=False)
    return df


def compare_rows(pairs: list[tuple[dict, np.ndarray, np.ndarray]]) -> list[dict]:
    """Paired comparisons with Holm correction across the listed conditions."""
    rows = [{**label, **paired_comparison(a, b)} for label, a, b in pairs]
    for r, p in zip(rows, holm([r["p_wilcoxon"] for r in rows])):
        r["p_holm"] = p
    return rows


def delta_row(label: dict, d: np.ndarray) -> dict:
    lo, hi = bootstrap_median_ci(d)
    return {**label, "n": len(d), "median_delta": float(np.median(d)), "ci_lo": lo, "ci_hi": hi}


# ---------------------------------------------------------------- Experiment 1 (H1)
def exp1(data: Data, Ds: list[int]) -> None:
    fig, axes = plt.subplots(2, len(Ds), figsize=(3.2 * len(Ds), 5.6), squeeze=False)
    summ, pairs = [], []
    for j, D in enumerate(Ds):
        Rs = {a: data.R(a, D=D, k=D, atype="aligned") for a in CORE}
        for a, R in Rs.items():
            if R is None:
                continue
            regret_band(axes[0, j], R, a)
            ecdf_curve(axes[1, j], R, a)
            summ.append({"D": D, "algo": a, **summarise(R)})
        axes[0, j].set_title(f"D = k = {D}")
        if Rs["bo_ard"] is not None and Rs["cmaes"] is not None:
            pairs.append(
                ({"D": D, "a": "bo_ard", "b": "cmaes"}, log_final(Rs["bo_ard"]), log_final(Rs["cmaes"]))
            )
    axes[0, 0].legend()
    save(fig, FIG / "fig1_exp1_regret_ecdf.png")
    table("exp1_summary", summ)
    table("exp1_paired_bo_vs_cma", compare_rows(pairs))
    finals = {a: [data.R(a, D=D, k=D, atype="aligned")[:, -1] for D in Ds] for a in CORE}
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for a in CORE:
        median_iqr_vs(ax, Ds, finals[a], a)
    ax.set(xlabel="ambient dimension D (= k)", ylabel=f"regret at B = {data.budget}")
    log_x(ax, Ds)
    ax.legend()
    save(fig, FIG / "fig3a_final_vs_D.png")


def exp1_extended(data: Data) -> None:
    Rs = {a: data.R(a, budget=BUDGET_EXT[a], D=20, k=20, atype="aligned") for a in CORE}
    if any(R is None for R in Rs.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, R in Rs.items():
        regret_band(axes[0], R, a)
        ecdf_curve(axes[1], R, a)
    axes[0].set_title("D = k = 20, extended budget")
    axes[0].legend()
    save(fig, FIG / "fig1b_exp1_extended.png")
    rows = []
    for t in [t for t in (50, 100, 200, 500, 1000) if t <= BUDGET_EXT["bo_ard"]]:
        L = {a: np.log10(R[:, t - 1] + FLOOR) for a, R in Rs.items()}
        rows.append(({"t": t, "a": "bo_ard", "b": "cmaes"}, L["bo_ard"], L["cmaes"]))
    table("exp1ext_paired_bo_vs_cma_by_t", compare_rows(rows))
    table("exp1ext_summary", [{"algo": a, **summarise(R)} for a, R in Rs.items()])


# ---------------------------------------------------------------- Experiment 2 (H2)
def exp2(data: Data, ks: list[int]) -> None:
    fig, axes = plt.subplots(2, len(ks), figsize=(3.2 * len(ks), 5.6), squeeze=False)
    summ, pairs = [], []
    for j, k in enumerate(ks):
        Rs = {a: data.R(a, D=20, k=k, atype="aligned") for a in CORE}
        for a, R in Rs.items():
            regret_band(axes[0, j], R, a)
            ecdf_curve(axes[1, j], R, a)
            summ.append({"k": k, "algo": a, **summarise(R)})
        axes[0, j].set_title(f"D = 20, k = {k} (aligned)")
        pairs.append(({"k": k, "a": "bo_ard", "b": "cmaes"}, log_final(Rs["bo_ard"]), log_final(Rs["cmaes"])))
    axes[0, 0].legend()
    save(fig, FIG / "fig2_exp2_regret_ecdf.png")
    table("exp2_summary", summ)
    table("exp2_paired_bo_vs_cma", compare_rows(pairs))
    # Improvement from k=20 -> k (paired by seed; x* shared by construction) and its BO-vs-CMA interaction.
    L = {(a, k): log_final(data.R(a, D=20, k=k, atype="aligned")) for a in CORE for k in ks}
    rows = []
    for k in ks[:-1]:
        for a in CORE:
            rows.append(delta_row({"k": k, "quantity": f"{a}: L(k) - L(k=20)"}, L[(a, k)] - L[(a, 20)]))
        rows.append(
            delta_row(
                {"k": k, "quantity": "interaction BO-ARD minus CMA-ES"},
                (L[("bo_ard", k)] - L[("bo_ard", 20)]) - (L[("cmaes", k)] - L[("cmaes", 20)]),
            )
        )
    table("exp2_improvement_k_vs_20", rows)
    # final regret vs k with privileged oracle (Exp 1 at D = k)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for a in CORE:
        median_iqr_vs(ax, ks, [data.R(a, D=20, k=k, atype="aligned")[:, -1] for k in ks], a)
    for a in ["bo_ard", "cmaes"]:
        orc = [data.R(a, D=k, k=k, atype="aligned") for k in ks]
        if all(o is not None for o in orc):
            median_iqr_vs(ax, ks, [o[:, -1] for o in orc], a, ls=":", label=f"{LABELS[a]} oracle (D = k)")
    ax.set(
        xlabel="effective dimension k (D = 20)",
        ylabel=f"regret at B = {data.budget}",
    )
    log_x(ax, ks)
    ax.legend(fontsize=7)
    save(fig, FIG / "fig3b_final_vs_k_with_oracle.png")
    lengthscale_figure(data, ks)


def lengthscale_figure(data: Data, ks: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.4))
    styles = ["-", "--", ":", "-."]
    rows = []
    for k, ls in zip(ks[:-1], styles):
        active = np.arange(20) < k
        S = [
            lengthscale_summary(pd.read_csv(f"{p}_lengthscales.csv"), active)
            for p in data.runs("bo_ard", D=20, k=k, atype="aligned")["path"]
        ]
        n = S[0]["n_train"].to_numpy()
        act = np.vstack([s["ls_active_median"] for s in S])
        ina = np.vstack([s["ls_inactive_median"] for s in S])
        median_trajectory(ax, n, act, COLORS["bo_ard"], f"k={k} active", ls)
        median_trajectory(ax, n, ina, "#52514e", f"k={k} inactive", ls)
        rat = np.vstack([s["ratio_min_active_to_min_inactive"] for s in S])
        rows.append(
            {
                "k": k,
                "final_ls_active_median": np.median(act[:, -1]),
                "final_ls_inactive_median": np.median(ina[:, -1]),
                "final_ratio_min_active_to_min_inactive_median": np.median(rat[:, -1]),
                "frac_runs_all_active_shorter_than_all_inactive": float(np.mean(rat[:, -1] < 1)),
            }
        )
    ax.set(yscale="log", xlabel="training points", ylabel="learned length scale (unit cube)")
    ax.set_title("Exp 2: BO-ARD length scales, D = 20")
    ax.legend(fontsize=7, ncol=2)
    save(fig, FIG / "fig6a_exp2_lengthscales.png")
    table("exp2_lengthscales_final", rows)


# ---------------------------------------------------------------- Experiments 3 & 5 (H3)
def orientation(data: Data, benchmark: str, tag: str) -> None:
    algos = [a for a in WITH_ISO if data.R(a, benchmark=benchmark, D=20, k=2, atype="rotated") is not None]
    if not algos:
        return
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
    L, summ, rows = {}, [], []
    for j, atype in enumerate(["aligned", "rotated"]):
        for a in algos:
            R = data.R(a, benchmark=benchmark, D=20, k=2, atype=atype)
            if R is None:
                continue
            regret_band(axes[0, j], R, a)
            ecdf_curve(axes[1, j], R, a)
            L[(a, atype)] = log_final(R)
            summ.append({"atype": atype, "algo": a, **summarise(R)})
        axes[0, j].set_title(f"{benchmark}, D = 20, k = 2, {atype}")
    axes[0, 0].legend()
    save(fig, FIG / f"fig4a_{tag}_regret_ecdf.png")
    table(f"{tag}_summary", summ)
    pairs = [
        ({"algo": a, "comparison": "rotated - aligned"}, L[(a, "rotated")], L[(a, "aligned")])
        for a in algos
        if (a, "aligned") in L
    ]
    pairs += [
        ({"algo": a, "comparison": f"{a} - cmaes, {at}"}, L[(a, at)], L[("cmaes", at)])
        for at in ["aligned", "rotated"]
        for a in algos
        if a != "cmaes" and (a, at) in L
    ]
    pairs += [
        ({"algo": "bo_ard", "comparison": f"bo_ard - bo_iso, {at}"}, L[("bo_ard", at)], L[("bo_iso", at)])
        for at in ["aligned", "rotated"]
        if ("bo_iso", at) in L
    ]
    table(f"{tag}_paired", compare_rows(pairs))
    deltas, cis = {}, {}
    for bo in ["bo_ard", "bo_iso"]:
        if (bo, "aligned") in L and (bo, "rotated") in L:
            d = interaction_delta(
                L[(bo, "rotated")], L[(bo, "aligned")], L[("cmaes", "rotated")], L[("cmaes", "aligned")]
            )
            deltas[bo], cis[bo] = d, bootstrap_median_ci(d)
            rows.append(delta_row({"bo": bo}, d))
    table(f"{tag}_interaction_delta", rows)
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    delta_strip(ax, deltas, cis)
    ax.set_title(f"{tag}: interaction Δ (>0: rotation hurts BO more)")
    save(fig, FIG / f"fig4b_{tag}_delta.png")


# ---------------------------------------------------------------- Experiment 4 (H3-mechanism)
def exp4(data: Data, thetas: list[float]) -> None:
    algos = ["bo_ard", "bo_iso", "cmaes"]
    deg = [np.degrees(t) for t in thetas]
    cond = {"D": 20, "k": 2, "atype": "partial"}
    L = {(a, i): log_final(data.R(a, theta=t, **cond)) for a in algos for i, t in enumerate(thetas)}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for a in algos:
        median_iqr_vs(axes[0], deg, [data.R(a, theta=t, **cond)[:, -1] for t in thetas], a)
    axes[0].set(xlabel="rotation angle θ (deg)", ylabel=f"regret at B = {data.budget}", xticks=deg)
    axes[0].legend()
    rows, diag = [], []
    for bo in ["bo_ard", "bo_iso"]:
        for i in range(1, len(thetas)):
            d = interaction_delta(L[(bo, i)], L[(bo, 0)], L[("cmaes", i)], L[("cmaes", 0)])
            rows.append(delta_row({"bo": bo, "theta_deg": deg[i]}, d))
    for a in algos:
        for i in range(1, len(thetas)):
            rows.append(
                delta_row({"bo": f"{a} alone: L(θ) - L(0)", "theta_deg": deg[i]}, L[(a, i)] - L[(a, 0)])
            )
    table("exp4_delta_vs_theta", rows)
    ratios, angles = [], []
    for i, t in enumerate(thetas):
        S = [
            lengthscale_summary(pd.read_csv(f"{p}_lengthscales.csv"), np.arange(20) < 2)
            for p in data.runs("bo_ard", theta=t, **cond)["path"]
        ]
        r = np.array([s["ratio_min_active_to_min_inactive"].iloc[-1] for s in S])
        A = [pd.read_csv(f"{p}_cma_state.csv") for p in data.runs("cmaes", theta=t, **cond)["path"]]
        ang = np.array([s["angle_bottom_k"].iloc[-1] for s in A])
        ratios.append(r)
        angles.append(ang)
        diag.append(
            {
                "theta_deg": deg[i],
                "bo_ard_ratio_min_active_min_inactive_median": np.median(r),
                "bo_ard_ls_active_median": np.median([s["ls_active_median"].iloc[-1] for s in S]),
                "cma_final_angle_bottom_k_median_deg": np.median(ang),
                "cma_final_angle_top_k_median_deg": np.median([s["angle_top_k"].iloc[-1] for s in A]),
            }
        )
        median_trajectory(
            axes[2],
            A[0]["evals"].to_numpy()[1:],  # generation 1 dropped: C = I, eigenvectors arbitrary
            np.vstack([s["angle_bottom_k"].to_numpy()[1:] for s in A]),
            plt.cm.viridis(i / (len(thetas) - 1)),
            f"θ = {deg[i]:.1f}°",
        )
    axes[1].boxplot(ratios, positions=deg, widths=6, manage_ticks=False)
    axes[1].axhline(1, color="#52514e", ls="--", lw=1)
    axes[1].set(yscale="log", xlabel="θ (deg)", ylabel="BO-ARD: min ls(e1,e2) / min ls(others)", xticks=deg)
    axes[2].set(
        xlabel="evaluations", ylabel="CMA-ES: max principal angle (deg)\nbottom-2 eigvecs of C vs span(A)"
    )
    axes[2].axhline(
        random_subspace_angle(20, 2), color="#52514e", ls=":", lw=1, label="chance (random subspace)"
    )
    axes[2].legend(fontsize=7)
    save(fig, FIG / "fig5_exp4_theta_dose_response.png")
    table("exp4_diagnostics", diag)
    table(
        "exp4_summary",
        [
            {"theta_deg": deg[i], "algo": a, **summarise(data.R(a, theta=t, **cond))}
            for a in algos
            for i, t in enumerate(thetas)
        ],
    )


def cma_alignment_figure(data: Data) -> None:
    """CMA-ES alignment over generations, aligned vs rotated (Exp 3 conditions)."""
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for atype, ls in [("aligned", "-"), ("rotated", "--")]:
        S = [pd.read_csv(f"{p}_cma_state.csv") for p in data.runs("cmaes", D=20, k=2, atype=atype)["path"]]
        median_trajectory(
            ax,
            S[0]["evals"].to_numpy()[1:],
            np.vstack([s["angle_bottom_k"].to_numpy()[1:] for s in S]),
            COLORS["cmaes"],
            f"{atype}: bottom-2 eigvecs",
            ls,
        )
        median_trajectory(
            ax,
            S[0]["evals"].to_numpy()[1:],
            np.vstack([s["angle_top_k"].to_numpy()[1:] for s in S]),
            "#52514e",
            f"{atype}: top-2 eigvecs",
            ls,
        )
    ax.axhline(random_subspace_angle(20, 2), color="#52514e", ls=":", lw=1, label="chance (random subspace)")
    ax.set(xlabel="evaluations", ylabel="max principal angle to span(A) (deg)")
    ax.set_title("CMA-ES covariance alignment, D = 20, k = 2")
    ax.legend(fontsize=7)
    save(fig, FIG / "fig6b_cma_alignment.png")


def cma_diagnostics(data: Data) -> None:
    """Final-generation CMA-ES state per condition: step size, conditioning, alignment, boundary activity."""
    conds = [{"D": D, "k": D, "atype": "aligned"} for D in (2, 5, 10, 20)]
    conds += [{"D": 20, "k": k, "atype": "aligned"} for k in (2, 5, 10)] + [
        {"D": 20, "k": 2, "atype": "rotated"}
    ]
    conds += [{"D": 20, "k": 2, "atype": at, "benchmark": "rosen"} for at in ("aligned", "rotated")]
    rows = []
    for c in conds:
        paths = data.runs("cmaes", **c)["path"]
        st = [pd.read_csv(f"{p}_cma_state.csv").iloc[-1] for p in paths]
        oob = [cma_coordinate_oob(dict(np.load(f"{p}_cma_eig.npz")), BOUNDS) for p in paths]
        rows.append(
            {
                "benchmark": c.get("benchmark", "quad"),
                "D": c["D"],
                "k": c["k"],
                "atype": c["atype"],
                "generations": int(np.median([s["generation"] for s in st])),
                "final_sigma_median": np.median([s["sigma"] for s in st]),
                "final_cond_C_median": np.median([s["cond_C"] for s in st]),
                "final_angle_bottom_k_median": np.median([s["angle_bottom_k"] for s in st]),
                "chance_angle_bottom_k": random_subspace_angle(c["D"], c["k"]) if c["k"] < c["D"] else 0.0,
                "coord_oob_prob_gen1_median": np.median([o[0] for o in oob]),
                "coord_oob_prob_final_median": np.median([o[-1] for o in oob]),
            }
        )
    table("cma_diagnostics", rows)


# ---------------------------------------------------------------- Sensitivity
def sensitivity(meta: pd.DataFrame) -> None:
    data = Data(meta, BUDGET, ["sens"])
    if data.R("cmaes", D=20, k=2, atype="aligned") is None:
        return
    L = {
        (a, at): log_final(data.R(a, D=20, k=2, atype=at))
        for a in ["bo_ard", "bo_ard_n0D", "cmaes", "cmaes_sig0.3", "cmaes_sig1.0", "cmaes_pop2x"]
        for at in ["aligned", "rotated"]
    }
    combos = [
        ("bo_ard", "cmaes"),
        ("bo_ard_n0D", "cmaes"),
        ("bo_ard", "cmaes_sig0.3"),
        ("bo_ard", "cmaes_sig1.0"),
        ("bo_ard", "cmaes_pop2x"),
    ]
    rows = []
    for bo, cma in combos:
        d = interaction_delta(
            L[(bo, "rotated")], L[(bo, "aligned")], L[(cma, "rotated")], L[(cma, "aligned")]
        )
        row = delta_row({"bo": bo, "cma": cma}, d)
        for at in ["aligned", "rotated"]:
            pc = paired_comparison(L[(bo, at)], L[(cma, at)])
            row[f"bo_minus_cma_{at}_median"] = pc["median_diff"]
            row[f"bo_minus_cma_{at}_ci"] = f"[{pc['ci_lo']:.2f}, {pc['ci_hi']:.2f}]"
            row[f"bo_minus_cma_{at}_p"] = pc["p_wilcoxon"]
        rows.append(row)
    table("sens_delta_and_bo_vs_cma", rows)
    table(
        "sens_median_final_regret",
        [{"algo": a, "atype": at, "median_r200": float(np.median(10**v))} for (a, at), v in L.items()],
    )


# ---------------------------------------------------------------- Runtime, initial values, bookkeeping
def runtime(data: Data, Ds: list[int]) -> None:
    m = data.meta
    rows = []
    for (a, D, k, b, bench, at), g in m.groupby(["algo", "D", "k", "budget", "benchmark", "atype"]):
        rows.append(
            {
                "algo": a,
                "benchmark": bench,
                "D": D,
                "k": k,
                "atype": at,
                "budget": b,
                "n": len(g),
                "wall_median_s": g["wall_time"].median(),
                "objective_median_s": g["objective_time"].median(),
                "overhead_median_s": g["overhead_time"].median(),
                "overhead_per_eval_ms": 1000 * g["overhead_time"].median() / b,
            }
        )
    table("runtime", rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for a in CORE:
        sub = [select(m, algo=a, D=D, k=D, atype="aligned", budget=data.budget, benchmark="quad") for D in Ds]
        median_iqr_vs(axes[0], Ds, [s["overhead_time"].to_numpy() for s in sub], a)
        ext = select(m, algo=a, D=20, k=20, atype="aligned", budget=BUDGET_EXT[a], benchmark="quad")
        if len(ext):
            H = [pd.read_csv(f"{p}.csv", usecols=["t_wall", "t_obj"]) for p in ext["path"]]
            Y = np.vstack([(h["t_wall"] - h["t_obj"]).to_numpy() for h in H])
            median_trajectory(axes[1], np.arange(1, Y.shape[1] + 1), Y, COLORS[a], LABELS[a])
    axes[0].set(xlabel="D (= k)", ylabel=f"optimiser overhead for B = {data.budget} (s)")
    log_x(axes[0], Ds)
    axes[0].legend()
    axes[1].set(
        yscale="log", xlabel="evaluations", ylabel="cumulative overhead (s)", title="D = 20, extended"
    )
    axes[1].legend()
    save(fig, FIG / "fig7_runtime.png")


def initial_values(data: Data) -> None:
    """§3.1/§7.5.8: distribution of initial f values by orientation (equal mean, different spread)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    rows = []
    for bench in ["quad", "rosen"]:
        for at, ls in [("aligned", "-"), ("rotated", "--")]:
            for j, (a, n_init) in enumerate([("random", None), ("bo_ard", 40), ("cmaes", 12)]):
                runs = data.runs(a, benchmark=bench, D=20, k=2, atype=at)
                if not len(runs):
                    continue
                f = np.concatenate(
                    [pd.read_csv(f"{p}.csv", usecols=["f"])["f"].to_numpy()[:n_init] for p in runs["path"]]
                )
                rows.append(
                    {
                        "benchmark": bench,
                        "atype": at,
                        "source": f"{a} first {n_init or 'all'}",
                        "mean": f.mean(),
                        "median": np.median(f),
                        "q1": np.quantile(f, 0.25),
                        "q3": np.quantile(f, 0.75),
                        "min": f.min(),
                    }
                )
                if bench == "quad":
                    axes[j].hist(
                        np.log10(f + FLOOR),
                        bins=40,
                        histtype="step",
                        density=True,
                        ls=ls,
                        lw=1.5,
                        color=COLORS[a],
                        label=at,
                    )
                    axes[j].set(
                        title=f"{LABELS[a]}: first {n_init or 'all'} values (quad)",
                        xlabel="log10 f",
                        ylabel="density",
                    )
    for ax in axes:
        ax.legend()
    save(fig, FIG / "fig8_initial_values.png")
    table("initial_values", rows)


def bookkeeping(meta: pd.DataFrame) -> None:
    rows = []
    for (exp, a), g in meta.groupby(["experiment", "algo"]):
        row = {
            "experiment": exp,
            "algo": a,
            "n_runs": len(g),
            "budget_exact": bool((g["evals_used"] == g["budget"]).all()),
            "terminations": ";".join(f"{k}:{v}" for k, v in g["termination"].value_counts().items()),
        }
        if "partial_last_generation" in g and g["optimizer"].iloc[0] == "cmaes":
            row["partial_last_gen_frac"] = g["partial_last_generation"].mean()
            row["frac_out_of_bounds_median"] = g["frac_out_of_bounds"].median()
        if g["optimizer"].iloc[0] == "bo":
            row["gp_fit_failures_total"] = int(g["fit_failures"].sum())
        rows.append(row)
    table("bookkeeping", rows)


def write_markdown() -> None:
    def fmt(v):
        return f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)

    lines = []
    for name, df in TABLES:
        lines += [f"### {name}", "", "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
        lines.append("")
    (SUM / "all_tables.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    global FIG, SUM
    if args.pilot:  # pilot outputs live in their own subfolders so they never mix with final results
        FIG, SUM = FIG / "pilot", SUM / "pilot"
    FIG.mkdir(parents=True, exist_ok=True)
    SUM.mkdir(parents=True, exist_ok=True)
    meta = load_runs(RESULTS / "raw")
    if args.pilot:
        data = Data(meta, PILOT_BUDGET, ["pilot"])
        exp1(data, [2, 10, 20])
        orientation(data, "quad", "pilot_orientation")
        runtime(data, [2, 10, 20])
        (SUM / "runtime.csv").replace(SUM.parent / "pilot_timing.csv")  # §8 output name
    else:
        data = Data(meta, BUDGET, ["exp1", "exp1ext", "exp2", "exp3", "exp4", "exp5"])
        exp1(data, [2, 5, 10, 20])
        exp1_extended(data)
        exp2(data, [2, 5, 10, 20])
        orientation(data, "quad", "exp3")
        exp4(data, THETAS)
        orientation(data, "rosen", "exp5")
        cma_alignment_figure(data)
        cma_diagnostics(data)
        sensitivity(meta)
        runtime(data, [2, 5, 10, 20])
        initial_values(data)
        bookkeeping(meta[meta["experiment"] != "pilot"])
    write_markdown()


if __name__ == "__main__":
    main()
