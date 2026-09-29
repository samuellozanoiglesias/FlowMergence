"""
analysis_bureaucracy.py — How much coordination is optimal, how much emerges, and how
useful is one more coordinator?

    python analysis_bureaucracy.py --fixed coord_value --endo coord_value_endogenous
    python analysis_bureaucracy.py --fixed coord_frontier_fixed --endo coord_frontier_endogenous

--fixed : experiment with coord_mode='fixed' (output measured for several fixed fractions)
--endo  : (optional) same points with emergent bureaucracy, for comparison
Targets are experiment names (newest exp_* launch) or paths, as in analysis.py.

For every point (the parameters that vary besides coord_fixed_frac, e.g. N and
centralization, or rho0 and exclusion) it computes:
    frac_opt       coordinator fraction that maximizes output per agent
    y_opt, y_zero  output per agent at the optimum and with no coordinators
    gain_opt       y_opt / y_zero − 1         (what coordination can buy at best)
    MV_first       (y(f1) − y(0)) / (100·f1)  marginal value of the first coordinators,
                   in output per agent per percentage point of coordinators
    MV_last        same, between the last two fractions (diminishing returns?)
and, with --endo:
    M_endo, y_endo  emergent coordinator fraction and output
    over_ratio      M_endo / frac_opt         (> 1: over-coordination)
    efficiency_gap  1 − y_endo / y_opt        (output lost relative to the optimum)

Output: DATA folder of the fixed experiment, analysis/bureaucracy/:
    optimum_table.csv, curves.png, and depending on the varying parameters:
    value_vs_N.png (if N varies), maps_*.png (if rho0 and exclusion vary).
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

import paths
from analysis import _resolve, load_results

CANDIDATES = ["N", "rho0", "exclusion", "centralization", "area_exponent", "spillover", "coord_strength"]


def _keys(df: pd.DataFrame) -> list:
    return [k for k in CANDIDATES if k in df and df[k].nunique() > 1]


def optimum_table(fixed: pd.DataFrame, endo: pd.DataFrame | None) -> pd.DataFrame:
    keys = _keys(fixed)
    rows = []
    for gk, g in (fixed.groupby(keys) if keys else [((), fixed)]):
        gk = gk if isinstance(gk, tuple) else (gk,)
        curve = g.groupby("coord_fixed_frac").y_per_capita.agg(["mean", "std"]).sort_index()
        f = curve.index.to_numpy(float)
        y = curve["mean"].to_numpy(float)
        i_opt = int(np.nanargmax(y))
        y0 = y[0] if f[0] == 0 else math.nan
        row = dict(zip(keys, gk))
        row.update({
            "frac_opt": f[i_opt], "y_opt": y[i_opt], "y_zero": y0,
            "gain_opt": y[i_opt] / y0 - 1 if y0 and y0 > 0 else math.inf,
            "MV_first": (y[1] - y[0]) / (100 * (f[1] - f[0])) if len(f) > 1 else math.nan,
            "MV_last": (y[-1] - y[-2]) / (100 * (f[-1] - f[-2])) if len(f) > 1 else math.nan,
            "opt_at_grid_edge": bool(i_opt == len(f) - 1),
        })
        if endo is not None:
            e = endo
            for k, v in zip(keys, gk):
                if k in e:
                    e = e[np.isclose(e[k].astype(float), float(v))]
            if len(e):
                row["M_endo"] = float(e.M_frac.mean())
                row["y_endo"] = float(e.y_per_capita.mean())
                row["over_ratio"] = row["M_endo"] / row["frac_opt"] if row["frac_opt"] > 0 else math.inf
                row["efficiency_gap"] = 1 - row["y_endo"] / row["y_opt"]
        rows.append(row)
    return pd.DataFrame(rows), keys


def plot_curves(fixed, endo, keys, path):
    import plots
    plt = plots.plt
    groups = list(fixed.groupby(keys)) if keys else [((), fixed)]
    n = len(groups)
    ncol = min(5, n)
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
    for ax, (gk, g) in zip(axes.ravel(), groups):
        gk = gk if isinstance(gk, tuple) else (gk,)
        c = g.groupby("coord_fixed_frac").y_per_capita.agg(["mean", "std"]).sort_index()
        ax.errorbar(100 * c.index, c["mean"], yerr=c["std"].fillna(0), fmt="o-", ms=3, capsize=2)
        i = int(np.nanargmax(c["mean"].to_numpy()))
        ax.plot(100 * c.index[i], c["mean"].iloc[i], "g^", ms=9, label="optimum")
        if endo is not None:
            e = endo
            for k, v in zip(keys, gk):
                if k in e:
                    e = e[np.isclose(e[k].astype(float), float(v))]
            if len(e):
                ax.plot(100 * e.M_frac.mean(), e.y_per_capita.mean(), "r*", ms=12, label="emergent")
        ax.set_title(", ".join(f"{k}={plots.fmt_value(v)}" for k, v in zip(keys, gk)), fontsize=8)
        ax.set_xlabel("coordinators (% of agents)", fontsize=8)
        ax.set_ylabel("output per agent", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_value_vs_N(tab, keys, path):
    import plots
    plt = plots.plt
    others = [k for k in keys if k != "N"]
    groups = list(tab.groupby(others)) if others else [((), tab)]
    cols = ["MV_first", "frac_opt", "gain_opt"] + (["efficiency_gap"] if "efficiency_gap" in tab else [])
    labels = {"MV_first": "marginal value of first coordinators\n(output/agent per % coordinators)",
              "frac_opt": "optimal coordinator fraction", "gain_opt": "best possible gain from coordination",
              "efficiency_gap": "output lost vs optimum (emergent)"}
    fig, axes = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 4))
    for (gk, g) in groups:
        gk = gk if isinstance(gk, tuple) else (gk,)
        lab = ", ".join(f"{k}={plots.fmt_value(v)}" for k, v in zip(others, gk)) or "all"
        g = g.sort_values("N")
        for ax, c in zip(axes, cols):
            vals = g[c].replace([np.inf], np.nan)
            ax.plot(g.N, vals, "o-", ms=4, label=lab)
    for ax, c in zip(axes, cols):
        ax.set(xscale="log", xlabel="N", title=labels[c])
        ax.axhline(0, color="k", lw=0.8, ls=":")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Optimal vs emergent bureaucracy.")
    ap.add_argument("--fixed", required=True, help="experiment with coord_mode=fixed (name or path)")
    ap.add_argument("--endo", default=None, help="matching experiment with emergent bureaucracy")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import plots

    fpath, _ = _resolve(args.fixed, args.root)
    fixed = load_results(fpath)
    endo = None
    if args.endo:
        epath, _ = _resolve(args.endo, args.root)
        endo = load_results(epath)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(fpath)), "analysis", "bureaucracy")
    paths.ensure(out)
    print(f"fixed: {fpath}" + (f"\nendogenous: {epath}" if endo is not None else ""))

    tab, keys = optimum_table(fixed, endo)
    tab.to_csv(os.path.join(out, "optimum_table.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(tab.round(4).to_string(index=False))
    if tab.get("opt_at_grid_edge", pd.Series(dtype=bool)).any():
        print("NOTE: some optima sit at the largest tested fraction; extend coord_fixed_frac there.")

    plot_curves(fixed, endo, keys, os.path.join(out, "curves.png"))
    if "N" in keys:
        plot_value_vs_N(tab, keys, os.path.join(out, "value_vs_N.png"))
    if "rho0" in keys and "exclusion" in keys:
        rest = [k for k in keys if k not in ("rho0", "exclusion")]
        subsets = list(tab.groupby(rest)) if rest else [((), tab)]
        for rk, sub in subsets:
            rk = rk if isinstance(rk, tuple) else (rk,)
            tag = "_".join(f"{k}{plots.fmt_value(v)}" for k, v in zip(rest, rk))
            for col, center in (("frac_opt", None), ("gain_opt", None), ("efficiency_gap", None), ("over_ratio", 1.0)):
                if col in sub and np.isfinite(sub[col].replace([np.inf], np.nan)).any():
                    s2 = sub.replace([np.inf], np.nan)
                    plots.plot_phase_diagram(s2, "rho0", "exclusion",
                                             os.path.join(out, f"maps_{col}{('_' + tag) if tag else ''}.png"),
                                             value=col, center=center)
    print(f"tables and figures in {out}/")


if __name__ == "__main__":
    main()
