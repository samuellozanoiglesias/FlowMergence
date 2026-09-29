"""
analysis_transition.py — Is the collapse a true phase transition, and does coordination
remove it?

    python analysis_transition.py jam_transition

For each (coordination, λ, N) it computes, as a function of density:
    y            mean output per agent (the "order parameter": ~0 when jammed)
    jam_prob     fraction of seeds that are jammed (blocked_frac > --jam-threshold)
    rel_sd       seed-to-seed variability sd(y)/mean(y) (peaks at a transition)
and the critical density rho_c (where jam_prob first reaches 0.5, linearly
interpolated) plus the width of the transition (density range with 0.1 < jam_prob < 0.9).

How to read it:
    A TRUE transition sharpens with system size: the y(ρ) curves get steeper, the jam
    window narrows (width ↓ as N ↑), and the variability peak grows. A finite-size
    crossover does not sharpen. BML also found rho_c drifting to lower density as the
    lattice grows, so watch rho_c(N) too.
    With coordinators, jam_prob should stay 0 everywhere if coordination suppresses it.

Output: <launch folder>/analysis/transition/: transition_curves.csv, critical.csv,
    order_parameter.png, jam_probability.png, variability.png, critical.png
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

import paths
from analysis import _resolve, load_results


def curves(df: pd.DataFrame, jam_thr: float) -> pd.DataFrame:
    df = df.copy()
    df["jammed"] = df.blocked_frac > jam_thr
    g = df.groupby(["coordination", "exclusion", "N", "rho0"])
    out = g.agg(density=("density", "mean"), y=("y_per_capita", "mean"), y_sd=("y_per_capita", "std"),
                jam_prob=("jammed", "mean"), blocked=("blocked_frac", "mean"),
                M_frac=("M_frac", "mean"), seeds=("seed", "count")).reset_index()
    out["rel_sd"] = out.y_sd / out.y.replace(0, np.nan)
    return out


def critical(cur: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (c, lam, N), g in cur.groupby(["coordination", "exclusion", "N"]):
        g = g.sort_values("density")
        d, p = g.density.to_numpy(), g.jam_prob.to_numpy()
        rho_c = math.nan
        for i in range(len(p)):
            if p[i] >= 0.5:
                rho_c = d[i] if i == 0 else d[i - 1] + (0.5 - p[i - 1]) * (d[i] - d[i - 1]) / max(p[i] - p[i - 1], 1e-12)
                break
        window = d[(p > 0.1) & (p < 0.9)]
        rows.append({"coordination": c, "exclusion": lam, "N": N, "rho_c": rho_c,
                     # 0 = the jam switches on between two neighbouring grid densities
                     # (sharper than the grid resolution); NaN = never jams
                     "width": (float(window.max() - window.min()) if window.size > 1
                               else (0.0 if (p > 0).any() else math.nan)),
                     "max_rel_sd": float(np.nanmax(g.rel_sd)) if g.rel_sd.notna().any() else math.nan,
                     "ever_jams": bool((p > 0).any())})
    return pd.DataFrame(rows)


def _panel_plot(cur, value, ylabel, path, logy=False):
    import plots
    plt = plots.plt
    for coord, sub in cur.groupby("coordination"):
        lams = sorted(sub.exclusion.unique())
        fig, axes = plt.subplots(1, len(lams), figsize=(3.6 * len(lams), 3.4), sharey=True, squeeze=False)
        Ns = sorted(sub.N.unique())
        colors = plots._colors(len(Ns))
        for ax, lam in zip(axes.ravel(), lams):
            for N, col in zip(Ns, colors):
                g = sub[(sub.exclusion == lam) & (sub.N == N)].sort_values("density")
                ax.plot(g.density, g[value], "o-", ms=3, color=col, label=f"N={N}")
            ax.set_title(f"λ={plots.fmt_value(lam)}", fontsize=9)
            ax.set_xlabel("density", fontsize=8)
            if logy:
                ax.set_yscale("log")
        axes.ravel()[0].set_ylabel(ylabel, fontsize=8)
        axes.ravel()[0].legend(fontsize=7)
        fig.suptitle(f"coordination = {coord}", fontsize=10)
        fig.tight_layout()
        base, ext = os.path.splitext(path)
        fig.savefig(f"{base}_coord{coord}{ext}", dpi=140)
        plt.close(fig)


def plot_critical(crit, path):
    import plots
    plt = plots.plt
    sub = crit[~crit.coordination.astype(bool)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    Ns = sorted(sub.N.unique())
    for N, col in zip(Ns, plots._colors(len(Ns))):
        g = sub[sub.N == N].sort_values("exclusion")
        a1.plot(g.exclusion, g.rho_c, "o-", color=col, label=f"N={N}")
        a2.plot(g.exclusion, g.width, "o-", color=col, label=f"N={N}")
    a1.set(xlabel="λ", ylabel="critical density ρ_c (jam_prob = 0.5)", title="Jamming frontier (no coordinators)")
    a2.set(xlabel="λ", ylabel="width of the jam window", title="Sharpness: shrinks with N if it is a true transition")
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Jamming-transition analysis.")
    ap.add_argument("target", help="experiment name or path to results.jsonl")
    ap.add_argument("--root", default=None)
    ap.add_argument("--launch", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--jam-threshold", type=float, default=0.9,
                    help="a run counts as jammed if its blocked fraction exceeds this")
    args = ap.parse_args()

    path, _ = _resolve(args.target, args.root, args.launch)
    df = load_results(path)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(path)), "analysis", "transition")
    paths.ensure(out)
    print(f"reading {path}: {len(df)} runs")

    cur = curves(df, args.jam_threshold)
    crit = critical(cur)
    cur.to_csv(os.path.join(out, "transition_curves.csv"), index=False)
    crit.to_csv(os.path.join(out, "critical.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(crit.round(4).to_string(index=False))

    _panel_plot(cur, "y", "output per agent", os.path.join(out, "order_parameter.png"))
    _panel_plot(cur, "jam_prob", "fraction of seeds jammed", os.path.join(out, "jam_probability.png"))
    _panel_plot(cur, "rel_sd", "seed variability sd/mean", os.path.join(out, "variability.png"))
    plot_critical(crit, os.path.join(out, "critical.png"))
    print(f"tables and figures in {out}/")


if __name__ == "__main__":
    main()
