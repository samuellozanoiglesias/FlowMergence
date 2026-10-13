"""
analysis_choice.py — Option B: centralization chosen by the agents vs the optimum.

    python analysis_choice.py --choice choice_sweep --fixed choice_fixed_premium \
                              [--lockin choice_lockin]

--choice : learned-choice runs (delivery_choice = learned), several hub premia π
--fixed  : fixed-centralization runs with the same settings (includes c = 0). For each π
           the PLANNER optimum at each N is the c maximizing value per agent
               y_value(c) = y(c) · (1 + π · hub share of completed jobs)
--lockin : (optional) learned-choice runs started centralized (p0 = 0.95) or
           decentralized (p0 = 0.05)

Outputs (in <choice launch folder>/analysis/choice/):
    choice_vs_planner.csv  per (π, N): emergent and optimal centralization, value per agent,
                           efficiency gap, over-centralization, Γ, polarization
    exponents.csv          β of value (emergent and planner) and of work output, per π
    lockin.csv             per (π, N): centralization from each start, hysteresis gap
    centralization_vs_N.png, value_vs_N.png, polarization.png, lockin.png

How to read it:
  * emergent c falls with N  -> agents decentralize as the system grows (city-like)
  * emergent c stays high     -> centralization persists (company-like, β < 1)
  * emergent c > optimal c    -> OVER-centralization (each agent ignores the queue it adds)
  * lock-in: a hysteresis gap > 0 means the final state depends on history
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

import paths
from analysis import _resolve, load_results


def _fit_beta(N, y):
    N, y = np.asarray(N, float), np.asarray(y, float)
    ok = (y > 0) & np.isfinite(y)
    if ok.sum() < 2:
        return math.nan
    return float(np.polyfit(np.log(N[ok]), np.log(y[ok] * N[ok]), 1)[0])


def planner(fixed: pd.DataFrame, premia) -> pd.DataFrame:
    share = fixed["hub_share_completed"] if "hub_share_completed" in fixed else fixed["centralization"]
    f = fixed.assign(share=share.fillna(fixed["centralization"]))
    m = f.groupby(["N", "centralization"]).agg(y=("y_per_capita", "mean"), share=("share", "mean")).reset_index()
    rows = []
    for pi in premia:
        m["v"] = m.y * (1 + pi * m.share)
        for N, g in m.groupby("N"):
            k = g.v.idxmax()
            rows.append({"hub_premium": pi, "N": N, "c_opt": g.centralization[k], "y_value_opt": g.v[k],
                         "y_value_c0": float(g[g.centralization == 0].v.mean()) if (g.centralization == 0).any() else math.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Emergent vs optimal centralization.")
    ap.add_argument("--choice", required=True)
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--lockin", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import plots
    plt = plots.plt

    cpath, _ = _resolve(args.choice, args.root)
    fpath, _ = _resolve(args.fixed, args.root)
    ch, fx = load_results(cpath), load_results(fpath)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(cpath)), "analysis", "choice")
    paths.ensure(out)
    print(f"choice: {cpath} ({len(ch)} runs)\nfixed : {fpath} ({len(fx)} runs)")

    premia = sorted(ch.hub_premium.unique())
    em = ch.groupby(["hub_premium", "N"]).agg(
        c_emergent=("hub_share_started", "mean"), y_value=("y_value_per_capita", "mean"),
        y_work=("y_per_capita", "mean"), Gamma=("Gamma_meas", "mean"), mean_p_hub=("mean_p_hub", "mean"),
        committed_hub=("frac_p_hub_high", "mean"), committed_local=("frac_p_hub_low", "mean"),
        DOL=("DOL_tasks", "mean")).reset_index()
    pl = planner(fx, premia)
    t = em.merge(pl, on=["hub_premium", "N"], how="left")
    t["efficiency_gap"] = 1 - t.y_value / t.y_value_opt
    t["over_centralization"] = t.c_emergent - t.c_opt
    t.to_csv(os.path.join(out, "choice_vs_planner.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(t[["hub_premium", "N", "c_emergent", "c_opt", "y_value", "y_value_opt", "efficiency_gap",
                 "Gamma", "committed_hub", "committed_local"]].round(3).to_string(index=False))

    ex = []
    for pi, g in t.groupby("hub_premium"):
        ex.append({"hub_premium": pi, "beta_value_emergent": _fit_beta(g.N, g.y_value),
                   "beta_value_planner": _fit_beta(g.N, g.y_value_opt), "beta_work_emergent": _fit_beta(g.N, g.y_work)})
    ex = pd.DataFrame(ex)
    ex.to_csv(os.path.join(out, "exponents.csv"), index=False)
    print("\n" + ex.round(3).to_string(index=False))

    cols = plots._colors(len(premia))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for pi, col in zip(premia, cols):
        g = t[t.hub_premium == pi].sort_values("N")
        ax.plot(g.N, g.c_emergent, "o-", ms=3, color=col, label=f"π={plots.fmt_value(pi)} agents")
        ax.plot(g.N, g.c_opt, "--", color=col, lw=1, drawstyle="steps-mid")
    ax.set(xscale="log", xlabel="N", ylabel="centralization (share of jobs to the hub)",
           title="Emergent (solid) vs planner-optimal (dashed) centralization")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "centralization_vs_N.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for pi, col in zip(premia, cols):
        g = t[t.hub_premium == pi].sort_values("N")
        ax.plot(g.N, g.y_value, "o-", ms=3, color=col, label=f"π={plots.fmt_value(pi)} agents")
        ax.plot(g.N, g.y_value_opt, "--", color=col, lw=1)
    ax.set(xscale="log", yscale="log", xlabel="N", ylabel="value per agent",
           title="Value per agent: emergent (solid) vs planner (dashed)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "value_vs_N.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for pi, col in zip(premia, cols):
        g = t[t.hub_premium == pi].sort_values("N")
        ax.plot(g.N, g.committed_hub, "o-", ms=3, color=col, label=f"π={plots.fmt_value(pi)} committed to hub")
        ax.plot(g.N, g.committed_local, "x:", ms=4, color=col)
    ax.set(xscale="log", xlabel="N", ylabel="fraction of agents", title="Polarization: p_hub > 0.8 (o) and < 0.2 (x)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "polarization.png"), dpi=150); plt.close(fig)

    if args.lockin:
        lpath, _ = _resolve(args.lockin, args.root)
        lk = load_results(lpath)
        l = lk.groupby(["hub_premium", "N", "choice_p0"]).hub_share_started.mean().unstack("choice_p0")
        l.columns = [f"c_from_p0_{c:g}" for c in l.columns]
        l = l.reset_index()
        hi, lo = [c for c in l.columns if c.startswith("c_from_p0_")][-1], [c for c in l.columns if c.startswith("c_from_p0_")][0]
        l["hysteresis_gap"] = l[hi] - l[lo]
        l.to_csv(os.path.join(out, "lockin.csv"), index=False)
        print("\nlock-in (hysteresis gap > 0: history matters)\n" + l.round(3).to_string(index=False))
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for pi, col in zip(sorted(l.hub_premium.unique()), plots._colors(l.hub_premium.nunique())):
            g = l[l.hub_premium == pi].sort_values("N")
            ax.plot(g.N, g[hi], "o-", color=col, label=f"π={plots.fmt_value(pi)}, started centralized")
            ax.plot(g.N, g[lo], "s--", color=col, label=f"π={plots.fmt_value(pi)}, started decentralized")
        ax.set(xscale="log", xlabel="N", ylabel="final centralization", title="Lock-in: does history matter?")
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "lockin.png"), dpi=150); plt.close(fig)
    print(f"tables and figures in {out}/")


if __name__ == "__main__":
    main()
