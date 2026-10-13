"""
analysis_gamma.py — The master variable Γ: one ratio from city-like to company-like.

    Γ = (time agents spend delivering to the centre) / (time they do not)
      = delivering_frac / (1 − delivering_frac)        MEASURED, not imposed

Claim tested here: output per agent, relative to the same system with no centralization,
follows a single curve
    y / y0 = (1 + Γ)^(−p)          with p ≈ 1
whatever produced Γ (a fixed c, a density path, a learned choice...). The size exponent
then decomposes as
    β_local = β0_local − p · d ln(1 + Γ) / d ln N
so a system is company-like (β < 1) exactly when Γ grows with N faster than the
spillover gain (β0 − 1) can compensate.

Usage (any number of experiments, pooled):
    python analysis_gamma.py centralization_sweep centralization_constant_density \
                             densification_x_centralization
    python analysis_gamma.py choice_sweep --baseline centralization_sweep

A run's baseline y0 is the mean output per agent of runs with NO centralization
(centralization = 0 and fixed delivery) that share the same --match parameters.
Use --baseline to add an experiment that only provides those baseline runs.

Output: <first target's launch folder>/analysis/gamma/
    gamma_points.csv, gamma_fit.txt, beta_decomposition.csv,
    collapse.png, gamma_vs_N.png, beta_decomposition.png, [choice_vs_N.png]
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

import paths
from analysis import _resolve, load_results

MATCH = ["N", "area_exponent", "rho0", "exclusion", "coordination", "spillover",
         "coord_mode", "coord_fixed_frac", "learning_exponent", "T_burn"]
TREAT = ["source", "delivery_choice", "centralization", "hub_premium", "choice_p0",
         "coordination", "area_exponent", "rho0", "exclusion"]


def _tag(target: str) -> str:
    """Readable label: the experiment name, also when a path is given."""
    if not os.path.exists(target):
        return target
    parts = os.path.normpath(os.path.abspath(target)).split(os.sep)
    parts = [x for x in parts if x not in ("results.jsonl",) and not x.startswith("combined")]
    parts = [x for x in parts if not x.startswith("exp_") and x != "runs"]
    return parts[-1] if parts else target


def _load(target, root, tag):
    path, _ = _resolve(target, root)
    df = load_results(path)
    df["source"] = tag
    if "delivery_choice" not in df:
        df["delivery_choice"] = "fixed"
    if "hub_premium" not in df:
        df["hub_premium"] = 0.0
    if "Gamma_meas" not in df:                      # older results: rebuild it
        d = df.get("delivering_frac", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["Gamma_meas"] = d / (1.0 - d).clip(lower=1e-12)
    return df, path


def build_points(df: pd.DataFrame, match: list) -> pd.DataFrame:
    match = [c for c in match if c in df]
    is_base = (df.centralization.fillna(0) == 0) & (df.delivery_choice == "fixed")
    base = df[is_base].groupby(match, dropna=False).y_per_capita.mean().rename("y0").reset_index()
    pts = df[~is_base].merge(base, on=match, how="left")
    missing = pts.y0.isna().sum()
    if missing:
        print(f"note: {missing} runs have no matching baseline (c=0) and are left out")
    pts = pts[pts.y0.notna() & (pts.y_per_capita > 0)].copy()
    pts["ratio"] = pts.y_per_capita / pts.y0
    return pts, base


def fit_collapse(pts: pd.DataFrame) -> dict:
    x = np.log1p(pts.Gamma_meas.to_numpy(float))
    y = -np.log(pts.ratio.to_numpy(float))
    ok = np.isfinite(x) & np.isfinite(y)
    p, c = np.polyfit(x[ok], y[ok], 1)
    pred1 = x[ok]                                   # prediction with p = 1, no intercept
    r2_free = 1 - np.sum((y[ok] - (p * x[ok] + c)) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
    r2_p1 = 1 - np.sum((y[ok] - pred1) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
    err = y[ok] - pred1
    return {"n_points": int(ok.sum()), "p_fit": float(p), "intercept": float(c),
            "R2_free_fit": float(r2_free), "R2_with_p_equal_1": float(r2_p1),
            "mean_log_error_p1": float(err.mean()), "sd_log_error_p1": float(err.std())}


def beta_decomposition(pts: pd.DataFrame, match: list, stride: int = 2) -> pd.DataFrame:
    """Observed local β vs β0_local − Δln(1+Γ)/Δln N, per treatment group."""
    groups = [c for c in TREAT if c in pts and pts[c].nunique() > 1]
    rest = [c for c in match if c in pts and c != "N" and c not in groups and pts[c].nunique() > 1]
    keys = groups + rest
    rows = []
    for gk, g in (pts.groupby(keys, dropna=False) if keys else [((), pts)]):
        gk = gk if isinstance(gk, tuple) else (gk,)
        m = g.groupby("N").agg(y=("y_per_capita", "mean"), y0=("y0", "mean"),
                               G=("Gamma_meas", "mean")).sort_index()
        if len(m) <= stride:
            continue
        N = m.index.to_numpy(float)
        for i in range(len(m) - stride):
            j = i + stride
            dlnN = math.log(N[j] / N[i])
            obs = 1 + (math.log(m.y.iloc[j]) - math.log(m.y.iloc[i])) / dlnN
            b0 = 1 + (math.log(m.y0.iloc[j]) - math.log(m.y0.iloc[i])) / dlnN
            dG = (math.log1p(m.G.iloc[j]) - math.log1p(m.G.iloc[i])) / dlnN
            row = dict(zip(keys, gk))
            row.update({"N_mid": math.sqrt(N[i] * N[j]), "beta_local_obs": obs,
                        "beta0_local": b0, "dln1pGamma_dlnN": dG, "beta_local_pred": b0 - dG,
                        "Gamma_mid": math.sqrt((1 + m.G.iloc[i]) * (1 + m.G.iloc[j])) - 1})
            rows.append(row)
    return pd.DataFrame(rows), keys


def main():
    ap = argparse.ArgumentParser(description="Γ collapse and β decomposition.")
    ap.add_argument("targets", nargs="+", help="experiment names or paths to results.jsonl")
    ap.add_argument("--baseline", nargs="*", default=[], help="extra experiments providing c=0 runs")
    ap.add_argument("--match", nargs="+", default=MATCH, help="parameters a baseline must share")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import plots
    plt = plots.plt

    frames, first = [], None
    for t in args.targets + args.baseline:
        tag = _tag(t)
        df, path = _load(t, args.root, tag)
        first = first or path
        frames.append(df)
        print(f"{tag}: {len(df)} runs from {path}")
    df = pd.concat(frames, ignore_index=True, sort=False)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(first)), "analysis", "gamma")
    paths.ensure(out)

    pts, base = build_points(df, args.match)
    pts.to_csv(os.path.join(out, "gamma_points.csv"), index=False)
    fit = fit_collapse(pts)
    with open(os.path.join(out, "gamma_fit.txt"), "w") as fh:
        fh.write("# y/y0 = (1+Γ)^(-p)   fitted on log scale\n")
        for k, v in fit.items():
            fh.write(f"{k:20s} = {v:.5g}\n")
        fh.write("\nper source (log-error with p = 1):\n")
        for s, g in pts.groupby("source"):
            e = -np.log(g.ratio) - np.log1p(g.Gamma_meas)
            fh.write(f"  {s:34s} n={len(g):4d}  mean {e.mean():+.4f}  sd {e.std():.4f}\n")
    print(open(os.path.join(out, "gamma_fit.txt")).read())

    dec, keys = beta_decomposition(pts, args.match)
    dec.to_csv(os.path.join(out, "beta_decomposition.csv"), index=False)

    # ---- figure 1: the collapse
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for (s, g), col in zip(pts.groupby("source"), plots._colors(pts.source.nunique())):
        ax.scatter(1 + g.Gamma_meas, g.ratio, s=9, alpha=0.6, color=col, label=s)
    xs = np.logspace(0, np.log10(max(2.0, 1 + pts.Gamma_meas.max())), 100)
    ax.plot(xs, 1 / xs, "k--", lw=1.2, label="(1+Γ)^-1")
    ax.set(xscale="log", yscale="log", xlabel="1 + Γ   (Γ = delivering time / other time)",
           ylabel="y / y0", title=f"Collapse: p = {fit['p_fit']:.3f}, R² (p=1) = {fit['R2_with_p_equal_1']:.3f}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "collapse.png"), dpi=150)
    plt.close(fig)

    # ---- figure 2: Γ vs N
    gcols = [c for c in TREAT if c in pts and pts[c].nunique() > 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = list(pts.groupby(gcols, dropna=False)) if gcols else [((), pts)]
    for (gk, g), col in zip(groups, plots._colors(len(groups))):
        gk = gk if isinstance(gk, tuple) else (gk,)
        m = g.groupby("N").Gamma_meas.mean()
        ax.plot(m.index, m.values, "o-", ms=3, color=col,
                label=", ".join(f"{k}={plots.fmt_value(v)}" for k, v in zip(gcols, gk)))
    ax.axhline(1, color="k", ls=":", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="N", ylabel="Γ (measured)",
           title="Γ grows with N → company-like; flat → city-like")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gamma_vs_N.png"), dpi=150)
    plt.close(fig)

    # ---- figure 3: β decomposition
    if not dec.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(dec.beta_local_pred, dec.beta_local_obs, c=np.log10(1 + dec.Gamma_mid), s=14, cmap="viridis")
        lo = min(dec.beta_local_pred.min(), dec.beta_local_obs.min()) - 0.02
        hi = max(dec.beta_local_pred.max(), dec.beta_local_obs.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.axhline(1, color="gray", lw=0.8, ls=":")
        ax.axvline(1, color="gray", lw=0.8, ls=":")
        ax.set(xlabel="predicted local β = β0 − d ln(1+Γ)/d ln N", ylabel="observed local β",
               title="β decomposition (colour: log10(1+Γ))")
        fig.colorbar(sc, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "beta_decomposition.png"), dpi=150)
        plt.close(fig)
        e = dec.beta_local_obs - dec.beta_local_pred
        print(f"β decomposition: {len(dec)} points, obs − pred mean {e.mean():+.4f}, sd {e.std():.4f}")

    # ---- figure 4: emergent centralization (learned choice only)
    lc = pts[pts.delivery_choice == "learned"] if "delivery_choice" in pts else pts.iloc[0:0]
    if len(lc) and "hub_share_started" in lc:
        cols = [c for c in ("hub_premium", "choice_p0", "coordination") if c in lc and lc[c].nunique() > 1]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
        groups = list(lc.groupby(cols)) if cols else [((), lc)]
        for (gk, g), col in zip(groups, plots._colors(len(groups))):
            gk = gk if isinstance(gk, tuple) else (gk,)
            lab = ", ".join(f"{k}={plots.fmt_value(v)}" for k, v in zip(cols, gk)) or "learned"
            m = g.groupby("N")[["hub_share_started", "Gamma_meas"]].mean()
            a1.plot(m.index, m.hub_share_started, "o-", ms=3, color=col, label=lab)
            a2.plot(m.index, m.Gamma_meas, "o-", ms=3, color=col, label=lab)
        a1.set(xscale="log", xlabel="N", ylabel="emergent centralization (share of jobs sent to hub)",
               title="Centralization chosen by the agents")
        a2.set(xscale="log", yscale="log", xlabel="N", ylabel="Γ (measured)", title="Emergent Γ")
        a1.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "choice_vs_N.png"), dpi=150)
        plt.close(fig)
    print(f"tables and figures in {out}/")


if __name__ == "__main__":
    main()
