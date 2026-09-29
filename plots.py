"""
plots.py — Figures (matplotlib, headless). All take the flat DataFrame from analysis.py.

    plot_scaling          Y vs N (log-log) and Y/N vs N, one colour per group
    plot_exponent_curve   β(λ) with CI and the β = 1 line; γ(λ) and γa(λ) below
    plot_phase_diagram    map of any value over a two-parameter plane
    plot_local_exponents  β_loc(N): where the crossover happens
    plot_bureaucracy      M/N, inactive bureaucracy fraction, career coordinators
    plot_specialization   DOL (Gorelick), focused fraction, mean efficiency
    plot_congestion       blocking fractions and density
    plot_timeseries       time series of one run (runner --timeseries)

Command line (time series of one stored run; the figure goes next to result.json):
    python plots.py /data/samuel_lozano/FlowMergence/single_runs/<run folder>
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402


def fmt_value(v) -> str:
    """Format a grouping value for labels/filenames: compact for numbers, plain for str/bool."""
    if isinstance(v, (bool,)) or type(v).__name__ == "bool_":
        return str(bool(v))
    if isinstance(v, (int, float)) or hasattr(v, "dtype"):
        try:
            return f"{float(v):g}"
        except (TypeError, ValueError):
            pass
    return str(v)


def _groups(df, by):
    for keys, g in df.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        yield keys, ", ".join(f"{c}={k}" for c, k in zip(by, keys)), g


def _colors(n):
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _mean_sd(g, col):
    s = g.groupby("N")[col].agg(["mean", "std"]).sort_index()
    return s.index.to_numpy(float), s["mean"].to_numpy(float), s["std"].fillna(0).to_numpy(float)


def plot_scaling(df, by, y, path):
    groups = list(_groups(df, by))
    cols = _colors(len(groups))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for (keys, label, g), c in zip(groups, cols):
        N, mu, sd = _mean_sd(g, y)
        a1.errorbar(N, mu, yerr=sd, fmt="o-", ms=3, color=c, label=label, capsize=2)
        a2.errorbar(N, mu / N, yerr=sd / N, fmt="o-", ms=3, color=c, capsize=2)
    Nall = np.sort(df.N.unique()).astype(float)
    ref = df.groupby("N")[y].mean().sort_index().to_numpy()
    if len(Nall) and ref[0] > 0:
        a1.plot(Nall, ref[0] * Nall / Nall[0], "k--", lw=1, label="slope 1")
    a1.set(xscale="log", yscale="log", xlabel="N", ylabel=y, title=f"{y} vs N")
    a2.set(xscale="log", yscale="log", xlabel="N", ylabel=f"{y}/N",
           title="per agent (slope = β − 1)")
    a1.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_exponent_curve(beta, gamma, gamma_a, col, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    b = beta.sort_values(col)
    a1.fill_between(b[col], b.ci_lo, b.ci_hi, alpha=0.25, color="C0")
    a1.plot(b[col], b.exponent, "o-", color="C0", label="β (output)")
    a1.axhline(1, color="k", ls="--", lw=1)
    a1.text(b[col].min(), 1.002, " linear", va="bottom", fontsize=8)
    a1.set(ylabel="β", title=f"Scaling exponent versus {col}")
    a1.legend()
    for tab, name, c in ((gamma, "γ (total M)", "C3"), (gamma_a, "γa (active M)", "C2")):
        t = tab.sort_values(col)
        a2.fill_between(t[col], t.ci_lo, t.ci_hi, alpha=0.2, color=c)
        a2.plot(t[col], t.exponent, "o-", color=c, label=name)
    a2.axhline(1, color="k", ls="--", lw=1)
    a2.set(xlabel=col, ylabel="exponent", title="Bureaucracy scaling")
    a2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_phase_diagram(table, xcol, ycol, path, value="exponent", center=1.0, title=None):
    """Heatmap of `value` over (xcol, ycol). With center set, uses a diverging map around it."""
    piv = table.pivot_table(index=ycol, columns=xcol, values=value, aggfunc="mean").sort_index()
    vals = piv.to_numpy(float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if center is not None:
        span = max(abs(finite.min() - center), abs(finite.max() - center), 1e-3)
        norm = TwoSlopeNorm(vcenter=center, vmin=center - span, vmax=center + span)
        im = ax.imshow(vals, origin="lower", cmap="RdBu_r", norm=norm, aspect="auto")
        default_title = f"{value}  (red: > {center:g} · blue: < {center:g})"
    else:
        im = ax.imshow(vals, origin="lower", cmap="viridis", aspect="auto")
        default_title = value
    ax.set_xticks(range(len(piv.columns)), [fmt_value(v) for v in piv.columns])
    ax.set_yticks(range(len(piv.index)), [fmt_value(v) for v in piv.index])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="k" if center is not None else "w")
    ax.set(xlabel=xcol, ylabel=ycol, title=title or default_title)
    fig.colorbar(im, ax=ax, label=value)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_density_response(resp, group, path):
    """Four panels vs measured density: y per agent, elasticity ε, blocking, coordinators."""
    groups = list(resp.groupby(group)) if group else [((), resp)]
    colors = _colors(len(groups))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    (a1, a2), (a3, a4) = axes
    for (keys, g), c in zip(groups, colors):
        keys = keys if isinstance(keys, tuple) else (keys,)
        label = ", ".join(f"{k}={fmt_value(v)}" for k, v in zip(group, keys)) or "all"
        g = g.sort_values("density")
        a1.errorbar(g.density, g.y_per_capita, yerr=g.y_sd, fmt="o-", ms=3, color=c, label=label, capsize=2)
        a2.plot(g.density_mid, g.epsilon, "o-", ms=3, color=c, label=label)
        a3.plot(g.density, g.blocked_frac, "o-", ms=3, color=c)
        if "M_frac" in g:
            a4.plot(g.density, g.M_frac, "o-", ms=3, color=c)
    a1.set(xscale="log", yscale="log", xlabel="density N/L²", ylabel="output per agent",
           title="Output per agent vs density")
    a2.axhline(0, color="k", ls="--", lw=1)
    a2.set(xscale="log", xlabel="density", ylabel="ε = dlog y / dlog ρ",
           title="Density elasticity  (β − 1 ≈ (1 − a)·ε)")
    a3.set(xscale="log", xlabel="density", ylabel="fraction of blocked moves", title="Blocking")
    a4.set(xscale="log", xlabel="density", ylabel="M / N", title="Coordinators")
    a1.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_local_exponents(loc, by, path):
    if loc.empty:
        return
    groups = list(_groups(loc, by))
    cols = _colors(len(groups))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (keys, label, g), c in zip(groups, cols):
        g = g.sort_values("N_mid")
        ax.plot(g.N_mid, g.local_exponent, "o-", ms=3, color=c, label=label)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set(xscale="log", xlabel="N", ylabel="local β = Δlog Y / Δlog N", title="Local exponent")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _panel_grid(df, by, cols_spec, path, suptitle):
    groups = list(_groups(df, by))
    colors = _colors(len(groups))
    fig, axes = plt.subplots(1, len(cols_spec), figsize=(5 * len(cols_spec), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (col, ylabel, logy) in zip(axes, cols_spec):
        for (keys, label, g), c in zip(groups, colors):
            if col not in g:
                continue
            N, mu, sd = _mean_sd(g, col)
            ax.errorbar(N, mu, yerr=sd, fmt="o-", ms=3, color=c, label=label, capsize=2)
        ax.set(xscale="log", xlabel="N", ylabel=ylabel)
        if logy:
            ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bureaucracy(df, by, path):
    _panel_grid(df, by, [("M_frac", "M / N", False),
                         ("bureau_inactive_frac", "inactive bureaucracy fraction", False),
                         ("frac_career_coord", "fraction of 'career bureaucrats'", False)],
                path, "Bureaucracy")


def plot_specialization(df, by, path):
    _panel_grid(df, by, [("DOL_tasks", "DOL (Gorelick) across tasks", False),
                         ("frac_focused", "fraction with >90% of time in one role", False),
                         ("mean_w", "mean efficiency w", True)],
                path, "Division of labour")


def plot_congestion(df, by, path):
    _panel_grid(df, by, [("blocked_frac", "fraction of blocked moves", False),
                         ("cross_blocked_frac", "crossing blocks", False),
                         ("density", "density N / L²", True)],
                path, "Congestion")


def plot_timeseries(result: dict, path: str):
    ts = result.get("timeseries")
    if not ts:
        raise ValueError("this run has no time series (use --timeseries)")
    t = np.asarray(ts["t"])
    N = result["params"]["N"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes[0, 0].plot(t, np.asarray(ts["work"]) / N)
    axes[0, 0].set(ylabel="output / N")
    axes[0, 1].plot(t, np.asarray(ts["M"]) / N, label="M/N")
    axes[0, 1].plot(t, np.asarray(ts["M_active"]) / N, label="active M/N")
    axes[0, 1].legend()
    axes[1, 0].plot(t, ts["blocked_rate"], label="total blocking")
    axes[1, 0].plot(t, ts["signal_rate"], label="coordination signal")
    axes[1, 0].legend()
    axes[1, 0].set(xlabel="t")
    s = np.asarray(ts["s_task"])
    for j in range(s.shape[1]):
        axes[1, 1].plot(t, s[:, j], label=f"s_{j}")
    axes[1, 1].plot(t, ts["s_coord"], "k--", label="s_C")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].set(xlabel="t", ylabel="stimuli")
    burn = result["params"]["T_burn"]
    for ax in axes.ravel():
        ax.axvline(burn, color="gray", ls=":", lw=1)
    fig.suptitle(f"N={N}, λ={result['params']['exclusion']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Time series of one stored run.")
    ap.add_argument("run", help="run folder or its result.json")
    ap.add_argument("--out", default=None, help="default: <run folder>/timeseries.png")
    a = ap.parse_args()
    rpath = a.run if a.run.endswith(".json") else os.path.join(a.run, "result.json")
    with open(rpath) as f:
        res = json.load(f)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(rpath)), "timeseries.png")
    plot_timeseries(res, out)
    print(f"saved {out}")