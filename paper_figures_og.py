"""
paper_figures.py — The MAIN figures of the paper (publication style), from the paper_* experiments.

    python paper_figures.py                 # newest launch of every experiment used
    python paper_figures.py --out <folder>  # default: <data root>/paper_figures/<timestamp>/
    python paper_figures.py --si            # also the supplementary figure (coordination)

Main figures (PDF with embedded fonts + PNG at 300 dpi; each skipped if its data is missing).
Fig. 1 of the paper is the model schematic (not drawn here).
    fig2_power_law_transition  THE ONE-KNOB RESULT. Area ∝ N^a: (a) output per agent vs N, one
                             straight log-log line per a, all through the common density point;
                             (b) local exponent β(N): flat for every a, above 1 for a < 1 and
                             below 1 for a > 1; (c) β vs a, measured (95% CI) vs the prediction
                             β = 1 + ε(1 − a) with ε measured independently at constant density
                             [paper_density_exponent, density_window_scan]
    fig3_scaling_transition  centralization c: (a) Y/N vs N per c, (b) local β(N), (c) global β
                             vs c with c*, (d) N_cross [paper_centralization_sweep]
    fig4_gamma_collapse      (a) y/y0 vs 1+Γ, (b) β decomposition
                             [paper_centralization_sweep, paper_constant_density,
                              paper_densification, paper_choice_fixed (+ choice sweep if it matches)]
    fig5_emergent_choice     option B: (a) centralization chosen by agents vs planner, (b) value,
                             (c) efficiency gap with the exploration cost, (d) β of value vs π
                             [paper_choice_logit_sweep, paper_choice_fixed]
    fig6_competency_trap     (a) committed to the hub from two starts (raw) vs skill-adjusted,
                             (b) belief vs realized hub advantage, (c) efficiency gap
                             [paper_choice_logit_lockin, paper_choice_logit_sweep,
                              choice_trap_test, paper_choice_fixed]
Supplementary (--si):
    figS1_coordination       [paper_coord_value, paper_coord_value_endogenous]
    figS2_densification      densification on the city path, with and without centralization
                             [paper_densification, paper_constant_density]
Only some figures:  python paper_figures.py --only power_laws   (keys: power_laws, scaling,
    gamma, choice, trap, coordination, densification)

Text outputs (for writing the paper):
    paper_numbers.txt        every number quoted from the figures, human readable
    paper_numbers.json       the same numbers as key -> value
    paper_macros.tex         \\FM{key} macros for LaTeX: \\input{paper_macros} then \\FM{beta_c0.2}
                             (an undefined key prints a bold ??key so it cannot go unnoticed)
    figures_manifest.txt     for each figure: files, panels, experiments and launches used
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os

import numpy as np
import pandas as pd

import paths
from analysis import _resolve, bootstrap_exponent, crossover_sizes, load_results, local_exponents

LINES, NUMBERS, MANIFEST, SOURCES = [], {}, [], {}

FULL, HALF = 7.0, 3.4          # figure widths (inches): double and single column
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
RAW_COL, SKILL_COL = "#D55E00", "#0072B2"


# ---------------------------------------------------------------------- bookkeeping
def note(s=""):
    print(s)
    LINES.append(s)


def num(key, value, fmt="{:.3f}"):
    """Record a number for paper_numbers.json / paper_macros.tex."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return
    try:
        NUMBERS[key] = fmt.format(value)
    except (ValueError, TypeError):
        NUMBERS[key] = str(value)


def get(name, root):
    try:
        path, _ = _resolve(name, root)
    except FileNotFoundError:
        note(f"[skip] {name}: not found")
        return None
    df = load_results(path)
    note(f"[data] {name}: {len(df)} runs  ({path})")
    SOURCES[name] = (str(path), len(df))
    return df


def first(*names_root):
    """First experiment that exists, e.g. the logit sweep, falling back to the old one."""
    *names, root = names_root
    for n in names:
        df = get(n, root)
        if df is not None:
            return n, df
    return None, None


def manifest(fig, panels, used):
    MANIFEST.append((fig, panels, [u for u in used if u]))


# ---------------------------------------------------------------------- style
def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "lines.linewidth": 1.2, "lines.markersize": 3.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100,
        "axes.prop_cycle": cycler(color=OKABE),
    })
    return plt


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    import matplotlib.pyplot as plt
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="left", va="bottom")


def colors_for(plt, values, cmap):
    cm = plt.get_cmap(cmap)
    n = len(values)
    return {v: cm(0.05 + 0.85 * i / max(n - 1, 1)) for i, v in enumerate(values)}


def discrete_cbar(plt, fig, axes, values, colors, label, fmt="{:g}"):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap
    n = len(values)
    sm = ScalarMappable(norm=BoundaryNorm(np.arange(n + 1) - 0.5, n),
                        cmap=ListedColormap([colors[v] for v in values]))
    cb = fig.colorbar(sm, ax=axes, ticks=range(n), fraction=0.04, pad=0.02, aspect=30)
    cb.ax.set_yticklabels([fmt.format(v) for v in values])
    cb.ax.tick_params(length=0)
    cb.set_label(label)
    cb.outline.set_linewidth(0.5)
    return cb


def unity(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(1, color="0.35", lw=0.7, ls="--", zorder=0)


def mean_sem(df, keys, col):
    g = df.groupby(keys)[col]
    t = g.agg(["mean", "std", "count"]).reset_index()
    t["se"] = (t["std"] / np.sqrt(t["count"])).fillna(0.0)
    return t


def logx(ax, xlabel=r"system size $N$"):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)


def ckey(c):
    return f"{c:g}"


# ---------------------------------------------------------------------- figure 2 (headline)
def _a_colors(plt, avals):
    """Diverging: a < 1 (city-like) red, a = 1 black, a > 1 (company-like) blue."""
    cm = plt.get_cmap("RdBu")
    lo = sorted(v for v in avals if v < 1 - 1e-9)
    hi = sorted(v for v in avals if v > 1 + 1e-9)
    col = {}
    for i, v in enumerate(lo):
        col[v] = cm(0.02 + 0.30 * i / max(len(lo) - 1, 1))
    for i, v in enumerate(hi):
        col[v] = cm(0.68 + 0.30 * i / max(len(hi) - 1, 1))
    for v in avals:
        col.setdefault(v, "0.15")
    return col


def epsilon_constant_density(W, rho_lo, rho_hi):
    """ε = d ln y / d ln ρ fitted on the constant-density scan over the densities visited."""
    if W is None:
        return math.nan, math.nan
    dcol = "density" if "density" in W else "rho0"
    g = W.groupby("rho0").agg(rho=(dcol, "mean"), y=("y_per_capita", "mean")).reset_index()
    g = g[(g.rho >= rho_lo * 0.999) & (g.rho <= rho_hi * 1.001)]
    if len(g) < 2:
        return math.nan, math.nan
    x, y = np.log(g.rho.values), np.log(g.y.values)
    slope = np.polyfit(x, y, 1)[0]
    loc = np.diff(y) / np.diff(x)
    return float(slope), float(loc.max() - loc.min()) if len(loc) > 1 else 0.0


def fig_power_laws(P, W, plt, out):
    avals = sorted(P.area_exponent.unique())
    col = _a_colors(plt, avals)
    y = mean_sem(P, ["area_exponent", "N"], "y_per_capita")
    loc = local_exponents(P, ["area_exponent"], "N", "Y", stride=2)

    rows = []
    for a, g in P.groupby("area_exponent"):
        est, lo, hi, r2, _ = bootstrap_exponent(g, "N", "Y", n_boot=500)
        lg = loc[loc.area_exponent == a].local_exponent
        rows.append({"a": a, "beta": est, "lo": lo, "hi": hi, "r2": r2,
                     "local_min": lg.min(), "local_max": lg.max(), "local_sd": lg.std()})
    bt = pd.DataFrame(rows).sort_values("a")
    x = 1 - bt.a.to_numpy()
    eps_fit = float(np.sum(x * (bt.beta - 1)) / np.sum(x * x)) if np.sum(x * x) > 0 else math.nan
    dcol = "density" if "density" in P else None
    rho_lo = P[dcol].min() if dcol else math.nan
    rho_hi = P[dcol].max() if dcol else math.nan
    eps_w, eps_w_spread = epsilon_constant_density(W, rho_lo, rho_hi)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(FULL, 2.55), constrained_layout=True)
    for a in avals:
        g = y[y.area_exponent == a].sort_values("N")
        b = bt[bt.a == a].beta.iloc[0]
        a1.plot(g.N, g["mean"], "o", color=col[a], ms=2.8)
        # fitted power law through the data (geometric-mean anchor)
        n0 = np.exp(np.log(g.N).mean())
        y0 = np.exp(np.log(g["mean"]).mean())
        nn = np.array([g.N.min(), g.N.max()])
        a1.plot(nn, y0 * (nn / n0) ** (b - 1), "-", lw=1.0, color=col[a])
        h = loc[loc.area_exponent == a].sort_values("N_mid")
        a2.plot(h.N_mid, h.local_exponent, "o-", ms=2.5, lw=0.9, color=col[a])
    a1.set_yscale("log")
    logx(a1)
    a1.set_ylabel("output per agent  $Y/N$")
    from matplotlib.ticker import FuncFormatter, LogLocator
    a1.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a1.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    unity(a2)
    logx(a2)
    a2.set_ylabel(r"local exponent $\beta(N)$")
    for a in avals:     # label each line at its right end
        h = loc[loc.area_exponent == a].sort_values("N_mid")
        if len(h):
            a2.annotate(f"$a={a:.2f}$", (h.N_mid.iloc[-1], h.local_exponent.iloc[-1]), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=5.8, color=col[a])
    a2.set_xlim(right=loc.N_mid.max() * 3.2)

    aa = np.linspace(min(avals) - 0.03, max(avals) + 0.03, 50)
    if np.isfinite(eps_w):
        a3.plot(aa, 1 + eps_w * (1 - aa), "k-", lw=0.9,
                label=rf"prediction $1+\varepsilon(1-a)$, $\varepsilon={eps_w:.2f}$" + "\n(constant-density runs)")
    a3.plot(aa, 1 + eps_fit * (1 - aa), "k:", lw=0.9, label=rf"fit, $\varepsilon={eps_fit:.2f}$")
    for _, r in bt.iterrows():
        a3.errorbar(r.a, r.beta, yerr=[[r.beta - r.lo], [r.hi - r.beta]], fmt="o", ms=3.5,
                    color=col[r.a], capsize=2, elinewidth=0.8, zorder=3)
    unity(a3)
    unity(a3, horizontal=False)
    a3.set(xlabel=r"area exponent $a$   (area $\propto N^a$)", ylabel=r"scaling exponent $\beta$")
    a3.legend(loc="upper right", fontsize=6.2)
    a3.text(0.03, 0.03, "city-like: $a<1$, $\\beta>1$\ncompany-like: $a>1$, $\\beta<1$",
            transform=a3.transAxes, ha="left", va="bottom", fontsize=6.2, color="0.35",
            bbox=dict(fc="white", ec="none", pad=1.0), zorder=5)
    for axx, l in zip((a1, a2, a3), "abc"):
        panel(axx, l)
    save(fig, out, "fig2_power_law_transition")
    manifest("fig2_power_law_transition",
             "(a) Y/N vs N per area exponent a (points: mean over seeds; lines: fitted power laws); "
             "(b) smoothed local exponent β(N) per a; (c) β vs a with 95% bootstrap CI, prediction "
             "1 + ε(1 − a) with ε from the constant-density scan (solid) and fitted ε (dotted)",
             ["paper_density_exponent", "density_window_scan" if W is not None else None])

    note("\n== Fig 2: one knob, pure power laws (paper_density_exponent)")
    note(f"  densities visited: {rho_lo:.4f} – {rho_hi:.4f}")
    note(f"  ε fitted from β vs a (through β = 1 at a = 1): {eps_fit:.3f}")
    note(f"  ε measured at constant density over the same densities: {eps_w:.3f} "
         f"(spread of local ε {eps_w_spread:.3f})")
    num("pl_rho_lo", rho_lo, "{:.3f}")
    num("pl_rho_hi", rho_hi, "{:.3f}")
    num("pl_eps_fit", eps_fit, "{:.2f}")
    num("pl_eps_window", eps_w, "{:.2f}")
    num("pl_N_min", P.N.min(), "{:,.0f}")
    num("pl_N_max", P.N.max(), "{:,.0f}")
    num("pl_decades", math.log10(P.N.max() / P.N.min()), "{:.1f}")
    worst_spread = 0.0
    for _, r in bt.iterrows():
        pred = 1 + eps_w * (1 - r.a) if np.isfinite(eps_w) else math.nan
        spread = r.local_max - r.local_min
        worst_spread = max(worst_spread, spread if np.isfinite(spread) else 0.0)
        note(f"  a = {r.a:.3f}  β = {r.beta:.3f} [{r.lo:.3f}, {r.hi:.3f}]  R² = {r.r2:.4f}  "
             f"local β {r.local_min:.3f}–{r.local_max:.3f} (s.d. {r.local_sd:.3f})  predicted {pred:.3f}")
        k = f"a{r.a:.3f}"
        num(f"pl_beta_{k}", r.beta)
        num(f"pl_beta_{k}_lo", r.lo)
        num(f"pl_beta_{k}_hi", r.hi)
        num(f"pl_R2_{k}", r.r2, "{:.4f}")
        num(f"pl_localspread_{k}", spread, "{:.3f}")
        num(f"pl_pred_{k}", pred)
    num("pl_localspread_max", worst_spread, "{:.3f}")
    num("pl_R2_min", bt.r2.min(), "{:.4f}")
    note(f"  largest spread of local β over N for any a: {worst_spread:.3f}; smallest R²: {bt.r2.min():.4f}")
    note("  (a pure power law: R² ≈ 1 and a local-β spread comparable to the seed noise)")


# ---------------------------------------------------------------------- figure 1
def fig_scaling(S, plt, out):
    cs = sorted(S.centralization.unique())
    col = colors_for(plt, cs, "plasma")
    y = mean_sem(S, ["centralization", "N"], "y_per_capita")
    loc = local_exponents(S, ["centralization"], "N", "Y", stride=2)
    cross = crossover_sizes(loc, ["centralization"]) if not loc.empty else pd.DataFrame()

    fig, ax = plt.subplots(2, 2, figsize=(FULL, 5.0), constrained_layout=True)
    (a, b), (c, d) = ax
    for cv in cs:
        g = y[y.centralization == cv].sort_values("N")
        a.plot(g.N, g["mean"], "o-", color=col[cv])
        a.fill_between(g.N, g["mean"] - 2 * g["se"], g["mean"] + 2 * g["se"], color=col[cv], alpha=0.2, lw=0)
        h = loc[loc.centralization == cv].sort_values("N_mid")
        b.plot(h.N_mid, h.local_exponent, "o-", color=col[cv])
    a.set_yscale("log")
    logx(a)
    a.set_ylabel("output per agent  $Y/N$")
    unity(b)
    logx(b)
    b.set_ylabel(r"local exponent $\beta(N)$")
    tr = b.get_yaxis_transform()
    b.text(0.02, 1.0, r"city-like ($\beta>1$)", transform=tr, ha="left", va="bottom", color="0.4", fontsize=6.5)
    b.text(0.02, 1.0, r"company-like ($\beta<1$)", transform=tr, ha="left", va="top", color="0.4", fontsize=6.5)
    discrete_cbar(plt, fig, [a, b], cs, col, "centralization $c$")

    rows = []
    for cv, g in S.groupby("centralization"):
        est, lo, hi, r2, _ = bootstrap_exponent(g, "N", "Y", n_boot=500)
        rows.append({"c": cv, "beta": est, "lo": lo, "hi": hi, "r2": r2})
    bt = pd.DataFrame(rows).sort_values("c")
    cstar = math.nan
    for i in range(len(bt) - 1):
        b0, b1 = bt.beta.iloc[i], bt.beta.iloc[i + 1]
        if (b0 - 1) * (b1 - 1) <= 0 and b0 != b1:
            cstar = bt.c.iloc[i] + (1 - b0) * (bt.c.iloc[i + 1] - bt.c.iloc[i]) / (b1 - b0)
            break
    c.fill_between(bt.c, bt.lo, bt.hi, color=OKABE[0], alpha=0.25, lw=0)
    c.plot(bt.c, bt.beta, "o-", color=OKABE[0])
    unity(c)
    if np.isfinite(cstar):
        c.axvline(cstar, color=OKABE[1], lw=0.8, ls=":")
        c.text(cstar, 0.97, rf" $c^*\approx{cstar:.2f}$", color=OKABE[1], transform=c.get_xaxis_transform(),
               va="top", fontsize=7)
    c.set_ylabel(r"scaling exponent $\beta$  (95% CI)")
    if not cross.empty:
        cr = cross.sort_values("centralization").dropna(subset=["N_cross"])
        d.plot(cr.centralization, cr.N_cross, "s-", color=OKABE[2])
    d.set_yscale("log", base=2)
    from matplotlib.ticker import FuncFormatter
    d.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    d.set_ylabel(r"$N_{\rm cross}$  (local $\beta<1$ above)")
    ticks = [t for t in (0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if t <= bt.c.max() + 1e-9]
    for axx in (c, d):
        axx.set_xscale("symlog", linthresh=0.02, linscale=0.5)
        axx.set_xlim(0, bt.c.max() * 1.1)
        axx.set_xticks(ticks)
        axx.set_xticklabels([f"{t:g}" for t in ticks])
        axx.minorticks_off()
        axx.set_xlabel("centralization $c$")
    for axx, l in zip((a, b, c, d), "abcd"):
        panel(axx, l)
    save(fig, out, "fig3_scaling_transition")
    manifest("fig3_scaling_transition",
             "(a) Y/N vs N per c (±2 s.e.m.); (b) smoothed local β(N); (c) global β vs c, 95% "
             "bootstrap CI, c* where β = 1; (d) N_cross(c)", ["paper_centralization_sweep"])

    note("\n== Fig 3: scaling transition (paper_centralization_sweep)")
    for _, r in bt.iterrows():
        note(f"  c = {r.c:<5g}  β = {r.beta:.3f}  [{r.lo:.3f}, {r.hi:.3f}]  R² = {r.r2:.4f}")
        num(f"beta_c{ckey(r.c)}", r.beta)
        num(f"beta_c{ckey(r.c)}_lo", r.lo)
        num(f"beta_c{ckey(r.c)}_hi", r.hi)
    note(f"  critical centralization c* (β = 1) ≈ {cstar:.3f}")
    num("cstar", cstar, "{:.2f}")
    if not cross.empty:
        for _, r in cross.sort_values("centralization").iterrows():
            note(f"  c = {r.centralization:<5g}  local β max {r.beta_local_max:.3f}, at largest N "
                 f"{r.beta_local_at_largest_N:.3f}, N_cross = {r.N_cross:g}")
            k = ckey(r.centralization)
            num(f"betalocal_max_c{k}", r.beta_local_max)
            num(f"betalocal_largestN_c{k}", r.beta_local_at_largest_N)
            num(f"Ncross_c{k}", r.N_cross, "{:,.0f}")


# ---------------------------------------------------------------------- figure 2
NICE = {"paper_centralization_sweep": "centralization sweep", "paper_constant_density": "constant density",
        "paper_densification": "densification", "paper_choice_fixed": "option B, fixed $c$",
        "paper_choice_logit_sweep": "option B, agents choose", "paper_choice_sweep": "option B, agents choose"}


def fig_gamma(frames, plt, out):
    import analysis_gamma as ag
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "delivery_choice" not in df:
        df["delivery_choice"] = "fixed"
    df["delivery_choice"] = df["delivery_choice"].fillna("fixed")
    if "Gamma_meas" not in df or df["Gamma_meas"].isna().any():
        dfr = df["delivering_frac"].fillna(0.0)
        df["Gamma_meas"] = df.get("Gamma_meas", pd.Series(np.nan, index=df.index)).fillna(
            dfr / (1 - dfr).clip(lower=1e-12))
    pts, _ = ag.build_points(df, ag.MATCH)
    fit = ag.fit_collapse(pts)
    p = fit["p_fit"]
    dec, _ = ag.beta_decomposition(pts, ag.MATCH)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.9), constrained_layout=True)
    srcs = [s for s in NICE if s in set(pts.source)] + sorted(set(pts.source) - set(NICE))
    for s, cl, mk in zip(srcs, OKABE, ["o", "s", "^", "D", "v", "P", "X", "*"]):
        g = pts[pts.source == s]
        a1.scatter(1 + g.Gamma_meas, g.ratio, s=6, alpha=0.55, color=cl, lw=0, marker=mk,
                   label=f"{NICE.get(s, s)} ({len(g)})")
    xs = np.logspace(0, np.log10(1 + pts.Gamma_meas.max()), 100)
    a1.plot(xs, xs ** (-p), "k-", lw=1.0, label=rf"$(1+\Gamma)^{{-{p:.2f}}}$")
    a1.plot(xs, 1 / xs, "k:", lw=0.9, label=r"$(1+\Gamma)^{-1}$")
    a1.set(xscale="log", yscale="log", xlabel=r"$1+\Gamma$   ($\Gamma$ = time delivering / other time)",
           ylabel=r"$y/y_0$")
    a1.text(0.03, 0.04, rf"$R^2={fit['R2_free_fit']:.3f}$", transform=a1.transAxes, fontsize=7)
    a1.legend(loc="upper right", fontsize=6.5)
    if not dec.empty:
        pred = dec.beta0_local - p * dec.dln1pGamma_dlnN
        sc = a2.scatter(pred, dec.beta_local_obs, s=7, c=np.log10(1 + dec.Gamma_mid), cmap="viridis", lw=0)
        lo = min(pred.min(), dec.beta_local_obs.min()) - 0.02
        hi = max(pred.max(), dec.beta_local_obs.max()) + 0.02
        a2.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        a2.axhline(1, color="0.6", lw=0.6, ls=":")
        a2.axvline(1, color="0.6", lw=0.6, ls=":")
        a2.set(xlabel=rf"predicted $\beta_0-{p:.2f}\,d\ln(1+\Gamma)/d\ln N$", ylabel=r"observed local $\beta$")
        cb = fig.colorbar(sc, ax=a2, fraction=0.05, pad=0.02, aspect=30)
        cb.set_label(r"$\log_{10}(1+\Gamma)$")
        cb.outline.set_linewidth(0.5)
    panel(a1, "a")
    panel(a2, "b")
    save(fig, out, "fig4_gamma_collapse")
    manifest("fig4_gamma_collapse", "(a) y/y0 vs 1+Γ, all runs, fit (1+Γ)^-p; (b) observed vs "
             "predicted local β (β decomposition)", sorted(set(pts.source)))

    note("\n== Fig 4: Γ collapse (pooled)")
    for s in srcs:
        note(f"  points from {s}: {int((pts.source == s).sum())}")
    note(f"  runs {fit['n_points']},  y/y0 = (1+Γ)^−p with p = {p:.3f},  R² = {fit['R2_free_fit']:.4f} "
         f"(R² with p = 1: {fit['R2_with_p_equal_1']:.4f})")
    num("gamma_p", p)
    num("gamma_R2", fit["R2_free_fit"], "{:.3f}")
    num("gamma_R2_p1", fit["R2_with_p_equal_1"], "{:.3f}")
    num("gamma_npoints", fit["n_points"], "{:,.0f}")
    if not dec.empty:
        err = dec.beta_local_obs - (dec.beta0_local - p * dec.dln1pGamma_dlnN)
        note(f"  β decomposition over {len(dec)} points: observed − predicted = {err.mean():+.4f} ± {err.std():.4f}")
        num("decomp_err_mean", err.mean(), "{:+.3f}")
        num("decomp_err_sd", err.std(), "{:.3f}")
        num("decomp_npoints", len(dec), "{:,.0f}")


# ---------------------------------------------------------------------- figure 3
def fig_densification(D, C, plt, out):
    rows = []
    for (a, cv), g in D.groupby(["area_exponent", "centralization"]):
        est, lo, hi, _, _ = bootstrap_exponent(g, "N", "Y", n_boot=500)
        rows.append({"a": a, "c": cv, "beta": est, "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    t0 = t[t.c == 0].sort_values("a")
    x = 1 - t0.a.to_numpy()
    yv = t0.beta.to_numpy() - 1
    eps = float(np.sum(x * yv) / np.sum(x * x)) if np.sum(x * x) > 0 else math.nan

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.7), constrained_layout=True)
    a1.errorbar(x, yv, yerr=[t0.beta - t0.lo, t0.hi - t0.beta], fmt="o", color=OKABE[0], capsize=2,
                elinewidth=0.8)
    xx = np.linspace(0, max(x.max(), 0.35) * 1.05, 50)
    a1.plot(xx, eps * xx, "--", color=OKABE[0], lw=0.9, label=rf"$\beta-1=\varepsilon(1-a)$, $\varepsilon={eps:.2f}$")
    a1.axhline(0, color="0.6", lw=0.6, ls=":")
    a1.set(xlabel=r"densification $1-a$   (density $\propto N^{1-a}$)", ylabel=r"$\beta-1$   ($c=0$)")
    a1.legend(loc="upper left")
    cs = sorted(t.c.unique())
    col = colors_for(plt, cs, "plasma")
    for cv in cs:
        g = t[t.c == cv].sort_values("a")
        a2.fill_between(g.a, g.lo, g.hi, alpha=0.2, color=col[cv], lw=0)
        a2.plot(g.a, g.beta, "o-", color=col[cv], label=f"$c={cv:g}$")
    unity(a2)
    a2.set(xlabel="area exponent $a$   (area $\\propto N^a$)", ylabel=r"$\beta$")
    a2.invert_xaxis()
    a2.legend(loc="best")
    panel(a1, "a")
    panel(a2, "b")
    save(fig, out, "figS2_densification")
    manifest("figS2_densification", "(a) β−1 vs 1−a at c = 0 with fit through origin; (b) β vs a "
             "per c (95% CI)", ["paper_densification", "paper_constant_density" if C is not None else None])
    note("\n== Fig S2: densification (paper_densification)")
    for _, r in t.sort_values(["c", "a"]).iterrows():
        note(f"  a = {r.a:.3f}  c = {r.c:<4g}  β = {r.beta:.3f}  [{r.lo:.3f}, {r.hi:.3f}]")
        num(f"beta_a{r.a:.3f}_c{ckey(r.c)}", r.beta)
    note(f"  density elasticity ε (fit through origin, c = 0) = {eps:.3f}  -> β(a = 2/3) ≈ {1 + eps / 3:.3f}")
    num("epsilon", eps)
    num("beta_city_path_pred", 1 + eps / 3)
    if C is not None:
        c0 = C[C.centralization == 0]
        if len(c0):
            est, lo, hi, _, _ = bootstrap_exponent(c0, "N", "Y", n_boot=500)
            note(f"  extensivity control (constant density, c = 0): β = {est:.3f} [{lo:.3f}, {hi:.3f}]  (expect 1)")
            num("beta_const_density", est)
            num("beta_const_density_lo", lo)
            num("beta_const_density_hi", hi)


# ---------------------------------------------------------------------- option B helpers
def _c_col(df):
    return "hub_share_completed" if "hub_share_completed" in df else "hub_share_started"


def _crossing_N(N, c, level=0.5):
    """First N (log-interpolated) where c falls below `level`; NaN if it never does."""
    N, c = np.asarray(N, float), np.asarray(c, float)
    for i in range(len(N) - 1):
        if c[i] >= level > c[i + 1]:
            f = (c[i] - level) / (c[i] - c[i + 1])
            return float(np.exp(np.log(N[i]) + f * (np.log(N[i + 1]) - np.log(N[i]))))
    return math.nan


def emergent_table(CH, FX):
    import analysis_choice as ac
    premia = sorted(CH.hub_premium.unique())
    cc = _c_col(CH)
    em = CH.groupby(["hub_premium", "N"]).agg(c=(cc, "mean"), c_sd=(cc, "std"),
                                              v=("y_value_per_capita", "mean"),
                                              v_sd=("y_value_per_capita", "std"),
                                              n=("y_value_per_capita", "size")).reset_index()
    pl = ac.planner(FX, premia)
    t = em.merge(pl, on=["hub_premium", "N"], how="left")
    t["gap"] = 1 - t.v / t.y_value_opt
    if 0.0 in premia:
        g0 = t[t.hub_premium == 0].set_index("N").gap
        t["gap_explore"] = t.N.map(g0)
        t["gap_net"] = t.gap - t.gap_explore
    return t, premia


# ---------------------------------------------------------------------- figure 4
def fig_choice(CH, FX, plt, out, ch_name):
    import analysis_choice as ac
    t, premia = emergent_table(CH, FX)
    col = colors_for(plt, premia, "viridis")
    fig, ax = plt.subplots(2, 2, figsize=(FULL, 5.0), constrained_layout=True)
    (a, b), (c, d) = ax
    for pi in premia:
        g = t[t.hub_premium == pi].sort_values("N")
        a.plot(g.N, g.c, "o-", color=col[pi])
        a.plot(g.N, g.c_opt, "--", lw=0.9, color=col[pi], drawstyle="steps-mid")
        b.plot(g.N, g.v, "o-", color=col[pi])
        b.plot(g.N, g.y_value_opt, "--", lw=0.9, color=col[pi])
        if pi == 0:
            c.plot(g.N, 100 * g.gap, "k--", lw=1.4, label=r"exploration cost ($\pi=0$)")
        else:
            c.plot(g.N, 100 * g.gap, "o-", color=col[pi])
    logx(a)
    a.set_ylabel("centralization (hub share of jobs)")
    a.set_ylim(-0.03, 1.03)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], color="0.3", marker="o", label="agents choose"),
                      Line2D([], [], color="0.3", ls="--", label="planner optimum")], loc="lower left")
    b.set_yscale("log")
    logx(b)
    b.set_ylabel("value per agent")
    logx(c)
    c.axhline(0, color="0.6", lw=0.6, ls=":")
    c.set_ylabel("efficiency gap to planner (%)")
    c.legend(loc="upper left")
    discrete_cbar(plt, fig, [a, b], premia, col, r"hub premium $\pi$")

    rows = []
    for pi in premia:
        g = CH[CH.hub_premium == pi]
        if "Y_value" in g and g.N.nunique() >= 3:
            est, lo, hi, _, _ = bootstrap_exponent(g, "N", "Y_value", n_boot=500)
        else:
            gg = t[t.hub_premium == pi].sort_values("N")
            est, lo, hi = ac._fit_beta(gg.N, gg.v), math.nan, math.nan
        gp = t[t.hub_premium == pi].sort_values("N")
        rows.append({"pi": pi, "beta_agents": est, "lo": lo, "hi": hi,
                     "beta_planner": ac._fit_beta(gp.N, gp.y_value_opt),
                     "N_half_agents": _crossing_N(gp.N, gp.c), "N_star_planner": _crossing_N(gp.N, gp.c_opt)})
    bt = pd.DataFrame(rows)
    xs = np.arange(len(bt))
    d.errorbar(xs, bt.beta_agents, yerr=[(bt.beta_agents - bt.lo).fillna(0), (bt.hi - bt.beta_agents).fillna(0)],
               fmt="o-", color=OKABE[0], capsize=2, elinewidth=0.8, label="agents choose")
    d.plot(xs, bt.beta_planner, "s--", color=OKABE[1], label="planner optimum")
    unity(d)
    d.set_xticks(xs)
    d.set_xticklabels([f"{v:g}" for v in bt.pi])
    d.set(xlabel=r"hub premium $\pi$", ylabel=r"$\beta$ of value")
    d.legend(loc="upper right")
    for axx, l in zip((a, b, c, d), "abcd"):
        panel(axx, l)
    save(fig, out, "fig5_emergent_choice")
    manifest("fig5_emergent_choice", "(a) hub share chosen by agents (solid) vs planner optimum "
             "(dashed) per π; (b) value per agent, agents vs planner; (c) efficiency gap, π = 0 is "
             "the pure exploration cost; (d) β of value vs π, agents (95% CI) vs planner",
             [ch_name, "paper_choice_fixed"])

    note(f"\n== Fig 5: emergent centralization ({ch_name} vs paper_choice_fixed)")
    rule = {k: sorted(map(str, CH[k].dropna().unique())) for k in
            ("choice_rule", "choice_payoff", "choice_explore", "T_burn") if k in CH}
    note(f"  learning settings: {rule}")
    for _, r in bt.iterrows():
        k = ckey(r.pi)
        note(f"  π = {r.pi:<5g}  β agents = {r.beta_agents:.3f} [{r.lo:.3f}, {r.hi:.3f}]   "
             f"β planner = {r.beta_planner:.3f}   agents switch (c = 0.5) at N ≈ {r.N_half_agents:,.0f}, "
             f"planner at N* ≈ {r.N_star_planner:,.0f}")
        num(f"choice_beta_agents_pi{k}", r.beta_agents)
        num(f"choice_beta_agents_pi{k}_lo", r.lo)
        num(f"choice_beta_agents_pi{k}_hi", r.hi)
        num(f"choice_beta_planner_pi{k}", r.beta_planner)
        num(f"choice_Nhalf_agents_pi{k}", r.N_half_agents, "{:,.0f}")
        num(f"choice_Nstar_planner_pi{k}", r.N_star_planner, "{:,.0f}")
    note("  per (π, N): c agents, c planner, efficiency gap, gap net of exploration (π = 0)")
    for _, r in t.sort_values(["hub_premium", "N"]).iterrows():
        gn = r.get("gap_net", math.nan)
        note(f"   π = {r.hub_premium:<4g} N = {int(r.N):>7d}  c = {r.c:.3f}  c_opt = {r.c_opt:.2f}  "
             f"gap = {100 * r.gap:6.2f}%  net = {100 * gn:6.2f}%")
        k = f"pi{ckey(r.hub_premium)}_N{int(r.N)}"
        num(f"choice_c_{k}", r.c)
        num(f"choice_copt_{k}", r.c_opt, "{:.2f}")
        num(f"choice_gap_{k}", 100 * r.gap, "{:.1f}")
        if np.isfinite(gn):
            num(f"choice_gapnet_{k}", 100 * gn, "{:.1f}")


# ---------------------------------------------------------------------- figure 5
def fig_trap(LK, CH, TT, FX, plt, out, ch_name):
    import analysis_choice as ac
    from matplotlib.lines import Line2D
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FULL, 2.6), constrained_layout=True)
    have = [x for x in (LK, CH) if x is not None]
    premia = sorted(set().union(*[set(x.hub_premium.unique()) for x in have]) - {0.0}) if have else []
    if LK is not None:
        premia = sorted(set(LK.hub_premium.unique()) - {0.0})
    col = colors_for(plt, premia, "viridis")
    commit = "frac_p_hub_high"

    # (a) committed to the hub, two starting points (raw payoff) vs skill-adjusted rule
    if LK is not None and commit in LK:
        for pi in premia:
            for p0, mk, ls, fill in ((0.95, "o", "-", col[pi]), (0.05, "o", ":", "white")):
                g = LK[(LK.hub_premium == pi) & (LK.choice_p0 == p0)].groupby("N")[commit].mean()
                a.plot(g.index, g.values, marker=mk, ls=ls, color=col[pi], mfc=fill, mew=0.8)
    if CH is not None and commit in CH:
        for pi in premia:
            g = CH[CH.hub_premium == pi].groupby("N")[commit].mean()
            if len(g):
                a.plot(g.index, g.values, ls="--", lw=0.8, color=col[pi], alpha=0.7)
    logx(a)
    a.set_ylabel("fraction committed to hub ($p_{\\rm hub}>0.8$)")
    a.set_ylim(-0.03, 1.03)
    a.legend(handles=[Line2D([], [], color="0.3", marker="o", label=r"raw, start $p_0=0.95$"),
                      Line2D([], [], color="0.3", marker="o", mfc="white", ls=":", label=r"raw, start $p_0=0.05$"),
                      Line2D([], [], color="0.3", ls="--", lw=0.8, label="skill-adjusted")],
             loc="lower left", fontsize=6.5)

    # (b) belief vs realized private advantage of the hub, one point per run
    def pts(df, payoff):
        if df is None or "mean_log_q_ratio" not in df or "hub_private_log_adv" not in df:
            return
        g = df if "choice_payoff" not in df else df[df.choice_payoff.fillna("raw") == payoff]
        g = g.dropna(subset=["mean_log_q_ratio", "hub_private_log_adv"])
        if len(g):
            b.scatter(g.hub_private_log_adv, g.mean_log_q_ratio, s=9, lw=0, alpha=0.7,
                      color=RAW_COL if payoff == "raw" else SKILL_COL)
    pts(LK, "raw")
    pts(TT, "raw")
    pts(CH, "per_skill")
    pts(TT, "per_skill")
    lim = b.get_xlim() + b.get_ylim()
    lo, hi = min(lim), max(lim)
    b.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    b.set_xlim(lo, hi)
    b.set_ylim(lo, hi)
    b.axhline(0, color="0.6", lw=0.5, ls=":")
    b.axvline(0, color="0.6", lw=0.5, ls=":")
    b.set(xlabel="realized hub advantage  $E\\ln r_{\\rm hub}-E\\ln r_{\\rm loc}$",
          ylabel="believed advantage  $\\langle\\ln Q_{\\rm hub}/Q_{\\rm loc}\\rangle$")
    b.text(0.04, 0.96, "over-optimistic\nabout the hub", transform=b.transAxes, va="top", fontsize=6.5, color="0.4")
    b.legend(handles=[Line2D([], [], color=RAW_COL, marker="o", ls="", label="raw payoff"),
                      Line2D([], [], color=SKILL_COL, marker="o", ls="", label="skill-adjusted")],
             loc="lower right", fontsize=6.5)

    # (c) efficiency gap, raw (lock-in, both starts pooled) vs skill-adjusted (sweep)
    if FX is not None:
        for df, ls, name in ((LK, "-", "raw"), (CH, "--", "per_skill")):
            if df is None:
                continue
            sub = df[df.hub_premium.isin(premia)]
            if not len(sub):
                continue
            pl = ac.planner(FX, premia)
            v = sub.groupby(["hub_premium", "N"]).y_value_per_capita.mean().reset_index()
            v = v.merge(pl, on=["hub_premium", "N"], how="inner")
            for pi in premia:
                g = v[v.hub_premium == pi].sort_values("N")
                c.plot(g.N, 100 * (1 - g.y_value_per_capita / g.y_value_opt), ls=ls,
                       marker="o" if name == "raw" else None, color=col[pi])
        logx(c)
        c.axhline(0, color="0.6", lw=0.6, ls=":")
        c.set_ylabel("efficiency gap to planner (%)")
        c.legend(handles=[Line2D([], [], color="0.3", marker="o", label="raw payoff"),
                          Line2D([], [], color="0.3", ls="--", label="skill-adjusted")], loc="upper left")
    if premia:
        discrete_cbar(plt, fig, [a, b, c], premia, col, r"hub premium $\pi$")
    for axx, l in zip((a, b, c), "abc"):
        panel(axx, l)
    save(fig, out, "fig6_competency_trap")
    manifest("fig6_competency_trap", "(a) fraction committed to the hub: raw payoff from p0 = 0.95 "
             "(filled, solid) and 0.05 (open, dotted) vs skill-adjusted payoff (dashed); (b) believed "
             "vs realized private advantage of the hub, one point per run (diagonal = correct beliefs); "
             "(c) efficiency gap to the planner, raw vs skill-adjusted",
             ["paper_choice_logit_lockin" if LK is not None else None, ch_name,
              "choice_trap_test" if TT is not None else None, "paper_choice_fixed" if FX is not None else None])

    note("\n== Fig 6: competency trap")
    if LK is not None:
        cc = _c_col(LK)
        note("  lock-in (raw payoff): hysteresis = c(p0 = 0.95) − c(p0 = 0.05), seed s.d. in brackets")
        worst = 0.0
        for (pi, N), g in LK.groupby(["hub_premium", "N"]):
            hi_ = g[g.choice_p0 == g.choice_p0.max()][cc]
            lo_ = g[g.choice_p0 == g.choice_p0.min()][cc]
            gap = hi_.mean() - lo_.mean()
            sd = float(np.nanmax([hi_.std(), lo_.std()]))
            fh = g[g.choice_p0 == g.choice_p0.max()].get(commit, pd.Series(dtype=float)).mean()
            fl = g[g.choice_p0 == g.choice_p0.min()].get(commit, pd.Series(dtype=float)).mean()
            note(f"   π = {pi:<4g} N = {int(N):>6d}  c: {hi_.mean():.3f} vs {lo_.mean():.3f}  "
                 f"gap {gap:+.3f} [{sd:.3f}]   committed hub: {fh:.3f} vs {fl:.3f}")
            k = f"pi{ckey(pi)}_N{int(N)}"
            num(f"lockin_gap_{k}", gap, "{:+.3f}")
            num(f"lockin_commit_hi_{k}", fh)
            num(f"lockin_commit_lo_{k}", fl)
            if abs(gap) > abs(worst):
                worst = gap
        num("lockin_gap_maxabs", worst, "{:+.3f}")
    if TT is not None and "choice_payoff" in TT:
        note("  trap test (raw vs per_skill): realized hub advantage, belief, committed, gap")
        pl = ac.planner(FX, sorted(TT.hub_premium.unique())) if FX is not None else None
        for (pi, N, pay), g in TT.groupby(["hub_premium", "N", "choice_payoff"]):
            row = {k: g[k].mean() for k in ("hub_private_log_adv", "mean_log_q_ratio", commit,
                                             "y_value_per_capita", _c_col(TT)) if k in g}
            gp = math.nan
            if pl is not None:
                m = pl[(pl.hub_premium == pi) & (pl.N == N)]
                if len(m):
                    gp = 1 - row["y_value_per_capita"] / m.y_value_opt.iloc[0]
            note(f"   π = {pi:<4g} N = {int(N):>6d} {pay:>9s}: realized {row.get('hub_private_log_adv', math.nan):+.3f}  "
                 f"believed {row.get('mean_log_q_ratio', math.nan):+.3f}  committed {row.get(commit, math.nan):.3f}  "
                 f"gap {100 * gp:.1f}%")
            k = f"pi{ckey(pi)}_N{int(N)}_{pay.replace('_', '')}"
            num(f"trap_realized_{k}", row.get("hub_private_log_adv"), "{:+.2f}")
            num(f"trap_believed_{k}", row.get("mean_log_q_ratio"), "{:+.2f}")
            num(f"trap_commit_{k}", row.get(commit), "{:.2f}")
            num(f"trap_gap_{k}", 100 * gp, "{:.1f}")


# ---------------------------------------------------------------------- supplementary
def fig_coordination(V, E, plt, out):
    import analysis_bureaucracy as ab
    tab, _ = ab.optimum_table(V, E)
    tab["MV_rel"] = tab.MV_first / tab.y_zero
    cs = sorted(tab.centralization.unique())
    col = colors_for(plt, cs, "plasma")
    ncol = 3 if "efficiency_gap" in tab else 2
    fig, axes = plt.subplots(1, ncol, figsize=(FULL, 2.6), constrained_layout=True)
    for cv in cs:
        g = tab[tab.centralization == cv].sort_values("N")
        axes[0].plot(g.N, g.MV_rel, "o-", color=col[cv])
        if ncol == 3:
            axes[2].plot(g.N, 100 * g.efficiency_gap, "o-", color=col[cv])
    axes[0].set_yscale("log")
    logx(axes[0])
    axes[0].set_ylabel("value of first coordinators")
    Nmax = V.N.max()
    for cv in cs:
        g = V[(V.N == Nmax) & (V.centralization == cv)].groupby("coord_fixed_frac").y_per_capita.mean()
        g0 = g.iloc[0] if len(g) else 1.0
        axes[1].plot(100 * g.index, g.values / g0, "o-", color=col[cv])
        if E is not None:
            e = E[(E.N == Nmax) & (E.centralization == cv)]
            if len(e):
                axes[1].plot(100 * e.M_frac.mean(), e.y_per_capita.mean() / g0, "*", ms=9,
                             color=col[cv], markeredgecolor="k", mew=0.5)
    axes[1].set(xlabel="coordinators (% of agents)", ylabel=f"output / no coordinators ($N={Nmax:g}$)")
    if ncol == 3:
        logx(axes[2])
        axes[2].set_ylabel("output lost by emergent bureaucracy (%)")
    discrete_cbar(plt, fig, list(axes), cs, col, "centralization $c$")
    for axx, l in zip(axes, "abc"):
        panel(axx, l)
    save(fig, out, "figS1_coordination")
    manifest("figS1_coordination", "(a) marginal value of the first coordinators vs N; (b) output vs "
             "coordinator % at the largest N, ★ = emergent; (c) output lost by emergent bureaucracy",
             ["paper_coord_value", "paper_coord_value_endogenous" if E is not None else None])
    note("\n== Fig S1: coordination (paper_coord_value)")
    show = [k for k in ("N", "centralization", "frac_opt", "gain_opt", "MV_first", "MV_rel", "M_endo",
                        "over_ratio", "efficiency_gap", "opt_at_grid_edge") if k in tab]
    for line in tab[show].round(4).to_string(index=False).splitlines():
        note("  " + line)


# ---------------------------------------------------------------------- outputs
def write_text_outputs(out):
    with open(os.path.join(out, "paper_numbers.txt"), "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(os.path.join(out, "paper_numbers.json"), "w") as fh:
        json.dump(NUMBERS, fh, indent=1, sort_keys=True)
    with open(os.path.join(out, "paper_macros.tex"), "w") as fh:
        fh.write("% Generated by paper_figures.py on " + _dt.datetime.now().isoformat(timespec="seconds") + "\n")
        fh.write("% Usage: \\input{paper_macros} in the preamble, then \\FM{key}, e.g. \\FM{cstar}.\n")
        fh.write("% An unknown key prints a bold ??key in the PDF.\n")
        fh.write("\\providecommand{\\FM}[1]{\\ifcsname FM@#1\\endcsname\\csname FM@#1\\endcsname"
                 "\\else\\textbf{??\\detokenize{#1}}\\fi}\n")
        for k, v in sorted(NUMBERS.items()):
            v = v.replace(",", "{,}")
            fh.write(f"\\expandafter\\def\\csname FM@{k}\\endcsname{{{v}}}\n")
    with open(os.path.join(out, "figures_manifest.txt"), "w") as fh:
        fh.write("Figures produced by paper_figures.py\n\n")
        for fig, panels, used in MANIFEST:
            fh.write(f"{fig}.pdf / {fig}.png\n  panels: {panels}\n  data:\n")
            for u in used:
                p, n = SOURCES.get(u, ("?", 0))
                fh.write(f"    {u}: {n} runs  ({p})\n")
            fh.write("\n")
        missing = sorted(set(EXPECTED) - set(SOURCES))
        if missing:
            fh.write("Experiments NOT found (their figures or panels are missing):\n")
            for m in missing:
                fh.write(f"    {m}\n")


EXPECTED = ["paper_density_exponent", "density_window_scan", "paper_centralization_sweep", "paper_constant_density", "paper_densification",
            "paper_choice_fixed", "paper_choice_logit_sweep", "paper_choice_logit_lockin", "choice_trap_test"]


def main():
    ap = argparse.ArgumentParser(description="Main paper figures (publication style).")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--si", action="store_true", help="also the supplementary figures")
    ap.add_argument("--only", nargs="*", default=None,
                    help="only these figures: power_laws scaling gamma choice trap coordination densification")
    args = ap.parse_args()
    out = args.out or os.path.join(paths.root(args.root), "paper_figures",
                                   _dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    paths.ensure(out)
    plt = setup_style()

    want = set(args.only) if args.only else None

    def on(key, si=False):
        return (key in want) if want is not None else (not si or args.si)

    P = W = S = C = D = FX = CH = LK = TT = None
    ch_name = None
    if on("power_laws"):
        P = get("paper_density_exponent", args.root)
        W = get("density_window_scan", args.root)
    if on("scaling") or on("gamma"):
        S = get("paper_centralization_sweep", args.root)
    if on("gamma") or on("densification", si=True):
        C = get("paper_constant_density", args.root)
        D = get("paper_densification", args.root)
    if on("gamma") or on("choice") or on("trap"):
        FX = get("paper_choice_fixed", args.root)
        ch_name, CH = first("paper_choice_logit_sweep", "paper_choice_sweep", args.root)
    if on("trap"):
        LK = get("paper_choice_logit_lockin", args.root)
        TT = get("choice_trap_test", args.root)

    if on("power_laws") and P is not None:
        fig_power_laws(P, W, plt, out)
    if on("scaling") and S is not None:
        fig_scaling(S, plt, out)
    if on("gamma"):
        frames = [df.assign(source=n) for n, df in (("paper_centralization_sweep", S),
                                                    ("paper_constant_density", C),
                                                    ("paper_densification", D),
                                                    ("paper_choice_fixed", FX)) if df is not None]
        if frames:
            try:   # option B choice runs join the collapse if analysis_gamma can match their baseline
                fig_gamma(frames + ([CH.assign(source=ch_name)] if CH is not None else []), plt, out)
            except Exception as e:  # noqa: BLE001
                note(f"[note] collapse with {ch_name} failed ({e!r}); retrying without it")
                fig_gamma(frames, plt, out)
    if on("choice") and CH is not None and FX is not None:
        fig_choice(CH, FX, plt, out, ch_name)
    if on("trap") and (LK is not None or TT is not None):
        fig_trap(LK, CH, TT, FX, plt, out, ch_name)
    if on("densification", si=True) and D is not None:
        fig_densification(D, C, plt, out)
    if on("coordination", si=True):
        V = get("paper_coord_value", args.root)
        E = get("paper_coord_value_endogenous", args.root)
        if V is not None:
            fig_coordination(V, E, plt, out)

    write_text_outputs(out)
    print(f"\nfigures (pdf + png), paper_numbers.txt/.json, paper_macros.tex and "
          f"figures_manifest.txt in {out}/")


if __name__ == "__main__":
    main()