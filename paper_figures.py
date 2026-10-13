"""
paper_figures.py — The MAIN figures of the paper (publication style), from the paper_* experiments.

    python paper_figures.py                 # newest launch of every experiment used
    python paper_figures.py --out <folder>  # default: <data root>/paper_figures/<timestamp>/
    python paper_figures.py --nmin 256      # main range of Fig. 2 from N = 256 (E1 decision rule)
    python paper_figures.py --companion     # also the companion-paper figures (choice, trap, coordination)
    Paper 1 analyses (default): power_laws epsilon ablations generality robustness scaling gamma
    transient seeds pace poly densification; each is skipped while its experiment is missing.

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


def epsilon_se(W, rho_lo, rho_hi):
    """Standard error of the window ε: regression s.e. over the densities in the window
    (reflects curvature and noise; seeds are averaged first)."""
    if W is None:
        return math.nan
    dcol = "density" if "density" in W else "rho0"
    g = W.groupby("rho0").agg(rho=(dcol, "mean"), y=("y_per_capita", "mean")).reset_index()
    g = g[(g.rho >= rho_lo * 0.999) & (g.rho <= rho_hi * 1.001)]
    if len(g) < 3:
        return math.nan
    x, y = np.log(g.rho.values), np.log(g.y.values)
    _, cov = np.polyfit(x, y, 1, cov=True)
    return float(np.sqrt(cov[0, 0]))


def power_law_stats(P, nmin=None, nmax=None, n_boot=500):
    """β per area exponent over N in [nmin, nmax]: bootstrap CI, R², local exponents (stride 2),
    ε fitted through (a = 1, β = 1) and with a free intercept β = β1 + ε(1 − a)."""
    sub = P
    if nmin is not None:
        sub = sub[sub.N >= nmin]
    if nmax is not None:
        sub = sub[sub.N <= nmax]
    loc = local_exponents(sub, ["area_exponent"], "N", "Y", stride=2)
    rows = []
    for a, g in sub.groupby("area_exponent"):
        est, lo, hi, r2, _ = bootstrap_exponent(g, "N", "Y", n_boot=n_boot)
        lg = loc[loc.area_exponent == a].local_exponent
        rows.append({"a": a, "beta": est, "lo": lo, "hi": hi, "r2": r2,
                     "local_min": lg.min(), "local_max": lg.max(), "local_sd": lg.std()})
    bt = pd.DataFrame(rows).sort_values("a")
    x = 1 - bt.a.to_numpy()
    eps_fit = float(np.sum(x * (bt.beta - 1)) / np.sum(x * x)) if np.sum(x * x) > 0 else math.nan
    if len(bt) >= 2:
        eps_off, beta1_off = np.polyfit(x, bt.beta.to_numpy(), 1)
    else:
        eps_off, beta1_off = math.nan, math.nan
    return sub, bt, loc, eps_fit, float(eps_off), float(beta1_off)


def joint_density_size_fit(P, nmin=None, n_boot=300, seed=0):
    """ln y = c + ε ln ρ + γ ln N over ALL runs of the a-sweep (actual densities). β = 1 + ε(1 − a)
    holds exactly when y depends on N only through ρ, i.e. γ = 0; γ ≠ 0 is a pure size effect.
    95% CI from a bootstrap over seed labels (common random numbers keep seeds paired)."""
    sub = P if nmin is None else P[P.N >= nmin]
    if "density" not in sub or sub.area_exponent.nunique() < 2:
        return dict(eps=math.nan, gamma=math.nan)
    def fit(d):
        m = d.groupby(["area_exponent", "N"]).agg(y=("y_per_capita", "mean"), rho=("density", "mean")).reset_index()
        X = np.column_stack([np.ones(len(m)), np.log(m.rho), np.log(m.N)])
        coef, *_ = np.linalg.lstsq(X, np.log(m.y), rcond=None)
        res = np.log(m.y) - X @ coef
        return coef[1], coef[2], float(np.sqrt(np.mean(res ** 2)))
    eps, gam, rmse = fit(sub)
    seeds = sorted(sub.seed.unique()) if "seed" in sub else []
    rs = np.random.RandomState(seed)
    be, bg = [], []
    if len(seeds) >= 2:
        for _ in range(n_boot):
            pick = rs.choice(seeds, len(seeds))
            e, g_, _ = fit(pd.concat([sub[sub.seed == s_] for s_ in pick]))
            be.append(e)
            bg.append(g_)
    q = (lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))) if be else (lambda v: (math.nan, math.nan))
    (elo, ehi), (glo, ghi) = q(be), q(bg)
    return dict(eps=eps, eps_lo=elo, eps_hi=ehi, gamma=gam, gamma_lo=glo, gamma_hi=ghi, rmse=rmse)


def record_power_law(bt, eps_w, prefix="pl", suffix=""):
    """Keys <prefix>_beta_a<a><suffix>, ... (suffix '' for the main range)."""
    worst = 0.0
    for _, r in bt.iterrows():
        pred = 1 + eps_w * (1 - r.a) if np.isfinite(eps_w) else math.nan
        spread = r.local_max - r.local_min
        worst = max(worst, spread if np.isfinite(spread) else 0.0)
        k = f"a{r.a:.3f}"
        num(f"{prefix}_beta_{k}{suffix}", r.beta)
        num(f"{prefix}_beta_{k}_lo{suffix}", r.lo)
        num(f"{prefix}_beta_{k}_hi{suffix}", r.hi)
        num(f"{prefix}_R2_{k}{suffix}", r.r2, "{:.4f}")
        num(f"{prefix}_localspread_{k}{suffix}", spread, "{:.3f}")
        num(f"{prefix}_pred_{k}{suffix}", pred)
        note(f"  a = {r.a:.3f}  β = {r.beta:.3f} [{r.lo:.3f}, {r.hi:.3f}]  R² = {r.r2:.4f}  "
             f"local β {r.local_min:.3f}–{r.local_max:.3f} (s.d. {r.local_sd:.3f})  predicted {pred:.3f}")
    num(f"{prefix}_localspread_max{suffix}", worst, "{:.3f}")
    num(f"{prefix}_R2_min{suffix}", bt.r2.min(), "{:.4f}")
    if np.isfinite(eps_w):
        dev = (bt.beta - (1 + eps_w * (1 - bt.a))).abs()
        num(f"{prefix}_meanabsdev{suffix}", dev.mean(), "{:.3f}")
    return worst


def fig_power_laws(P, W, plt, out, nmin=64, mf=None):
    """Fig. 2. Main range N >= nmin (--nmin); keys for N >= 64, 256, 512 are always written with
    suffixes _N64/_N256/_N512 so the decision rule of E1 can be applied from paper_numbers."""
    avals = sorted(P.area_exponent.unique())
    col = _a_colors(plt, avals)
    dcol = "density" if "density" in P else None
    rho_lo = P[dcol].min() if dcol else math.nan
    rho_hi = P[dcol].max() if dcol else math.nan
    eps_w, eps_w_spread = epsilon_constant_density(W, rho_lo, rho_hi)
    eps_se = epsilon_se(W, rho_lo, rho_hi)
    num("pl_eps_window_se", eps_se, "{:.2f}")
    Pm, bt, loc, eps_fit, eps_off, beta1_off = power_law_stats(P, nmin=nmin)
    y = mean_sem(Pm, ["area_exponent", "N"], "y_per_capita")

    fig, ax = plt.subplots(2, 2, figsize=(FULL, 5.0), constrained_layout=True)
    (a1, a2), (a4, a3) = ax
    for a in avals:
        g = y[y.area_exponent == a].sort_values("N")
        b = bt[bt.a == a].beta.iloc[0]
        a1.plot(g.N, g["mean"], "o", color=col[a], ms=2.8)
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
    for axx in (a1, a4):
        axx.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        axx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axx.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    unity(a2)
    logx(a2)
    a2.set_ylabel(r"local exponent $\beta(N)$")
    for a in avals:
        h = loc[loc.area_exponent == a].sort_values("N_mid")
        if len(h):
            a2.annotate(f"$a={a:.2f}$", (h.N_mid.iloc[-1], h.local_exponent.iloc[-1]), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=5.8, color=col[a])
    a2.set_xlim(right=loc.N_mid.max() * 3.2)

    # (c) collapse: output per agent depends on N only through density
    if dcol:
        m = Pm.groupby(["area_exponent", "N"]).agg(rho=(dcol, "mean"), y=("y_per_capita", "mean")).reset_index()
        for a in avals:
            g = m[m.area_exponent == a]
            a4.plot(g.rho, g.y, "o", ms=2.6, color=col[a], alpha=0.85, lw=0)
    if W is not None:
        wc = "density" if "density" in W else "rho0"
        gw = W.groupby("rho0").agg(rho=(wc, "mean"), y=("y_per_capita", "mean")).reset_index()
        a4.plot(gw.rho, gw.y, "s", ms=4, mfc="none", mec="k", mew=0.7,
                label=r"constant density, $N=4096$")
    if mf is not None and len(mf):
        a4.plot(mf.rho, mf.y, "--", color="0.35", lw=0.9, label="mean field (no fit)")
    a4.set(xscale="log", yscale="log", xlabel=r"density $\rho=N/L^2$", ylabel="output per agent  $Y/N$")
    a4.axvspan(rho_lo, rho_hi, color="0.92", zorder=0, lw=0)
    a4.legend(loc="upper left", fontsize=6.2)

    aa = np.linspace(min(avals) - 0.03, max(avals) + 0.03, 50)
    if np.isfinite(eps_w):
        a3.plot(aa, 1 + eps_w * (1 - aa), "k-", lw=0.9,
                label=rf"prediction $1+\varepsilon(1-a)$, $\varepsilon={eps_w:.2f}$" + "\n(constant-density runs)")
    a3.plot(aa, 1 + eps_fit * (1 - aa), "k:", lw=0.9, label=rf"fit, $\varepsilon={eps_fit:.2f}$")
    if mf is not None and "eps_window" in mf.attrs and np.isfinite(mf.attrs["eps_window"]):
        em = mf.attrs["eps_window"]
        a3.plot(aa, 1 + em * (1 - aa), "--", color="0.45", lw=0.9, label=rf"mean field, $\varepsilon={em:.2f}$")
    for _, r in bt.iterrows():
        a3.errorbar(r.a, r.beta, yerr=[[r.beta - r.lo], [r.hi - r.beta]], fmt="o", ms=3.5,
                    color=col[r.a], capsize=2, elinewidth=0.8, zorder=3)
    unity(a3)
    unity(a3, horizontal=False)
    a3.set(xlabel=r"area exponent $a$   (area $\propto N^a$)", ylabel=r"scaling exponent $\beta$")
    a3.legend(loc="upper right", fontsize=6.0)
    a3.text(0.03, 0.03, "city-like: $a<1$, $\\beta>1$\ncompany-like: $a>1$, $\\beta<1$",
            transform=a3.transAxes, ha="left", va="bottom", fontsize=6.2, color="0.35",
            bbox=dict(fc="white", ec="none", pad=1.0), zorder=5)
    for axx, l in zip((a1, a2, a4, a3), "abcd"):
        panel(axx, l)
    save(fig, out, "fig2_power_law_transition")
    manifest("fig2_power_law_transition",
             f"N >= {nmin}. (a) Y/N vs N per a; (b) smoothed local exponent; (c) Y/N vs density, all a "
             "and the constant-density scan, mean field dashed, Fig.-2 window shaded; (d) β vs a, "
             "95% bootstrap CI, prediction with ε from the constant-density scan, fit, mean field",
             ["paper_density_exponent", "density_window_scan" if W is not None else None])

    note(f"\n== Fig 2: one knob, pure power laws (paper_density_exponent), main range N >= {nmin}")
    note(f"  densities visited: {rho_lo:.4f} – {rho_hi:.4f}")
    note(f"  ε fitted from β vs a (through β = 1 at a = 1): {eps_fit:.3f}; with free intercept: "
         f"ε = {eps_off:.3f}, β(a=1) = {beta1_off:.3f}")
    note(f"  ε measured at constant density over the same densities: {eps_w:.3f} "
         f"(spread of local ε {eps_w_spread:.3f})")
    num("pl_rho_lo", rho_lo, "{:.3f}")
    num("pl_rho_hi", rho_hi, "{:.3f}")
    num("pl_eps_fit", eps_fit, "{:.2f}")
    num("pl_eps_offset", eps_off, "{:.2f}")
    num("pl_beta1_offset", beta1_off, "{:.3f}")
    num("pl_eps_window", eps_w, "{:.2f}")
    num("pl_N_min", Pm.N.min(), "{:,.0f}")
    num("pl_N_max", Pm.N.max(), "{:,.0f}")
    num("pl_decades", math.log10(Pm.N.max() / Pm.N.min()), "{:.1f}")
    num("pl_nmin_main", nmin, "{:,.0f}")
    num("pl_runs", len(P), "{:,.0f}")
    num("pl_seeds", P.seed.nunique() if "seed" in P else math.nan, "{:.0f}")
    worst = record_power_law(bt, eps_w)
    note(f"  largest spread of local β over N for any a: {worst:.3f}; smallest R²: {bt.r2.min():.4f}")
    jf = joint_density_size_fit(P, nmin=nmin)
    note(f"  joint fit ln y = c + ε ln ρ + γ ln N (all a, actual densities): ε = {jf['eps']:.3f} "
         f"[{jf.get('eps_lo', math.nan):.3f}, {jf.get('eps_hi', math.nan):.3f}], γ = {jf['gamma']:+.4f} "
         f"[{jf.get('gamma_lo', math.nan):+.4f}, {jf.get('gamma_hi', math.nan):+.4f}], rmse {jf.get('rmse', math.nan):.4f}")
    num("pl_joint_eps", jf["eps"])
    num("pl_joint_eps_lo", jf.get("eps_lo"))
    num("pl_joint_eps_hi", jf.get("eps_hi"))
    num("pl_joint_gamma", jf["gamma"], "{:+.3f}")
    num("pl_joint_gamma_lo", jf.get("gamma_lo"), "{:+.3f}")
    num("pl_joint_gamma_hi", jf.get("gamma_hi"), "{:+.3f}")
    num("pl_joint_rmse", jf.get("rmse"), "{:.3f}")

    # E1: the same numbers for several lower cut-offs (suffix _N<nmin>), and the decision rule
    note("\n== E1: small-N offset — refits for N >= 64, 256, 512")
    res = {}
    for nm in (64, 256, 512):
        if nm >= P.N.max() / 16:
            continue
        note(f" N >= {nm}:")
        _, b2, _, e2, eo2, b1o2 = power_law_stats(P, nmin=nm)
        record_power_law(b2, eps_w, suffix=f"_N{nm}")
        num(f"pl_eps_fit_N{nm}", e2, "{:.2f}")
        num(f"pl_eps_offset_N{nm}", eo2, "{:.2f}")
        num(f"pl_beta1_offset_N{nm}", b1o2, "{:.3f}")
        num(f"pl_decades_N{nm}", math.log10(P.N.max() / nm), "{:.1f}")
        j2 = joint_density_size_fit(P, nmin=nm)
        num(f"pl_joint_gamma_N{nm}", j2["gamma"], "{:+.3f}")
        num(f"pl_joint_gamma_lo_N{nm}", j2.get("gamma_lo"), "{:+.3f}")
        num(f"pl_joint_gamma_hi_N{nm}", j2.get("gamma_hi"), "{:+.3f}")
        r1 = b2[np.isclose(b2.a, 1.0)]
        dev = (b2.beta - (1 + eps_w * (1 - b2.a))).abs().mean() if np.isfinite(eps_w) else math.nan
        res[nm] = dict(b1=r1.beta.iloc[0] if len(r1) else math.nan,
                       lo=r1.lo.iloc[0] if len(r1) else math.nan,
                       hi=r1.hi.iloc[0] if len(r1) else math.nan, dev=dev)
        note(f"   β(a=1) = {res[nm]['b1']:.3f} [{res[nm]['lo']:.3f}, {res[nm]['hi']:.3f}],  "
             f"mean |β − prediction| = {dev:.3f},  joint γ = {j2['gamma']:+.4f}")
    if 64 in res and 256 in res:
        adopt = (res[256]["lo"] <= 1.0 <= res[256]["hi"]) and (res[256]["dev"] < res[64]["dev"])
        NUMBERS["E1_adopt_N256"] = "yes" if adopt else "no"
        note(f"  DECISION E1: adopt N >= 256 as the main range? {'YES' if adopt else 'NO'} "
             f"(rule: β(a=1) CI contains 1 AND mean |β − prediction| drops)")
    # diagnostics of the residual size effect at constant density (a = 1)
    d1 = P[np.isclose(P.area_exponent, 1.0)]
    if len(d1):
        cols = [c for c in ("y_per_capita", "mean_w", "working_frac", "idle_frac", "blocked_frac",
                            "mean_nsame", "DOL_tasks", "mean_top_share", "density") if c in d1]
        t = d1.groupby("N")[cols].mean()
        note("  a = 1 (constant density): what changes with N?")
        for line in t.round(4).to_string().splitlines():
            note("   " + line)
        for c in ("mean_w", "working_frac", "blocked_frac", "DOL_tasks", "density", "mean_nsame"):
            if c in t and (t[c] > 0).all() and len(t) > 2:
                sl = np.polyfit(np.log(t.index.values), np.log(t[c].values), 1)[0]
                num(f"pl_a1_slope_{c}", sl, "{:+.4f}")
                if c == "DOL_tasks":
                    num("pl_a1_DOL_min", t[c].min(), "{:.3f}")
                    num("pl_a1_DOL_max", t[c].max(), "{:.3f}")


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
        num(f"beta_c{ckey(r.c)}_R2", r.r2, "{:.4f}")
    note(f"  critical centralization c* (β = 1) ≈ {cstar:.3f}")
    num("cs_runs", len(S), "{:,.0f}")
    num("cs_seeds", S.seed.nunique() if "seed" in S else math.nan, "{:.0f}")
    c0 = S[S.centralization == 0]
    if len(c0):
        big = c0[c0.N == c0.N.max()]
        for k_, c_ in (("density", "density"), ("blocked", "blocked_frac"), ("idle", "idle_frac"),
                       ("nsame", "mean_nsame"), ("DOL", "DOL_tasks")):
            if c_ in big:
                num(f"city_c0_{k_}_Nmax", big[c_].mean(), "{:.3f}")
        small = c0[c0.N == c0.N.min()]
        if "blocked_frac" in small:
            num("city_c0_blocked_Nmin", small["blocked_frac"].mean(), "{:.3f}")
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
    A = math.exp(-fit.get("intercept", 0.0))       # analysis_gamma fits with a free intercept
    a1.plot(xs, A * xs ** (-p), "k-", lw=1.0, label=rf"${A:.2f}\,(1+\Gamma)^{{-{p:.2f}}}$")
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
    num("gamma_intercept", fit.get("intercept", math.nan), "{:+.3f}")
    num("gamma_prefactor", math.exp(-fit.get("intercept", 0.0)), "{:.3f}")
    num("gamma_R2", fit["R2_free_fit"], "{:.3f}")
    num("gamma_R2_p1", fit["R2_with_p_equal_1"], "{:.3f}")
    num("gamma_npoints", fit["n_points"], "{:,.0f}")
    if not dec.empty:
        err = dec.beta_local_obs - (dec.beta0_local - p * dec.dln1pGamma_dlnN)
        note(f"  β decomposition over {len(dec)} points: observed − predicted = {err.mean():+.4f} ± {err.std():.4f}")
        num("decomp_err_mean", err.mean(), "{:+.3f}")
        num("decomp_err_sd", err.std(), "{:.3f}")
        num("decomp_npoints", len(dec), "{:,.0f}")
    for s_ in srcs:
        num(f"gamma_n_{s_}", int((pts.source == s_).sum()), "{:,.0f}")
    return fit, pts, dec


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
        num(f"beta_a{r.a:.3f}_c{ckey(r.c)}_lo", r.lo)
        num(f"beta_a{r.a:.3f}_c{ckey(r.c)}_hi", r.hi)
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


# ====================================================================== PAPER 1 analyses
# Each function writes keys to paper_numbers.json (\FM{key}) and, where a table is long, a
# LaTeX tabular body to <out>/tables/<name>.tex (the manuscript \input's it).
TABLES = {}


def write_table(out, name, header, rows):
    """rows: list of lists of strings. Writes <out>/tables/<name>.tex (tabular body only)."""
    d = paths.ensure(os.path.join(out, "tables"))
    with open(os.path.join(d, f"{name}.tex"), "w") as fh:
        fh.write("% generated by paper_figures.py; do not edit. Complete tabular body with rules:\n")
        fh.write("% read it with the TeX primitive (\\csname @@input\\endcsname file) inside tabular.\n")
        fh.write(r"\toprule" + "\n" + " & ".join(header) + r" \\ \midrule" + "\n")
        for r in rows:
            fh.write(" & ".join(r) + r" \\" + "\n")
        fh.write(r"\bottomrule" + "\n")
    TABLES[name] = len(rows)


def _f(v, fmt="{:.3f}"):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def check_pinned(df, name):
    """Warn if an experiment's stored parameters differ from config.MODEL / BEST (the values the
    paper reports in its parameter table). Keys missing from old runs are skipped."""
    try:
        import config
    except Exception:  # noqa: BLE001
        return
    exp = config.EXPERIMENTS.get(name, {})
    varied = set(exp.get("grid", {})) | set(exp.get("base", {}))
    derived = {"route_moves": "route_length", "rho_at_Nref": "rho0", "phi_over_xi": "phi"}
    varied |= {v for k, v in derived.items() if k in varied}
    bad = []
    for k, v in dict(config.BEST, **config.MODEL).items():
        if k in varied or k not in df:
            continue
        vals = df[k].dropna().unique()
        if len(vals) and not all((x == v) or (isinstance(x, float) and np.isclose(x, v)) for x in vals):
            bad.append(f"{k}={list(vals)[:3]} (paper: {v})")
    if bad:
        note(f"[WARNING] {name}: parameters differ from the paper's table: " + "; ".join(bad))


def little(df):
    """Per-run times from Little's law (steps per job): cycle, on lattice, idle, delivering."""
    d = df.copy()
    comp = d["completions"].clip(lower=1e-12)
    d["T_cycle"] = d["N"] / comp
    d["T_lattice"] = d["working_frac"] * d["N"] / comp
    d["T_idle"] = d["idle_frac"] * d["N"] / comp
    dl = d["delivering_frac"] if "delivering_frac" in d else 0.0
    d["T_work"] = (d["working_frac"] - dl) * d["N"] / comp
    return d


def _mf_params(df):
    import meanfield as mfm
    return mfm.MFParams.from_params(df.iloc[0].to_dict())


# ---------------------------------------------------------------------- E2: ε window and mean field
def epsilon_window(W, P, out, plt):
    """Per-density ε from the constant-density scan, its exact accounting decomposition, and the
    mean-field predictions (crude, closed, semi-empirical with measured n, q, B)."""
    import meanfield as mfm
    MFP = _mf_params(W)
    rho_lo = P["density"].min() if P is not None and "density" in P else 0.018
    rho_hi = P["density"].max() if P is not None and "density" in P else 0.082
    wc = "density" if "density" in W else "rho0"
    g = W.groupby("rho0").agg(rho=(wc, "mean"), y=("y_per_capita", "mean"), Y=("Y", "mean"),
                              N=("N", "mean"), work=("working_frac", "mean"), idle=("idle_frac", "mean"),
                              B=("blocked_frac", "mean"), w=("mean_w", "mean"), n=("mean_nsame", "mean"),
                              ysd=("y_per_capita", "std")).reset_index().sort_values("rho")
    g["Yacc"] = g.N * g.work * 0.5 * (1 - g.B) * g.w
    g["acc_ratio"] = g.Y / g.Yacc
    lr = np.log(g.rho.values)
    mid = np.exp(0.5 * (lr[1:] + lr[:-1]))
    dl = np.diff(lr)
    loc = np.diff(np.log(g.y.values)) / dl
    parts = {k: np.diff(np.log(v)) / dl for k, v in
             (("work", g.work.values), ("unblocked", 1 - g.B.values), ("w", g.w.values),
              ("ratio", g.acc_ratio.values))}
    # mean field on the same densities
    mf = pd.DataFrame(mfm.curve(g.rho.values, MFP))
    mfmid = pd.DataFrame(mfm.curve(mid, MFP))
    n_mid = np.exp(0.5 * (np.log(g.n.values[1:]) + np.log(g.n.values[:-1])))
    q_mid = 0.5 * (g.idle.values[1:] + g.idle.values[:-1])
    semi = mfm.analytic_epsilon(n_mid, np.diff(np.log(g.n.values)) / dl, q_mid, parts["unblocked"], MFP)
    crude = mfm.crude_epsilon(mid, MFP)
    inwin = (mid >= rho_lo * 0.999) & (mid <= rho_hi * 1.001)
    T = pd.DataFrame(dict(rho=mid, eps=loc, eps_work=parts["work"], eps_unblocked=parts["unblocked"],
                          eps_w=parts["w"], eps_ratio=parts["ratio"], eps_mf=mfmid.eps.values,
                          eps_semi=semi, eps_crude=crude, inwin=inwin))
    # window statistics (densities of the Fig. 2 runs)
    ew = T[T.inwin]
    note("\n== E2: density elasticity in the constant-density scan (density_window_scan)")
    for line in T.round(3).to_string(index=False).splitlines():
        note("  " + line)
    note("  accounting identity y = ½·working·(1−B)·w̄·ratio; ratio (overshoot, timing) per density: "
         + ", ".join(f"{v:.3f}" for v in g.acc_ratio))
    num("epsw_acc_ratio_min", g.acc_ratio.min())
    num("epsw_acc_ratio_max", g.acc_ratio.max())
    if len(ew):
        for k in ("eps", "eps_w", "eps_work", "eps_unblocked", "eps_ratio", "eps_mf", "eps_semi", "eps_crude"):
            num(f"epsw_{k}_win_mean", ew[k].mean(), "{:.2f}")
            num(f"epsw_{k}_win_min", ew[k].min(), "{:.2f}")
            num(f"epsw_{k}_win_max", ew[k].max(), "{:.2f}")
        for k in ("eps_mf", "eps_semi", "eps_crude"):
            dev = (ew[k] - ew.eps).abs().max()
            num(f"epsw_{k}_win_maxdev", dev, "{:.2f}")
        rule = (ew.eps_mf - ew.eps).abs().max() <= 0.05
        NUMBERS["E2_firstprinciples"] = "yes" if rule else "no"
        note(f"  DECISION E2: closed mean field within ±0.05 of measured ε everywhere in the window? "
             f"{'YES' if rule else 'NO'} (max |Δ| = {(ew.eps_mf - ew.eps).abs().max():.3f}); semi-empirical "
             f"max |Δ| = {(ew.eps_semi - ew.eps).abs().max():.3f}")
    # flat window: the same definition as analysis_density_window.py (widest run of consecutive
    # intervals whose local ε all lie within ±0.05 of their mean, at least 3 intervals)
    try:
        import analysis_density_window as adw
        _, e_adw = adw.elasticity(W)
        fw = adw.widest_window(e_adw, 0.05, 3)
        T["eps_err"] = e_adw.epsilon_err.values
        if fw:
            num("epsw_flat_rho_lo", fw["rho_lo"], "{:.3f}")
            num("epsw_flat_rho_hi", fw["rho_hi"], "{:.3f}")
            num("epsw_flat_eps", fw["epsilon"], "{:.2f}")
            num("epsw_flat_spread", fw["epsilon_spread"], "{:.2f}")
            note(f"  flat window (analysis_density_window): ρ = {fw['rho_lo']:.3f}–{fw['rho_hi']:.3f}, "
                 f"ε = {fw['epsilon']:.3f}, spread {fw['epsilon_spread']:.3f}")
    except Exception as ex:  # noqa: BLE001
        note(f"[note] analysis_density_window unavailable ({ex!r})")
    hi_rows = T[T.rho > rho_hi]
    if len(hi_rows):
        num("epsw_eps_highest_rho", hi_rows.eps.iloc[-1], "{:.2f}")
        num("epsw_highest_rho", hi_rows.rho.iloc[-1], "{:.2f}")
    # level: mean field y against measured y in the window (no fit)
    lev = g.y.values / mf.y.values
    num("epsw_mf_level_ratio_min", lev[(g.rho >= rho_lo) & (g.rho <= rho_hi)].min() if ((g.rho >= rho_lo) & (g.rho <= rho_hi)).any() else math.nan, "{:.2f}")
    num("epsw_mf_level_ratio_max", lev[(g.rho >= rho_lo) & (g.rho <= rho_hi)].max() if ((g.rho >= rho_lo) & (g.rho <= rho_hi)).any() else math.nan, "{:.2f}")
    num("epsw_w_meas_over_mf_win", np.mean((g.w.values / mf.w.values)[(g.rho >= rho_lo) & (g.rho <= rho_hi)])
        if ((g.rho >= rho_lo) & (g.rho <= rho_hi)).any() else math.nan, "{:.2f}")
    num("epsw_n_meas_over_mf_win", np.mean((g.n.values / mf.n.values)[(g.rho >= rho_lo) & (g.rho <= rho_hi)])
        if ((g.rho >= rho_lo) & (g.rho <= rho_hi)).any() else math.nan, "{:.2f}")
    write_table(out, "tab_epsilon_window",
                [r"$\rho$", r"$\varepsilon$", r"$\varepsilon_w$", r"$\varepsilon_{\rm work}$",
                 r"$\varepsilon_{1-B}$", r"$\varepsilon_{\rm MF}$", r"$\varepsilon_{\rm semi}$",
                 r"$\kappa n/(g+\kappa n)$"],
                [[_f(r.rho, "{:.3f}"), _f(r.eps, "{:.2f}"), _f(r.eps_w, "{:.2f}"), _f(r.eps_work, "{:.2f}"),
                  _f(r.eps_unblocked, "{:.2f}"), _f(r.eps_mf, "{:.2f}"), _f(r.eps_semi, "{:.2f}"),
                  _f(r.eps_crude, "{:.2f}")] for r in T.itertuples()])
    # SI figure: ε(ρ) measured vs mean field, and the decomposition
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.6), constrained_layout=True)
    a1.axvspan(rho_lo, rho_hi, color="0.92", lw=0, zorder=0)
    a1.plot(T.rho, T.eps, "ko-", ms=3, label="measured (constant density)")
    a1.plot(T.rho, T.eps_mf, "--", color=OKABE[0], label="mean field, closed")
    a1.plot(T.rho, T.eps_semi, ":", color=OKABE[2], label=r"mean field, measured $n,q,B$")
    a1.plot(T.rho, T.eps_crude, "-.", color="0.5", lw=0.8, label=r"$b\kappa n/(g+\kappa n)$")
    a1.set(xscale="log", xlabel=r"density $\rho$", ylabel=r"elasticity $\varepsilon=d\ln y/d\ln\rho$")
    a1.legend(fontsize=6, loc="lower left")
    for k, cl, lab in (("eps_w", OKABE[0], r"efficiency $\bar w$"), ("eps_work", OKABE[1], "working share"),
                       ("eps_unblocked", OKABE[3], "unblocked moves"), ("eps_ratio", "0.5", "remainder")):
        a2.plot(T.rho, T[k], "o-", ms=2.5, color=cl, label=lab)
    a2.plot(T.rho, T.eps, "k-", lw=1.2, label=r"total $\varepsilon$")
    a2.axhline(0, color="0.6", lw=0.5, ls=":")
    a2.set(xscale="log", xlabel=r"density $\rho$", ylabel="contribution to $\\varepsilon$")
    a2.legend(fontsize=6, loc="lower left")
    panel(a1, "a")
    panel(a2, "b")
    save(fig, out, "figS_epsilon_window")
    manifest("figS_epsilon_window", "(a) ε(ρ) measured vs mean field; (b) exact decomposition of ε",
             ["density_window_scan"])
    mfline = pd.DataFrame(mfm.curve(np.geomspace(max(g.rho.min(), 1e-3), g.rho.max(), 40), MFP))
    mfline.attrs["eps_window"] = float(ew.eps_mf.mean()) if len(ew) else math.nan
    return T, mfline


# ---------------------------------------------------------------------- E3: ablations
ABL_NICE = {"base": "full model", "kappa0": r"no spillovers ($\kappa=0$)",
            "nodol": r"no division of labour ($\xi=\varphi=0$)", "lambda0": r"crossings never block ($\lambda=0$)",
            "b0": r"no learning ($b=0$)", "lambda1": r"hard crossings ($\lambda=1$)"}


def _short_pl(df, n_boot=300):
    rows = []
    for a, g in df.groupby("area_exponent"):
        est, lo, hi, r2, _ = bootstrap_exponent(g, "N", "Y", n_boot=n_boot)
        rows.append(dict(a=a, beta=est, lo=lo, hi=hi))
    t = pd.DataFrame(rows).sort_values("a")
    eps, b1 = (np.polyfit(1 - t.a, t.beta, 1) if len(t) >= 2 else (math.nan, math.nan))
    return t, float(eps), float(b1)


def _short_c(df, n_boot=300):
    rows = []
    loc = local_exponents(df, ["centralization"], "N", "Y", stride=2)
    cr = crossover_sizes(loc, ["centralization"]) if not loc.empty else pd.DataFrame()
    for c, g in df.groupby("centralization"):
        est, lo, hi, r2, _ = bootstrap_exponent(g, "N", "Y", n_boot=n_boot)
        r = dict(c=c, beta=est, lo=lo, hi=hi)
        if not cr.empty and (cr.centralization == c).any():
            rr = cr[cr.centralization == c].iloc[0]
            r.update(bloc_last=rr.beta_local_at_largest_N, Ncross=rr.N_cross)
        big = g[g.N == g.N.max()]
        r["Gamma_Nmax"] = big.Gamma_meas.mean() if "Gamma_meas" in big else math.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values("c"), loc


def ablations(root, P, S, out, plt):
    """E3. Each ablation is compared with the full experiment over the seeds they SHARE (common
    random numbers: same seed, same initial state), so differences are paired."""
    import config
    names = list(config.ABLATIONS) + ["lambda1"]
    res = {}
    avals = config.PL_AVALS_SHORT
    # load every ablation first: the shared seeds define the baseline subsets
    APL, AC = {}, {}
    for nm in names:
        A = get(f"abl_pl_{nm}", root) if nm != "lambda1" else None
        if A is not None:
            check_pinned(A, f"abl_pl_{nm}")
            APL[nm] = A
        C = get(f"abl_c_{nm}", root)
        if C is not None:
            check_pinned(C, f"abl_c_{nm}")
            AC[nm] = C
    def shared(base, abls):
        if base is None:
            return []
        s0 = set(base.seed.unique())
        for A in abls:
            s0 &= set(A.seed.unique())
        return sorted(int(x) for x in s0)
    seeds_pl = shared(P, APL.values()) if APL else (sorted(int(x) for x in P.seed.unique()) if P is not None else [])
    seeds_c = shared(S, AC.values()) if AC else (sorted(int(x) for x in S.seed.unique()) if S is not None else [])
    num("abl_seeds_pl", len(seeds_pl), "{:.0f}")
    num("abl_seeds_c", len(seeds_c), "{:.0f}")
    note(f"\n== E3 seeds: power-law comparisons over seeds {seeds_pl}, centralization over {seeds_c}")
    if P is not None:
        b = P[P.area_exponent.round(3).isin([round(x, 3) for x in avals]) & P.seed.isin(seeds_pl)]
        res["base"] = dict(pl=_short_pl(b), dol=b.DOL_tasks.mean() if "DOL_tasks" in b else math.nan,
                           y1024=b[(np.isclose(b.area_exponent, 1.0)) & (b.N == 1024)].y_per_capita.mean())
    for nm in names:
        r = {}
        if nm in APL:
            A = APL[nm][APL[nm].seed.isin(seeds_pl)]
            r["pl"] = _short_pl(A)
            r["dol"] = A.DOL_tasks.mean() if "DOL_tasks" in A else math.nan
            r["y1024"] = A[(np.isclose(A.area_exponent, 1.0)) & (A.N == 1024)].y_per_capita.mean()
        if nm in AC:
            C = AC[nm][AC[nm].seed.isin(seeds_c)]
            r["c"] = _short_c(C)
            r["cNmax"] = C.N.max()
        if r:
            res[nm] = r
    if S is not None:
        for nm, r in list(res.items()):
            if "c" in r or nm == "base":
                nmax = r.get("cNmax", 32768)
                bS = S[S.centralization.isin(config.C_SHORT) & S.seed.isin(seeds_c) & (S.N <= nmax)]
                res.setdefault(f"base_N{int(nmax)}", {})["c"] = _short_c(bS)
    if len(res) <= 1:
        return
    note("\n== E3: ablations (2 seeds each; baseline = seeds 0–1 of the full experiments)")
    rows = []
    for nm, r in res.items():           # baselines over the N range of each ablation
        if nm.startswith("base_N"):
            for _, x in r["c"][0].iterrows():
                kb = f"abl_base{nm[6:]}_c{ckey(x.c)}"
                num(f"{kb}_beta", x.beta)
                num(f"{kb}_bloclast", x.get("bloc_last", math.nan))
                num(f"{kb}_Ncross", x.get("Ncross", math.nan), "{:,.0f}")
    for nm, r in res.items():
        if nm.startswith("base_N"):
            continue
        line = [ABL_NICE.get(nm, nm)]
        if "pl" in r:
            t, eps, b1 = r["pl"]
            num(f"abl_{nm}_eps", eps, "{:.2f}")
            num(f"abl_{nm}_beta1", b1)
            for _, x in t.iterrows():
                num(f"abl_{nm}_beta_a{x.a:.3f}", x.beta)
            num(f"abl_{nm}_dol", r.get("dol"), "{:.2f}")
            if "base" in res and np.isfinite(res["base"].get("y1024", math.nan)):
                num(f"abl_{nm}_ylevel", r["y1024"] / res["base"]["y1024"], "{:.2f}")
                if nm != "base":
                    num(f"abl_{nm}_eps_ratio", eps / res["base"]["pl"][1], "{:.2f}")
            line += [_f(eps, "{:.2f}"), _f(b1), _f(r.get("dol"), "{:.2f}"),
                     _f(r["y1024"] / res["base"]["y1024"] if "base" in res else math.nan, "{:.2f}")]
            note(f"  {nm:8s} power law: ε = {eps:.3f}, β(a=1) = {b1:.3f}, DOL = {r.get('dol', math.nan):.2f}, "
                 + ", ".join(f"β(a={x.a:.3f}) = {x.beta:.3f}" for _, x in t.iterrows()))
        else:
            line += ["--"] * 4
        cres = r.get("c") if nm != "base" else res.get("base_N32768", {}).get("c")
        if cres is not None:
            ct, _ = cres
            for _, x in ct.iterrows():
                k = ckey(x.c)
                num(f"abl_{nm}_c{k}_beta", x.beta)
                num(f"abl_{nm}_c{k}_bloclast", x.get("bloc_last", math.nan))
                num(f"abl_{nm}_c{k}_Ncross", x.get("Ncross", math.nan), "{:,.0f}")
                num(f"abl_{nm}_c{k}_GammaNmax", x.get("Gamma_Nmax", math.nan), "{:.2f}")
            g1 = ct[ct.c == 1.0]
            line += [_f(ct[ct.c == 0.0].beta.iloc[0] if (ct.c == 0).any() else math.nan),
                     _f(g1.beta.iloc[0] if len(g1) else math.nan),
                     _f(g1.bloc_last.iloc[0] if len(g1) and "bloc_last" in g1 else math.nan)]
            note(f"  {nm:8s} centralization: " + ", ".join(
                f"c={x.c:g}: β={x.beta:.3f}, local β(Nmax)={x.get('bloc_last', math.nan):.3f}, "
                f"N_cross={x.get('Ncross', math.nan):g}" for _, x in ct.iterrows()))
        else:
            line += ["--"] * 3
        rows.append(line)
    write_table(out, "tab_ablations", ["variant", r"$\varepsilon$", r"$\beta(a{=}1)$", "DOL",
                                       r"$y/y_{\rm full}$", r"$\beta$, $c{=}0$", r"$\beta$, $c{=}1$",
                                       r"$\beta_{\rm loc}(N_{\max})$, $c{=}1$"], rows)
    # figure: (a) β vs a per variant; (b) local β vs N at c = 1 per variant
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.6), constrained_layout=True)
    for (nm, r), cl, mk in zip([(k, v) for k, v in res.items() if not k.startswith("base_N")], OKABE, "osD^vP"):
        if "pl" in r:
            t = r["pl"][0]
            a1.plot(t.a, t.beta, marker=mk, color=cl, label=ABL_NICE.get(nm, nm), ms=3.5)
        cres = r.get("c") if nm != "base" else res.get("base_N32768", {}).get("c")
        if cres is not None:
            loc = cres[1]
            h = loc[loc.centralization == 1.0].sort_values("N_mid")
            a2.plot(h.N_mid, h.local_exponent, marker=mk, color=cl, ms=3, label=ABL_NICE.get(nm, nm))
    unity(a1)
    unity(a1, horizontal=False)
    a1.set(xlabel="area exponent $a$", ylabel=r"scaling exponent $\beta$")
    a1.legend(fontsize=5.8, loc="upper right")
    unity(a2)
    logx(a2)
    a2.set_ylabel(r"local exponent, $c=1$")
    a2.legend(fontsize=5.8, loc="lower left")
    panel(a1, "a")
    panel(a2, "b")
    save(fig, out, "fig_ablations")
    manifest("fig_ablations", "(a) β vs a (short Fig. 2) per ablation; (b) local β vs N at c = 1 "
             "(short Fig. 3) per ablation", [f"abl_pl_{n}" for n in names] + [f"abl_c_{n}" for n in names])


# ---------------------------------------------------------------------- E4: generality, robustness
def generality(PB, WB, out, plt):
    """Fig. 2 with the second learning setting (b = 0.75, κ = 4, R = 2) inside its own window."""
    rho_lo, rho_hi = PB.density.min(), PB.density.max()
    eps_w, _ = epsilon_constant_density(WB, rho_lo, rho_hi)
    _, bt, loc, eps_fit, eps_off, b1 = power_law_stats(PB)
    note(f"\n== E4: generality (paper_density_exponent_b075), densities {rho_lo:.3f}–{rho_hi:.3f}")
    note(f"  ε window (density_window_scan_b075) = {eps_w:.3f};  fit through origin {eps_fit:.3f}; "
         f"free intercept ε = {eps_off:.3f}, β(a=1) = {b1:.3f}")
    record_power_law(bt, eps_w, prefix="g")
    num("g_eps_window", eps_w, "{:.2f}")
    num("g_eps_fit", eps_fit, "{:.2f}")
    num("g_eps_offset", eps_off, "{:.2f}")
    num("g_beta1_offset", b1, "{:.3f}")
    num("g_rho_lo", rho_lo, "{:.3f}")
    num("g_rho_hi", rho_hi, "{:.3f}")
    jf = joint_density_size_fit(PB)
    num("g_joint_eps", jf["eps"])
    num("g_joint_gamma", jf["gamma"], "{:+.3f}")
    se = epsilon_se(WB, rho_lo, rho_hi)
    num("g_eps_window_se", se, "{:.2f}")
    # prediction error bar from ε (1.96 s.e.) combined with the CI half-width of β
    tol = [math.hypot((r.hi - r.lo) / 2, 1.96 * se * abs(1 - r.a)) if np.isfinite(se) else (r.hi - r.lo) / 2
           for r in bt.itertuples()]
    comb = all(abs(r.beta - (b1 + eps_w * (1 - r.a))) <= t_ for r, t_ in zip(bt.itertuples(), tol))
    NUMBERS["E4_pass_combined"] = "yes" if comb else "no"
    strict = all(r.lo <= 1 + eps_w * (1 - r.a) <= r.hi for r in bt.itertuples())
    offs = all(r.lo <= b1 + eps_w * (1 - r.a) <= r.hi for r in bt.itertuples())
    halfwidth = (bt.hi - bt.lo).mean() / 2
    near = (bt.beta - (b1 + eps_w * (1 - bt.a))).abs().max()
    NUMBERS["E4_pass_strict"] = "yes" if strict else "no"
    NUMBERS["E4_pass_offset"] = "yes" if offs else "no"
    num("g_maxdev_offset", near, "{:.3f}")
    num("g_ci_halfwidth", halfwidth, "{:.3f}")
    note(f"  DECISION E4 (use 'combined'): β(a=1) + ε(1−a) within the combined uncertainty of β and of "
         f"ε (s.e. {se:.3f})? {'YES' if comb else 'NO'}")
    note(f"  (strict) prediction 1 + ε(1−a) inside every CI? {'YES' if strict else 'NO'};  "
         f"with the a = 1 offset, β(a=1) + ε(1−a)? {'YES' if offs else 'NO'} (max |Δ| {near:.3f}, "
         f"mean CI half-width {halfwidth:.3f})")


def robustness(T, C, out):
    """Robustness table from superlinear_tuning (ε at constant density, N = 4096) and the measured
    β on the city path from superlinear_city. β on the city path from ε is a PREDICTION."""
    rows, out_rows = [], []
    if T is not None:
        g0 = T.copy()
        w_ref = ((g0.practice_gain + g0.spillover) / g0.exp_decay) ** g0.learning_exponent
        g0["moves"] = (g0.route_length / w_ref).round().astype(int)
        keys = ["learning_exponent", "spillover", "spillover_radius", "moves"]
        for k, g in g0.groupby(keys):
            m = g.groupby("rho0").agg(rho=("density", "mean"), y=("y_per_capita", "mean")).reset_index()
            def sl(lo, hi):
                mm = m[(m.rho >= lo * 0.999) & (m.rho <= hi * 1.001)]
                return np.polyfit(np.log(mm.rho), np.log(mm.y), 1)[0] if len(mm) >= 2 else math.nan
            e_low, e_path = sl(0.02, 0.11), sl(0.04, 0.45)
            rows.append(dict(zip(keys, k), eps_low=e_low, eps_path=e_path, beta_city_pred=1 + e_path / 3))
    R = pd.DataFrame(rows)
    if len(R):
        R = R.sort_values("beta_city_pred", ascending=False)
        note("\n== E4b: robustness (superlinear_tuning, N = 4096, constant density)")
        for line in R.round(3).to_string(index=False).splitlines():
            note("  " + line)
        num("rob_n_settings", len(R), "{:.0f}")
        num("rob_eps_low_min", R.eps_low.min(), "{:.2f}")
        num("rob_eps_low_max", R.eps_low.max(), "{:.2f}")
        num("rob_betapred_min", R.beta_city_pred.min())
        num("rob_betapred_max", R.beta_city_pred.max())
        band = R[(R.beta_city_pred >= 1.12) & (R.beta_city_pred <= 1.18)]
        num("rob_n_cityband", len(band), "{:.0f}")
        out_rows = [[_f(r.learning_exponent, "{:g}"), _f(r.spillover, "{:g}"), _f(r.spillover_radius, "{:g}"),
                     _f(r.moves, "{:g}"), _f(r.eps_low, "{:.2f}"), _f(r.eps_path, "{:.2f}"),
                     _f(r.beta_city_pred)] for r in R.itertuples()]
        write_table(out, "tab_robustness", ["$b$", r"$\kappa$", "$R$", "moves/job",
                                            r"$\varepsilon$, $\rho\le0.1$", r"$\varepsilon$, $\rho=0.05$--$0.4$",
                                            r"$1+\varepsilon/3$ (pred.)"], out_rows)
    if C is not None:
        w_ref = ((C.practice_gain + C.spillover) / C.exp_decay) ** C.learning_exponent
        C = C.assign(moves=(C.route_length / w_ref).round().astype(int))
        for mv, g in C.groupby("moves"):
            est, lo, hi, r2, _ = bootstrap_exponent(g, "N", "Y", n_boot=300)
            note(f"  superlinear_city: {mv} moves/job: β = {est:.3f} [{lo:.3f}, {hi:.3f}] (measured, city path, N ≤ {g.N.max()})")
            num(f"rob_city_beta_moves{mv}", est)
            num(f"rob_city_beta_moves{mv}_lo", lo)
            num(f"rob_city_beta_moves{mv}_hi", hi)


# ---------------------------------------------------------------------- E5: Γ_ff, slowdown, asymptote
def gamma_extras(frames, fit, S, out, plt, NL=None):
    """Free-flow Γ for every run, delivery slowdown S = Γ_meas / Γ_ff(measured rest), idle time,
    efficiency and neighbour ratios to c = 0, and the asymptote β0 − p(1 − a/2)."""
    import meanfield as mfm
    p = fit["p_fit"]
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = little(df)
    if "hub_mean_dist" not in df:
        df["hub_mean_dist"] = (df["L"] - 1) / 2.0
    df["hub_mean_dist"] = df["hub_mean_dist"].fillna((df["L"] - 1) / 2.0)
    key = ["source", "N", "area_exponent", "rho0"]
    base = df[df.centralization == 0].groupby(key).agg(
        y0=("y_per_capita", "mean"), w0=("mean_w", "mean"), n0=("mean_nsame", "mean"),
        B0=("blocked_frac", "mean"), idle0=("idle_frac", "mean")).reset_index()
    d = df[df.centralization > 0].merge(base, on=key, how="inner")
    hs = d["hub_share_completed"].clip(lower=1e-9)
    comp = d["completions"].clip(lower=1e-12)
    d["T_deliv_hub"] = d["delivering_frac"] * d["N"] / (hs * comp)       # steps per hub job
    d["slowdown"] = d["T_deliv_hub"] / (2.0 * d["hub_mean_dist"])        # 1 = free flow
    MFP = mfm.MFParams.from_params(df.iloc[0].to_dict())
    tau = {}
    for r in d[["density"]].drop_duplicates().itertuples():
        tau[r.density] = MFP.D / mfm.solve(float(r.density), MFP)["w"]     # moves per job, mean field
    d["tau_mf"] = d["density"].map(tau)
    d["Gamma_ff"] = d["centralization"] * d["hub_mean_dist"] / d["tau_mf"]
    d["w_ratio"] = d["mean_w"] / d["w0"]
    d["n_ratio"] = d["mean_nsame"] / d["n0"]
    d["B_ratio"] = d["blocked_frac"] / d["B0"].clip(lower=1e-9)
    note("\n== E5: Γ_ff, delivery slowdown, idle time (runs with c > 0)")
    ok = d[(d.Gamma_meas > 0) & (d.Gamma_ff > 0)]
    if len(ok):
        sl = np.polyfit(np.log(ok.Gamma_ff), np.log(ok.Gamma_meas), 1)
        r = np.corrcoef(np.log(ok.Gamma_ff), np.log(ok.Gamma_meas))[0, 1]
        note(f"  ln Γ_meas vs ln Γ_ff: slope {sl[0]:.3f}, r = {r:.3f}, median Γ_meas/Γ_ff = "
             f"{np.median(ok.Gamma_meas / ok.Gamma_ff):.2f}")
        num("gff_slope", sl[0])
        num("gff_corr", r)
        num("gff_ratio_median", np.median(ok.Gamma_meas / ok.Gamma_ff), "{:.2f}")
    src = "paper_centralization_sweep"
    cs = d[d.source == src]
    if len(cs):
        t = cs.groupby(["centralization", "N"]).agg(G=("Gamma_meas", "mean"), Gff=("Gamma_ff", "mean"),
                                                    Sd=("slowdown", "mean"), idle=("idle_frac", "mean"),
                                                    wr=("w_ratio", "mean"), nr=("n_ratio", "mean"),
                                                    Br=("B_ratio", "mean")).reset_index()
        for line in t.round(3).to_string(index=False).splitlines():
            note("  " + line)
        Nmax, Nmin = t.N.max(), t.N.min()
        for c in sorted(t.centralization.unique()):
            k = ckey(c)
            hi_, lo_ = t[(t.centralization == c) & (t.N == Nmax)], t[(t.centralization == c) & (t.N == Nmin)]
            for nm, col_ in (("G", "G"), ("Gff", "Gff"), ("S", "Sd"), ("idle", "idle"), ("wratio", "wr"),
                             ("nratio", "nr"), ("Bratio", "Br")):
                num(f"gx_{nm}_c{k}_Nmax", hi_[col_].iloc[0] if len(hi_) else math.nan, "{:.2f}")
                num(f"gx_{nm}_c{k}_Nmin", lo_[col_].iloc[0] if len(lo_) else math.nan, "{:.2f}")
        if S is not None:
            c0 = S[S.centralization == 0].groupby("N")
            for nm_, col_ in (("idle", "idle_frac"),):
                num(f"gx_{nm_}_c0_Nmax", c0[col_].mean().iloc[-1], "{:.2f}")
                num(f"gx_{nm_}_c0_Nmin", c0[col_].mean().iloc[0], "{:.2f}")
        # asymptote: β0 at the largest pair, minus p times the free-flow growth of Γ
        loc0 = local_exponents(S[S.centralization == 0], ["centralization"], "N", "Y", stride=2) if S is not None else None
        a_ = float(cs.area_exponent.iloc[0])
        if loc0 is not None and len(loc0):
            b0 = loc0.sort_values("N_mid").local_exponent.iloc[-1]
            asym_ff = b0 - p * (1 - a_ / 2)
            g1 = t[t.centralization == t.centralization.max()].sort_values("N")
            Ns = g1.N.values
            if len(Ns) >= 3:
                slope_mf = math.log(g1.Gff.values[-1] / g1.Gff.values[-3]) / math.log(Ns[-1] / Ns[-3])
            else:
                slope_mf = math.nan
            asym_mf = b0 - p * slope_mf
            note(f"  asymptote (Γ ≫ 1): β0 − p(1 − a/2) = {b0:.3f} − {p:.3f}·{1 - a_ / 2:.3f} = {asym_ff:.3f};  "
                 f"with the mean-field growth of Γ_ff at the largest sizes ({slope_mf:.3f}): {asym_mf:.3f}")
            num("gx_beta0_last", b0)
            num("gx_asym_ff", asym_ff)
            num("gx_gff_slope_last", slope_mf)
            num("gx_asym_mf", asym_mf)
        # SI figure
        fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.4), constrained_layout=True)
        cvals = sorted(t.centralization.unique())
        col = colors_for(plt, cvals, "plasma")
        ax[0].scatter(ok.Gamma_ff, ok.Gamma_meas, s=5, c=np.log10(ok.N), cmap="viridis", lw=0)
        lim = [min(ok.Gamma_ff.min(), ok.Gamma_meas.min()), max(ok.Gamma_ff.max(), ok.Gamma_meas.max())]
        ax[0].plot(lim, lim, "k--", lw=0.7)
        ax[0].set(xscale="log", yscale="log", xlabel=r"free-flow $\Gamma_{\rm ff}$ (Eq. 5)", ylabel=r"measured $\Gamma$")
        for c in cvals:
            g = t[t.centralization == c]
            ax[1].plot(g.N, g.Sd, "o-", ms=2.5, color=col[c])
            ax[2].plot(g.N, g.wr, "o-", ms=2.5, color=col[c])
        ax[1].axhline(1, color="0.5", lw=0.6, ls=":")
        ax[1].set(xscale="log", yscale="log", xlabel=r"$N$", ylabel=r"delivery slowdown $S$")
        ax[2].axhline(1, color="0.5", lw=0.6, ls=":")
        ax[2].set(xscale="log", xlabel=r"$N$", ylabel=r"$\bar w(c)/\bar w(0)$")
        discrete_cbar(plt, fig, list(ax[1:]), cvals, col, "centralization $c$")
        for axx, l in zip(ax, "abc"):
            panel(axx, l)
        save(fig, out, "figS_gamma_ff")
        manifest("figS_gamma_ff", "(a) Γ measured vs free-flow; (b) delivery slowdown vs N per c; (c) mean "
                 "efficiency relative to c = 0", sorted(set(d.source)))
    # E5b: no learning while delivering
    if NL is not None and S is not None:
        import analysis_gamma as ag
        base_S = S[(S.centralization == 0)].assign(source="nl")
        both = pd.concat([NL.assign(source="nl"), base_S], ignore_index=True, sort=False)
        if "delivery_choice" not in both:
            both["delivery_choice"] = "fixed"
        both["delivery_choice"] = both["delivery_choice"].fillna("fixed")
        pts, _ = ag.build_points(both, ag.MATCH)
        f2 = ag.fit_collapse(pts)
        note(f"  E5b: without learning while delivering: p = {f2['p_fit']:.3f} (R² {f2['R2_free_fit']:.3f}, "
             f"R² with p = 1: {f2['R2_with_p_equal_1']:.3f}) over {f2['n_points']} runs; with learning p = {p:.3f}")
        num("nl_p", f2["p_fit"])
        num("nl_R2", f2["R2_free_fit"], "{:.3f}")
        num("nl_R2_p1", f2["R2_with_p_equal_1"], "{:.3f}")
        num("nl_npoints", f2["n_points"], "{:,.0f}")
        num("nl_seeds", NL.seed.nunique(), "{:.0f}")
        ref = S[S.centralization.isin(NL.centralization.unique()) & S.N.isin(NL.N.unique()) & S.seed.isin(NL.seed.unique())]
        for (c, N), g in NL.groupby(["centralization", "N"]):
            r0 = ref[(ref.centralization == c) & (ref.N == N)]
            if len(r0):
                num(f"nl_yratio_c{ckey(c)}_N{int(N)}", g.y_per_capita.mean() / r0.y_per_capita.mean(), "{:.2f}")
                num(f"nl_wratio_c{ckey(c)}_N{int(N)}", g.mean_w.mean() / r0.mean_w.mean(), "{:.2f}")
                note(f"   c = {c:g} N = {int(N)}: y(no learning)/y(learning) = {g.y_per_capita.mean() / r0.y_per_capita.mean():.3f}, "
                     f"w̄ ratio {g.mean_w.mean() / r0.mean_w.mean():.3f}")
        NUMBERS["E5b_p_close_to_1"] = "yes" if abs(f2["p_fit"] - 1) < abs(p - 1) / 2 else "no"
    return d


# ---------------------------------------------------------------------- E6, E7: convergence and seeds
def _paired_rel(A, B, keys):
    """Relative difference of y (B/A − 1) per cell, seeds matched; also the seed CV of A."""
    m = A.merge(B, on=keys + ["seed"], suffixes=("_a", "_b"))
    if not len(m):
        return pd.DataFrame()
    m["rel"] = m.y_per_capita_b / m.y_per_capita_a - 1
    cv = A.groupby(keys).y_per_capita.agg(lambda v: v.std() / v.mean()).rename("cv").reset_index()
    t = m.groupby(keys).rel.agg(["mean", "std", "count"]).reset_index().merge(cv, on=keys)
    t["se"] = t["std"] / np.sqrt(t["count"])
    return t


def transient_checks(S, FX, P, TP):
    note("\n== E6: transient convergence")
    if S is not None and FX is not None:
        t = _paired_rel(S, FX, ["N", "centralization"])
        if len(t):
            num("tr_c_meanabs_pct", 100 * t["mean"].abs().mean(), "{:.1f}")
            num("tr_c_maxabs_pct", 100 * t["mean"].abs().max(), "{:.1f}")
            num("tr_c_cv_median_pct", 100 * t.cv.median(), "{:.1f}")
            num("tr_c_ncells", len(t), "{:.0f}")
            big = t[t.N == t.N.max()]
            num("tr_c_maxabs_Nmax_pct", 100 * big["mean"].abs().max(), "{:.1f}")
            note(f"  Fig. 3 cells, T_burn 10000 (paper_choice_fixed) vs 6000 (paper_centralization_sweep), "
                 f"{len(t)} cells: mean |Δy/y| = {100 * t['mean'].abs().mean():.2f}%, max {100 * t['mean'].abs().max():.2f}%, "
                 f"median seed CV {100 * t.cv.median():.2f}%")
            # do the exponents change?
            for c in sorted(set(S.centralization) & set(FX.centralization)):
                e1 = bootstrap_exponent(S[S.centralization == c], "N", "Y", n_boot=1)[0]
                e2 = bootstrap_exponent(FX[FX.centralization == c], "N", "Y", n_boot=1)[0]
                num(f"tr_c_dbeta_c{ckey(c)}", e2 - e1, "{:+.3f}")
    if P is not None and TP is not None:
        t = _paired_rel(P, TP, ["N", "area_exponent"])
        num("tr_pl_seeds", TP.seed.nunique(), "{:.0f}")
        if len(t):
            num("tr_pl_maxabs_pct", 100 * t["mean"].abs().max(), "{:.1f}")
            num("tr_pl_meanabs_pct", 100 * t["mean"].abs().mean(), "{:.1f}")
            num("tr_pl_cv_median_pct", 100 * t.cv.median(), "{:.1f}")
            note(f"  Fig. 2 cells, T_burn 12000 vs 6000: max |Δy/y| = {100 * t['mean'].abs().max():.2f}% "
                 f"(seed CV {100 * t.cv.median():.2f}%)")


def seed_blocks(P, PB):
    """E7: β per a from seeds 0–2 (block A) and 100–104 (block B)."""
    _, A, _, eA, _, _ = power_law_stats(P)
    _, B, _, eB, _, _ = power_law_stats(PB)
    m = A.merge(B, on="a", suffixes=("_A", "_B"))
    inside = ((m.beta_B >= m.lo_A) & (m.beta_B <= m.hi_A)).mean()
    _, AB, _, eAB, _, _ = power_law_stats(pd.concat([P, PB], ignore_index=True, sort=False))
    note("\n== E7: independent seed blocks")
    for r in m.itertuples():
        note(f"  a = {r.a:.3f}: A {r.beta_A:.3f} [{r.lo_A:.3f}, {r.hi_A:.3f}]   B {r.beta_B:.3f} [{r.lo_B:.3f}, {r.hi_B:.3f}]")
        num(f"sb_beta_a{r.a:.3f}_B", r.beta_B)
    num("sb_inside_frac", inside, "{:.2f}")
    num("sb_maxdiff", (m.beta_A - m.beta_B).abs().max(), "{:.3f}")
    num("sb_eps_fit_A", eA, "{:.2f}")
    num("sb_eps_fit_B", eB, "{:.2f}")
    num("sb_eps_fit_AB", eAB, "{:.2f}")
    for r in AB.itertuples():
        num(f"sb_beta_a{r.a:.3f}_AB", r.beta)
        num(f"sb_beta_a{r.a:.3f}_AB_lo", r.lo)
        num(f"sb_beta_a{r.a:.3f}_AB_hi", r.hi)


# ---------------------------------------------------------------------- E11: pace of life, DOL
def pace_of_life(P, W):
    """Job duration (steps on the lattice per job, Little's law) against N for each a; predicted
    exponent −ε_T(1 − a), ε_T = −d ln T_job / d ln ρ from the constant-density scan."""
    if P is None or W is None:
        return
    Pl, Wl = little(P), little(W)
    rho_lo, rho_hi = P.density.min(), P.density.max()
    gw = Wl.groupby("rho0").agg(rho=("density", "mean"), T=("T_lattice", "mean")).reset_index()
    gw = gw[(gw.rho >= rho_lo * 0.999) & (gw.rho <= rho_hi * 1.001)]
    epsT = -np.polyfit(np.log(gw.rho), np.log(gw["T"]), 1)[0] if len(gw) >= 2 else math.nan
    note(f"\n== E11: pace of life — job duration ∝ N^(−ε_T(1−a)), ε_T = {epsT:.3f}")
    num("pace_epsT", epsT, "{:.2f}")
    for a, g in Pl.groupby("area_exponent"):
        m = g.groupby("N").T_lattice.mean()
        sl = np.polyfit(np.log(m.index.values), np.log(m.values), 1)[0]
        num(f"pace_slope_a{a:.3f}", sl, "{:+.3f}")
        num(f"pace_pred_a{a:.3f}", -epsT * (1 - a), "{:+.3f}")
        num(f"pace_T_a{a:.3f}_Nmin", m.iloc[0], "{:.0f}")
        num(f"pace_T_a{a:.3f}_Nmax", m.iloc[-1], "{:.0f}")
        note(f"  a = {a:.3f}: d ln T_job/d ln N = {sl:+.3f}  (predicted {-epsT * (1 - a):+.3f}); "
             f"T_job {m.iloc[0]:.0f} -> {m.iloc[-1]:.0f} steps")


def dol_keys(P, S):
    for nm, df in (("pl", P), ("city", S)):
        if df is None or "DOL_tasks" not in df:
            continue
        sub = df if nm == "pl" else df[df.centralization == 0]
        num(f"dol_{nm}_min", sub.DOL_tasks.min(), "{:.2f}")
        num(f"dol_{nm}_mean", sub.DOL_tasks.mean(), "{:.2f}")
        if "frac_focused" in sub:
            num(f"dol_{nm}_focused_mean", sub.frac_focused.mean(), "{:.2f}")


# ---------------------------------------------------------------------- E9: polycentric delivery
def poly_pilot(PP, PPB):
    """Pilot table: Γ at the smallest N and slowdown at the largest N per (eta, moves)."""
    if PP is None:
        return
    d = little(PP)
    w_ref = ((d.practice_gain + d.spillover) / d.exp_decay) ** d.learning_exponent
    d["moves"] = (d.route_length / w_ref).round().astype(int)
    hs = d.hub_share_completed.clip(lower=1e-9)
    d["slowdown"] = d.delivering_frac * d.N / (hs * d.completions.clip(lower=1e-12)) / (2 * d.hub_mean_dist.clip(lower=0.5))
    t = d.groupby(["moves", "hub_count_exponent", "N"]).agg(G=("Gamma_meas", "mean"), S=("slowdown", "mean"),
                                                            K=("n_hubs", "mean"), y=("y_per_capita", "mean")).reset_index()
    note("\n== E9 pilot (poly_pilot): choose route_moves with Γ ≫ 1 at N = 64 and S not exploding")
    for line in t.round(3).to_string(index=False).splitlines():
        note("  " + line)


def polycentric(PM, PMB, p_fit, out, plt):
    import analysis_gamma as ag
    import hubs as hb
    both = pd.concat([PM.assign(source="poly"), PMB.assign(source="poly")], ignore_index=True, sort=False)
    both["delivery_choice"] = "fixed"
    etas = sorted(PM.hub_count_exponent.unique())
    note("\n== E9: polycentric delivery (poly_main)")
    loc = local_exponents(PM, ["hub_count_exponent"], "N", "Y", stride=2)
    b0loc = local_exponents(PMB.assign(hub_count_exponent=-1), ["hub_count_exponent"], "N", "Y", stride=2)
    beta0 = b0loc.local_exponent.mean() if len(b0loc) else 1.0
    n_const = 0
    rows = []
    for eta in etas:
        g = PM[PM.hub_count_exponent == eta]
        est, lo, hi, r2, _ = bootstrap_exponent(g, "N", "Y", n_boot=300)
        lg = loc[loc.hub_count_exponent == eta].sort_values("N_mid")
        # longest run of local exponents within ±0.025 of their median (2 decades = factor 100)
        spread = lg.local_exponent.max() - lg.local_exponent.min()
        dm = g.groupby("N").hub_mean_dist.mean()
        sl_d = np.polyfit(np.log(dm.index.values), np.log(dm.values), 1)[0]
        pred = beta0 - p_fit * sl_d                       # a = 1: τ constant, Γ_ff ∝ d̄
        pred_an = beta0 - p_fit * hb.gamma_ff_exponent(1.0, eta)
        ok = spread <= 0.05 and math.log10(lg.N_mid.max() / lg.N_mid.min()) + math.log10(4) >= 2
        n_const += int(ok)
        k = f"eta{eta:g}"
        num(f"poly_beta_{k}", est)
        num(f"poly_beta_{k}_lo", lo)
        num(f"poly_beta_{k}_hi", hi)
        num(f"poly_R2_{k}", r2, "{:.4f}")
        num(f"poly_spread_{k}", spread, "{:.3f}")
        num(f"poly_pred_{k}", pred)
        num(f"poly_predan_{k}", pred_an)
        num(f"poly_Gmin_{k}", g[g.N == g.N.min()].Gamma_meas.mean(), "{:.1f}")
        rows.append([f"{eta:g}", _f(est), f"[{_f(lo)}, {_f(hi)}]", _f(r2, "{:.4f}"), _f(spread), _f(pred), _f(pred_an)])
        note(f"  eta = {eta:g}: β = {est:.3f} [{lo:.3f}, {hi:.3f}], local spread {spread:.3f}, predicted "
             f"{pred:.3f} (trip-length slope {sl_d:.3f}; analytic {pred_an:.3f})")
    NUMBERS["E9_include"] = "yes" if n_const >= 3 else "no"
    num("poly_beta0", beta0)
    note(f"  DECISION E9: constant local exponent (spread ≤ 0.05 over ≥ 2 decades) for {n_const} values of eta "
         f"-> {'INCLUDE section' if n_const >= 3 else 'outlook paragraph only'}")
    write_table(out, "tab_polycentric", [r"$\eta$", r"$\beta$", "95\\% CI", "$R^2$", "local spread",
                                         "predicted", "predicted (ideal)"], rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.6), constrained_layout=True)
    col = colors_for(plt, etas, "viridis")
    for eta in etas:
        h = loc[loc.hub_count_exponent == eta].sort_values("N_mid")
        a1.plot(h.N_mid, h.local_exponent, "o-", ms=2.5, color=col[eta])
        a1.axhline(beta0 - p_fit * hb.gamma_ff_exponent(1.0, eta), color=col[eta], lw=0.6, ls="--")
    logx(a1)
    unity(a1)
    a1.set_ylabel(r"local exponent $\beta(N)$")
    for eta in etas:
        g = PM[PM.hub_count_exponent == eta]
        est = bootstrap_exponent(g, "N", "Y", n_boot=1)[0]
        a2.plot(eta, est, "o", color=col[eta])
    ee = np.linspace(0, 1, 20)
    a2.plot(ee, [beta0 - p_fit * hb.gamma_ff_exponent(1.0, e) for e in ee], "k--", lw=0.8,
            label=r"$\beta_0-p(1-\eta)/2$")
    a2.set(xlabel=r"hub-count exponent $\eta$ ($K\propto L^\eta$)", ylabel=r"$\beta$")
    a2.legend(fontsize=6.5)
    discrete_cbar(plt, fig, [a1, a2], etas, col, r"$\eta$")
    panel(a1, "a")
    panel(a2, "b")
    save(fig, out, "fig_polycentric")
    manifest("fig_polycentric", "(a) local β vs N per eta, dashed: asymptote; (b) β vs eta", ["poly_main", "poly_main_base"])


# ---------------------------------------------------------------------- λ dependence (SI)
def lambda_dependence(S, A0, LC, A1, W, W0, W1, out, plt):
    """β, local β and output per agent against the crossing-block probability λ on the city path
    (c ∈ {0, 0.2, 1}, N = 64–32768, seeds paired across λ), and ε(ρ) at constant density for
    λ = 0, 0.5, 1. λ = 0.5 is the main sweep; λ = 0 and 1 the ablations; 0.25, 0.75 lambda_city."""
    import config
    nmax = 32768
    parts = [df for df in (A0, LC, A1) if df is not None]
    if S is not None:
        parts.append(S[S.centralization.isin(config.C_SHORT)])
    if not parts:
        return
    D = pd.concat(parts, ignore_index=True, sort=False)
    D = D[(D.N <= nmax) & D.centralization.isin(config.C_SHORT)]
    seeds = None
    for lam, g in D.groupby("exclusion"):            # seeds shared by every λ (pairing)
        ss = set(g.seed.unique())
        seeds = ss if seeds is None else seeds & ss
    D = D[D.seed.isin(seeds)]
    lams = sorted(float(x) for x in D.exclusion.unique())
    cs = sorted(D.centralization.unique())
    note(f"\n== λ dependence: city path, N ≤ {nmax}, seeds {sorted(int(x) for x in seeds)}, λ = {lams}")
    num("lam_seeds", len(seeds), "{:.0f}")
    rows, res = [], {}
    for lam in lams:
        for c in cs:
            g = D[(D.exclusion == lam) & (D.centralization == c)]
            if g.N.nunique() < 3:
                continue
            est, lo, hi, r2, _ = bootstrap_exponent(g, "N", "Y", n_boot=300)
            loc = local_exponents(g.assign(k=0), ["k"], "N", "Y", stride=2).sort_values("N_mid")
            bl = loc.local_exponent.iloc[-1] if len(loc) else math.nan
            ybig = g[g.N == g.N.max()].y_per_capita.mean()
            y1k = g[g.N == 1024].y_per_capita.mean()
            res[(lam, c)] = dict(beta=est, lo=lo, hi=hi, bloc=bl, ybig=ybig, y1k=y1k)
            k = f"c{ckey(c)}_l{lam:g}"
            num(f"lam_beta_{k}", est)
            num(f"lam_beta_{k}_lo", lo)
            num(f"lam_beta_{k}_hi", hi)
            num(f"lam_bloc_{k}", bl)
            num(f"lam_y_{k}_Nmax", ybig, "{:.0f}")
            num(f"lam_y_{k}_N1024", y1k, "{:.0f}")
            note(f"  λ = {lam:g}  c = {c:g}:  β = {est:.3f} [{lo:.3f}, {hi:.3f}]  local β(N = {g.N.max()}) = {bl:.3f}"
                 f"  y(1024) = {y1k:.0f}  y({g.N.max()}) = {ybig:.0f}")
    for (lam, c), r in res.items():                  # output relative to the model (λ = 0.5)
        ref = res.get((0.5, c))
        if ref:
            num(f"lam_yrel_c{ckey(c)}_l{lam:g}_Nmax", r["ybig"] / ref["ybig"], "{:.2f}")
            num(f"lam_yrel_c{ckey(c)}_l{lam:g}_N1024", r["y1k"] / ref["y1k"], "{:.2f}")
    eps_curves = {}
    for lam, Wl in ((0.0, W0), (0.5, W), (1.0, W1)):
        if Wl is None:
            continue
        e_win, _ = epsilon_constant_density(Wl, 0.018, 0.082)
        dcol = "density" if "density" in Wl else "rho0"
        gw = Wl.groupby("rho0").agg(rho=(dcol, "mean"), y=("y_per_capita", "mean")).reset_index().sort_values("rho")
        lr, ly = np.log(gw.rho.values), np.log(gw.y.values)
        mid, loc = np.exp(0.5 * (lr[1:] + lr[:-1])), np.diff(ly) / np.diff(lr)
        eps_curves[lam] = (mid, loc)
        num(f"lam_epsw_l{lam:g}", e_win, "{:.2f}")
        num(f"lam_eps_densest_l{lam:g}", loc[-1], "{:.2f}")
        note(f"  constant density, λ = {lam:g}: ε (ρ = 0.018–0.082) = {e_win:.3f}; ε at the densest interval "
             f"(ρ ≈ {mid[-1]:.2f}) = {loc[-1]:.3f}")
    for lam in lams:
        r = [f"{lam:g}"]
        for c in cs:
            x = res.get((lam, c))
            r.append(_f(x["beta"]) if x else "--")
        x1, ref = res.get((lam, 1.0)), res.get((0.5, 1.0))
        r.append(_f(x1["bloc"]) if x1 else "--")
        r.append(_f(x1["ybig"] / ref["ybig"], "{:.2f}") if (x1 and ref) else "--")
        r.append(_f(float(np.interp(np.log(0.04), np.log(eps_curves[lam][0]), eps_curves[lam][1])), "{:.2f}")
                 if lam in eps_curves else "--")
        rows.append(r)
    write_table(out, "tab_lambda", [r"$\lambda$"] + [rf"$\beta$, $c={c:g}$" for c in cs] +
                [r"$\beta_{\rm loc}(N_{\max})$, $c=1$", r"$y/y_{\lambda=0.5}$, $c=1$, $N_{\max}$",
                 r"$\varepsilon$ at $\rho=0.04$"], rows)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(FULL, 2.5), constrained_layout=True)
    col = colors_for(plt, cs, "plasma")
    for c in cs:
        xs = [l for l in lams if (l, c) in res]
        a1.errorbar(xs, [res[(l, c)]["beta"] for l in xs],
                    yerr=[[res[(l, c)]["beta"] - res[(l, c)]["lo"] for l in xs],
                          [res[(l, c)]["hi"] - res[(l, c)]["beta"] for l in xs]],
                    fmt="o-", ms=3, color=col[c], capsize=2, label=f"$c={c:g}$")
        a2.plot(xs, [res[(l, c)]["ybig"] for l in xs], "o-", ms=3, color=col[c], label=f"$c={c:g}$")
    unity(a1)
    a1.set(xlabel=r"crossing-block probability $\lambda$", ylabel=r"global $\beta$ (N = 64–32768)")
    a1.legend(fontsize=6.5)
    a2.set(yscale="log", xlabel=r"crossing-block probability $\lambda$", ylabel=rf"output per agent, $N={nmax}$")
    lc = {0.0: OKABE[1], 0.5: "k", 1.0: OKABE[3]}
    for lam, (mid, loc) in sorted(eps_curves.items()):
        a3.plot(mid, loc, "o-", ms=3, color=lc.get(lam, "0.4"), label=rf"$\lambda={lam:g}$")
    a3.axvspan(0.018, 0.082, color="0.92", lw=0, zorder=0)
    a3.set(xscale="log", xlabel=r"density $\rho$ (constant)", ylabel=r"elasticity $\varepsilon$")
    a3.legend(fontsize=6.5)
    for axx, l in zip((a1, a2, a3), "abc"):
        panel(axx, l)
    save(fig, out, "figS_lambda")
    manifest("figS_lambda", "(a) global β vs λ per c, city path; (b) output per agent at the largest N vs λ; "
             "(c) ε(ρ) at constant density for λ = 0, 0.5, 1",
             ["lambda_city", "abl_c_lambda0", "abl_c_lambda1", "paper_centralization_sweep",
              "density_window_scan", "density_window_scan_lam0", "density_window_scan_lam1"])


# ---------------------------------------------------------------------- outputs
def _numf(key):
    """A number from NUMBERS as float, or None if missing / not a number."""
    v = NUMBERS.get(key)
    try:
        x = float(str(v).replace(",", "").replace("+", ""))
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def decision_switches():
    """The manuscript's \\if switches from the decision rules. Missing data -> False (weaker
    claim), with the reason. Returns [(switch, value, explanation)]."""
    out = []
    def yes(key):
        v = NUMBERS.get(key)
        return (v == "yes", f"{key} = {v}") if v in ("yes", "no") else (False, f"{key} missing: not decided")
    out.append(("Eoneadopt",) + yes("E1_adopt_N256"))
    out.append(("Etwofp",) + yes("E2_firstprinciples"))
    r = _numf("abl_nodol_eps_ratio")
    out.append(("Ethreedol", r is not None and r < 0.8,
                f"abl_nodol_eps_ratio = {r} (rule: < 0.8)" if r is not None else "abl_nodol_eps_ratio missing: not decided"))
    lam, base, asym = _numf("abl_lambda0_c1_bloclast"), _numf("abl_base131072_c1_bloclast"), _numf("gx_asym_ff")
    if None in (lam, base, asym):
        out.append(("Ethreesat", False, "abl_lambda0_c1_bloclast / abl_base131072_c1_bloclast / gx_asym_ff missing: not decided"))
    else:
        ok = abs(lam - asym) <= 0.05 and (asym - base) > 0.05
        out.append(("Ethreesat", ok, f"lambda=0: {lam:.3f}, full model: {base:.3f}, free-flow limit: {asym:.3f} "
                                     "(rule: |lambda0 - limit| <= 0.05 and limit - full > 0.05)"))
    out.append(("Efourgeneral",) + yes("E4_pass_combined"))
    out.append(("Efiveb",) + yes("E5b_p_close_to_1"))
    out.append(("Enine",) + yes("E9_include"))
    return out


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
    with open(os.path.join(out, "paper_switches.tex"), "w") as fh:
        fh.write("% Generated by paper_figures.py on " + _dt.datetime.now().isoformat(timespec="seconds") + "\n")
        fh.write("% Decision switches of main.tex / si.tex, set by the decision rules (README, section 4).\n")
        fh.write("% Usage: AFTER the \\newif lines:  \\IfFileExists{paper_switches}{\\input{paper_switches}}{}\n")
        fh.write("% A switch whose data is missing stays false (the weaker claim).\n")
        for sw, val, why in decision_switches():
            fh.write(f"\\{sw}{'true' if val else 'false'}   % {why}\n")
    with open(os.path.join(out, "figures_manifest.txt"), "w") as fh:
        fh.write("Figures produced by paper_figures.py\n\n")
        for fig, panels, used in MANIFEST:
            fh.write(f"{fig}.pdf / {fig}.png\n  panels: {panels}\n  data:\n")
            for u in used:
                p, n = SOURCES.get(u, ("?", 0))
                fh.write(f"    {u}: {n} runs  ({p})\n")
            fh.write("\n")
        if TABLES:
            fh.write("Tables (LaTeX tabular bodies in tables/):\n")
            for t_, n_ in sorted(TABLES.items()):
                fh.write(f"    {t_}.tex: {n_} rows\n")
        missing = sorted(set(EXPECTED) - set(SOURCES))
        if missing:
            fh.write("Experiments NOT found (their figures or panels are missing):\n")
            for m in missing:
                fh.write(f"    {m}\n")


EXPECTED = ["paper_density_exponent", "density_window_scan", "paper_centralization_sweep",
            "paper_constant_density", "paper_densification", "paper_choice_fixed",
            # Paper 1 additions (each analysis is skipped while its data is missing)
            "abl_pl_kappa0", "abl_pl_nodol", "abl_pl_lambda0", "abl_pl_b0",
            "abl_c_kappa0", "abl_c_nodol", "abl_c_lambda0", "abl_c_b0",
            "paper_density_exponent_b075", "density_window_scan_b075", "superlinear_tuning",
            "superlinear_city", "deliver_learning_check", "transient_check_pl",
            "paper_density_exponent_seedsB", "poly_pilot", "poly_main", "poly_main_base",
            "lambda_city", "density_window_scan_lam0", "density_window_scan_lam1"]
PAPER1 = ["power_laws", "epsilon", "ablations", "generality", "robustness", "scaling", "gamma",
          "transient", "seeds", "pace", "poly", "densification", "lambda"]
COMPANION = ["choice", "trap", "coordination"]        # agents choosing centralization (Paper 2)


def main():
    ap = argparse.ArgumentParser(description="Paper figures and numbers (publication style).")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="only these analyses: " + " ".join(PAPER1 + COMPANION))
    ap.add_argument("--nmin", type=int, default=64,
                    help="smallest N of the main range of Fig. 2 (E1 decision rule: 64 or 256)")
    ap.add_argument("--companion", action="store_true",
                    help="also the figures of the companion paper (choice, trap, coordination)")
    args = ap.parse_args()
    out = args.out or os.path.join(paths.root(args.root), "paper_figures",
                                   _dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    paths.ensure(out)
    plt = setup_style()
    want = set(args.only) if args.only else set(PAPER1) | (set(COMPANION) if args.companion else set())
    on = want.__contains__
    R = args.root
    cache = {}

    def G(name):
        if name not in cache:
            cache[name] = get(name, R)
            if cache[name] is not None:
                check_pinned(cache[name], name)
        return cache[name]

    P = G("paper_density_exponent") if want & {"power_laws", "epsilon", "ablations", "seeds", "pace", "transient"} else None
    W = G("density_window_scan") if want & {"power_laws", "epsilon", "pace"} else None
    S = G("paper_centralization_sweep") if want & {"scaling", "gamma", "ablations", "transient", "densification"} else None
    C = D = FX = None
    if on("gamma") or on("densification"):
        C, D = G("paper_constant_density"), G("paper_densification")
    if on("gamma") or on("transient") or on("choice") or on("trap"):
        FX = G("paper_choice_fixed")

    mfline = None
    if on("epsilon") and W is not None:
        try:
            _, mfline = epsilon_window(W, P, out, plt)
        except Exception as e:  # noqa: BLE001
            note(f"[note] epsilon_window failed: {e!r}")
    if on("power_laws") and P is not None:
        fig_power_laws(P, W, plt, out, nmin=args.nmin, mf=mfline)
        dol_keys(P, S)
    if on("scaling") and S is not None:
        fig_scaling(S, plt, out)
    fit = None
    if on("gamma"):
        frames = [df.assign(source=n) for n, df in (("paper_centralization_sweep", S),
                                                    ("paper_constant_density", C),
                                                    ("paper_densification", D),
                                                    ("paper_choice_fixed", FX)) if df is not None]
        if frames:
            fit, _, _ = fig_gamma(frames, plt, out)
            try:
                gamma_extras(frames, fit, S, out, plt, NL=G("deliver_learning_check"))
            except Exception as e:  # noqa: BLE001
                note(f"[note] gamma_extras failed: {e!r}")
    if on("ablations"):
        ablations(R, P, S, out, plt)
    if on("generality"):
        PB, WB = G("paper_density_exponent_b075"), G("density_window_scan_b075")
        if PB is not None and WB is not None:
            generality(PB, WB, out, plt)
    if on("robustness"):
        robustness(G("superlinear_tuning"), G("superlinear_city"), out)
    if on("transient"):
        transient_checks(S, FX, P, G("transient_check_pl"))
    if on("seeds") and P is not None:
        PB2 = G("paper_density_exponent_seedsB")
        if PB2 is not None:
            seed_blocks(P, PB2)
    if on("pace"):
        pace_of_life(P, W)
    if on("poly"):
        poly_pilot(G("poly_pilot"), G("poly_pilot_base"))
        PM, PMB = G("poly_main"), G("poly_main_base")
        if PM is not None and PMB is not None:
            p_ = fit["p_fit"] if fit else float(NUMBERS.get("gamma_p", "nan"))
            polycentric(PM, PMB, p_, out, plt)
    if on("lambda"):
        Wl = W if W is not None else G("density_window_scan")
        Sl = S if S is not None else G("paper_centralization_sweep")
        lambda_dependence(Sl, G("abl_c_lambda0"), G("lambda_city"), G("abl_c_lambda1"),
                          Wl, G("density_window_scan_lam0"), G("density_window_scan_lam1"), out, plt)
    if on("densification") and D is not None:
        fig_densification(D, C, plt, out)
    # companion paper (agents choosing centralization): only on request
    if on("choice") or on("trap"):
        ch_name, CH = first("paper_choice_logit_sweep", "paper_choice_sweep", R)
        if on("choice") and CH is not None and FX is not None:
            fig_choice(CH, FX, plt, out, ch_name)
        if on("trap"):
            LK, TT = G("paper_choice_logit_lockin"), G("choice_trap_test")
            if LK is not None or TT is not None:
                fig_trap(LK, CH, TT, FX, plt, out, ch_name)
    if on("coordination"):
        V, E = G("paper_coord_value"), G("paper_coord_value_endogenous")
        if V is not None:
            fig_coordination(V, E, plt, out)

    write_text_outputs(out)
    print(f"\nfigures (pdf + png), tables/, paper_numbers.txt/.json, paper_macros.tex and "
          f"figures_manifest.txt in {out}/")


if __name__ == "__main__":
    main()