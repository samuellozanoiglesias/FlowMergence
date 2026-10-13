#!/usr/bin/env python3
"""
analysis_density_window.py -- where is the density elasticity ε constant?

Usage (experiment folders: the newest exp_*/results.jsonl is used; or results.jsonl files):
  python analysis_density_window.py \
      $DATA/experiments/density_window_scan $DATA/experiments/density_window_scan_b075 \
      --out analysis/density_window

For each experiment (constant density, a = 1, one N):
  y(ρ)       output per agent vs density (mean ± s.e.m. over seeds)
  ε(ρ)       local elasticity d ln y / d ln ρ between neighbouring densities (± error)
  window     the WIDEST run of consecutive intervals whose ε all lie within ±tol of the
             window's mean ε (at least --min-intervals intervals)
Then, for paper_density_exponent (β = 1 + ε(1 − a), density = ρ_mid at the middle N):
  rho_mid    geometric centre of the window
  |1−a|max   largest |1 − a| whose density sweep fits inside the window, for N = 64…32768
             (N_GRID) and N = 64…131072 (N_WIDE): (N_max/N_min)^|1−a| ≤ ρ_hi/ρ_lo
  β range    1 ± ε·|1−a|max: the most city-like / company-like exponents that stay pure
             power laws in this setting
Writes window_summary.csv, epsilon_by_density.csv and density_window.png.
"""
import argparse, math, os
import numpy as np
import pandas as pd

import glob, json

def load(p):
    """results.jsonl (or experiment folder -> newest exp_*/results.jsonl), nested keys flattened
    to their last component (parameter blocks win over summary blocks on duplicates)."""
    path = p
    if os.path.isdir(p):
        direct = os.path.join(p, "results.jsonl")
        launches = sorted(glob.glob(os.path.join(p, "exp_*", "results.jsonl")))
        path = direct if os.path.isfile(direct) else (launches[-1] if launches else None)
        if path is None:
            raise SystemExit(f"no results.jsonl under {p}")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "error" not in r]
    df = pd.json_normalize(rows)
    prio = ("params", "config", "summary")
    short = {}
    for c in df.columns:
        short.setdefault(c.split(".")[-1], []).append(c)
    flat = {}
    for name, cols in short.items():
        cols = sorted(cols, key=lambda c: prio.index(c.split(".")[0]) if c.split(".")[0] in prio else 9)
        col = df[cols[0]]
        for c in cols[1:]:
            col = col.combine_first(df[c])
        flat[name] = col
    print(f"loaded {len(rows):4d} runs from {path}")
    return pd.DataFrame(flat)

GRIDS = {"N_GRID (64…32768)": 32768 / 64, "N_WIDE (64…131072)": 131072 / 64}


def elasticity(df):
    dcol = "density" if "density" in df else "rho0"
    g = df.groupby("rho0").agg(density=(dcol, "mean"), y=("y_per_capita", "mean"),
                               y_sd=("y_per_capita", "std"), n=("y_per_capita", "size"))
    for extra in ("blocked_frac", "mean_nsame", "DOL_tasks", "idle_frac"):
        if extra in df:
            g[extra] = df.groupby("rho0")[extra].mean()
    g = g.reset_index().sort_values("density")
    g["y_se"] = (g.y_sd / np.sqrt(g.n)).fillna(0.0)
    lr, ly = np.log(g.density.values), np.log(g.y.values)
    rel = (g.y_se / g.y).values                       # s.e. of ln y
    e = pd.DataFrame({
        "rho_lo": g.density.values[:-1], "rho_hi": g.density.values[1:],
        "rho_mid": np.exp((lr[:-1] + lr[1:]) / 2),
        "epsilon": np.diff(ly) / np.diff(lr),
        "epsilon_err": np.sqrt(rel[:-1] ** 2 + rel[1:] ** 2) / np.diff(lr),
    })
    return g, e


def widest_window(e, tol, min_int):
    best = None
    eps = e.epsilon.values
    for i in range(len(eps)):
        for j in range(i + min_int, len(eps) + 1):
            w = eps[i:j]
            if np.all(np.abs(w - w.mean()) <= tol):
                lo, hi = e.rho_lo.iloc[i], e.rho_hi.iloc[j - 1]
                if best is None or hi / lo > best["factor"]:
                    best = {"rho_lo": lo, "rho_hi": hi, "factor": hi / lo, "epsilon": w.mean(),
                            "epsilon_spread": w.max() - w.min(), "n_intervals": j - i}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--out", default="analysis/density_window")
    ap.add_argument("--tol", type=float, default=0.05, help="allowed |ε − mean ε| inside the window")
    ap.add_argument("--min-intervals", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows, tables = [], []
    for path in a.experiments:
        name = os.path.basename(os.path.normpath(path)).replace("results.jsonl", "") or path
        df = load(path)
        if "rho0" not in df or "y_per_capita" not in df:
            raise SystemExit(f"{path}: needs rho0 and y_per_capita")
        if df.get("area_exponent", pd.Series([1.0])).nunique() > 1 or df.N.nunique() > 1:
            print(f"  warning: {name} has several a or N values; they are pooled per rho0")
        settings = {k: getattr(df[k].iloc[0], "item", lambda: df[k].iloc[0])() for k in ("learning_exponent", "spillover", "spillover_radius",
                                                "route_length", "exclusion", "N") if k in df}
        g, e = elasticity(df)
        e.insert(0, "experiment", name)
        tables.append((name, g, e))
        w = widest_window(e, a.tol, a.min_intervals)
        print(f"\n== {name}  {settings}")
        with pd.option_context("display.width", 200):
            print(e.drop(columns="experiment").round(3).to_string(index=False))
        if w is None:
            print(f"  no window of ≥ {a.min_intervals} intervals with ε within ±{a.tol}")
            rows.append({"experiment": name, **settings})
            continue
        r = {"experiment": name, **settings, **w, "rho_mid": math.sqrt(w["rho_lo"] * w["rho_hi"])}
        print(f"  flat window: density {w['rho_lo']:.3g} – {w['rho_hi']:.3g} (factor {w['factor']:.1f}), "
              f"ε = {w['epsilon']:.3f} (spread {w['epsilon_spread']:.3f}), centre ρ_mid = {r['rho_mid']:.3g}")
        for lab, span in GRIDS.items():
            m = math.log(w["factor"]) / math.log(span)
            key = lab.split()[0]
            r[f"max_abs_1_minus_a_{key}"] = m
            r[f"beta_city_{key}"] = 1 + w["epsilon"] * m
            r[f"beta_company_{key}"] = 1 - w["epsilon"] * m
            print(f"  {lab}: |1 − a| ≤ {m:.3f}  ->  a ∈ [{1 - m:.3f}, {1 + m:.3f}],  "
                  f"β from {1 + w['epsilon'] * m:.3f} down to {1 - w['epsilon'] * m:.3f}")
        need = 0.15 / w["epsilon"]
        print(f"  β = 1.15 / 0.85 needs |1 − a| = {need:.3f}; it fits N_GRID: "
              f"{'yes' if need <= r['max_abs_1_minus_a_N_GRID'] else 'no'}")
        rows.append(r)

    pd.DataFrame(rows).round(4).to_csv(os.path.join(a.out, "window_summary.csv"), index=False)
    pd.concat([e for _, _, e in tables]).round(4).to_csv(os.path.join(a.out, "epsilon_by_density.csv"), index=False)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        for (name, g, e), r in zip(tables, rows):
            l = a1.errorbar(g.density, g.y, yerr=2 * g.y_se, fmt="o-", ms=3, label=name)[0]
            a2.errorbar(e.rho_mid, e.epsilon, yerr=2 * e.epsilon_err, fmt="o-", ms=3, color=l.get_color())
            if "rho_lo" in r:
                a2.axvspan(r["rho_lo"], r["rho_hi"], color=l.get_color(), alpha=0.12)
                a2.axhline(r["epsilon"], color=l.get_color(), ls="--", lw=0.8)
        a1.set(xscale="log", yscale="log", xlabel="density ρ", ylabel="output per agent")
        a2.set(xscale="log", xlabel="density ρ", ylabel="ε = d ln y / d ln ρ",
               title="shaded: widest window with constant ε")
        a1.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "density_window.png"), dpi=130)
    except Exception as ex:  # noqa: BLE001
        print("plot skipped:", ex)
    print(f"\nwritten to {a.out}/ (window_summary.csv, epsilon_by_density.csv, density_window.png)")


if __name__ == "__main__":
    main()
