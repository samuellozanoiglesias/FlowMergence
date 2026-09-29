"""
analysis.py — Loads results, fits scaling exponents and produces figures.

The target can be an experiment NAME (looked up under DATA_ROOT) or a path to a
results.jsonl file:
    python analysis.py lambda_sweep --by exclusion
    python analysis.py phase_diagram --by spillover exclusion
    python analysis.py control_density --by exclusion --Nmin 256
    python analysis.py /some/path/results.jsonl --by exclusion --out /some/figs

Output (default): DATA_ROOT/experiments/<exp>/analysis/by_<columns>[_N<min>-<max>]/
    exponents.csv, local_exponents.csv, runs_flat.csv, *.png, analysis_config.txt

Exponents (linear fit in log-log of the MEAN over seeds at each N):
    β  : Y ∝ N^β            (total output)
    γ  : M ∝ N^γ            (total bureaucracy)
    γa : M_active ∝ N^γa    (active bureaucracy)
95% CI by bootstrap: seeds are resampled within each N.

LOCAL exponent: β_loc(N) = Δlog Y / Δlog N between consecutive N values. Since the
crossover between regimes happens over a finite range of N, the β_loc(N) curve is
usually more informative than a single global β.
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


def load_results(path: str) -> pd.DataFrame:
    rows, n_err = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "error" in rec:
                n_err += 1
                continue
            row = dict(rec["params"])
            row.update(rec["summary"])
            row["seed"] = rec["seed"]
            row["runtime_s"] = rec.get("runtime_s", math.nan)
            row["run_dir"] = rec.get("run_dir")
            rows.append(row)
    if n_err:
        print(f"warning: ignored {n_err} error records")
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"no valid results in {path}")
    if "phi" in df and "xi" in df:
        df["phi_over_xi"] = (df["phi"] / df["xi"]).round(4)   # derived: the master knob
    return df


def fit_loglog(x, y):
    """Return (slope, intercept, R²) of log y = a log x + c, ignoring y ≤ 0."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = (x > 0) & (y > 0) & np.isfinite(y)
    if ok.sum() < 2:
        return math.nan, math.nan, math.nan
    lx, ly = np.log(x[ok]), np.log(y[ok])
    a, c = np.polyfit(lx, ly, 1)
    pred = a * lx + c
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return float(a), float(c), r2


def bootstrap_exponent(g: pd.DataFrame, x: str, y: str, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    by_x = {xv: sub[y].to_numpy(float) for xv, sub in g.groupby(x)}
    xs = np.array(sorted(by_x))
    means = np.array([np.nanmean(by_x[v]) for v in xs])
    est, _, r2 = fit_loglog(xs, means)
    boots = []
    for _ in range(n_boot):
        ys = [np.nanmean(rng.choice(by_x[v], size=len(by_x[v]), replace=True)) for v in xs]
        boots.append(fit_loglog(xs, ys)[0])
    boots = np.asarray(boots)
    boots = boots[np.isfinite(boots)]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots.size else (math.nan, math.nan))
    return est, float(lo), float(hi), r2, int((means > 0).sum())


def scaling_table(df: pd.DataFrame, by: list, x: str = "N", y: str = "Y", n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        est, lo, hi, r2, npts = bootstrap_exponent(g, x, y, n_boot)
        row = dict(zip(by, keys))
        row.update({"observable": y, "exponent": est, "ci_lo": lo, "ci_hi": hi,
                    "r2": r2, "n_points": npts, "N_min": g[x].min(), "N_max": g[x].max()})
        rows.append(row)
    return pd.DataFrame(rows)


def local_exponents(df: pd.DataFrame, by: list, x: str = "N", y: str = "Y", stride: int = 1) -> pd.DataFrame:
    """Local exponent between N_i and N_{i+stride}. stride=2 (a factor 4 in N for a
    powers-of-two grid) smooths the lattice-size zigzag."""
    rows = []
    for keys, g in df.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        m = g.groupby(x)[y].mean().sort_index()
        xs, ys = m.index.to_numpy(float), m.to_numpy(float)
        for i in range(len(xs) - stride):
            j = i + stride
            if ys[i] > 0 and ys[j] > 0:
                slope = (math.log(ys[j]) - math.log(ys[i])) / (math.log(xs[j]) - math.log(xs[i]))
                row = dict(zip(by, keys))
                row.update({"N_mid": math.sqrt(xs[i] * xs[j]), "local_exponent": slope, "observable": y})
                rows.append(row)
    return pd.DataFrame(rows)


def crossover_sizes(loc: pd.DataFrame, by: list) -> pd.DataFrame:
    """For each group: the first N_mid from which the (smoothed) local β stays below 1,
    the largest local β (city strength) and the smallest (company strength)."""
    rows = []
    for keys, g in loc.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        g = g.sort_values("N_mid")
        b = g.local_exponent.to_numpy()
        n = g.N_mid.to_numpy()
        cross = math.nan
        for i in range(len(b)):
            if np.all(b[i:] < 1):
                cross = float(n[i])
                break
        row = dict(zip(by, keys))
        row.update({"N_cross": cross, "beta_local_max": float(np.nanmax(b)),
                    "beta_local_min": float(np.nanmin(b)), "beta_local_at_largest_N": float(b[-1])})
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve(target: str, root: str | None, launch: str | None = None):
    """Return (results.jsonl path, experiment name or None).

    `target` may be:
      * a path to a results.jsonl file;
      * a path to a launch folder (…/experiments/<exp>/exp_<stamp>/) or to an experiment folder;
      * an experiment NAME, looked up under DATA_ROOT/experiments/<name>/.
    Inside an experiment folder, every launch subfolder exp_*/ that contains a results.jsonl
    is a candidate; the newest one is used (by folder name, which starts with a sortable
    timestamp, then by modification time). Use --launch to pick a specific one.
    A results.jsonl directly inside the experiment folder (flat layout) is used as a fallback.
    """
    import glob

    if os.path.isfile(target):
        return target, None

    if os.path.isdir(target):
        exp_dir, exp = os.path.abspath(target), None
    else:
        exp_dir, exp = os.path.join(paths.root(root), "experiments", target), target
    if os.path.isfile(os.path.join(exp_dir, "results.jsonl")) and os.path.basename(exp_dir).startswith("exp_"):
        return os.path.join(exp_dir, "results.jsonl"), exp       # target was a launch folder

    if launch:
        path = os.path.join(exp_dir, launch, "results.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no results.jsonl in the requested launch folder: {path}")
        return path, exp

    candidates = [d for d in glob.glob(os.path.join(exp_dir, "exp_*"))
                  if os.path.isfile(os.path.join(d, "results.jsonl"))]
    if candidates:
        candidates.sort(key=lambda d: (os.path.basename(d), os.path.getmtime(d)))
        latest = candidates[-1]
        if len(candidates) > 1:
            others = ", ".join(os.path.basename(d) for d in candidates[:-1])
            print(f"using the latest launch {os.path.basename(latest)} (others: {others}; "
                  f"pick one with --launch)")
        return os.path.join(latest, "results.jsonl"), exp

    flat = os.path.join(exp_dir, "results.jsonl")
    if os.path.isfile(flat):
        return flat, exp
    raise FileNotFoundError(f"no results found for '{target}': looked for {exp_dir}/exp_*/results.jsonl "
                            f"and {flat}")


def density_response(df: pd.DataFrame, group: list) -> pd.DataFrame:
    """Output per agent vs MEASURED density, and its local elasticity
        ε(ρ) = Δlog y / Δlog ρ   between consecutive densities.
    With density growing as N^(1−a), the size exponent follows β − 1 ≈ (1 − a)·ε.
    The implied β for two reference area exponents (a = 2/3, urban-like, and 0.85) is reported.
    """
    cols = ["density", "y_per_capita", "blocked_frac", "cross_blocked_frac", "M_frac",
            "DOL_tasks", "idle_frac", "mean_w", "mean_nsame"]
    cols = [c for c in cols if c in df]
    keys = group + ["rho0"]
    m = df.groupby(keys)[cols].mean().reset_index()
    sd = df.groupby(keys)["y_per_capita"].std().reset_index(name="y_sd")
    m = m.merge(sd, on=keys)
    rows = []
    for gkeys, g in (m.groupby(group) if group else [((), m)]):
        g = g.sort_values("density").reset_index(drop=True)
        eps = [math.nan]
        for i in range(1, len(g)):
            y0, y1 = g.y_per_capita[i - 1], g.y_per_capita[i]
            r0, r1 = g.density[i - 1], g.density[i]
            eps.append((math.log(y1) - math.log(y0)) / (math.log(r1) - math.log(r0))
                       if y0 > 0 and y1 > 0 and r1 > r0 > 0 else math.nan)
        g["epsilon"] = eps
        g["density_mid"] = [math.nan] + [math.sqrt(g.density[i - 1] * g.density[i]) for i in range(1, len(g))]
        g["beta_if_a_0.67"] = 1 + (1 - 2 / 3) * g["epsilon"]
        g["beta_if_a_0.85"] = 1 + (1 - 0.85) * g["epsilon"]
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def _write_analysis_config(out: str, path: str, by: list, args, df: pd.DataFrame):
    varying = [c for c in df.columns if c in paths.ABBREV and df[c].nunique() > 1]
    with open(os.path.join(out, "analysis_config.txt"), "w") as fh:
        fh.write("# FlowMergence — analysis configuration\n")
        fh.write(f"created   : {_dt.datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"source    : {os.path.abspath(path)}\n")
        fh.write(f"group by  : {by}\n")
        fh.write(f"N window  : [{args.Nmin or df.N.min()}, {args.Nmax or df.N.max()}]\n")
        fh.write(f"bootstrap : {args.nboot}\n")
        fh.write(f"runs used : {len(df)}\n")
        fh.write(f"parameters varying in the data: {varying}\n")
        not_grouped = [c for c in varying if c not in by and c != "N"]
        if "phi_over_xi" in by and "xi" in by and "phi" in not_grouped:
            not_grouped.remove("phi")   # phi is determined by (phi_over_xi, xi)
        if not_grouped:
            fh.write(f"WARNING: these vary but are not in --by (they are averaged together): {not_grouped}\n")
            print(f"WARNING: {not_grouped} vary in the data but are not in --by")


def main():
    ap = argparse.ArgumentParser(description="Scaling analysis.")
    ap.add_argument("target", help="experiment name or path to results.jsonl")
    ap.add_argument("--by", nargs="+", default=["exclusion"])
    ap.add_argument("--root", default=None, help=f"data root (default {paths.DATA_ROOT})")
    ap.add_argument("--out", default=None, help="output folder (default: inside the experiment)")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--Nmin", type=int, default=None, help="fit window")
    ap.add_argument("--Nmax", type=int, default=None)
    ap.add_argument("--launch", default=None,
                    help="specific launch subfolder, e.g. exp_2026_09_29-01_57_34 (default: latest)")
    args = ap.parse_args()

    import plots  # imported here so the analysis functions work without matplotlib

    path, exp = _resolve(args.target, args.root, args.launch)
    print(f"reading {path}")
    by = args.by
    if args.out:
        out = args.out
    else:
        # always next to the results actually used (inside the exp_* launch folder if any)
        base = os.path.join(os.path.dirname(os.path.abspath(path)), "analysis", "by_" + "_".join(by))
        if args.Nmin or args.Nmax:
            base += f"_N{args.Nmin or 'min'}-{args.Nmax or 'max'}"
        out = base
    paths.ensure(out)

    df = load_results(path)
    if args.Nmin:
        df = df[df.N >= args.Nmin]
    if args.Nmax:
        df = df[df.N <= args.Nmax]
    print(f"{len(df)} runs, N ∈ [{df.N.min()}, {df.N.max()}], grouped by {by}")
    _write_analysis_config(out, path, by, args, df)

    if df.N.nunique() < 2:
        # Single system size (e.g. calibration_xi_phi): no exponents, compare metrics instead.
        cols = [c for c in ("DOL_tasks", "mean_top_share", "frac_mostly_one", "frac_focused",
                            "frac_specialist_any", "mean_specialties", "y_per_capita", "mean_w",
                            "idle_frac", "blocked_frac", "M_frac") if c in df]
        tab = df.groupby(by)[cols].agg(["mean", "std"])
        tab.columns = [f"{a}" if b == "mean" else f"{a}_sd" for a, b in tab.columns]
        tab = tab.reset_index()
        tab.to_csv(os.path.join(out, "metrics_by_group.csv"), index=False)
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            show = by + [c for c in ("DOL_tasks", "DOL_tasks_sd", "mean_top_share", "idle_frac",
                                     "y_per_capita") if c in tab]
            print(tab[show].sort_values("DOL_tasks", ascending=False).head(25).round(3).to_string(index=False))
        map_cols = ("DOL_tasks", "mean_top_share", "idle_frac", "y_per_capita")
        if "rho0" in by and df["density"].nunique() > 1:
            others = [c for c in by if c != "rho0"]
            resp = density_response(df, others)
            resp.to_csv(os.path.join(out, "density_response.csv"), index=False)
            with pd.option_context("display.width", 200, "display.max_rows", 500):
                print(resp.round(4).to_string(index=False))
            plots.plot_density_response(resp, others, os.path.join(out, "density_response.png"))
        if len(by) == 2:
            for col in map_cols:
                plots.plot_phase_diagram(tab, by[0], by[1], os.path.join(out, f"map_{col}.png"),
                                         value=col, center=None)
        elif len(by) >= 3:
            # one map over (by[0], by[1]) for every combination of the remaining columns
            for keys, sub in tab.groupby(by[2:]):
                keys = keys if isinstance(keys, tuple) else (keys,)
                tag = "_".join(f"{c}{plots.fmt_value(v)}" for c, v in zip(by[2:], keys))
                label = ", ".join(f"{c}={plots.fmt_value(v)}" for c, v in zip(by[2:], keys))
                for col in map_cols:
                    plots.plot_phase_diagram(sub, by[0], by[1], os.path.join(out, f"map_{col}_{tag}.png"),
                                             value=col, center=None, title=f"{col}  ({label})")
        print(f"single N: no exponents fitted. Tables and maps in {out}/")
        return

    beta = scaling_table(df, by, "N", "Y", args.nboot)
    gamma = scaling_table(df, by, "N", "M", args.nboot)
    gamma_a = scaling_table(df, by, "N", "M_active", args.nboot)
    loc = local_exponents(df, by, "N", "Y")
    loc2 = local_exponents(df, by, "N", "Y", stride=2)
    cross = crossover_sizes(loc2, by) if not loc2.empty else pd.DataFrame()
    table = pd.concat([beta, gamma, gamma_a], ignore_index=True)
    table.to_csv(os.path.join(out, "exponents.csv"), index=False)
    loc.to_csv(os.path.join(out, "local_exponents.csv"), index=False)
    loc2.to_csv(os.path.join(out, "local_exponents_smooth.csv"), index=False)
    if not cross.empty:
        cross.to_csv(os.path.join(out, "crossover.csv"), index=False)
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print("\ncrossover (smoothed local β stays < 1 from N_cross on):")
            print(cross.round(3).to_string(index=False))
    df.to_csv(os.path.join(out, "runs_flat.csv"), index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(beta[by + ["exponent", "ci_lo", "ci_hi", "r2"]].to_string(index=False))

    plots.plot_scaling(df, by, "Y", os.path.join(out, "scaling_Y.png"))
    plots.plot_local_exponents(loc, by, os.path.join(out, "local_beta.png"))
    if not loc2.empty:
        plots.plot_local_exponents(loc2, by, os.path.join(out, "local_beta_smooth.png"))
    plots.plot_bureaucracy(df, by, os.path.join(out, "bureaucracy.png"))
    plots.plot_specialization(df, by, os.path.join(out, "specialization.png"))
    plots.plot_congestion(df, by, os.path.join(out, "congestion.png"))
    if len(by) == 1:
        plots.plot_exponent_curve(beta, gamma, gamma_a, by[0], os.path.join(out, "beta_gamma.png"))
    elif len(by) == 2:
        plots.plot_phase_diagram(beta, by[0], by[1], os.path.join(out, "phase_beta.png"))
    print(f"figures and tables in {out}/")


if __name__ == "__main__":
    main()