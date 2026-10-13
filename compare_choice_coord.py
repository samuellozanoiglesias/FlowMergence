#!/usr/bin/env python3
"""
compare_choice_coord.py -- learned choice WITH coordinators vs WITHOUT (Step 12, option B).

Usage (pass experiment folders or results.jsonl files; for a folder the newest
exp_<stamp>/results.jsonl is used, same convention as the other analyses):

  python compare_choice_coord.py \
      --base  $DATA/experiments/choice_sweep \
      --coord $DATA/experiments/choice_with_coordination \
      --out   analysis/choice_coord

Writes:
  coord_vs_nocoord.csv   per (hub_premium, N): c, value, Gamma, M_frac, block split, gain
  exponents_coord.csv    beta of value per premium, with and without coordinators
  coord_gain.png         value gain from coordination vs N, and c_emergent vs N
Prints a config diff (parameters that differ between the two experiments) first,
so an accidental config mismatch is caught before any comparison is read.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

METRICS = ["hub_share_completed", "mean_p_hub", "y_value_per_capita", "y_per_capita",
           "Gamma_meas", "M_frac", "blocked_frac", "cross_blocked_frac",
           "DOL_tasks", "frac_p_hub_high", "frac_p_hub_low"]
KEYS = ["hub_premium", "N"]


def find_jsonl(p):
    if os.path.isfile(p):
        return p
    direct = os.path.join(p, "results.jsonl")
    if os.path.isfile(direct):
        return direct
    launches = sorted(glob.glob(os.path.join(p, "exp_*", "results.jsonl")))
    if not launches:
        sys.exit(f"no results.jsonl under {p}")
    return launches[-1]          # timestamped names sort chronologically


def load(p):
    path = find_jsonl(p)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.json_normalize(rows)
    # flatten nested keys (params.N, summary.Y, ...) to their last component. When the same
    # name appears in several blocks (e.g. params.N and summary.N), take the first non-missing
    # value in the order: parameter-like blocks first, then the rest.
    prio = ("params", "config", "p", "parameters", "derived", "summary", "result", "metrics")
    def rank(c):
        head = c.split(".")[0] if "." in c else ""
        return (prio.index(head) if head in prio else len(prio)) if head else -1
    short = {}
    for c in df.columns:
        short.setdefault(c.split(".")[-1], []).append(c)
    flat = {}
    for name, cols in short.items():
        cols = sorted(cols, key=rank)
        col = df[cols[0]]
        for c in cols[1:]:
            col = col.combine_first(df[c])
        flat[name] = col
    df = pd.DataFrame(flat)
    print(f"loaded {len(df):5d} runs from {path}")
    missing = [k for k in KEYS + ["hub_share_completed", "y_value_per_capita"] if k not in df]
    if missing:
        sys.exit(f"columns {missing} not found in {path}.\n"
                 f"first-row keys: {list(rows[0].keys())}\n"
                 f"all columns: {sorted(df.columns)[:80]}")
    return df


def config_diff(a, b):
    """Parameters = columns constant within every (hub_premium, N) cell (measured
    quantities vary across seeds). Report those whose values differ between runs sets."""
    skip = {"seed", "run_dir", "runtime_s", "L", "density"} | set(KEYS)
    def params(df):
        g = df.groupby(KEYS)
        return {c for c in df.columns if c not in skip
                and (g[c].nunique(dropna=False) <= 1).all()}
    diffs = []
    for c in sorted(params(a) & params(b)):
        norm = lambda v: "nan" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else (f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v))
        va = sorted({norm(v) for v in a[c]}); vb = sorted({norm(v) for v in b[c]})
        if va != vb:
            diffs.append((c, va[:5], vb[:5]))
    return diffs


def summarize(df, tag):
    cols = [m for m in METRICS if m in df.columns]
    g = df.groupby(KEYS)
    out = g[cols].mean().add_suffix(f"_{tag}")
    out[f"seeds_{tag}"] = g.size()
    if "y_value_per_capita" in cols:
        out[f"y_value_sd_{tag}"] = g["y_value_per_capita"].std()
    return out


def beta(t, col):
    res = []
    for pi, s in t.reset_index().groupby("hub_premium"):
        s = s.dropna(subset=[col])
        s = s[s[col] > 0]
        if len(s) < 3:
            res.append((pi, np.nan)); continue
        slope = np.polyfit(np.log(s.N), np.log(s[col]), 1)[0]
        res.append((pi, 1 + slope))          # per-capita slope -> beta = 1 + slope
    return pd.Series(dict(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--coord", required=True)
    ap.add_argument("--out", default="analysis/choice_coord")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    base, coord = load(a.base), load(a.coord)
    d = config_diff(base, coord)
    print("\nparameters that differ (base -> coord):")
    for c, va, vb in d:
        print(f"  {c:24s} {va} -> {vb}")
    if not d:
        print("  none (check that 'coordination' is really on in the coord runs!)")

    t = summarize(base, "base").join(summarize(coord, "coord"), how="inner")
    if t.empty:
        sys.exit("no common (hub_premium, N) cells between the two experiments")
    t["value_gain"] = t.y_value_per_capita_coord / t.y_value_per_capita_base - 1
    t["dc"] = t.hub_share_completed_coord - t.hub_share_completed_base
    for tag in ("base", "coord"):
        if f"cross_blocked_frac_{tag}" in t:
            t[f"cross_share_of_blocks_{tag}"] = (t[f"cross_blocked_frac_{tag}"]
                                                 / t[f"blocked_frac_{tag}"].replace(0, np.nan))
    # is the gain larger than seed noise? (2 standard errors of the difference)
    se = np.sqrt(t.y_value_sd_base**2 / t.seeds_base + t.y_value_sd_coord**2 / t.seeds_coord)
    t["gain_significant"] = (t.y_value_per_capita_coord - t.y_value_per_capita_base).abs() > 2 * se
    t.round(5).to_csv(os.path.join(a.out, "coord_vs_nocoord.csv"))

    ex = pd.DataFrame({"beta_value_base": beta(t, "y_value_per_capita_base"),
                       "beta_value_coord": beta(t, "y_value_per_capita_coord")})
    ex.index.name = "hub_premium"
    ex.round(4).to_csv(os.path.join(a.out, "exponents_coord.csv"))

    show = ["hub_share_completed_base", "hub_share_completed_coord", "value_gain",
            "gain_significant", "M_frac_coord", "Gamma_meas_base", "Gamma_meas_coord"]
    show = [c for c in show if c in t]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print("\n", t[show].round(3).to_string())
        print("\n", ex.round(3).to_string())

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for pi, s in t.reset_index().groupby("hub_premium"):
            ax[0].plot(s.N, 100 * s.value_gain, "o-", label=f"π={pi:g}")
            ax[1].plot(s.N, s.hub_share_completed_base, "--", color=ax[0].lines[-1].get_color())
            ax[1].plot(s.N, s.hub_share_completed_coord, "o-", color=ax[0].lines[-1].get_color())
        ax[0].axhline(0, color="k", lw=0.8)
        for x in ax:
            x.set_xscale("log", base=2); x.set_xlabel("N")
        ax[0].set_ylabel("value gain from coordination (%)"); ax[0].legend(fontsize=8)
        ax[1].set_ylabel("c_emergent (dashed: no coord, solid: coord)")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "coord_gain.png"), dpi=130)
    except Exception as e:
        print("plot skipped:", e)
    print(f"\nwritten to {a.out}/")


if __name__ == "__main__":
    main()