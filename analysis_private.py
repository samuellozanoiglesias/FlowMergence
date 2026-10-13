#!/usr/bin/env python3
"""
analysis_private.py -- does the hub still pay PRIVATELY where the planner says c = 0?

Usage:
  python analysis_private.py --choice $DATA/experiments/choice_logit_robust \
      --planner analysis/choice/choice_vs_planner.csv --out analysis/private

--choice   experiment folder (newest exp_*/results.jsonl) or a results.jsonl file
--planner  optional choice_vs_planner.csv from analysis_choice.py (adds c_opt, efficiency_gap)

Per (hub_premium, N) it reports the realized private payoff of each delivery mode
(hub_private_log_adv = E ln r_hub − E ln r_local, measured on jobs finished in the window),
what agents believe (logit rule: mean ln Q_hub/Q_local, fraction believing the hub is
better) and the convergence diagnostics (median jobs per agent, fraction with < 20 jobs).

Verdict per cell:
  competency trap  committed hub users believe the hub pays (ln Q_hub/Q_loc > 0) but hub jobs
                   actually pay less (log_adv < 0): stale estimates of the unused mode
  externality      agents centralize ≥ 0.1 more than the planner and the hub pays privately
                   (log_adv > 0): each agent ignores the queue it adds
  aligned          |c − c_opt| < 0.1 and efficiency gap < 5%
  exploration cost planner c_opt = 0, only exploring agents use the hub, gap ≥ 5%
  not settled      > 20% of agents finished fewer than 20 jobs
  other            anything else (inspect the numbers)
"""
import argparse, os
import numpy as np
import pandas as pd

from compare_choice_coord import load

COLS = ["hub_share_completed", "mean_p_hub", "hub_private_log_adv", "hub_private_ratio",
        "rate_hub_mean", "rate_local_mean", "frac_tried_both", "mean_log_q_ratio",
        "frac_believe_hub_better", "log_q_ratio_committed_hub", "log_q_ratio_committed_local",
        "frac_p_hub_high", "frac_p_hub_low", "jobs_per_agent_median", "frac_agents_lt20_jobs",
        "local_info_age_committed_hub", "hub_info_age_committed_local",
        "frac_never_local_committed_hub",
        "y_value_per_capita", "Gamma_meas"]
# extra grid parameters kept apart (e.g. choice_explore, T_burn in choice_logit_robust)
EXTRA = ["choice_payoff", "choice_p0", "choice_explore", "T_burn", "choice_temperature",
         "choice_rule", "coordination", "coord_fixed_frac"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--choice", required=True)
    ap.add_argument("--planner", default=None)
    ap.add_argument("--out", default="analysis/private")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load(a.choice)
    if "hub_private_log_adv" not in df:
        raise SystemExit("no hub_private_log_adv in these results: they were run before the "
                         "metrics.py / dynamics.py update. Relaunch the experiment.")
    keys = ["hub_premium", "N"] + [k for k in EXTRA if k in df and df[k].nunique(dropna=False) > 1]
    print("grouping by:", keys)
    cols = [c for c in COLS if c in df]
    t = df.groupby(keys)[cols].mean()
    t["seeds"] = df.groupby(keys).size()
    t["log_adv_sd"] = df.groupby(keys)["hub_private_log_adv"].std()
    t = t.reset_index()

    explore = float(df["choice_explore"].iloc[0]) if "choice_explore" in df else 0.02
    if a.planner:
        pl = pd.read_csv(a.planner)[["hub_premium", "N", "c_opt", "y_value_opt"]]
        t = t.merge(pl, on=["hub_premium", "N"], how="left")
        # gap per row (per T_burn / ε), not the planner file's average over them
        t["efficiency_gap"] = 1.0 - t.y_value_per_capita / t.y_value_opt
    if "log_q_ratio_committed_hub" in t:
        # belief of committed hub users minus what hub jobs actually paid (log units)
        t["belief_bias_hub_users"] = t.log_q_ratio_committed_hub - t.hub_private_log_adv

    def verdict(r):
        if r.get("frac_agents_lt20_jobs", 0) > 0.2:
            return "not settled"
        if (r.get("frac_p_hub_high", 0) > 0.1 and r.get("log_q_ratio_committed_hub", -1) > 0
                and r.hub_private_log_adv < 0):
            return "competency trap"     # hub users believe it pays; it does not
        if "c_opt" in r and pd.notna(r["c_opt"]):
            over = r.hub_share_completed - r.c_opt
            if abs(over) < 0.1 and r.efficiency_gap < 0.05:
                return "aligned"
            if over >= 0.1 and r.hub_private_log_adv > 0:
                return "externality"     # hub pays privately, planner wants less of it
            ex = r.get("choice_explore", explore)
            if r.c_opt == 0 and r.hub_share_completed <= 2 * ex:
                return "exploration cost"  # only exploring agents use the hub
        return "other"
    t["verdict"] = t.apply(verdict, axis=1)
    t.round(4).to_csv(os.path.join(a.out, "private_payoff.csv"), index=False)

    show = keys + [c for c in ["hub_share_completed", "c_opt", "mean_p_hub", "hub_private_log_adv",
                               "log_adv_sd", "log_q_ratio_committed_hub", "belief_bias_hub_users",
                               "local_info_age_committed_hub", "efficiency_gap",
                               "jobs_per_agent_median", "frac_agents_lt20_jobs", "verdict"] if c in t]
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(t[show].round(3).to_string(index=False))
    print(f"\nwritten to {a.out}/private_payoff.csv")
    print("hub_private_log_adv > 0: a hub job paid its agent more than a local one "
          "(e.g. 0.1 = 10% more, −0.7 = half).")


if __name__ == "__main__":
    main()