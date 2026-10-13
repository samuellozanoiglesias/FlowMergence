"""Synthetic test of the new private-payoff / belief summaries (no lattice needed)."""
import numpy as np, math, json
from types import SimpleNamespace
from metrics import Accumulator, summarize
rng = np.random.default_rng(0)
N, m = 1000, 4
acc = Accumulator()
true_h, true_l = [], []
for t in range(200):                         # mimic the per-step info of dynamics.step
    n = 30; hub = rng.random(n) < 0.3
    rate = np.where(hub, 3.0, 1.0) * np.exp(0.2 * rng.standard_normal(n)).astype(np.float32)
    lr = np.log(rate)
    true_h += list(rate[hub]); true_l += list(rate[~hub])
    acc.add({"work": 10.0, "completions": float(n), "attempts": 100.0,
             "rate_hub_sum": (rate * hub).sum(), "rate_hub_log": (lr * hub).sum(), "rate_hub_n": hub.sum(),
             "rate_loc_sum": (rate * ~hub).sum(), "rate_loc_log": (lr * ~hub).sum(), "rate_loc_n": n - hub.sum()})
p = SimpleNamespace(theta_min=1.0, theta_max=4000.0, m=m)
qh = np.where(rng.random(N) < 0.9, rng.uniform(1, 4, N), 0).astype(np.float32)
ql = np.ones(N, np.float32); ph = np.where(qh > 2, 0.98, 0.02).astype(np.float32)
sim = SimpleNamespace(N=N, L=50, m=m, p=p, learned_choice=True, choice_rule="logit",
                      p_hub=ph, q_hub=qh, q_loc=ql, n_jobs=rng.integers(0, 60, N),
                      time_budget=rng.integers(0, 10, (N, m + 2)), theta=np.full((N, m + 1), 100.0))
s = summarize(sim, acc)
keys = ["rate_hub_mean", "rate_local_mean", "hub_private_ratio", "hub_private_log_adv",
        "frac_tried_both", "mean_log_q_ratio", "frac_believe_hub_better",
        "log_q_ratio_committed_hub", "log_q_ratio_committed_local",
        "jobs_per_agent_median", "frac_agents_lt20_jobs"]
for k in keys: print(f"{k:28s} {s[k]:.4f}")
assert abs(s["rate_hub_mean"] - np.mean(true_h)) < 1e-3 and abs(s["rate_local_mean"] - np.mean(true_l)) < 1e-3
assert abs(s["hub_private_log_adv"] - (np.mean(np.log(true_h)) - np.mean(np.log(true_l)))) < 1e-4
assert abs(s["hub_private_log_adv"] - math.log(3)) < 0.05
json.dumps(s)                                   # must stay JSON-serializable
# a window with no hub jobs, and a non-logit run: NaN, no crash
acc2 = Accumulator(); acc2.add({"rate_loc_sum": 5.0, "rate_loc_n": 5, "rate_loc_log": 0.0})
sim.choice_rule = "bush_mosteller"; s2 = summarize(sim, acc2)
print("no hub jobs ->", s2["hub_private_log_adv"], s2["hub_private_ratio"], "| logit keys present:", "mean_log_q_ratio" in s2)
print("ALL OK")
