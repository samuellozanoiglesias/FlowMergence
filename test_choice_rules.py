"""Synthetic test: stationary mean p_hub vs the hub/local payoff-rate ratio, both rules.
Payoff rates are lognormal with sd `sd` around 1 (local) and `ratio` (hub)."""
import numpy as np, mechanism as mech
def run(rule, ratio, sd, n=20000, jobs=1500, p0=0.5, seed=0, T=0.1):
    r = np.random.default_rng(seed)
    p = np.full(n, p0, np.float32); A = np.zeros(n, np.float32)
    qh = np.zeros(n, np.float32); ql = np.zeros(n, np.float32); out = []
    for t in range(jobs):
        hub = r.random(n) < p
        rate = (np.where(hub, ratio, 1.0) * np.exp(sd * r.standard_normal(n))).astype(np.float32)
        if rule == "bm":
            p, A = mech.update_delivery_choice(p, A, hub, rate, 0.3, 0.1, 0.02)
        else:
            p, qh, ql = mech.update_delivery_values(p, qh, ql, hub, rate, 0.1, T, 0.02)
        if t >= jobs - 300: out.append(p.mean())
    return np.mean(out)
ratios = (0.1, 0.5, 0.8, 1.0, 1.25, 2, 5)
print("ratio:           " + "".join(f"{q:>7}" for q in ratios))
for sd in (0.05, 0.3, 0.6):
    for rule in ("bm", "logit"):
        print(f"sd={sd:4} {rule:6}: " + "".join(f"{run(rule,q,sd):7.3f}" for q in ratios))
# start-point independence (lock-in test of the rule itself) and speed
for p0 in (0.05, 0.95):
    print(f"logit, ratio 0.5, p0={p0}: after 50 jobs {run('logit',0.5,0.3,jobs=50,p0=p0):.3f}, "
          f"after 300 {run('logit',0.5,0.3,jobs=300,p0=p0):.3f}")
