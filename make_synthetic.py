"""Synthetic results.jsonl for every experiment paper_figures.py reads (TEST DATA ONLY).
  CODE_DIR=. SYN_ROOT=/tmp/fmsyn python tests/make_synthetic.py
  python paper_figures.py --root /tmp/fmsyn --out /tmp/fmout      # exercises every analysis
Values come from meanfield.py plus planted effects (small-N deficit, centralization, ablations);
they are NOT results and must never be mixed with real data."""
import sys, os, json, math, warnings
warnings.simplefilter("ignore")
sys.path[:0] = [os.environ.get("CODE_DIR", ".")]
import numpy as np, config, meanfield as mfm, hubs
from params import Params
ROOT = os.environ.get("SYN_ROOT", "/tmp/fmsyn")
NAMES = [n for n in config.EXPERIMENTS if n.startswith(("paper_", "abl_", "density_window", "superlinear_",
         "deliver_", "transient_", "poly_"))]
def row(job):
    pd_ = job["params"]; p = Params.from_dict(pd_); rs = np.random.RandomState(abs(hash((job["job_id"]))) % 2**31)
    sr = np.random.RandomState(job["seed"] + 1000)                   # common seed effect
    P = mfm.MFParams.from_params(pd_)
    st = mfm.solve(p.density, P)
    y = st["y"] * (1 - 0.04 * math.sqrt(64 / p.N))                   # small-N deficit
    dol = 0.9
    if p.xi == 0 and p.phi == 0: y *= 0.35; dol = 0.03
    c = p.centralization
    dbar = hubs.mean_hub_distance(p.L, p.hub_count_exponent, p.hub_spacing_ref)
    G = 0.0
    if c > 0:
        tau = p.route_length / st["w"]
        S = 1 + 0.5 * c * (p.N / 131072) ** 0.7
        G = c * dbar / tau * S
        y *= (1 + G) ** (-(1.0 if not p.learn_while_delivering else 0.89))
        if not p.learn_while_delivering: y *= 0.97
    y *= math.exp(0.006 * sr.randn() + 0.004 * rs.randn())
    d = G / (1 + G)
    comp = y * p.N / p.route_length
    summ = dict(N=p.N, L=p.L, density=p.density, Y=y * p.N, y_per_capita=y, completions=comp,
                working_frac=1 - st["q"] * (1 - d), idle_frac=st["q"] * (1 - d), blocked_frac=st["B"] * (1 + 0.3 * c),
                mean_w=st["w"] * (1 + 0.2 * c), mean_nsame=st["n"] * 1.05 * (1 + 0.5 * c), delivering_frac=d,
                Gamma_meas=G, hub_share_completed=c, DOL_tasks=dol, frac_focused=0.8, mean_top_share=0.95,
                hub_mean_dist=dbar, n_hubs=hubs.n_hubs(p.L, p.hub_count_exponent, p.hub_spacing_ref))
    return dict(params=pd_, summary=summ, seed=job["seed"], job_id=job["job_id"])
for n in NAMES:
    jobs = config.build_jobs(n, "20261004-000000", root=ROOT)
    d = os.path.join(ROOT, "experiments", n, "exp_20261004-000000"); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.jsonl"), "w") as fh:
        for j in jobs: fh.write(json.dumps(row(j)) + "\n")
print("synthetic experiments:", len(NAMES))
