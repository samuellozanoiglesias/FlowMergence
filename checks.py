"""
checks.py — Sanity checks. Run them BEFORE launching large sweeps:

    python checks.py            # all (~a few minutes on one core)
    python checks.py --quick    # invariants only

Everything is stored in DATA_ROOT/checks/<timestamp>/: a report.txt with all printed
lines and one folder (config.txt + result.json) per full run used by the checks.

1. Lattice invariants at every step (no duplicates, consistent positions).
2. Without coordination, no coordinator ever appears.
3. With λ = 0 there are no crossing blocks.
4. With λ = 1 and high density there are crossing blocks (the BML substrate works).
5. Emergent specialization: DOL across tasks clearly > 0 (reproduces T98), with
   diagnostics on the stimuli if it fails.
6. Scaling control: with a = 1, β should come out ≈ 1 (indicative; short runs).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os

import numpy as np

import paths
from analysis import fit_loglog
from dynamics import Simulation
from params import Params
from runner import run_single

OUT = None
_LOG = []


def log(msg: str):
    print(msg, flush=True)
    _LOG.append(msg)


def _run(name: str, p: Params, seed: int, **kw) -> dict:
    out_dir = os.path.join(OUT, name, paths.run_name(p), f"seed{seed:03d}")
    return run_single(p, seed=seed, out_dir=out_dir, experiment=f"checks/{name}", **kw)


def check_invariants():
    p = Params(N=400, rho0=0.15, T_burn=0, T_measure=600)
    sim = Simulation(p, seed=1)
    for _ in range(600):
        sim.step(measure=True)
        sim.consistency_check()
    log("[ok] lattice invariants hold for 600 steps")


def check_no_coordination():
    p = Params(N=300, coordination=False, T_burn=0, T_measure=400)
    sim = Simulation(p, seed=2)
    for _ in range(400):
        sim.step()
        assert int((sim.state == sim.COORD).sum()) == 0
    log("[ok] without coordination no coordinator appears")


def check_zero_exclusion():
    p = Params(N=600, rho0=0.3, exclusion=0.0, coordination=False, T_burn=0, T_measure=300)
    sim = Simulation(p, seed=3)
    for _ in range(300):
        info = sim.step()
        assert int(info["cross_blocked"]) == 0
    log("[ok] λ=0 -> no crossing blocks")


def check_hard_exclusion():
    p = Params(N=2000, rho0=0.45, exclusion=1.0, coordination=False, T_burn=300, T_measure=300)
    s = _run("hard_exclusion", p, 4)["summary"]
    assert s["cross_blocked_frac"] > 0, s
    log(f"[ok] λ=1 density {s['density']:.2f}: total blocking {s['blocked_frac']:.3f}, "
        f"crossing {s['cross_blocked_frac']:.3f}")


def check_specialization():
    p = Params(N=200, coordination=False, T_burn=6000, T_measure=2000, record_every=50)
    res = _run("specialization", p, 5, record_timeseries=True)
    s = res["summary"]
    ts = res["timeseries"]
    burn_idx = [i for i, t in enumerate(ts["t"]) if t >= p.T_burn]
    stim = np.asarray(ts["s_task"])[burn_idx]
    log(f"[info] DOL_tasks={s['DOL_tasks']:.3f}  frac_focused={s['frac_focused']:.3f}  "
        f"top_share={s['mean_top_share']:.3f}  specialists={s['frac_specialist_any']:.3f}  "
        f"specialties/agent={s['mean_specialties']:.2f}")
    log(f"[diag] stimuli during measurement: min={stim.min():.1f}  max={stim.max():.1f}  "
        f"(s_base={p.s_init}, s_max={p.s_max}). If they touch 0 or s_max, the regulator oscillates.")
    log(f"[diag] mean efficiency w={s['mean_w']:.2f}  idle fraction={s['idle_frac']:.3f}")
    f_star = p.phi / (p.xi + p.phi)
    log(f"[diag] condition 1: f* = φ/(ξ+φ) = {f_star:.3f} vs even rotation 1/m = {1 / p.m:.3f} "
        f"-> {'OK' if f_star > 1 / p.m else 'VIOLATED'} (need f* > 1/m).")
    log(f"[diag] condition 2: threshold scale θ_max/s_init = {p.theta_max / p.s_init:.1f} "
        f"(want a few, not ≫ 1); per-step drop ξ = {p.xi} vs θ_max = {p.theta_max}.")
    if s["DOL_tasks"] <= 0.2:
        if s["mean_specialties"] > 0.5 * p.m:
            regime = ("GENERALIST regime: agents learned every task. Condition 1 fails in "
                      "practice: raise φ/ξ.")
        elif s["mean_specialties"] < 0.5 and s["idle_frac"] > 0.2:
            regime = ("DEAF regime: thresholds sit at θ_max, in the flat tail of the response "
                      "curve. Condition 2 fails: lower θ_max towards s_init and/or raise ξ "
                      "(keeping φ/ξ > 1/(m−1)).")
        else:
            regime = "MIXED: symmetry breaking too slow; try a longer T_burn or a larger ξ."
        log(f"[FAIL] no division of labour emerges. {regime} "
            "Run the calibration_xi_phi experiment rather than guessing single values.")
        raise AssertionError("DOL_tasks <= 0.2")
    log("[ok] division of labour emerges from identical agents")


def check_control_scaling():
    Ns = [64, 128, 256, 512, 1024]
    ys = []
    for N in Ns:
        vals = [_run("control_scaling", Params(N=N, area_exponent=1.0, T_burn=3000, T_measure=1000), s)
                ["summary"]["Y"] for s in range(2)]
        ys.append(np.mean(vals))
    beta, _, r2 = fit_loglog(Ns, ys)
    log(f"[info] control a=1: β = {beta:.3f} (R²={r2:.3f}); ≈ 1 is expected. "
        "A large deviation at small N is usually a finite-size effect.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--root", default=None, help=f"data root (default {paths.DATA_ROOT})")
    a = ap.parse_args()
    OUT = paths.ensure(os.path.join(paths.checks_dir(a.root), _dt.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")))
    log(f"checks output folder: {OUT}")
    try:
        check_invariants()
        check_no_coordination()
        check_zero_exclusion()
        if not a.quick:
            check_hard_exclusion()
            check_specialization()
            check_control_scaling()
        log("all checks finished")
    finally:
        with open(os.path.join(OUT, "report.txt"), "w") as fh:
            fh.write("\n".join(_LOG) + "\n")
        print(f"report saved to {os.path.join(OUT, 'report.txt')}")