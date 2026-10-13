"""Checks of the dynamics.py / params.py changes.
  ORIG_DIR : directory with the ORIGINAL params.py, dynamics.py, mechanism.py (e.g. a git worktree
             of the previous commit). NEW_DIR: the patched code (default ".").
  STANDINS : optional directory of stand-in traffic/backend/grid/bureaucracy modules. Leave it
             UNSET to test with your real modules (the stronger test).
  Example:  ORIG_DIR=../code_before_paper1 python tests/test_dynamics.py"""
import os, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
ORIG, NEW, ST = os.environ.get("ORIG_DIR"), os.environ.get("NEW_DIR", "."), os.environ.get("STANDINS")
BASE = list(sys.path)
MODS = ("dynamics", "params", "mechanism", "hubs", "metrics", "traffic", "backend", "grid", "bureaucracy")
def load(d):
    for k in MODS:
        sys.modules.pop(k, None)
    sys.path[:] = [os.path.abspath(d)] + ([os.path.abspath(ST)] if ST else []) + BASE
    import params, dynamics
    return params, dynamics
BEST = dict(learning_exponent=1.0, spillover=8.0, spillover_radius=4, route_length=2000.0, w_max=1e12)
def run(d, steps=400, **kw):
    params, dyn = load(d)
    p = params.Params(**dict(dict(BEST, exclusion=0.5, coordination=False, area_exponent=1.0,
                                  rho0=0.05, N=256), **kw))
    sim = dyn.Simulation(p, seed=3)
    out = [sim.step(measure=True) for _ in range(steps)]
    sim.consistency_check()
    return sim, out
if ORIG:
    for kw in (dict(centralization=0.0), dict(centralization=0.5), dict(centralization=1.0)):
        s0, o0 = run(ORIG, **kw)
        s1, o1 = run(NEW, **kw)
        assert np.array_equal(np.asarray(s0.exp), np.asarray(s1.exp))
        assert np.array_equal(np.asarray(s0.state), np.asarray(s1.state))
        assert [float(o["work"]) for o in o0] == [float(o["work"]) for o in o1]
        print("bit-identical to the original with default parameters:", kw)
else:
    print("ORIG_DIR not set: bit-identity test skipped")
d = {}
for eta in (0.0, 0.5, 1.0):
    s, o = run(NEW, steps=600, centralization=1.0, hub_count_exponent=eta, hub_spacing_ref=4.0)
    d[eta] = (s.n_hubs, np.mean([float(x["n_delivering"]) for x in o[200:]]))
print("eta -> (hub lines K, mean number delivering):", d)
assert d[0.0][0] == 1 and d[1.0][0] > d[0.5][0] > 1 and d[0.0][1] > d[0.5][1] > d[1.0][1]
params, dyn = load(NEW)
p = params.Params(**dict(BEST, exclusion=0.5, coordination=False, area_exponent=1.0, rho0=0.05,
                         N=256, centralization=1.0, learn_while_delivering=False))
sim = dyn.Simulation(p, seed=1)
checked = 0
for t in range(500):
    st = np.asarray(sim.state).copy()
    w = np.nonzero((st >= 0) & (st < sim.m))[0]
    hj, pr, D = np.asarray(sim.hub_job), np.asarray(sim.progress), np.asarray(sim.D)
    car = w[hj[w] & (pr[w] >= D[st[w]])]
    tk = st[car]
    before = np.asarray(sim.exp)[car, tk].copy()
    sim.step()
    after = np.asarray(sim.exp)[car, tk]
    assert np.allclose(after, before * np.float32(1 - p.exp_decay), rtol=1e-6), "gain while carrying"
    checked += car.size
print("no experience gain while carrying a job:", checked, "agent-steps checked")
for bad in (dict(hub_count_exponent=1.5), dict(hub_spacing_ref=0.5), dict(xi=-1.0)):
    try:
        params.Params(**bad).validate()
        raise SystemExit(f"validate() accepted {bad}")
    except ValueError:
        pass
print("test_dynamics: PASSED")