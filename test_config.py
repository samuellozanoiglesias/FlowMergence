"""config.py: every OLD experiment builds the same grids and seeds; MODEL pins equal the defaults;
every new experiment builds and validates. Needs your paths.py on the path.
  ORIG_DIR=../code_before_paper1 python tests/test_config.py"""
import os, sys, warnings
warnings.simplefilter("ignore")
ORIG, NEW = os.environ["ORIG_DIR"], os.environ.get("NEW_DIR", ".")
BASE = list(sys.path)
def load(d):
    for k in ("config", "params"):
        sys.modules.pop(k, None)
    sys.path[:] = [os.path.abspath(d)] + BASE
    import config
    return config
co = load(ORIG)
old = {n: co.build_jobs(n, "T") for n in co.EXPERIMENTS}
cn = load(NEW)
from params import DEFAULTS
NEWF = {"hub_count_exponent", "hub_spacing_ref", "learn_while_delivering"}
for n, jobs in old.items():
    jn = cn.build_jobs(n, "T")
    assert len(jn) == len(jobs), n
    for a, b in zip(jobs, jn):
        assert {k: v for k, v in b["params"].items() if k not in NEWF} == a["params"], n
        assert a["seed"] == b["seed"], n
for k, v in cn.MODEL.items():
    assert getattr(DEFAULTS, k) == v, k
new = [n for n in cn.EXPERIMENTS if n not in old]
for n in new:
    cn.build_jobs(n, "T")
print(f"test_config: PASSED ({len(old)} old experiments unchanged, {len(new)} new ones valid)")