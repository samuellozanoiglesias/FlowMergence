"""Tests of meanfield.py (pure). From the code directory:  python tests/test_meanfield.py"""
import os, sys
sys.path.insert(0, os.environ.get("CODE_DIR", "."))
import meanfield as m
P = m.MFParams()
for r in (0.01, 0.04, 0.1, 0.4):                       # the fixed point satisfies its equations
    st = m.solve(r, P)
    n = m.neighbours(r, st["q"], P)
    w = (1 + (1 - st["q"]) * (P.g + P.kappa * n) / P.rho_E) ** P.b
    Tj = 2 * P.D / w / (1 - st["B"])
    assert abs(st["q"] - st["T_i"] / (st["T_i"] + Tj)) < 1e-9
    assert abs(st["y"] - P.D / (st["T_i"] + Tj)) < 1e-6 * st["y"]
for r in (0.02, 0.05):                                 # no idle time, no blocking -> crude formula
    e = m.epsilon(r, P, T_idle=0.0, B=0.0)
    assert abs(e - float(m.crude_epsilon(r, P))) < 2e-3, (e, m.crude_epsilon(r, P))
st = m.solve(0.05, P, T_idle=3.0, B=0.0)               # analytic form vs numerical derivative
e_an = float(m.analytic_epsilon(st["n"], 1.0, st["q"], 0.0, P))
e_num = m.epsilon(0.05, P, T_idle=3.0, B=0.0)
assert abs(e_an - e_num) < 0.02, (e_an, e_num)
assert abs(m.epsilon(0.05, m.MFParams(b=0.0, D=50.0), T_idle=3.0, B=0.0)) < 1e-6   # b = 0
print("test_meanfield: PASSED")
