"""
meanfield.py — Mean-field prediction of output per agent y(ρ) and of the density elasticity
ε(ρ) = d ln y / d ln ρ  (experiment E2). Pure functions; no simulation needed.

Life cycle of a fully specialized agent (see dynamics.py):
    idle (off the lattice) for T_i steps  ->  works a job of D route units  ->  idle ...
Working agents attempt a move every other step (traffic light) and advance w per successful
move, so a job lasts T_j = 2 (D / w) / (1 − B) steps, B = probability that a move is blocked.
Experience is gained only while working on the task (a fraction 1 − q of the time) and decays
by ρ_E per step:
    e = (1 − q)(g + κ n) / ρ_E,   w = (1 + e)^b,
    n = ((2R+1)² − 1) · ρ (1 − q) / m      (same-task agents in the neighbourhood; only the
                                             working agents, a fraction 1 − q, are on the lattice)
    q = T_i / (T_i + T_j),                  y = D / (T_i + T_j)          (Little's law)
Closures used when no measurement is supplied ("closed" mean field):
    B   = λ ρ (1 − q) / 2       crossing blocks only (other layer occupied), free flow in lanes
    T_i = (1 − p_r) / p_r,  p_r = (1/m)(1 − ρ(1 − q)/2)
          an idle specialist meets its own task with probability ≈ 1/m per step (encounters ∝
          stimulus, balanced stimuli), accepts it (θ ≈ θ_min ≪ s), and enters if the drawn cell
          of its layer is free.
Analytic elasticity (T_i fixed), useful to read the result:
    ε = (1 − q)/(1 + b q) · [ b · κn/(g + κn) · dln n/dln ρ  +  dln(1 − B)/dln ρ ].
For q -> 0 and B -> 0 this is the crude estimate ε ≈ b κn/(g + κn).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MFParams:
    g: float = 1.0          # practice_gain
    kappa: float = 8.0      # spillover
    R: int = 4              # spillover_radius
    m: int = 4              # tasks
    rho_E: float = 0.002    # exp_decay
    b: float = 1.0          # learning_exponent
    D: float = 225000.0     # route_length
    lam: float = 0.5        # exclusion (crossing-block probability)
    w_max: float = 1e12

    @classmethod
    def from_params(cls, p: dict) -> "MFParams":
        """Build from a params dict (a results.jsonl row or Params.to_dict())."""
        return cls(g=float(p["practice_gain"]), kappa=float(p["spillover"]),
                   R=int(p["spillover_radius"]), m=int(p["m"]), rho_E=float(p["exp_decay"]),
                   b=float(p["learning_exponent"]), D=float(p["route_length"]),
                   lam=float(p["exclusion"]), w_max=float(p["w_max"]))


def neighbours(rho, q, P: MFParams):
    return ((2 * P.R + 1) ** 2 - 1) * rho * (1.0 - q) / P.m


def crude_epsilon(rho, P: MFParams):
    """The quick estimate b·κn/(g+κn) with n = ((2R+1)²−1)ρ/m (no idle time, no blocking)."""
    n = neighbours(np.asarray(rho, float), 0.0, P)
    return P.b * P.kappa * n / (P.g + P.kappa * n)


def solve(rho: float, P: MFParams, T_idle: float | None = None, B: float | None = None,
          tol: float = 1e-12, max_iter: int = 10000) -> dict:
    """Self-consistent state at density rho. T_idle / B: measured values, or None for the
    closures above. Returns y, w, q, n, B, T_i, T_j."""
    q = 0.0
    for _ in range(max_iter):
        n = neighbours(rho, q, P)
        e = (1.0 - q) * (P.g + P.kappa * n) / P.rho_E
        w = min((1.0 + e) ** P.b, P.w_max)
        Bq = P.lam * rho * (1.0 - q) / 2.0 if B is None else B
        Bq = min(max(Bq, 0.0), 0.999)
        if T_idle is None:
            pr = (1.0 / P.m) * max(1.0 - rho * (1.0 - q) / 2.0, 1e-6)
            Ti = (1.0 - pr) / pr
        else:
            Ti = float(T_idle)
        Tj = 2.0 * (P.D / w) / (1.0 - Bq)
        q_new = Ti / (Ti + Tj)
        if abs(q_new - q) < tol:
            q = q_new
            break
        q = 0.5 * q + 0.5 * q_new          # damped
    n = neighbours(rho, q, P)
    return dict(rho=rho, y=P.D / (Ti + Tj), w=w, q=q, n=n, B=Bq, T_i=Ti, T_j=Tj)


def epsilon(rho: float, P: MFParams, h: float = 0.01, **kw) -> float:
    """ε = d ln y / d ln ρ by central differences on the closed (or partly measured) model."""
    lo, hi = solve(rho * math.exp(-h), P, **kw), solve(rho * math.exp(h), P, **kw)
    return (math.log(hi["y"]) - math.log(lo["y"])) / (2 * h)


def analytic_epsilon(n, dlnn_dlnrho, q, dln1mB_dlnrho, P: MFParams):
    """ε = (1−q)/(1+bq)·[b·κn/(g+κn)·dln n/dln ρ + dln(1−B)/dln ρ] (T_i held fixed).
    Feed it measured n (mean_nsame), q (idle_frac) and B (blocked_frac) to get the
    'semi-empirical' prediction."""
    n = np.asarray(n, float)
    q = np.asarray(q, float)
    s = P.kappa * n / (P.g + P.kappa * n)
    return (1 - q) / (1 + P.b * q) * (P.b * s * dlnn_dlnrho + dln1mB_dlnrho)


def curve(rhos, P: MFParams, **kw):
    """y and ε of the closed mean field on a grid of densities (for figures)."""
    rows = []
    for r in rhos:
        st = solve(float(r), P, **kw)
        st["eps"] = epsilon(float(r), P, **kw)
        st["eps_crude"] = float(crude_epsilon(r, P))
        rows.append(st)
    return rows


if __name__ == "__main__":       # python meanfield.py
    P = MFParams()
    for r in (0.01, 0.018, 0.03, 0.0386, 0.05, 0.082, 0.1, 0.2, 0.4):
        st = solve(r, P)
        print(f"rho={r:<7g} y={st['y']:8.1f} w={st['w']:8.1f} q={st['q']:.3f} n={st['n']:.2f} "
              f"B={st['B']:.3f}  eps={epsilon(r, P):.3f}  crude={crude_epsilon(r, P):.3f}")
