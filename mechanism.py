"""
mechanism.py — Microscopic rules (what each agent does). Pure functions, no state.

1. Response to stimuli [B96, eq. 1]
       T(s, θ) = s² / (s² + θ²)

2. Threshold reinforcement [T98, eqs. 2-3], in discrete time (Δt = 1):
       θ_ij <- θ_ij − ξ   if i performs task j during this step
       θ_ij <- θ_ij + φ   otherwise
   bounded in [θ_min, θ_max]. Coordination is one more "task" (column m), so
   bureaucratic ossification comes out of the SAME rule: whoever coordinates a lot
   ends up with θ_iC ≈ θ_min and takes the role again at the slightest signal.

3. Learning by doing [NEW]. T98 keeps the efficiency α fixed and only notes that
   "α can vary as a result of specialization". Here it is modelled:
       e_ij <- (1−ρ) e_ij + [i does j]·(g + κ·n_ij)
       w_ij  = min((1 + e_ij)^b, w_max)                   (Wright's law)
   where n_ij is the number of OTHER agents doing j within radius R (spillover).

4. Task stimulus [NEW]: proportional to the relative composition mismatch
       s_j = clip(s_base · (1 + δ·(π_j − share_j)/π_j), 0, s_max)
   It does not cap total output, it only allocates it. (An earlier version was a
   pure integrator; it oscillated between 0 and s_max and destroyed specialization.)

5. Coordination stimulus [NEW]: s_C = min(g_C · signal_rate, s_max).

6. Crossing-block probability with coordinator coverage k [NEW, on top of BML]:
       b(k) = λ · [(1−q)^k + η·max(k−1, 0)]    clipped to [0, 1]
"""
from __future__ import annotations

from backend import get_array_module


def response_probability(s, theta):
    """[B96] Probability of taking up a task given stimulus s and threshold θ."""
    s2 = s * s
    return s2 / (s2 + theta * theta + 1e-12)


def update_thresholds(theta, act_idx, act_role, xi, phi, tmin, tmax):
    """[T98] In-place reinforcement. act_idx/act_role: active agents and the role they performed.

    Cost O(N·(m+1)). Every threshold rises by φ, and the one of the performed role
    drops by ξ+φ, i.e. −ξ in net terms: exactly T98 eq. (3) with x_ij ∈ {0,1}.
    """
    xp = get_array_module(theta)
    theta += phi
    if act_idx.size:
        theta[act_idx, act_role] -= (xi + phi)
    xp.clip(theta, tmin, tmax, out=theta)


def efficiency(exp, b, w_max):
    """[NEW] Efficiency (route units per move) from experience."""
    xp = get_array_module(exp)
    return xp.minimum(xp.power(1.0 + exp, b), w_max).astype(xp.float32)


def update_task_stimuli(share, pi, gain, s_base, s_max):
    """[NEW] Stimulus PROPORTIONAL to the relative composition mismatch:
        s_j = clip(s_base · (1 + gain · (π_j − share_j) / π_j), 0, s_max)
    No integration, so it can neither wind up nor oscillate. Does not cap total output.
    """
    xp = get_array_module(share)
    rel = (pi - share) / xp.maximum(pi, 1e-9)
    return xp.clip(s_base * (1.0 + gain * rel), 0.0, s_max).astype(xp.float32)


def coordination_stimulus(rate, gain, s_max):
    """[NEW] Coordination stimulus, proportional to the signal (blocking) rate."""
    xp = get_array_module(rate)
    return xp.minimum(xp.asarray(gain * rate, dtype=xp.float32), xp.float32(s_max))


def crossing_block_probability(exclusion, coverage, strength, conflict):
    """[NEW] Per-cell crossing-block probability given the coverage k (L×L array)."""
    xp = get_array_module(coverage)
    k = coverage.astype(xp.float32)
    b = xp.power(xp.float32(1.0 - strength), k)
    if conflict > 0:
        b = b + conflict * xp.maximum(k - 1.0, 0.0)
    return xp.clip(exclusion * b, 0.0, 1.0).astype(xp.float32)
