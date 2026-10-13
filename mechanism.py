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

7. Delivery choice (option B) [NEW]: update_delivery_choice (Bush–Mosteller, one shared
   aspiration; choice_rule = 'bush_mosteller') or update_delivery_values (payoff estimate
   per mode + logit on their log ratio; choice_rule = 'logit').
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


def update_delivery_choice(p_hub, aspiration, chose_hub, rate, lr, asp_lr, explore):
    """[NEW] Bush–Mosteller reinforcement of the probability of choosing hub delivery.

    For each agent that just finished a job, with payoff rate r = value / duration and
    aspiration A (the agent's running average payoff rate):
        r ≥ A: the chosen mode is reinforced;  r < A: it is weakened.
        step  η_eff = lr · min(1, |r − A| / A)       (bigger surprise, bigger step)
        reinforce hub   / weaken local : p ← p + η_eff (1 − p)
        reinforce local / weaken hub   : p ← p − η_eff p
    then A ← (1 − asp_lr) A + asp_lr · r, and p is kept in [explore, 1 − explore] so every
    agent keeps sampling both modes. Agents with no aspiration yet (A ≤ 0) only set A = r.
    All arguments are 1-D arrays over the agents that finished (except the scalars).
    Returns (new_p_hub, new_aspiration).
    """
    xp = get_array_module(p_hub)
    has_a = aspiration > 0
    rel = xp.abs(rate - aspiration) / xp.maximum(aspiration, 1e-12)
    eta = lr * xp.minimum(rel, 1.0) * has_a
    good = rate >= aspiration
    towards_hub = (chose_hub & good) | (~chose_hub & ~good)
    p_new = xp.where(towards_hub, p_hub + eta * (1.0 - p_hub), p_hub - eta * p_hub)
    p_new = xp.clip(p_new, explore, 1.0 - explore)
    a_new = xp.where(has_a, (1.0 - asp_lr) * aspiration + asp_lr * rate, rate)
    return p_new.astype(xp.float32), a_new.astype(xp.float32)


def update_delivery_values(p_hub, q_hub, q_loc, chose_hub, rate, lr, temperature, explore):
    """[NEW] Value-based (logit) learning of the hub-delivery probability.

    Alternative to update_delivery_choice. Bush–Mosteller with ONE shared aspiration has a
    noise-dependent floor and ceiling (a hub that pays 10x less is still chosen 6–20% of
    the time; one that always pays more plateaus at 0.77–0.94), because disappointing
    LOCAL jobs push p_hub up no matter how bad the hub is. Here each agent keeps a running
    payoff-rate estimate PER MODE and chooses by a logit on their log ratio:
        Q_mode <- (1 − lr) Q_mode + lr · r      for the mode just used (Q = r the first time)
        p_hub   = clip( 1 / (1 + exp(−ln(Q_hub / Q_loc) / T)), explore, 1 − explore )
    The log ratio makes T scale-free (payoffs grow with D and π): T = 0.1 means a hub
    paying 10% more gives p_hub ≈ 0.73, 30% more ≈ 0.93. The first job is chosen with
    choice_p0; an agent that has tried only one mode then tries the other (p = 1 − explore
    towards it); after that the logit applies. Q = 0 means "never tried".
    Agents still learn only from their own experience: nobody sees the queue they add.
    Returns (new_p_hub, new_q_hub, new_q_loc).
    """
    xp = get_array_module(p_hub)
    rate = xp.maximum(rate, 1e-12)
    upd_h = xp.where(q_hub > 0, (1.0 - lr) * q_hub + lr * rate, rate)
    upd_l = xp.where(q_loc > 0, (1.0 - lr) * q_loc + lr * rate, rate)
    q_hub = xp.where(chose_hub, upd_h, q_hub)
    q_loc = xp.where(chose_hub, q_loc, upd_l)
    both = (q_hub > 0) & (q_loc > 0)
    z = xp.log(xp.maximum(q_hub, 1e-12) / xp.maximum(q_loc, 1e-12)) / max(temperature, 1e-9)
    p_logit = 1.0 / (1.0 + xp.exp(-xp.clip(z, -50.0, 50.0)))
    # an agent that has tried only one mode tries the other one next ("try each once",
    # as in bandit algorithms); otherwise extreme starts would need ~50–80 jobs to relax,
    # more than agents complete at large N
    p_try = xp.where(q_hub > 0, 0.0, 1.0)
    p_new = xp.clip(xp.where(both, p_logit, xp.where((q_hub > 0) | (q_loc > 0), p_try, p_hub)),
                    explore, 1.0 - explore)
    return p_new.astype(xp.float32), q_hub.astype(xp.float32), q_loc.astype(xp.float32)


def crossing_block_probability(exclusion, coverage, strength, conflict):
    """[NEW] Per-cell crossing-block probability given the coverage k (L×L array)."""
    xp = get_array_module(coverage)
    k = coverage.astype(xp.float32)
    b = xp.power(xp.float32(1.0 - strength), k)
    if conflict > 0:
        b = b + conflict * xp.maximum(k - 1.0, 0.0)
    return xp.clip(exclusion * b, 0.0, 1.0).astype(xp.float32)