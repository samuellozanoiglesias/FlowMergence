"""
bureaucracy.py — Endogenous coordination ("crossing police") and its bookkeeping.

Idea
----
Coordinating is one more task within the threshold model (column m of θ). Any idle
agent may "encounter" the coordination stimulus s_C and take the role with
probability T(s_C, θ_iC). Nobody sets how many coordinators there are: M(N) EMERGES.

While coordinating, an agent:
  * does NOT produce, but counts in N, so output per agent goes down;
  * covers a square zone of radius r around its centre;
  * in cells with coverage k, the crossing-block probability goes from λ to
        b(k) = λ·[(1−q)^k + η·max(k−1,0)]
    with q = individual effectiveness and η = conflict between overlapping coordinators.

ACTIVE / INACTIVE coordinator (measured every step)
---------------------------------------------------
  ACTIVE   : at least one signal event (block) happened in its zone this step,
             i.e. it is actually doing something.
  INACTIVE : its zone was quiet; it is pure bureaucratic overhead.
The inactive fraction measures OSSIFICATION: through T98 threshold reinforcement,
whoever coordinates a lot ends up with θ_iC ≈ θ_min and takes the role again as soon
as s_C > 0, even when it is not needed. Relocation (coord_relocate_prob) is the
antidote: a bureaucracy that moves to where the problem is.

Scaling observables: M ∝ N^γ, M_active ∝ N^γa, M_inactive/M.
Cost: O(L²) per step (box sums) + O(M).
"""
from __future__ import annotations

import mechanism as mech
from grid import box_sum_periodic, count_grid


class Bureaucracy:
    def __init__(self, xp, L, radius, strength, conflict, exclusion):
        self.xp = xp
        self.L = int(L)
        self.radius = int(max(0, min(radius, (L - 1) // 2)))
        self.strength = float(strength)
        self.conflict = float(conflict)
        self.exclusion = float(exclusion)

    # ------------------------------------------------------------------ space
    def coverage(self, cy, cx):
        """k(cell) = number of coordinators whose zone contains the cell."""
        return box_sum_periodic(count_grid(self.xp, cy, cx, self.L), self.radius)

    def block_probability(self, coverage):
        """L×L grid with the crossing-block probability of each cell."""
        return mech.crossing_block_probability(self.exclusion, coverage, self.strength, self.conflict)

    def zone_load(self, event_grid, cy, cx):
        """Signal events inside each coordinator's zone (for active/inactive)."""
        return box_sum_periodic(event_grid, self.radius)[cy, cx]

    # ------------------------------------------------------------------ placement
    def place(self, rs, n, sig_y, sig_x, mode):
        """Centres for n new or relocated coordinators.

        'congestion': chosen at random among this step's signal events (coordinators
        go where the problems are); if there are none, at random.
        """
        xp = self.xp
        if mode == "congestion" and sig_y.size > 0:
            pick = rs.randint(0, int(sig_y.size), size=n)
            return sig_y[pick].astype(xp.int32), sig_x[pick].astype(xp.int32)
        return (rs.randint(0, self.L, size=n).astype(xp.int32),
                rs.randint(0, self.L, size=n).astype(xp.int32))
