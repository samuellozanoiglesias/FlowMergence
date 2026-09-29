"""
dynamics.py — Simulation state and one time step.

Per-agent state
    state[i]   : −1 idle | j ∈ [0, m) doing task j | m coordinating
    pos_y/x[i] : lattice position (worker) or zone centre (coordinator)
    progress[i]: progress of the current job (route units)
    theta[i,:] : response thresholds for the m tasks + coordination        [T98]
    exp[i,:]   : experience in each task                                  [NEW]

Order of one step t
    1. BML movement (one direction per step if traffic_light).
    2. Coordinator bookkeeping: active if there was signal in its zone; relocation.
    3. Learning: practice + spillover; forgetting for everyone.
    4. Job completion: output; the agent leaves the lattice. With centralization c > 0,
       a fraction c of jobs must first be delivered to the hub line (L//2).
    5. Quitting (work with p_quit_work, coordination with p_quit_coord).
    6. Stimuli: composition -> s_j; signal rate -> s_C.
    7. Recruitment of idle agents: each one "encounters" one role (with probability ∝
       stimulus by default, see Params.encounter) and takes it with probability
       T(s, θ) [B96]; workers try to enter the lattice.
    8. Threshold reinforcement with the role each agent PERFORMED this step [T98].
    9. Coordinator coverage -> block probability for the next step.

Cost per step: O(N·m + m·L²); since L² ∝ N^a with a ≤ 1, this is O(N·m).
Memory: O(N·m + L²).
"""
from __future__ import annotations

import numpy as np

import mechanism as mech
import traffic
from backend import get_xp, to_numpy
from bureaucracy import Bureaucracy
from grid import count_grid
from params import Params

IDLE = -1


class Simulation:
    def __init__(self, params: Params, seed: int = 0):
        params.validate()
        self.p = p = params
        self.xp = xp = get_xp(p.backend)
        self.rs = xp.random.RandomState(int(seed) % (2 ** 32))
        self.N, self.m, self.L = int(p.N), int(p.m), int(p.L)
        N, m, L = self.N, self.m, self.L
        self.COORD = m
        self.coordination = bool(p.coordination)
        self.fixed_coord = self.coordination and p.coord_mode == "fixed"
        # in fixed mode coordination is not a recruitable role
        self.n_roles = m + 1 if (self.coordination and not self.fixed_coord) else m

        # --- agents
        self.state = xp.full(N, IDLE, dtype=xp.int32)
        self.pos_y = xp.zeros(N, dtype=xp.int32)
        self.pos_x = xp.zeros(N, dtype=xp.int32)
        self.progress = xp.zeros(N, dtype=xp.float32)
        self.theta = xp.full((N, m + 1), p.theta_init, dtype=xp.float32)
        if p.theta_init_noise > 0:
            span = (p.theta_max - p.theta_min) * p.theta_init_noise
            noise = (self.rs.random_sample((N, m + 1)) * 2.0 - 1.0) * span
            self.theta += noise.astype(xp.float32)
            xp.clip(self.theta, p.theta_min, p.theta_max, out=self.theta)
        self.exp = xp.zeros((N, m), dtype=xp.float32)

        # --- tasks
        self.task_dir_host = [j % 2 for j in range(m)]
        self.task_dir = xp.asarray(self.task_dir_host, dtype=xp.int32)
        self.D = xp.asarray(p.route_lengths(), dtype=xp.float32)
        self.pi = xp.asarray(p.demand_shares(), dtype=xp.float32)

        # --- stimuli and signals
        self.s = xp.full(m, p.s_init, dtype=xp.float32)
        self.s_coord = xp.zeros((), dtype=xp.float32)
        self.share = self.pi.copy()
        self.blocked_rate = xp.zeros((), dtype=xp.float32)
        self.signal_rate = xp.zeros((), dtype=xp.float32)

        # --- space and bureaucracy
        self.occ = xp.full((2, L, L), -1, dtype=xp.int32)
        self.bureau = (Bureaucracy(xp, L, p.coord_radius, p.coord_strength, p.coord_conflict, p.exclusion)
                       if self.coordination else None)
        self.block_grid = None          # None => uniform λ (no coordinators)
        self.covered_frac = 0.0

        # --- centralization: which jobs must be delivered to the hub line
        self.hub_job = xp.zeros(N, dtype=bool)
        self.hub_line = L // 2

        # --- fixed bureaucracy: permanent coordinators from t = 0
        if self.fixed_coord:
            M0 = int(round(p.coord_fixed_frac * N))
            if M0 > 0:
                idx = self.rs.permutation(N)[:M0]
                self.state[idx] = self.COORD
                self.pos_y[idx] = self.rs.randint(0, L, size=M0).astype(xp.int32)
                self.pos_x[idx] = self.rs.randint(0, L, size=M0).astype(xp.int32)

        # --- per-agent time budget: column 0 = idle, 1..m = tasks, m+1 = coordination
        self.time_budget = xp.zeros((N, m + 2), dtype=xp.int32)
        self._arangeN = xp.arange(N)
        self.t = 0

    # ================================================================== one step
    def step(self, measure: bool = False) -> dict:
        xp, p, rs = self.xp, self.p, self.rs
        m, L = self.m, self.L
        st = self.state
        role_start = st.copy()

        # ---------------------------------------------------------- 1. movement
        working = xp.nonzero((st >= 0) & (st < m))[0]
        wdir = self.task_dir[st[working]]
        dirs = (self.t % 2,) if p.traffic_light else (0, 1)
        snapshot = None if p.traffic_light else self.occ.copy()
        bp = float(p.exclusion) if self.block_grid is None else self.block_grid

        n_att = 0
        n_blk = xp.zeros((), dtype=xp.int64)
        n_cross = xp.zeros((), dtype=xp.int64)
        sig_y_parts, sig_x_parts = [], []
        event_grid = xp.zeros((L, L), dtype=xp.int64) if self.coordination else None
        for d in dirs:
            movers = working[wdir == d]
            if movers.size == 0:
                continue
            cross_layer = self.occ[1 - d] if snapshot is None else snapshot[1 - d]
            res = traffic.move_layer(xp, rs, self.occ, d, cross_layer, movers,
                                     self.pos_y, self.pos_x, L, bp)
            ids = movers[res.moved]
            if ids.size:
                self.progress[ids] += mech.efficiency(self.exp[ids, st[ids]], p.learning_exponent, p.w_max)
            blocked = ~res.moved
            n_att += int(movers.size)
            n_blk = n_blk + blocked.sum()
            n_cross = n_cross + res.cross_blocked.sum()
            sig = res.cross_blocked if p.coord_signal == "cross" else blocked
            sy, sx = res.y0[sig], res.x0[sig]
            sig_y_parts.append(sy)
            sig_x_parts.append(sx)
            if event_grid is not None and sy.size:
                event_grid += count_grid(xp, sy, sx, L)
        if sig_y_parts:
            sig_y, sig_x = xp.concatenate(sig_y_parts), xp.concatenate(sig_x_parts)
        else:
            sig_y = sig_x = xp.zeros(0, dtype=xp.int32)
        n_sig = int(sig_y.size)

        # ---------------------------------------------------------- 2. bureaucracy: active / inactive
        M, M_active = 0, 0
        if self.coordination:
            coords = xp.nonzero(role_start == self.COORD)[0]
            M = int(coords.size)
            if M:
                load = self.bureau.zone_load(event_grid, self.pos_y[coords], self.pos_x[coords])
                active = load > 0
                M_active = active.sum()
                if p.coord_relocate_prob > 0 and n_sig > 0:
                    reloc = coords[(~active) & (rs.random_sample(M) < p.coord_relocate_prob)]
                    if reloc.size:
                        cy, cx = self.bureau.place(rs, int(reloc.size), sig_y, sig_x, "congestion")
                        self.pos_y[reloc] = cy
                        self.pos_x[reloc] = cx

        # ---------------------------------------------------------- 3. learning
        self.exp *= (1.0 - p.exp_decay)
        mean_w, mean_n = 0.0, 0.0
        if working.size:
            tasks = st[working]
            n_same = traffic.same_task_counts(xp, working, tasks, self.pos_y, self.pos_x,
                                              m, L, p.spillover_radius)
            self.exp[working, tasks] += (p.practice_gain + p.spillover * n_same).astype(xp.float32)
            mean_w = mech.efficiency(self.exp[working, tasks], p.learning_exponent, p.w_max).mean()
            mean_n = n_same.mean()

        # ---------------------------------------------------------- 4. finished jobs
        comp = xp.zeros(m, dtype=xp.int64)
        n_delivering = 0
        if working.size:
            wt_ = st[working]
            work_done = self.progress[working] >= self.D[wt_]
            if p.centralization > 0:
                # hub jobs only count once the agent is ON the hub line (column for
                # horizontal agents, row for vertical ones)
                at_hub = xp.where(self.task_dir[wt_] == 0,
                                  self.pos_x[working] == self.hub_line,
                                  self.pos_y[working] == self.hub_line)
                hubj = self.hub_job[working]
                n_delivering = (work_done & hubj & ~at_hub).sum()
                work_done = work_done & (~hubj | at_hub)
            done = working[work_done]
            if done.size:
                dt = st[done]
                comp = xp.bincount(dt, minlength=m)[:m]
                traffic.remove(self.occ, self.task_dir[dt], self.pos_y[done], self.pos_x[done])
                st[done] = IDLE
                self.progress[done] = 0.0
        work_units = (comp * self.D).sum()      # output of this step (route units)

        # ---------------------------------------------------------- 5. quitting
        if p.p_quit_work > 0:
            still = xp.nonzero((st >= 0) & (st < m))[0]
            if still.size:
                q = still[rs.random_sample(int(still.size)) < p.p_quit_work]
                if q.size:
                    traffic.remove(self.occ, self.task_dir[st[q]], self.pos_y[q], self.pos_x[q])
                    st[q] = IDLE
                    self.progress[q] = 0.0
        if self.coordination and not self.fixed_coord:
            cs = xp.nonzero(st == self.COORD)[0]
            if cs.size:
                st[cs[rs.random_sample(int(cs.size)) < p.p_quit_coord]] = IDLE

        # ---------------------------------------------------------- 6. stimuli
        inst = xp.where(work_units > 0, comp * self.D / xp.maximum(work_units, 1e-9), self.share)
        self.share = ((1.0 - p.share_ema) * self.share + p.share_ema * inst).astype(xp.float32)
        self.s = mech.update_task_stimuli(self.share, self.pi, p.s_gain, p.s_init, p.s_max)
        a = p.blocked_ema
        denom = max(n_att, 1)
        self.blocked_rate = xp.asarray((1 - a) * self.blocked_rate + a * (n_blk / denom), dtype=xp.float32)
        self.signal_rate = xp.asarray((1 - a) * self.signal_rate + a * (n_sig / denom), dtype=xp.float32)
        if self.coordination:
            self.s_coord = mech.coordination_stimulus(self.signal_rate, p.coord_stimulus_gain, p.s_max)

        # ---------------------------------------------------------- 7. recruitment
        n_fail, n_new_coord = 0, 0
        idle = xp.nonzero(st == IDLE)[0]
        if idle.size:
            s_all = xp.concatenate([self.s, xp.asarray(self.s_coord, dtype=xp.float32).reshape(1)])
            if p.encounter == "stimulus":
                # encounter probability ∝ stimulus: silent roles are never encountered
                w_enc = s_all[:self.n_roles].astype(xp.float64)
                cdf = xp.cumsum(w_enc) / xp.maximum(w_enc.sum(), 1e-12)
                u = rs.random_sample(int(idle.size))
                k = xp.minimum(xp.searchsorted(cdf, u, side="right"), self.n_roles - 1)
            else:
                k = rs.randint(0, self.n_roles, size=int(idle.size))
            prob = mech.response_probability(s_all[k], self.theta[idle, k])
            eng = rs.random_sample(int(idle.size)) < prob
            cand, kc = idle[eng], k[eng]
            if cand.size:
                wsel = kc < m
                wid, wt = cand[wsel], kc[wsel]
                if wid.size:
                    sel, ey, ex, n_fail = traffic.try_enter(xp, rs, self.occ, wid, self.task_dir[wt], L)
                    ids = wid[sel]
                    st[ids] = wt[sel].astype(xp.int32)
                    self.pos_y[ids] = ey
                    self.pos_x[ids] = ex
                    self.progress[ids] = 0.0
                    if p.centralization > 0 and ids.size:
                        self.hub_job[ids] = rs.random_sample(int(ids.size)) < p.centralization
                cid = cand[~wsel]
                if cid.size:
                    cy, cx = self.bureau.place(rs, int(cid.size), sig_y, sig_x, p.coord_placement)
                    st[cid] = self.COORD
                    self.pos_y[cid] = cy
                    self.pos_x[cid] = cx
                    n_new_coord = int(cid.size)

        # ---------------------------------------------------------- 8. threshold reinforcement
        act = xp.nonzero(role_start >= 0)[0]
        mech.update_thresholds(self.theta, act, role_start[act], p.xi, p.phi, p.theta_min, p.theta_max)

        # ---------------------------------------------------------- 9. coverage for t+1
        if self.coordination:
            cs = xp.nonzero(st == self.COORD)[0]
            if cs.size:
                cov = self.bureau.coverage(self.pos_y[cs], self.pos_x[cs])
                self.block_grid = self.bureau.block_probability(cov)
                self.covered_frac = (cov > 0).mean()
            else:
                self.block_grid = None
                self.covered_frac = 0.0

        if measure:
            self.time_budget[self._arangeN, role_start + 1] += 1
        self.t += 1

        return {
            "work": work_units,
            "completions": comp.sum(),
            "attempts": n_att,
            "blocked": n_blk,
            "cross_blocked": n_cross,
            "signal": n_sig,
            "entry_fail": n_fail,
            "n_working": int(working.size),
            "n_idle": (role_start == IDLE).sum(),
            "M": M,
            "M_active": M_active,
            "new_coord": n_new_coord,
            "mean_w": mean_w,
            "mean_nsame": mean_n,
            "blocked_rate": self.blocked_rate,
            "signal_rate": self.signal_rate,
            "s_coord": self.s_coord,
            "covered_frac": self.covered_frac,
            "n_delivering": n_delivering,
        }

    # ================================================================== utilities
    def snapshot(self) -> dict:
        """Visual state (on CPU): task in each layer (−1 = empty) and coordinator centres."""
        xp = self.xp
        layers = []
        for d in (0, 1):
            o = self.occ[d]
            tg = xp.where(o >= 0, self.state[xp.maximum(o, 0)], -1).astype(xp.int8)
            layers.append(to_numpy(tg))
        cs = xp.nonzero(self.state == self.COORD)[0]
        return {"t": self.t, "h": layers[0], "v": layers[1],
                "cy": to_numpy(self.pos_y[cs]).astype(np.int32),
                "cx": to_numpy(self.pos_x[cs]).astype(np.int32)}

    def consistency_check(self):
        """Invariants: each worker sits exactly in its cell and layer; no ghosts."""
        st = to_numpy(self.state)
        occ = to_numpy(self.occ)
        py, px = to_numpy(self.pos_y), to_numpy(self.pos_x)
        dirs = np.asarray(self.task_dir_host)
        working = np.nonzero((st >= 0) & (st < self.m))[0]
        on_grid = occ[occ >= 0]
        assert on_grid.size == working.size, f"occupancy {on_grid.size} != workers {working.size}"
        assert np.unique(on_grid).size == on_grid.size, "an agent appears twice on the lattice"
        if working.size:
            assert np.all(occ[dirs[st[working]], py[working], px[working]] == working), "inconsistent position"
        assert np.all(st >= -1) and np.all(st <= self.m), "state out of range"