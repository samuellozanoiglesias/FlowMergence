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
    4. Job completion: output; the agent leaves the lattice. Hub jobs (a fraction c, or
       chosen by the agent when delivery_choice='learned') must first be delivered to the
       hub line (L//2). With learned choice, each finished job updates the agent's p_hub
       (Bush–Mosteller, or per-mode payoff estimates + logit if choice_rule = 'logit').
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

import hubs
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
        # polycentric delivery (E9): K evenly spaced hub lines per direction, K = 1 (default)
        # is exactly the single line L // 2. With K = 1 the original comparison is used, so
        # default runs are bit-identical to the original model.
        eta = float(getattr(p, "hub_count_exponent", 0.0))
        self.hub_positions = hubs.hub_positions(L, eta, float(getattr(p, "hub_spacing_ref", 16.0)))
        self.n_hubs = int(self.hub_positions.size)
        self.hub_is = xp.asarray(hubs.hub_mask(L, eta, float(getattr(p, "hub_spacing_ref", 16.0))))
        self.learn_while_delivering = bool(getattr(p, "learn_while_delivering", True))
        self.learned_choice = p.delivery_choice == "learned"
        self.any_hub = self.learned_choice or p.centralization > 0
        self.job_start = xp.zeros(N, dtype=xp.int32)
        self.n_jobs = xp.zeros(N, dtype=xp.int32)      # jobs completed per agent (whole run)
        if self.learned_choice:
            self.p_hub = xp.full(N, p.choice_p0, dtype=xp.float32)
            xp.clip(self.p_hub, p.choice_explore, 1.0 - p.choice_explore, out=self.p_hub)
            self.aspiration = xp.zeros(N, dtype=xp.float32)   # 0 = no payoff seen yet
            # choice_rule: 'bush_mosteller' (default, shared aspiration) or 'logit'
            # (payoff-rate estimate per mode; see mechanism.update_delivery_values)
            self.choice_rule = getattr(p, "choice_rule", "bush_mosteller")
            if self.choice_rule == "logit":
                self.q_hub = xp.zeros(N, dtype=xp.float32)    # 0 = mode never tried
                self.q_loc = xp.zeros(N, dtype=xp.float32)
            # what the choice rule learns from: 'raw' payoff rate (value / duration) or
            # 'per_skill' = raw / own efficiency w at completion, which removes the drift of
            # all payoffs with experience (a rarely tried mode otherwise keeps a stale,
            # novice-level estimate: competency trap)
            self.choice_payoff = getattr(p, "choice_payoff", "raw")
            # step at which each mode was last tried (−1 = never): staleness diagnostic
            self.t_tried_hub = xp.full(N, -1, dtype=xp.int32)
            self.t_tried_loc = xp.full(N, -1, dtype=xp.int32)
        else:
            self.p_hub = None
            self.aspiration = None

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
            gain = (p.practice_gain + p.spillover * n_same).astype(xp.float32)
            if self.any_hub and not self.learn_while_delivering:
                # E5 ablation: an agent whose work is done and that is only carrying the job to
                # the hub gains no experience (it is still counted as a spillover source)
                carrying = self.hub_job[working] & (self.progress[working] >= self.D[tasks])
                gain = xp.where(carrying, xp.float32(0.0), gain).astype(xp.float32)
            self.exp[working, tasks] += gain
            mean_w = mech.efficiency(self.exp[working, tasks], p.learning_exponent, p.w_max).mean()
            mean_n = n_same.mean()

        # ---------------------------------------------------------- 4. finished jobs
        comp = xp.zeros(m, dtype=xp.int64)
        n_delivering = 0
        work_value, n_done_hub, n_start_hub, n_start = 0.0, 0, 0, 0
        # realized PRIVATE payoff rates (value / job duration) of the jobs finished this step,
        # by delivery mode: the fresh private return of each mode, for any rule or fixed c
        r_hub_sum, r_hub_log, r_hub_n = 0.0, 0.0, 0
        r_loc_sum, r_loc_log, r_loc_n = 0.0, 0.0, 0
        if working.size:
            wt_ = st[working]
            work_done = self.progress[working] >= self.D[wt_]
            if self.any_hub:
                # hub jobs only count once the agent is ON the hub line (column for
                # horizontal agents, row for vertical ones)
                if self.n_hubs == 1:
                    at_hub = xp.where(self.task_dir[wt_] == 0,
                                      self.pos_x[working] == self.hub_line,
                                      self.pos_y[working] == self.hub_line)
                else:
                    at_hub = xp.where(self.task_dir[wt_] == 0,
                                      self.hub_is[self.pos_x[working]],
                                      self.hub_is[self.pos_y[working]])
                hubj = self.hub_job[working]
                n_delivering = (work_done & hubj & ~at_hub).sum()
                work_done = work_done & (~hubj | at_hub)
            done = working[work_done]
            if done.size:
                dt = st[done]
                comp = xp.bincount(dt, minlength=m)[:m]
                hub_done = self.hub_job[done]
                val = self.D[dt] * (1.0 + p.hub_premium * hub_done)
                work_value = val.sum()
                n_done_hub = hub_done.sum()
                self.n_jobs[done] += 1
                # payoff rate of the finished job = value / steps since it was taken
                dur = (self.t - self.job_start[done] + 1).astype(xp.float32)
                rate = (val / xp.maximum(dur, 1.0)).astype(xp.float32)
                lrate = xp.log(xp.maximum(rate, 1e-12))
                r_hub_n, r_loc_n = n_done_hub, done.size - n_done_hub
                r_hub_sum, r_loc_sum = (rate * hub_done).sum(), (rate * ~hub_done).sum()
                r_hub_log, r_loc_log = (lrate * hub_done).sum(), (lrate * ~hub_done).sum()
                if self.learned_choice:
                    t_now = xp.int32(self.t)
                    self.t_tried_hub[done] = xp.where(hub_done, t_now, self.t_tried_hub[done])
                    self.t_tried_loc[done] = xp.where(hub_done, self.t_tried_loc[done], t_now)
                    if self.choice_payoff == "per_skill":
                        w_done = mech.efficiency(self.exp[done, dt], p.learning_exponent, p.w_max)
                        rate = (rate / xp.maximum(w_done, 1e-6)).astype(xp.float32)
                    if self.choice_rule == "logit":
                        self.p_hub[done], self.q_hub[done], self.q_loc[done] = \
                            mech.update_delivery_values(
                                self.p_hub[done], self.q_hub[done], self.q_loc[done], hub_done,
                                rate, p.choice_aspiration_lr, p.choice_temperature,
                                p.choice_explore)
                    else:
                        self.p_hub[done], self.aspiration[done] = mech.update_delivery_choice(
                            self.p_hub[done], self.aspiration[done], hub_done, rate,
                            p.choice_lr, p.choice_aspiration_lr, p.choice_explore)
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
                    if ids.size:
                        self.job_start[ids] = self.t
                        if self.learned_choice:
                            self.hub_job[ids] = rs.random_sample(int(ids.size)) < self.p_hub[ids]
                        elif p.centralization > 0:
                            self.hub_job[ids] = rs.random_sample(int(ids.size)) < p.centralization
                        else:
                            self.hub_job[ids] = False
                        n_start_hub = self.hub_job[ids].sum()
                        n_start = int(ids.size)
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
            "work_value": work_value,
            "done_hub": n_done_hub,
            "started_hub": n_start_hub,
            "started": n_start,
            "mean_p_hub": (self.p_hub.mean() if self.learned_choice else 0.0),
            "rate_hub_sum": r_hub_sum, "rate_hub_log": r_hub_log, "rate_hub_n": r_hub_n,
            "rate_loc_sum": r_loc_sum, "rate_loc_log": r_loc_log, "rate_loc_n": r_loc_n,
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