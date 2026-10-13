"""
metrics.py — Observables, accumulators and division-of-labour indices.

Main observables (averages over the measurement window)
    Y            output per step (completed route units)          -> Y ∝ N^β
    M            number of coordinators                           -> M ∝ N^γ
    M_active     coordinators with signal in their zone
    blocked_frac fraction of move attempts that were blocked
    DOL          division-of-labour index of Gorelick et al. (2004):
                 DOL = I(individual; task) / H(task) ∈ [0, 1]
                 (0 = everyone does everything, 1 = each individual does a single task)
"""
from __future__ import annotations

import math

import numpy as np

from backend import to_numpy


class Accumulator:
    """Sums the per-step `info` dicts (on device) and converts them at the end."""

    def __init__(self):
        self.sums = {}
        self.n = 0

    def add(self, info: dict):
        for k, v in info.items():
            self.sums[k] = self.sums.get(k, 0) + v
        self.n += 1

    def means(self) -> dict:
        n = max(self.n, 1)
        return {k: float(to_numpy(v)) / n for k, v in self.sums.items()}

    def reset(self):
        self.sums, self.n = {}, 0


class TimeseriesRecorder:
    """Stores window averages every `every` steps, plus the stimulus of each task."""

    def __init__(self, every: int):
        self.every = max(1, int(every))
        self.window = Accumulator()
        self.data = {"t": [], "s_task": []}

    def add(self, t: int, info: dict, sim):
        self.window.add(info)
        if (t + 1) % self.every == 0:
            for k, v in self.window.means().items():
                self.data.setdefault(k, []).append(v)
            self.data["t"].append(t)
            self.data["s_task"].append([float(x) for x in to_numpy(sim.s)])
            self.window.reset()


# ---------------------------------------------------------------------- division of labour
def _gorelick(c: np.ndarray) -> float:
    tot = c.sum()
    if tot <= 0:
        return math.nan
    p = c / tot
    pi = p.sum(axis=1, keepdims=True)
    pj = p.sum(axis=0, keepdims=True)
    nz = p > 0
    denom = (pi * pj)[nz]
    mi = float(np.sum(p[nz] * np.log(p[nz] / denom)))
    pjv = pj.ravel()
    pjv = pjv[pjv > 0]
    H = float(-np.sum(pjv * np.log(pjv)))
    return mi / H if H > 0 else math.nan


def division_of_labour(tb: np.ndarray, m: int) -> dict:
    """tb: N×(m+2) time budget [idle, tasks..., coordination]."""
    c = tb[:, 1:].astype(np.float64)       # active roles
    active = c.sum(axis=1)
    mask = active > 0
    out = {
        "DOL_all": _gorelick(c),
        "DOL_tasks": _gorelick(c[:, :m]),
        "frac_time_idle": float(tb[:, 0].sum() / max(tb.sum(), 1)),
    }
    if mask.any():
        top = c[mask].max(axis=1) / active[mask]
        out["mean_top_share"] = float(top.mean())
        out["frac_focused"] = float((top > 0.9).mean())
        out["frac_mostly_one"] = float((top > 0.7).mean())
        out["frac_career_coord"] = float((c[mask, m] / active[mask] > 0.5).mean())
    else:
        out["mean_top_share"] = out["frac_focused"] = out["frac_career_coord"] = math.nan
        out["frac_mostly_one"] = math.nan
    return out


def threshold_stats(theta: np.ndarray, p) -> dict:
    cut = p.theta_min + 0.1 * (p.theta_max - p.theta_min)
    spec = theta < cut
    prod = spec[:, :p.m]
    return {
        "frac_specialist_any": float(prod.any(axis=1).mean()),
        "mean_specialties": float(prod.sum(axis=1).mean()),
        "frac_specialist_coord": float(spec[:, p.m].mean()),
    }


def private_payoff_stats(mu: dict) -> dict:
    """[NEW] Realized private payoff rate (value / job duration) of the jobs finished in the
    measurement window, by delivery mode. hub_private_log_adv = E[ln r | hub] − E[ln r | local]:
    > 0 means a hub job paid its agent more than a local one. Over-centralization (the
    externality) = agents still use the hub and it still pays privately while the planner's
    optimum is c = 0. NaN when a mode had no completed job in the window."""
    nan = math.nan
    nh, nl = mu.get("rate_hub_n", 0.0), mu.get("rate_loc_n", 0.0)
    out = {
        "rate_hub_mean": mu.get("rate_hub_sum", 0.0) / nh if nh > 0 else nan,
        "rate_local_mean": mu.get("rate_loc_sum", 0.0) / nl if nl > 0 else nan,
        "jobs_hub_per_step": nh, "jobs_local_per_step": nl,
    }
    lh = mu.get("rate_hub_log", 0.0) / nh if nh > 0 else nan
    ll = mu.get("rate_loc_log", 0.0) / nl if nl > 0 else nan
    out["hub_private_log_adv"] = lh - ll
    out["hub_private_ratio"] = (out["rate_hub_mean"] / out["rate_local_mean"]
                                if nh > 0 and nl > 0 and out["rate_local_mean"] > 0 else nan)
    return out


def belief_stats(ph: np.ndarray, qh: np.ndarray, ql: np.ndarray) -> dict:
    """[NEW] Logit rule: what agents BELIEVE (their per-mode payoff estimates Q).
    Only agents that have tried both modes enter the log ratio."""
    both = (qh > 0) & (ql > 0)
    out = {"frac_tried_both": float(both.mean())}
    if not both.any():
        for k in ("mean_log_q_ratio", "median_log_q_ratio", "frac_believe_hub_better",
                  "log_q_ratio_committed_hub", "log_q_ratio_committed_local"):
            out[k] = math.nan
        return out
    r = np.log(qh[both] / ql[both])
    pb = ph[both]
    out["mean_log_q_ratio"] = float(r.mean())
    out["median_log_q_ratio"] = float(np.median(r))
    out["frac_believe_hub_better"] = float((r > 0).mean())
    out["log_q_ratio_committed_hub"] = float(r[pb > 0.8].mean()) if (pb > 0.8).any() else math.nan
    out["log_q_ratio_committed_local"] = float(r[pb < 0.2].mean()) if (pb < 0.2).any() else math.nan
    return out


def staleness_stats(ph: np.ndarray, th: np.ndarray, tl: np.ndarray, t: int) -> dict:
    """[NEW] Age (steps) of each agent's information about the mode it does NOT use.
    Large ages for committed agents = their estimate of the other mode is stale
    (competency trap: skills grow, the unused mode is judged at an old skill level)."""
    hub, loc = ph > 0.8, ph < 0.2
    def med_age(mask, tt):
        tried = mask & (tt >= 0)
        return float(np.median(t - tt[tried])) if tried.any() else math.nan
    return {
        "local_info_age_committed_hub": med_age(hub, tl),   # hub users: last local job
        "hub_info_age_committed_local": med_age(loc, th),   # local users: last hub job
        "frac_never_local_committed_hub": float((hub & (tl < 0)).sum() / max(hub.sum(), 1)),
    }


def _hub_mean_dist(sim) -> float:
    pos = getattr(sim, "hub_positions", None)
    if pos is None:
        return (sim.L - 1) / 2.0
    pos = np.sort(np.asarray(to_numpy(pos)))
    gaps = np.diff(np.concatenate([pos, [pos[0] + sim.L]]))
    return float(np.sum(gaps * (gaps - 1)) / (2.0 * sim.L))


def summarize(sim, acc: Accumulator) -> dict:
    """Summary of one run: plain Python float/int only, JSON-serializable."""
    mu = acc.means()
    N = sim.N
    att = max(mu.get("attempts", 0.0), 1e-12)
    M = mu.get("M", 0.0)
    Ma = mu.get("M_active", 0.0)
    out = {
        "N": int(N), "L": int(sim.L), "density": float(N / sim.L ** 2),
        "Y": mu.get("work", 0.0),
        "y_per_capita": mu.get("work", 0.0) / N,
        "Y_per_worker": mu.get("work", 0.0) / max(mu.get("n_working", 0.0), 1e-12),
        "completions": mu.get("completions", 0.0),
        "blocked_frac": mu.get("blocked", 0.0) / att,
        "cross_blocked_frac": mu.get("cross_blocked", 0.0) / att,
        "signal_frac": mu.get("signal", 0.0) / att,
        "entry_fail": mu.get("entry_fail", 0.0),
        "working_frac": mu.get("n_working", 0.0) / N,
        "idle_frac": mu.get("n_idle", 0.0) / N,
        "M": M, "M_frac": M / N,
        "M_active": Ma, "M_inactive": M - Ma,
        "bureau_inactive_frac": (M - Ma) / M if M > 0 else math.nan,
        "new_coord_rate": mu.get("new_coord", 0.0),
        "mean_w": mu.get("mean_w", 0.0),
        "mean_nsame": mu.get("mean_nsame", 0.0),
        "covered_frac": mu.get("covered_frac", 0.0),
        "s_coord": mu.get("s_coord", 0.0),
        "delivering_frac": mu.get("n_delivering", 0.0) / N,
        "Gamma_meas": (mu.get("n_delivering", 0.0) / N) / max(1.0 - mu.get("n_delivering", 0.0) / N, 1e-12),
        "Y_value": mu.get("work_value", 0.0),
        "y_value_per_capita": mu.get("work_value", 0.0) / N,
        "hub_share_completed": mu.get("done_hub", 0.0) / max(mu.get("completions", 0.0), 1e-12),
        "hub_share_started": mu.get("started_hub", 0.0) / max(mu.get("started", 0.0), 1e-12),
        "mean_p_hub": mu.get("mean_p_hub", 0.0),
        "steps_measured": acc.n,
        # hub geometry (E9): number of hub lines per direction and mean trip length to the
        # next hub ahead from a random cell (moves); 1 and (L-1)/2 for the original model
        "n_hubs": int(getattr(sim, "n_hubs", 1)),
        "hub_mean_dist": float(_hub_mean_dist(sim)),
    }
    if getattr(sim, "learned_choice", False):
        ph = to_numpy(sim.p_hub)
        out["frac_p_hub_high"] = float((ph > 0.8).mean())   # committed to the hub
        out["frac_p_hub_low"] = float((ph < 0.2).mean())    # committed to local delivery
        if getattr(sim, "choice_rule", "") == "logit":
            out.update(belief_stats(ph, to_numpy(sim.q_hub), to_numpy(sim.q_loc)))
        if hasattr(sim, "t_tried_hub"):
            out.update(staleness_stats(ph, to_numpy(sim.t_tried_hub), to_numpy(sim.t_tried_loc),
                                       int(sim.t)))
    out.update(private_payoff_stats(mu))
    if hasattr(sim, "n_jobs"):
        nj = to_numpy(sim.n_jobs)
        out["jobs_per_agent_median"] = float(np.median(nj))
        out["frac_agents_lt20_jobs"] = float((nj < 20).mean())   # choice may not have settled
    out.update(division_of_labour(to_numpy(sim.time_budget), sim.m))
    out.update(threshold_stats(to_numpy(sim.theta), sim.p))
    return out