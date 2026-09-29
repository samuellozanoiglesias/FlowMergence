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
        "steps_measured": acc.n,
    }
    out.update(division_of_labour(to_numpy(sim.time_budget), sim.m))
    out.update(threshold_stats(to_numpy(sim.theta), sim.p))
    return out