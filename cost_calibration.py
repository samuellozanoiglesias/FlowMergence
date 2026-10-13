#!/usr/bin/env python3
"""
cost_calibration.py — How many hours will an experiment take? Calibrated on finished launches.

Reads the newest launch of one or more finished reference experiments (results.jsonl:
runtime_s of every job; manifest.json: start time; config.txt: workers), fits the run time of
one job as
        runtime_s ≈ steps · (α · N(m+1) + β · (m+2) L²)        (the two terms of estimated_cost)
and predicts, for each target experiment, its CPU-hours and its wall-clock time with W workers,
simulating the launcher's schedule (most expensive job first, a job starts as soon as a worker
is free). α and β are fitted separately because cost per agent and cost per cell differ, and the
new experiments sit at other densities than the reference ones.

Usage
    python cost_calibration.py                 # references paper_density_exponent + paper_centralization_sweep
    python cost_calibration.py --ref paper_density_exponent paper_centralization_sweep --workers 38
    python cost_calibration.py --targets abl_pl_kappa0 poly_main --workers 38
Prints the reference launch's real wall-clock time, the fit and a table of predictions; writes
nothing.
"""
from __future__ import annotations

import argparse
import glob
import heapq
import json
import math
import os
import re
import time

import numpy as np

import paths
from config import EXPERIMENTS, build_jobs
from params import Params

PAPER1 = ["abl_pl_kappa0", "abl_pl_nodol", "abl_pl_lambda0", "abl_pl_b0",
          "abl_c_kappa0", "abl_c_nodol", "abl_c_lambda0", "abl_c_b0", "abl_c_lambda1",
          "deliver_learning_check", "transient_check_pl", "paper_density_exponent_b075",
          "poly_pilot", "poly_pilot_base", "paper_density_exponent_seedsB",
          "poly_main", "poly_main_base"]


def terms(params: dict) -> tuple:
    """The two cost terms of config.estimated_cost (agents, cells), per run."""
    p = Params.from_dict(params)
    steps = p.T_burn + p.T_measure
    return steps * p.N * (p.m + 1), steps * (p.m + 2) * p.L ** 2


def newest_launch(name: str, root: str | None, launch: str | None = None) -> str:
    d = os.path.join(paths.root(root), "experiments", name)
    if launch:
        return os.path.join(d, launch)
    cands = sorted(x for x in glob.glob(os.path.join(d, "exp_*")) if os.path.isfile(os.path.join(x, "results.jsonl")))
    if not cands:
        raise FileNotFoundError(f"no finished launch of {name} under {d}")
    return cands[-1]


def launch_info(folder: str) -> dict:
    """Rows (params, runtime_s) and the launch's real wall-clock time and worker count."""
    rows = []
    with open(os.path.join(folder, "results.jsonl")) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                rows.append((r["params"], r.get("runtime_s")))
    info = {"rows": rows, "workers": None, "wall_s": None}
    cfg = os.path.join(folder, "config.txt")
    if os.path.isfile(cfg):
        m = re.search(r"^workers\s*:\s*(\d+)", open(cfg).read(), re.M)
        if m:
            info["workers"] = int(m.group(1))
    man = os.path.join(folder, "manifest.json")
    if os.path.isfile(man):
        try:
            t0 = time.mktime(time.strptime(json.load(open(man))["started"], "%Y-%m-%d %H:%M:%S"))
            info["wall_s"] = os.path.getmtime(os.path.join(folder, "results.jsonl")) - t0
        except Exception:  # noqa: BLE001
            pass
    return info


def fit(rows) -> tuple:
    """Least squares runtime = α·A + β·C (α, β ≥ 0). Returns (α, β, median |rel. error|)."""
    X = np.array([terms(p) for p, _ in rows], float)
    y = np.array([rt for _, rt in rows], float)
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = sol
    if a < 0 or b < 0:                    # fall back to one ratio on the total cost
        tot = X.sum(axis=1)
        r = float(np.sum(y * tot) / np.sum(tot * tot))
        a = b = r
    pred = a * X[:, 0] + b * X[:, 1]
    return float(a), float(b), float(np.median(np.abs(pred / y - 1)))


def makespan(durations, workers: int) -> float:
    """Launcher schedule: most expensive first, each job on the first free worker."""
    free = [0.0] * max(1, workers)
    heapq.heapify(free)
    for d in sorted(durations, reverse=True):
        t = heapq.heappop(free)
        heapq.heappush(free, t + d)
    return max(free)


def hms(s: float) -> str:
    return "—" if s is None or not math.isfinite(s) else f"{s / 3600:7.1f} h"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ref", nargs="+", default=["paper_density_exponent", "paper_centralization_sweep"],
                    help="finished experiments used for calibration (newest launch of each). A sparse "
                         "and a dense one together pin down both cost terms")
    ap.add_argument("--launch", default=None, help="specific launch folder of a single --ref")
    ap.add_argument("--targets", nargs="*", default=PAPER1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--root", default=None)
    a = ap.parse_args()

    rows = []
    for name in a.ref:
        try:
            folder = newest_launch(name, a.root, a.launch if len(a.ref) == 1 else None)
        except FileNotFoundError as ex:
            print(f"reference {name}: skipped ({ex})")
            continue
        info = launch_info(folder)
        rr = [(p, rt) for p, rt in info["rows"] if rt is not None and rt > 0]
        cpu = sum(rt for _, rt in rr)
        print(f"reference {name}: {os.path.basename(folder)}, {len(info['rows'])} runs, "
              f"{len(rr)} with runtime_s")
        print(f"    real wall-clock time (manifest start -> last result): {hms(info['wall_s'])} "
              f"with {info['workers'] or '?'} workers;  sum of run times {hms(cpu)} CPU")
        if rr and info["workers"]:
            print(f"    predicted by the schedule model at {info['workers']} workers: "
                  f"{hms(makespan([rt for _, rt in rr], info['workers']))} (check against the real time)")
        rows += rr
    if not rows:
        raise SystemExit("no runtime_s in the reference results: use the real wall-clock time above, "
                         "or rerun one small experiment to calibrate")
    al, be, err = fit(rows)
    print(f"\nfit: runtime_s = steps·({al:.3g}·N(m+1) + {be:.3g}·(m+2)L²);  median |error| per run "
          f"{100 * err:.0f}%  ({len(rows)} runs)\n")
    print(f"{'experiment':34s} {'runs':>5s} {'CPU':>9s} {'wall @ %d' % a.workers:>11s}   longest run")
    tot_cpu = tot_wall = 0.0
    for name in a.targets:
        if name not in EXPERIMENTS:
            print(f"{name:34s}  (not in config.py)")
            continue
        jobs = build_jobs(name, experiment_datetime="calib", root=a.root)
        d = [al * A + be * C for A, C in (terms(j["params"]) for j in jobs)]
        w = makespan(d, a.workers)
        tot_cpu += sum(d)
        tot_wall += w
        print(f"{name:34s} {len(jobs):5d} {hms(sum(d))} {hms(w):>11s}   {hms(max(d))}")
    print(f"{'TOTAL (one after the other)':34s} {'':5s} {hms(tot_cpu)} {hms(tot_wall):>11s}")
    print("\nWall times assume the experiments run one after the other, as in run_all.sh, and that the "
          "machine is not shared. A launch never takes less than its longest run.")


if __name__ == "__main__":
    main()