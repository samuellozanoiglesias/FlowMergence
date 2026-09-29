"""
paths.py — Where everything is stored, and how run folders are named.

Everything lives under DATA_ROOT (override with the FLOWMERGENCE_DATA environment
variable or with --root on the command line):

DATA_ROOT/                                   default: /data/samuel_lozano/FlowMergence
├── experiments/
│   └── <experiment>/
│       ├── config.txt              experiment definition, overrides, machine info
│       ├── manifest.json           same, machine-readable
│       ├── results.jsonl           one line per finished run (summary only, no time series)
│       ├── errors.jsonl            failed runs (retried on relaunch)
│       ├── runs/
│       │   └── <run name>/         e.g. N0001024_lam0.5_a0.85_kap2_coordon
│       │       └── seed000/
│       │           ├── config.txt  full parameter list of THIS run (* = non-default)
│       │           ├── result.json summary + time series (if recorded)
│       │           ├── snapshots.npz  lattice snapshots (if recorded)
│       │           └── error.txt   only if the run failed
│       └── analysis/
│           └── by_<columns>/       figures, tables, analysis_config.txt
├── single_runs/
│   └── <YYYYmmdd-HHMMSS>_<run name>_seed000/   runs launched with runner.py
└── checks/                          output of checks.py

Run names always contain the core parameters (N, λ, a, κ, coordination) and, after
them, every parameter that differs from its default (abbreviated). Runs that differ
in any parameter therefore never share a folder.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import socket
import sys
from dataclasses import fields

from params import DEFAULTS, Params

DATA_ROOT = os.environ.get("FLOWMERGENCE_DATA", "/data/samuel_lozano/FlowMergence")

CORE = ("N", "exclusion", "area_exponent", "spillover", "coordination")
SKIP_IN_NAME = {"backend", "record_every"}
ABBREV = {
    "N": "N", "exclusion": "lam", "area_exponent": "a", "spillover": "kap", "coordination": "coord",
    "rho0": "rho", "L_min": "Lmin", "m": "m", "route_length": "D", "route_length_spread": "Dspr",
    "demand_skew": "skew", "traffic_light": "tl", "practice_gain": "g", "spillover_radius": "R",
    "exp_decay": "rhoE", "learning_exponent": "b", "w_max": "wmax", "theta_min": "thmin",
    "theta_max": "thmax", "theta_init": "th0", "theta_init_noise": "thnoise", "xi": "xi",
    "phi": "phi", "s_init": "s0", "s_max": "smax", "s_gain": "sgain", "share_ema": "shema",
    "p_quit_work": "pqw", "coord_radius": "r", "coord_strength": "q", "coord_conflict": "eta",
    "coord_stimulus_gain": "gC", "coord_signal": "sig", "p_quit_coord": "pqc",
    "coord_placement": "place", "coord_relocate_prob": "reloc", "blocked_ema": "bema",
    "T_burn": "Tb", "T_measure": "Tm", "encounter": "enc",
}


def root(r: str | None = None) -> str:
    return os.path.abspath(r or DATA_ROOT)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v).replace(os.sep, "-").replace(" ", "")


def run_name(p: Params) -> str:
    """Descriptive, unique folder name for a parameter set (seed not included)."""
    tokens = [f"N{p.N:07d}"]  # zero-padded so folders sort by N
    tokens += [ABBREV[k] + _fmt(getattr(p, k)) for k in CORE[1:]]
    for f in fields(Params):
        if f.name in CORE or f.name in SKIP_IN_NAME:
            continue
        v = getattr(p, f.name)
        if v != getattr(DEFAULTS, f.name):
            tokens.append(ABBREV.get(f.name, f.name) + _fmt(v))
    return "_".join(tokens)


def experiment_dir(experiment: str, experiment_datetime: str, r: str | None = None) -> str:
    return os.path.join(root(r), "experiments", experiment, f"exp_{experiment_datetime}")



def experiment_run_dir(experiment: str, experiment_datetime: str, p: Params, seed: int, r: str | None = None) -> str:
    return os.path.join(experiment_dir(experiment, experiment_datetime, r), "runs", run_name(p), f"seed{seed:03d}")


def single_run_dir(p: Params, seed: int, r: str | None = None) -> str:
    stamp = _dt.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    return os.path.join(root(r), "single_runs", f"{stamp}_{run_name(p)}_seed{seed:03d}")


def analysis_dir(experiment: str, by: list, r: str | None = None) -> str:
    return os.path.join(experiment_dir(experiment, r), "analysis", "by_" + "_".join(by))


def checks_dir(r: str | None = None) -> str:
    return os.path.join(root(r), "checks")


def ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------- config.txt
def _machine_lines() -> list:
    import numpy as np
    return [f"host        : {socket.gethostname()}",
            f"python      : {sys.version.split()[0]}",
            f"numpy       : {np.__version__}",
            f"platform    : {platform.platform()}"]


def write_run_config(path: str, p: Params, seed: int, experiment: str | None = None,
                     job_id: str | None = None, extra: dict | None = None) -> str:
    """Human-readable configuration of ONE run. Non-default parameters are marked with *."""
    lines = ["# FlowMergence — run configuration",
             f"created     : {_dt.datetime.now().isoformat(timespec='seconds')}",
             f"experiment  : {experiment or '(single run)'}",
             f"job_id      : {job_id or '-'}",
             f"seed        : {seed}",
             f"run name    : {run_name(p)}",
             f"params hash : {p.key()}"]
    lines += _machine_lines()
    lines += ["", "[derived]",
              f"L (lattice side)      = {p.L}",
              f"cells L^2             = {p.L ** 2}",
              f"density N/L^2         = {p.density:.6g}",
              f"route lengths D_j     = {p.route_lengths()}",
              f"demand shares pi_j    = {[round(x, 6) for x in p.demand_shares()]}",
              f"total steps           = {p.T_burn + p.T_measure}",
              "", "[parameters]   (* = differs from default)"]
    width = max(len(f.name) for f in fields(Params))
    for f in fields(Params):
        v = getattr(p, f.name)
        d = getattr(DEFAULTS, f.name)
        mark = "*" if v != d else " "
        tail = f"   (default: {d!r})" if v != d else ""
        lines.append(f"{mark} {f.name:<{width}} = {v!r}{tail}")
    if extra:
        lines += ["", "[extra]"] + [f"{k} = {v!r}" for k, v in extra.items()]
    ensure(os.path.dirname(path) or ".")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_experiment_config(path: str, experiment: str, definition: dict, overrides: dict,
                            n_jobs: int, n_pending: int, workers: int) -> str:
    lines = ["# FlowMergence — experiment configuration",
             f"created     : {_dt.datetime.now().isoformat(timespec='seconds')}",
             f"experiment  : {experiment}",
             f"description : {definition.get('description', '')}",
             f"jobs        : {n_jobs} total, {n_pending} pending at launch",
             f"workers     : {workers}"]
    lines += _machine_lines()
    lines += ["", "[base overrides]"] + [f"{k} = {v!r}" for k, v in definition.get("base", {}).items()]
    lines += ["", "[command-line overrides]"] + [f"{k} = {v!r}" for k, v in overrides.items()]
    lines += ["", "[grid]"] + [f"{k} = {v!r}" for k, v in definition.get("grid", {}).items()]
    lines += ["", f"seeds       = {definition.get('seeds', 1)}",
              f"timeseries  = {definition.get('timeseries', False)}",
              f"snapshots   = every {definition.get('snapshot_every', 0)} steps (0 = off)",
              "", "[all other parameters: defaults]"]
    width = max(len(f.name) for f in fields(Params))
    for f in fields(Params):
        lines.append(f"  {f.name:<{width}} = {getattr(DEFAULTS, f.name)!r}")
    ensure(os.path.dirname(path) or ".")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_json(path: str, obj) -> str:
    ensure(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written result.json
    return path
