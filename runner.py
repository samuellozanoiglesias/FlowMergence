"""
runner.py — Runs ONE simulation (params, seed) and stores it in its own folder.

Command line (single exploratory run, stored under DATA_ROOT/single_runs/):
    python runner.py --set N=2048 exclusion=0.5 --seed 0 --timeseries --snapshots 50

Each run folder contains:
    config.txt      full parameter list (* = non-default), derived quantities, machine
    result.json     summary (+ time series if recorded)
    snapshots.npz   lattice snapshots (if requested)
    error.txt       only if the run failed

The launcher uses `job_worker`, which never raises: it returns an error record so that
one failure does not stop a sweep of thousands of runs.
"""
from __future__ import annotations

import argparse
import os
import time
import traceback

import numpy as np

import paths
from dynamics import Simulation
from metrics import Accumulator, TimeseriesRecorder, summarize
from params import Params, parse_overrides


def run_single(params: Params, seed: int = 0, out_dir: str | None = None,
               record_timeseries: bool = False, snapshot_every: int = 0,
               max_snapshots: int = 400, experiment: str | None = None,
               job_id: str | None = None, verbose: bool = False) -> dict:
    """Run one simulation. If out_dir is given, config.txt / result.json / snapshots.npz
    are written there (config.txt first, so crashed runs are still documented)."""
    t0 = time.time()
    if out_dir:
        paths.ensure(out_dir)
        paths.write_run_config(os.path.join(out_dir, "config.txt"), params, seed,
                               experiment=experiment, job_id=job_id,
                               extra={"record_timeseries": record_timeseries,
                                      "snapshot_every": snapshot_every})
    sim = Simulation(params, seed)
    acc = Accumulator()
    rec = TimeseriesRecorder(params.record_every) if record_timeseries else None
    snaps = []
    total = params.T_burn + params.T_measure
    for t in range(total):
        measure = t >= params.T_burn
        info = sim.step(measure=measure)
        if measure:
            acc.add(info)
        if rec is not None:
            rec.add(t, info, sim)
        if snapshot_every and t % snapshot_every == 0 and len(snaps) < max_snapshots:
            snaps.append(sim.snapshot())
        if verbose and t % 500 == 0:
            print(f"  t={t:6d}  working={info['n_working']:7d}  M={info['M']:6d}  "
                  f"output={float(info['work']):9.1f}", flush=True)
    result = {
        "job_id": job_id,
        "experiment": experiment,
        "params": params.to_dict(),
        "seed": int(seed),
        "summary": summarize(sim, acc),
        "runtime_s": time.time() - t0,
        "run_dir": out_dir,
    }
    if rec is not None:
        result["timeseries"] = rec.data
    if out_dir:
        if snaps:
            save_snapshots(os.path.join(out_dir, "snapshots.npz"), snaps, params)
            result["snapshot_path"] = os.path.join(out_dir, "snapshots.npz")
        paths.write_json(os.path.join(out_dir, "result.json"), result)
    return result


def save_snapshots(path: str, snaps: list, params: Params):
    counts = np.array([len(s["cy"]) for s in snaps], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    np.savez_compressed(
        path,
        t=np.array([s["t"] for s in snaps]),
        h=np.stack([s["h"] for s in snaps]),
        v=np.stack([s["v"] for s in snaps]),
        cy=np.concatenate([s["cy"] for s in snaps]) if offsets[-1] else np.zeros(0, np.int32),
        cx=np.concatenate([s["cx"] for s in snaps]) if offsets[-1] else np.zeros(0, np.int32),
        offsets=offsets,
        radius=params.coord_radius,
        m=params.m,
        L=params.L,
    )


def job_worker(job: dict) -> dict:
    """Entry point for the process pool. Never raises.

    Writes the full result (with time series) to the run folder and returns a LEAN
    record (no time series) for the experiment-level results.jsonl.
    """
    run_dir = job.get("run_dir")
    try:
        p = Params.from_dict(job["params"])
        res = run_single(p, seed=job["seed"], out_dir=run_dir,
                         record_timeseries=job.get("record_timeseries", False),
                         snapshot_every=job.get("snapshot_every", 0),
                         experiment=job.get("experiment"), job_id=job["job_id"])
        res.pop("timeseries", None)
        return res
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        if run_dir:
            try:
                paths.ensure(run_dir)
                with open(os.path.join(run_dir, "error.txt"), "w") as fh:
                    fh.write(tb)
            except OSError:
                pass
        return {"job_id": job.get("job_id"), "experiment": job.get("experiment"),
                "params": job.get("params"), "seed": job.get("seed"), "run_dir": run_dir,
                "error": repr(e), "traceback": tb}


def main():
    ap = argparse.ArgumentParser(description="Single model run, stored under DATA_ROOT/single_runs/.")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeseries", action="store_true")
    ap.add_argument("--snapshots", type=int, default=0, help="store a snapshot every k steps")
    ap.add_argument("--root", default=None, help=f"data root (default {paths.DATA_ROOT})")
    ap.add_argument("--out-dir", default=None, help="explicit output folder (overrides the automatic one)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    p = Params(**parse_overrides(args.set))
    out_dir = args.out_dir or paths.single_run_dir(p, args.seed, args.root)
    print(f"N={p.N}  L={p.L}  density={p.density:.3f}  λ={p.exclusion}  coordination={p.coordination}")
    print(f"output folder: {out_dir}")
    res = run_single(p, seed=args.seed, out_dir=out_dir, record_timeseries=args.timeseries,
                     snapshot_every=args.snapshots, verbose=args.verbose)
    s = res["summary"]
    print(f"Y={s['Y']:.2f}  Y/N={s['y_per_capita']:.4f}  blocked={s['blocked_frac']:.3f}  "
          f"M/N={s['M_frac']:.4f}  DOL={s['DOL_tasks']:.3f}  t={res['runtime_s']:.1f}s")


if __name__ == "__main__":
    main()
