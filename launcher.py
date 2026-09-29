"""
launcher.py — Runs an experiment from config.py in parallel, with resumption.

Examples
    nohup python launcher.py --exp lambda_sweep --workers 20 > log_lambda_sweep.out 2>&1 &
    nohup python launcher.py --exp lambda_sweep_nocoord --workers 20 > log_lambda_sweep_nocoord.out 2>&1 &
    nohup python launcher.py --exp lambda_sweep --workers 20 --max-N 32768           # CPU
    nohup python launcher.py --exp large_N --backend cupy --workers 8                  # GPU

Output (see paths.py for the full layout):
    DATA_ROOT/experiments/<exp>/exp_<timestamp>/config.txt, manifest.json, results.jsonl, errors.jsonl
    DATA_ROOT/experiments/<exp>/exp_<timestamp>/runs/<run name>/seedXXX/{config.txt, result.json, ...}

Resumption: if interrupted, relaunching skips the job_ids already in results.jsonl.
Failed runs go to errors.jsonl (and error.txt in their folder) and are retried on the
next launch.

Resources: each NumPy worker uses 1 thread (OMP_NUM_THREADS=1 is set BEFORE importing
numpy so the 200 cores are not oversubscribed). On GPU each process creates its own
CUDA context (~0.3–0.5 GB), so 8–16 workers fit easily in 40 GB; the GPU only pays
off for N ≳ 1e5.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import datetime as _dt
import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import paths  # noqa: E402
from config import EXPERIMENTS, build_jobs, estimated_cost  # noqa: E402
from params import parse_overrides  # noqa: E402
from runner import job_worker  # noqa: E402


def _read_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "error" not in rec:
                        done.add(rec["job_id"])
                except json.JSONDecodeError:
                    pass  # truncated line from an interruption
    return done


def main(experiment_datetime: str):
    ap = argparse.ArgumentParser(description="Sweep launcher.")
    ap.add_argument("--exp", help="experiment name (see --list)")
    ap.add_argument("--list", action="store_true", help="list experiments")
    ap.add_argument("--root", default=None, help=f"data root (default {paths.DATA_ROOT})")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--backend", choices=["numpy", "cupy"], default=None)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides on the base")
    ap.add_argument("--seeds", type=int, default=None, help="override the number of seeds")
    ap.add_argument("--min-N", type=int, default=None)
    ap.add_argument("--max-N", type=int, default=None)
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list or not args.exp:
        for k, v in EXPERIMENTS.items():
            print(f"{k:24s} {v['description']}")
        return

    extra = parse_overrides(args.set)
    if args.backend:
        extra["backend"] = args.backend
    jobs = build_jobs(args.exp, experiment_datetime=experiment_datetime, extra_base=extra, seeds=args.seeds, root=args.root)
    if args.min_N is not None:
        jobs = [j for j in jobs if j["params"]["N"] >= args.min_N]
    if args.max_N is not None:
        jobs = [j for j in jobs if j["params"]["N"] <= args.max_N]

    outdir = paths.ensure(paths.experiment_dir(args.exp, experiment_datetime, args.root))
    res_path = os.path.join(outdir, "results.jsonl")
    err_path = os.path.join(outdir, "errors.jsonl")
    done = _read_done(res_path)
    pending = [j for j in jobs if j["job_id"] not in done]
    pending.sort(key=estimated_cost, reverse=True)     # expensive first: better load balance
    if args.max_jobs:
        pending = pending[:args.max_jobs]

    total_cost = sum(estimated_cost(j) for j in pending)
    print(f"[{args.exp}] root={paths.root(args.root)}")
    print(f"[{args.exp}] total={len(jobs)}  done={len(done)}  pending={len(pending)}  "
          f"workers={args.workers}  relative cost={total_cost:.3g}")
    if args.dry_run or not pending:
        if pending:
            print(f"  e.g. first run folder: {pending[0]['run_dir']}")
        return

    definition = EXPERIMENTS[args.exp]
    paths.write_experiment_config(os.path.join(outdir, "config.txt"), args.exp, definition,
                                  extra, len(jobs), len(pending), args.workers)
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump({"experiment": args.exp, "definition": definition, "overrides": extra,
                   "root": paths.root(args.root),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S"), "n_jobs": len(jobs)},
                  f, indent=2, default=str)

    backend = extra.get("backend", definition.get("base", {}).get("backend", "numpy"))
    ctx = mp.get_context("spawn" if backend == "cupy" else "fork")
    t0 = time.time()
    n_ok = n_err = 0
    cost_done = 0.0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex, \
            open(res_path, "a") as fres, open(err_path, "a") as ferr:
        futs = {ex.submit(job_worker, j): j for j in pending}
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001  (e.g. a worker killed by the OOM killer)
                res = {"job_id": job["job_id"], "params": job["params"], "seed": job["seed"],
                       "run_dir": job["run_dir"], "error": f"worker crash: {e!r}"}
            if "error" in res:
                n_err += 1
                ferr.write(json.dumps(res) + "\n")
                ferr.flush()
            else:
                n_ok += 1
                fres.write(json.dumps(res) + "\n")
                fres.flush()
            cost_done += estimated_cost(job)
            el = time.time() - t0
            frac = cost_done / max(total_cost, 1e-12)
            eta = el * (1 - frac) / max(frac, 1e-9)
            print(f"  ok={n_ok} err={n_err}  {100 * frac:5.1f}%  "
                  f"elapsed={el / 60:6.1f} min  ETA≈{eta / 60:6.1f} min  "
                  f"(N={job['params']['N']}, λ={job['params']['exclusion']}, seed={job['seed']})",
                  flush=True)
    print(f"Finished: {n_ok} ok, {n_err} errors. Results in {res_path}")


if __name__ == "__main__":
    experiment_datetime = _dt.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    main(experiment_datetime=experiment_datetime)
