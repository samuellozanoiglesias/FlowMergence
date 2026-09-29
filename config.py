"""
config.py — Experiment definitions (sweeps). Each experiment has:
    base           : fixed overrides on Params
    grid           : {parameter: [values]}, expanded as a Cartesian product
    seeds          : replicas per point. The SAME seeds are used at every point (common
                     random numbers), which reduces the variance of differences between points.
    timeseries     : (optional) store time series in each run's result.json
    snapshot_every : (optional) store lattice snapshots every k steps (snapshots.npz)

Recommended order:
    smoke -> calibration_fine -> calibration_robustness -> density_scan (+ _nocoord,
    _spillover) -> control_density -> lambda_sweep (densities chosen from density_scan)
    -> lambda_sweep_nocoord -> phase_diagram -> area_exponent -> ossification
    -> useless_bureaucracy -> learning_ablation -> large_N (GPU)
"""
from __future__ import annotations

import hashlib
import itertools
import json

import paths
from params import Params

N_GRID = [2 ** k for k in range(6, 16)]         # 64 … 32768  (2.7 decades)
N_COARSE = [2 ** k for k in range(6, 16, 2)]    # 64, 256, 1024, 4096, 16384
LAMBDAS = [round(0.1 * i, 3) for i in range(11)]

EXPERIMENTS = {
    "smoke": dict(
        description="Quick end-to-end test that the whole pipeline runs.",
        base=dict(T_burn=300, T_measure=200),
        grid=dict(N=[64, 128, 256, 512], exclusion=[0.0, 1.0]),
        seeds=2,
    ),
    "calibration_xi_phi": dict(
        description="(coarse, superseded by calibration_fine) Locate where division of labour "
                    "emerges over θ_max × ξ × φ.",
        base=dict(N=1024, exclusion=0.5, theta_init=200.0),
        grid=dict(theta_max=[400.0, 600.0, 1000.0],
                  xi=[1.0, 3.0, 10.0],
                  phi=[0.5, 1.5, 5.0]),
        seeds=2,
        timeseries=True,
    ),
    "calibration_fine": dict(
        description="Fine calibration. The ratio φ/ξ is the master knob (window found around "
                    "0.5); θ_max helps (higher = more selective); θ_init near the stimulus "
                    "avoids the deaf trap; small ξ = cleaner but slower. 'phi_over_xi' is a "
                    "derived key: φ = ratio × ξ.",
        base=dict(N=1024, exclusion=0.5),
        grid=dict(phi_over_xi=[0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2],
                  xi=[0.3, 1.0, 3.0],
                  theta_max=[1000.0, 2000.0, 4000.0],
                  theta_init=[100.0, 200.0, 400.0]),
        seeds=4,
        timeseries=True,
    ),
    "calibration_robustness": dict(
        description="Does the chosen (ξ, φ, θ_max, θ_init) keep specializing across N and λ? "
                    "Pass the candidate with --set xi=... phi=... theta_max=... theta_init=...",
        grid=dict(N=[64, 256, 1024, 4096, 16384], exclusion=[0.0, 0.5, 1.0]),
        seeds=4,
        timeseries=True,
    ),
    "density_scan": dict(
        description="Density response at FIXED N: with a=1, density = rho0 exactly. Measures "
                    "ε(ρ, λ) = d log y / d log ρ, so that β − 1 ≈ (1 − a)·ε for any area exponent a. "
                    "Spans from sparse to beyond the BML jam (~0.3–0.4).",
        base=dict(N=4096, area_exponent=1.0),
        grid=dict(rho0=[0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
                  exclusion=[0.0, 0.25, 0.5, 0.75, 1.0]),
        seeds=4,
    ),
    "density_scan_nocoord": dict(
        description="Same as density_scan without coordinators: what do the cops buy at each density?",
        base=dict(N=4096, area_exponent=1.0, coordination=False),
        grid=dict(rho0=[0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
                  exclusion=[0.0, 0.25, 0.5, 0.75, 1.0]),
        seeds=4,
    ),
    "density_scan_spillover": dict(
        description="Benefit side: how ε depends on the spillover strength κ (λ fixed at 0.5).",
        base=dict(N=4096, area_exponent=1.0, exclusion=0.5),
        grid=dict(rho0=[0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6],
                  spillover=[0.0, 1.0, 2.0, 4.0, 8.0]),
        seeds=4,
    ),
    "control_density": dict(
        description="CONTROL: constant density (a=1). β ≈ 1 is expected for every λ. "
                    "If not, there is a finite-size effect or a bug.",
        base=dict(area_exponent=1.0),
        grid=dict(N=N_GRID, exclusion=[0.0, 0.5, 1.0]),
        seeds=6,
    ),
    "lambda_sweep": dict(
        description="MAIN: β(λ) and γ(λ) with growing density (a=0.85).",
        base=dict(area_exponent=0.85),
        grid=dict(N=N_GRID, exclusion=LAMBDAS),
        seeds=8,
    ),
    "lambda_sweep_nocoord": dict(
        description="Ablation: same as lambda_sweep but without the coordinator role.",
        base=dict(area_exponent=0.85, coordination=False),
        grid=dict(N=N_GRID, exclusion=LAMBDAS),
        seeds=8,
    ),
    "phase_diagram": dict(
        description="β on the (κ spillover, λ exclusion) plane: benefit versus cost.",
        base=dict(area_exponent=0.85),
        grid=dict(N=N_COARSE, spillover=[0.0, 0.5, 1.0, 2.0, 4.0],
                  exclusion=[0.0, 0.25, 0.5, 0.75, 1.0]),
        seeds=4,
    ),
    "area_exponent": dict(
        description="How much it matters that density grows with N (a = 2/3 … 1).",
        grid=dict(N=N_GRID, area_exponent=[0.67, 0.75, 0.85, 1.0], exclusion=[0.2, 0.8]),
        seeds=6,
    ),
    "ossification": dict(
        description="Bureaucracy: tenure (p_quit_coord), conflict (η) and relocation.",
        base=dict(area_exponent=0.85, exclusion=0.8),
        grid=dict(N=N_COARSE, p_quit_coord=[0.001, 0.01, 0.1],
                  coord_conflict=[0.0, 0.1, 0.3], coord_relocate_prob=[0.0, 0.05]),
        seeds=4,
    ),
    "useless_bureaucracy": dict(
        description="Control: coordinators that do not help (q=0). Cost of the role by itself.",
        base=dict(area_exponent=0.85, coord_strength=0.0, coord_signal="all"),
        grid=dict(N=N_GRID, exclusion=[0.2, 0.8]),
        seeds=6,
    ),
    "learning_ablation": dict(
        description="No payoff from specializing (b=0) versus b>0.",
        base=dict(area_exponent=0.85),
        grid=dict(N=N_COARSE, learning_exponent=[0.0, 0.25, 0.5, 0.75], exclusion=[0.2, 0.8]),
        seeds=4,
    ),
    "dynamics_examples": dict(
        description="A few runs with time series and snapshots to inspect transients.",
        base=dict(area_exponent=0.85),
        grid=dict(N=[1024, 8192], exclusion=[0.0, 0.5, 1.0]),
        seeds=2,
        timeseries=True,
        snapshot_every=100,
    ),
    "large_N": dict(
        description="Extends the N range (recommended on GPU: --backend cupy --workers 8).",
        base=dict(area_exponent=0.85),
        grid=dict(N=[2 ** 16, 2 ** 17, 2 ** 18], exclusion=[0.0, 0.5, 1.0]),
        seeds=3,
    ),
}


def _to_py(v):
    """Convert numpy types to plain Python types (for JSON and stable hashing)."""
    if hasattr(v, "item"):
        return v.item()
    return v


def job_id(params_dict: dict, seed: int) -> str:
    blob = json.dumps({"p": params_dict, "seed": seed}, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def build_jobs(name: str, experiment_datetime: str, extra_base: dict | None = None, seeds: int | None = None,
               root: str | None = None) -> list:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment: {name}. Available: {sorted(EXPERIMENTS)}")
    exp = EXPERIMENTS[name]
    base = {**exp.get("base", {}), **(extra_base or {})}
    grid = exp["grid"]
    keys = list(grid)
    n_seeds = seeds if seeds is not None else exp.get("seeds", 1)
    jobs = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        kw = {**base, **{k: _to_py(v) for k, v in zip(keys, combo)}}
        # derived keys (not Params fields): resolved into real parameters here
        if "phi_over_xi" in kw:
            ratio = kw.pop("phi_over_xi")
            xi = kw.get("xi", Params().xi)
            kw["phi"] = round(ratio * xi, 6)
        p = Params(**kw)
        p.validate()
        pd = p.to_dict()
        for s in range(n_seeds):
            jobs.append({
                "job_id": job_id(pd, s),
                "experiment": name,
                "params": pd,
                "seed": s,
                "experiment_datetime": experiment_datetime,
                # keyword arguments: robust to the parameter order of paths.experiment_run_dir
                "run_dir": paths.experiment_run_dir(name, experiment_datetime=experiment_datetime,
                                                    p=p, seed=s, r=root),
                "record_timeseries": bool(exp.get("timeseries", False)),
                "snapshot_every": int(exp.get("snapshot_every", 0)),
            })
    return jobs


def estimated_cost(job: dict) -> float:
    """Relative cost ≈ steps × (N·m + m·L²). Used to launch the most expensive jobs first."""
    p = Params.from_dict(job["params"])
    return (p.T_burn + p.T_measure) * (p.N * (p.m + 1) + (p.m + 2) * p.L ** 2)