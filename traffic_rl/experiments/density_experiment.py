"""
density_experiment.py – Traffic density sweep experiment.

Trains or evaluates a MAPPO agent across a range of vehicle spawn densities
and aggregates the results to study how emergent coordination patterns scale
with density.

Research question
-----------------
Do emergent priority rules become stronger or weaker as traffic density
increases?  Is there a phase transition where gridlock overwhelms coordination?

Usage
-----
    python -m traffic_rl.experiments.density_experiment --train --steps 500000
    python -m traffic_rl.experiments.density_experiment --eval \\
        --checkpoint checkpoints/density_0.30_final.pt
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from traffic_rl.config import (
    EnvConfig, MAPPOConfig, TrainingConfig,
    ExperimentConfig, make_density_config,
)
from traffic_rl.train import Trainer, set_seed
from traffic_rl.evaluate import run_evaluation
from traffic_rl.utils.visualization import plot_density_sweep


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_density_sweep(
    density_values: List[float],
    steps_per_point: int = 500_000,
    seed: int = 42,
    plots_dir: str = "plots",
    results_dir: str = "results",
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, Any]:
    """Train one MAPPO agent per density level and collect metrics.

    Parameters
    ----------
    density_values:
        List of spawn-rate values (probability per step) to sweep.
    steps_per_point:
        Training steps for each density level.
    seed:
        Base random seed (incremented for each density point).

    Returns
    -------
    Dict mapping density → evaluation metrics.
    """
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    all_results: Dict[float, Dict[str, float]] = {}

    for i, density in enumerate(density_values):
        print(f"\n{'='*60}")
        print(f"  Density sweep: density={density:.2f}  ({i+1}/{len(density_values)})")
        print(f"{'='*60}")

        run_seed = seed + i * 100
        cfg = make_density_config(density, seed=run_seed)
        cfg.training.total_steps = steps_per_point
        cfg.training.checkpoint_dir = checkpoint_dir
        cfg.training.plots_dir = plots_dir
        cfg.training.results_dir = results_dir

        # Train
        trainer = Trainer(experiment_cfg=cfg)
        trainer.train()

        # Evaluate
        ckpt = os.path.join(checkpoint_dir, f"{cfg.name}_final.pt")
        if os.path.exists(ckpt):
            eval_results = run_evaluation(
                checkpoint_path=ckpt,
                env_cfg=cfg.env,
                n_episodes=5,
                seed=run_seed + 9999,
                results_dir=results_dir,
                plots_dir=plots_dir,
                run_name=cfg.name,
            )
            all_results[density] = eval_results["summary"]

    # ---- Aggregate plots ----
    if all_results:
        densities = sorted(all_results.keys())
        metrics_by_key: Dict[str, List[float]] = {}

        sample = next(iter(all_results.values()))
        for key in sample:
            try:
                metrics_by_key[key] = [float(all_results[d].get(key, 0)) for d in densities]
            except (TypeError, ValueError):
                pass

        plot_density_sweep(
            densities=densities,
            metrics_by_density=metrics_by_key,
            save_path=os.path.join(plots_dir, "density_sweep_summary.png"),
            title="Traffic Density Sweep – Performance vs Density",
        )

    # Save JSON
    results_path = os.path.join(results_dir, "density_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, default=str)
    print(f"\n[Density Sweep] Results saved → {results_path}")

    return {str(k): v for k, v in all_results.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Traffic density sweep experiment")
    p.add_argument("--train", action="store_true", help="Run training sweep")
    p.add_argument("--eval", action="store_true", help="Only run evaluation")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--densities", type=float, nargs="+",
        default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    )
    p.add_argument("--plots-dir", type=str, default="plots")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.train or not args.eval:
        run_density_sweep(
            density_values=args.densities,
            steps_per_point=args.steps,
            seed=args.seed,
            plots_dir=args.plots_dir,
            results_dir=args.results_dir,
            checkpoint_dir=args.checkpoint_dir,
        )


if __name__ == "__main__":
    main()
