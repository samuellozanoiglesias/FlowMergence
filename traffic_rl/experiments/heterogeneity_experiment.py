"""
heterogeneity_experiment.py – Cooperative vs. aggressive agent mix experiment.

Research question
-----------------
How does the fraction of aggressive vehicles affect:
* Overall system throughput?
* Collision rate?
* Emergence of priority rules (do aggressive agents dominate)?
* Whether cooperative agents learn to avoid aggressive ones?

Methodology
-----------
For each aggressive_fraction value in [0.0, 0.1, 0.2, …, 1.0] a fresh MAPPO
agent is trained in a mixed environment.  Both agent types share the same
policy network (parameter sharing); they are distinguished by the
``agent_type`` feature in their observation.

Usage
-----
    python -m traffic_rl.experiments.heterogeneity_experiment --steps 500000
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt

from traffic_rl.config import (
    EnvConfig, MAPPOConfig, TrainingConfig,
    ExperimentConfig, make_heterogeneous_config,
)
from traffic_rl.train import Trainer
from traffic_rl.evaluate import run_evaluation
from traffic_rl.utils.visualization import plot_density_sweep


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_heterogeneity_sweep(
    aggressive_fractions: List[float],
    steps_per_point: int = 500_000,
    seed: int = 42,
    plots_dir: str = "plots",
    results_dir: str = "results",
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, Any]:
    """Sweep over aggressive-agent fractions and collect metrics.

    Returns
    -------
    Dict mapping fraction (as string) → evaluation metrics.
    """
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    all_results: Dict[float, Dict[str, float]] = {}

    for i, frac in enumerate(aggressive_fractions):
        print(f"\n{'='*60}")
        print(f"  Heterogeneity: aggressive_fraction={frac:.2f}  ({i+1}/{len(aggressive_fractions)})")
        print(f"{'='*60}")

        run_seed = seed + i * 100
        cfg = make_heterogeneous_config(aggressive_frac=frac)
        cfg.training.total_steps = steps_per_point
        cfg.training.seed = run_seed
        cfg.env.seed = run_seed
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
                aggressive_fraction=frac,
                seed=run_seed + 9999,
                results_dir=results_dir,
                plots_dir=plots_dir,
                run_name=cfg.name,
            )
            all_results[frac] = eval_results["summary"]

    # ---- Aggregate visualisation ----
    _plot_heterogeneity_summary(all_results, plots_dir)

    # Save JSON
    out_path = os.path.join(results_dir, "heterogeneity_results.json")
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, default=str)
    print(f"\n[Heterogeneity] Results → {out_path}")

    return {str(k): v for k, v in all_results.items()}


def _plot_heterogeneity_summary(
    results: Dict[float, Dict[str, float]],
    plots_dir: str,
) -> None:
    """Create a 2×2 summary figure for the heterogeneity experiment."""
    if not results:
        return

    fracs = sorted(results.keys())
    metrics_of_interest = [
        ("mean_throughput", "Throughput", "seagreen"),
        ("mean_collision_rate", "Collision Rate", "crimson"),
        ("mean_waiting_time", "Mean Waiting Time", "darkorange"),
        ("mean_gridlock_prob", "Gridlock Probability", "purple"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Heterogeneity Experiment – System Performance vs Aggressive Fraction", fontsize=12)

    for ax, (key, label, color) in zip(axes.flatten(), metrics_of_interest):
        values = [float(results[f].get(key, 0.0)) for f in fracs]
        ax.plot(fracs, values, marker="o", color=color, linewidth=2)
        ax.fill_between(fracs, values, alpha=0.15, color=color)
        ax.set_xlabel("Aggressive agent fraction")
        ax.set_ylabel(label)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(plots_dir, "heterogeneity_summary.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Heterogeneity] Summary plot → {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Heterogeneity experiment")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fractions", type=float, nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    )
    p.add_argument("--plots-dir", type=str, default="plots")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_heterogeneity_sweep(
        aggressive_fractions=args.fractions,
        steps_per_point=args.steps,
        seed=args.seed,
        plots_dir=args.plots_dir,
        results_dir=args.results_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
