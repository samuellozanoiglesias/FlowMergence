"""
ablation_study.py – Systematic ablation of MAPPO components.

Ablation targets
----------------
1. ``no_entropy``            – entropy coefficient set to 0 (no exploration bonus).
2. ``no_centralised_critic`` – each agent uses its own local-state value function.
3. ``no_comm``               – observation radius set to zero (fully blind to others).
4. ``no_gae``                – GAE λ set to 0 (plain TD-lambda advantage).
5. ``small_net``             – reduced network capacity (64 hidden units, 2 layers).

Each ablation trains a fresh agent with one component disabled and then
evaluates it with the same metrics as the baseline.  The aggregate comparison
is visualised as a radar / bar chart.

Usage
-----
    python -m traffic_rl.experiments.ablation_study --steps 500000
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from traffic_rl.config import (
    EnvConfig, MAPPOConfig, TrainingConfig, ExperimentConfig,
    make_default_config,
)
from traffic_rl.train import Trainer
from traffic_rl.evaluate import run_evaluation


# ---------------------------------------------------------------------------
# Ablation configuration factory
# ---------------------------------------------------------------------------

def make_ablation_configs(
    base_cfg: ExperimentConfig,
    targets: List[str],
) -> Dict[str, ExperimentConfig]:
    """Build a dict of ablation configurations from a baseline.

    Parameters
    ----------
    base_cfg:
        Fully specified baseline experiment configuration.
    targets:
        List of ablation target names (see module docstring for valid names).

    Returns
    -------
    Dict mapping ablation name → modified ExperimentConfig.
    """
    configs: Dict[str, ExperimentConfig] = {"baseline": deepcopy(base_cfg)}
    configs["baseline"].name = f"{base_cfg.name}_baseline"

    for target in targets:
        cfg = deepcopy(base_cfg)

        if target == "no_entropy":
            cfg.mappo.entropy_coef = 0.0
            cfg.name = f"{base_cfg.name}_no_entropy"

        elif target == "no_centralised_critic":
            cfg.mappo.use_centralised_critic = False
            cfg.name = f"{base_cfg.name}_no_cc"

        elif target == "no_comm":
            # Blind agents – set obs_radius to very small value
            cfg.env.obs_radius = 1.0
            cfg.name = f"{base_cfg.name}_no_comm"

        elif target == "no_gae":
            cfg.mappo.gae_lambda = 0.0
            cfg.name = f"{base_cfg.name}_no_gae"

        elif target == "small_net":
            cfg.mappo.hidden_dim = 64
            cfg.mappo.n_layers = 2
            cfg.name = f"{base_cfg.name}_small_net"

        else:
            print(f"[Ablation] Unknown target '{target}' – skipping.")
            continue

        configs[target] = cfg

    return configs


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_ablation_study(
    ablation_targets: List[str],
    base_cfg: ExperimentConfig,
    steps: int = 500_000,
    seed: int = 42,
    plots_dir: str = "plots",
    results_dir: str = "results",
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, Any]:
    """Train and evaluate each ablation variant.

    Returns
    -------
    Dict mapping ablation name → evaluation summary.
    """
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    all_cfgs = make_ablation_configs(base_cfg, ablation_targets)
    all_results: Dict[str, Dict[str, float]] = {}

    for ablation_name, cfg in all_cfgs.items():
        print(f"\n{'='*60}")
        print(f"  Ablation: {ablation_name}")
        print(f"{'='*60}")

        cfg.training.total_steps = steps
        cfg.training.seed = seed
        cfg.env.seed = seed
        cfg.training.checkpoint_dir = checkpoint_dir
        cfg.training.plots_dir = plots_dir
        cfg.training.results_dir = results_dir

        trainer = Trainer(experiment_cfg=cfg)
        trainer.train()

        ckpt = os.path.join(checkpoint_dir, f"{cfg.name}_final.pt")
        if os.path.exists(ckpt):
            eval_results = run_evaluation(
                checkpoint_path=ckpt,
                env_cfg=cfg.env,
                n_episodes=5,
                seed=seed + 9999,
                results_dir=results_dir,
                plots_dir=plots_dir,
                run_name=cfg.name,
            )
            all_results[ablation_name] = eval_results["summary"]

    # ---- Comparison plots ----
    _plot_ablation_comparison(all_results, plots_dir)

    # Save JSON
    out_path = os.path.join(results_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[Ablation] Results → {out_path}")

    return all_results


def _plot_ablation_comparison(
    results: Dict[str, Dict[str, float]],
    plots_dir: str,
) -> None:
    """Side-by-side bar chart comparing all ablations on key metrics."""
    if not results:
        return

    metrics_of_interest = [
        "mean_throughput",
        "mean_collision_rate",
        "mean_waiting_time",
        "mean_gridlock_prob",
        "mean_speed",
    ]
    labels = list(results.keys())
    n_metrics = len(metrics_of_interest)
    n_variants = len(labels)

    x = np.arange(n_metrics)
    width = 0.8 / max(n_variants, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, n_variants))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, variant in enumerate(labels):
        vals = [float(results[variant].get(m, 0.0)) for m in metrics_of_interest]
        bars = ax.bar(x + i * width - (n_variants - 1) * width / 2, vals, width,
                      label=variant, color=colors[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("mean_", "").replace("_", " ").title() for m in metrics_of_interest])
    ax.set_ylabel("Metric value")
    ax.set_title("Ablation Study – Component Contribution")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(plots_dir, "ablation_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Ablation] Comparison plot → {save_path}")

    # Radar chart
    _plot_radar(results, metrics_of_interest, plots_dir)


def _plot_radar(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    plots_dir: str,
) -> None:
    """Radar (spider) chart for ablation comparison."""
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Normalise per metric across variants for radar readability
    metric_vals: Dict[str, List[float]] = {m: [] for m in metrics}
    for variant in results:
        for m in metrics:
            metric_vals[m].append(float(results[variant].get(m, 0.0)))

    normed: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        mx = max(metric_vals[m]) or 1.0
        for variant in results:
            normed.setdefault(variant, {})[m] = float(results[variant].get(m, 0.0)) / mx

    for i, variant in enumerate(results):
        values = [normed[variant][m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2, label=variant)
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    ax.set_thetagrids(
        np.degrees(np.array(angles[:-1])),
        [m.replace("mean_", "").replace("_", "\n") for m in metrics],
        fontsize=8,
    )
    ax.set_title("Ablation – Normalised Metric Radar", pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    save_path = os.path.join(plots_dir, "ablation_radar.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Ablation] Radar chart → {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAPPO ablation study")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--targets", type=str, nargs="+",
        default=["no_entropy", "no_centralised_critic", "no_comm", "no_gae", "small_net"],
    )
    p.add_argument("--plots-dir", type=str, default="plots")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = make_default_config()
    base.training.total_steps = args.steps
    run_ablation_study(
        ablation_targets=args.targets,
        base_cfg=base,
        steps=args.steps,
        seed=args.seed,
        plots_dir=args.plots_dir,
        results_dir=args.results_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
