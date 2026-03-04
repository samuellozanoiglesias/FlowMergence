"""
evaluate.py – Standalone model evaluation script.

Runs trained MAPPO checkpoints in the traffic environment and produces:
* Aggregate metric tables (throughput, travel time, collision rate, etc.)
* Per-episode metric CSV
* Emergent pattern analysis report
* Optional video / GIF of an evaluation episode

Usage
-----
    python -m traffic_rl.evaluate \\
        --checkpoint checkpoints/mappo_run_final.pt \\
        --episodes 10 \\
        --video plots/eval_episode.gif
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from traffic_rl.config import EnvConfig, MAPPOConfig, TrainingConfig, make_default_config
from traffic_rl.env.traffic_env import TrafficEnv
from traffic_rl.marl.mappo import MAPPO
from traffic_rl.utils.metrics import MetricsTracker, EpisodeStats
from traffic_rl.utils.visualization import (
    animate_episode,
    plot_intersection_usage,
    plot_emergent_patterns,
)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: str,
    env_cfg: Optional[EnvConfig] = None,
    mappo_cfg: Optional[MAPPOConfig] = None,
    n_episodes: int = 10,
    aggressive_fraction: float = 0.0,
    seed: int = 2024,
    capture_frames: bool = False,
    results_dir: str = "results",
    plots_dir: str = "plots",
    run_name: str = "eval",
) -> Dict[str, Any]:
    """Evaluate a saved checkpoint and return a results summary.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` checkpoint produced by ``train.py``.
    env_cfg, mappo_cfg:
        Configuration objects.  Defaults are used when omitted.
    n_episodes:
        Number of evaluation episodes.
    aggressive_fraction:
        Fraction of aggressive agents (used for heterogeneity analysis).
    seed:
        Base RNG seed.
    capture_frames:
        If True, render and store all frames of the first episode for
        video export.
    results_dir, plots_dir:
        Output directories.
    run_name:
        Prefix for saved files.

    Returns
    -------
    Dict with ``summary`` and ``per_episode`` keys.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    env_cfg = env_cfg or EnvConfig()
    mappo_cfg = mappo_cfg or MAPPOConfig()

    # Build environment
    env = TrafficEnv(
        config=env_cfg,
        aggressive_fraction=aggressive_fraction,
        render_mode="rgb_array" if capture_frames else None,
    )
    obs_dim = env.observation_dim
    n_actions = env_cfg.n_actions
    global_state_dim = env.global_state_dim

    # Load policy
    algo = MAPPO(
        obs_dim=obs_dim,
        n_actions=n_actions,
        global_state_dim=global_state_dim,
        mappo_cfg=mappo_cfg,
    )
    algo.load(checkpoint_path)
    print(f"[Eval] Loaded checkpoint: {checkpoint_path}")

    metrics = MetricsTracker(window=n_episodes)
    per_episode_records: List[Dict[str, float]] = []

    # Emergent pattern accumualtion
    entropy_hist: List[float] = []
    oscillation_hist: List[float] = []
    priority_hist: List[float] = []

    frames: List[np.ndarray] = []  # only for episode 0 if capture_frames

    for ep in range(n_episodes):
        ep_stats = EpisodeStats()
        obs_dict, _ = env.reset(seed=seed + ep)
        global_state = env.get_global_state()
        is_capturing = capture_frames and ep == 0

        for step in range(env_cfg.max_steps):
            if not obs_dict:
                break

            actions_dict = algo.select_actions(obs_dict, global_state, deterministic=True)
            obs_dict, rewards_dict, terminated, truncated, info = env.step(actions_dict)
            global_state = env.get_global_state()

            metrics.record_step(rewards_dict, info, ep_stats)

            if is_capturing:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if not obs_dict or all(truncated.get(a, False) for a in obs_dict):
                break

        # Per-episode stats
        metrics.record_episode(ep_stats)
        ep_dict = {f"ep_{k}": v for k, v in ep_stats.to_dict().items()}
        per_episode_records.append(ep_dict)

        # Emergent patterns
        emerg = metrics.detect_priority_emergence(env._intersections)
        entropy_hist.append(emerg["mean_entropy"])
        oscillation_hist.append(emerg["mean_oscillation"])
        priority_hist.append(emerg["priority_emergence"])

        print(
            f"  ep={ep+1}/{n_episodes} | "
            f"reward={ep_stats.total_reward:.2f} | "
            f"arrivals={ep_stats.n_arrivals} | "
            f"collisions={ep_stats.n_collisions} | "
            f"throughput={ep_stats.throughput():.4f}"
        )

    env.close()

    # ---- Save video ----
    if frames:
        vid_path = os.path.join(plots_dir, f"{run_name}_episode.gif")
        animate_episode(frames, vid_path, fps=15)

    # ---- Save intersection usage plot ----
    plot_intersection_usage(
        intersections=env._intersections,
        save_path=os.path.join(plots_dir, f"{run_name}_intersection_usage.png"),
        title=f"{run_name} – Intersection Usage",
    )

    # ---- Save emergent patterns plot ----
    plot_emergent_patterns(
        entropy_history=entropy_hist,
        oscillation_history=oscillation_hist,
        priority_emergence=priority_hist,
        save_path=os.path.join(plots_dir, f"{run_name}_emergent.png"),
        title=f"{run_name} – Emergent Patterns (Eval)",
    )

    # ---- Yield sequence analysis ----
    yield_analysis = metrics.yield_sequence_analysis(env._intersections)
    flow_oscillations = metrics.detect_flow_oscillations()

    # ---- Summary ----
    summary = metrics.summary()
    full_results = {
        "summary": summary,
        "per_episode": per_episode_records,
        "emergent": {
            "entropy_history": entropy_hist,
            "oscillation_history": oscillation_hist,
            "priority_emergence": priority_hist,
        },
        "yield_analysis": yield_analysis,
        "flow_oscillations": flow_oscillations,
    }

    # ---- Save JSON ----
    results_file = os.path.join(results_dir, f"{run_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"[Eval] Results saved → {results_file}")

    # Print summary table
    print("\n===== Evaluation Summary =====")
    for k, v in summary.items():
        print(f"  {k:35s} {v:.4f}")
    print("==============================\n")

    return full_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a FlowMergence MAPPO checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--aggressive", type=float, default=0.0)
    p.add_argument("--video", action="store_true", help="Capture and save video of episode 0")
    p.add_argument("--grid", type=int, default=None)
    p.add_argument("--vehicles", type=int, default=None)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--plots-dir", type=str, default="plots")
    p.add_argument("--name", type=str, default="eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = EnvConfig()
    if args.grid:
        env_cfg.grid_size = args.grid
    if args.vehicles:
        env_cfg.max_vehicles = args.vehicles

    run_evaluation(
        checkpoint_path=args.checkpoint,
        env_cfg=env_cfg,
        n_episodes=args.episodes,
        aggressive_fraction=args.aggressive,
        seed=args.seed,
        capture_frames=args.video,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
