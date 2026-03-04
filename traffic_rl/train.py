"""
train.py – Main training loop for FlowMergence MAPPO.

Usage
-----
    python -m traffic_rl.train             # default config
    python -m traffic_rl.train --steps 2000000 --seed 123

The training loop follows the standard MAPPO on-policy pipeline:

  while steps < total_steps:
      collect rollout of length T
      compute GAE advantages
      run K PPO epochs on mini-batches
      log metrics & save checkpoints
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from traffic_rl.config import (
    EnvConfig, MAPPOConfig, TrainingConfig, ExperimentConfig, make_default_config
)
from traffic_rl.env.traffic_env import TrafficEnv
from traffic_rl.marl.mappo import MAPPO
from traffic_rl.utils.logger import ExperimentLogger
from traffic_rl.utils.metrics import MetricsTracker, EpisodeStats
from traffic_rl.utils.visualization import (
    plot_training_curves, plot_emergent_patterns, plot_intersection_usage
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all RNG seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env(
    env_cfg: EnvConfig,
    aggressive_fraction: float = 0.0,
    render_mode: Optional[str] = None,
) -> TrafficEnv:
    """Factory to create and reset a TrafficEnv."""
    env = TrafficEnv(
        config=env_cfg,
        aggressive_fraction=aggressive_fraction,
        render_mode=render_mode,
    )
    return env


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class Trainer:
    """Wrapper that manages the full training lifecycle.

    Parameters
    ----------
    experiment_cfg:
        Complete experiment configuration.
    resume_from:
        Path to a checkpoint to resume training from.
    """

    def __init__(
        self,
        experiment_cfg: Optional[ExperimentConfig] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        self.cfg = experiment_cfg or make_default_config()
        self.env_cfg = self.cfg.env
        self.mappo_cfg = self.cfg.mappo
        self.train_cfg = self.cfg.training

        set_seed(self.train_cfg.seed)

        # Device
        self.device = torch.device(
            "cuda" if self.mappo_cfg.use_gpu and torch.cuda.is_available() else "cpu"
        )
        print(f"[Trainer] Using device: {self.device}")

        # Environment
        self.env = make_env(self.env_cfg, aggressive_fraction=self.cfg.aggressive_fraction)
        obs_dim = self.env.observation_dim
        n_actions = self.env_cfg.n_actions
        global_state_dim = self.env.global_state_dim

        # Algorithm
        self.algo = MAPPO(
            obs_dim=obs_dim,
            n_actions=n_actions,
            global_state_dim=global_state_dim,
            mappo_cfg=self.mappo_cfg,
            training_cfg=self.train_cfg,
            device=self.device,
        )
        print(f"[Trainer] {self.algo.policy}")

        if resume_from and os.path.exists(resume_from):
            self.algo.load(resume_from)
            print(f"[Trainer] Resumed from {resume_from} (step {self.algo.total_steps:,})")

        # Logging
        os.makedirs(self.train_cfg.log_dir, exist_ok=True)
        os.makedirs(self.train_cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(self.train_cfg.plots_dir, exist_ok=True)
        os.makedirs(self.train_cfg.results_dir, exist_ok=True)

        self.logger = ExperimentLogger(
            log_dir=self.train_cfg.log_dir,
            run_name=self.cfg.name,
            config_dict=self.cfg.to_dict(),
        )

        # Metrics
        self.metrics = MetricsTracker(window=100)

        # Running state
        self._global_step = self.algo.total_steps
        self._episode = 0
        self._current_episode_stats = EpisodeStats()

        # History for plots
        self._ep_rewards: List[float] = []
        self._ep_throughputs: List[float] = []
        self._ep_collisions: List[float] = []
        self._ep_travel_times: List[float] = []
        self._entropy_history: List[float] = []
        self._oscillation_history: List[float] = []
        self._priority_history: List[float] = []

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop until *total_steps* is reached."""
        total_steps = self.train_cfg.total_steps
        rollout_len = self.train_cfg.rollout_length
        eval_interval = self.train_cfg.eval_interval
        ckpt_interval = self.train_cfg.checkpoint_interval

        print(f"[Trainer] Starting training for {total_steps:,} steps.")
        start_time = time.time()

        # Initial reset
        obs_dict, _ = self.env.reset()
        global_state = self.env.get_global_state()

        while self._global_step < total_steps:
            # ---- Rollout collection ----
            rollout_start = self._global_step
            for _ in range(rollout_len):
                if self._global_step >= total_steps:
                    break

                # Select actions
                actions_dict = self.algo.select_actions(obs_dict, global_state)

                # Step environment
                next_obs_dict, rewards_dict, terminated, truncated, info = self.env.step(
                    actions_dict
                )
                next_global = self.env.get_global_state()

                # Store transition
                dones_dict: Dict[str, bool] = {
                    aid: (terminated.get(aid, False) or truncated.get(aid, False))
                    for aid in obs_dict
                }
                self.algo.store_transition(
                    obs_dict, global_state, actions_dict,
                    rewards_dict, dones_dict,
                )

                # Update metrics
                self.metrics.record_step(rewards_dict, info, self._current_episode_stats)
                self.logger.set_step(self._global_step)
                self._global_step += 1

                # Handle episode end (all agents truncated, or empty observation)
                episode_ended = (
                    not next_obs_dict
                    or all(truncated.get(aid, False) for aid in obs_dict)
                )
                if episode_ended:
                    # Record finished episode
                    self._end_episode()
                    obs_dict, _ = self.env.reset()
                    global_state = self.env.get_global_state()
                    self._current_episode_stats = EpisodeStats()
                else:
                    obs_dict = next_obs_dict
                    global_state = next_global

            # ---- Policy update ----
            update_metrics = self.algo.update(obs_dict, global_state)
            self.logger.log_training(update_metrics, step=self._global_step)

            # ---- Logging ----
            running = self.metrics.summary()
            self.logger.log_scalars("running", running, step=self._global_step)

            # Emergent pattern detection
            emergent = self.metrics.detect_priority_emergence(self.env._intersections)
            self.logger.log_scalars("emergent", emergent, step=self._global_step)
            self._entropy_history.append(emergent["mean_entropy"])
            self._oscillation_history.append(emergent["mean_oscillation"])
            self._priority_history.append(emergent["priority_emergence"])

            # Progress print every ~10 updates
            if self.algo.update_count % 10 == 0:
                elapsed = time.time() - start_time
                sps = self._global_step / (elapsed + 1e-8)
                eta = (total_steps - self._global_step) / (sps + 1e-8)
                print(
                    f"  step={self._global_step:>8,} | "
                    f"ep={self._episode:>5} | "
                    f"rew={running.get('mean_reward', 0):.2f} | "
                    f"tp={running.get('mean_throughput', 0):.4f} | "
                    f"sps={sps:.0f} | "
                    f"eta={eta/60:.1f}min"
                )

            # ---- Evaluation ----
            if self._global_step % eval_interval < rollout_len:
                eval_metrics = evaluate(
                    self.algo,
                    self.env_cfg,
                    n_episodes=self.train_cfg.eval_episodes,
                    aggressive_fraction=self.cfg.aggressive_fraction,
                    seed=self.train_cfg.seed + 1000,
                )
                self.logger.log_scalars("eval", eval_metrics, step=self._global_step)
                print(
                    f"[Eval] step={self._global_step:,}  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items())
                )

            # ---- Checkpoint ----
            if self._global_step % ckpt_interval < rollout_len:
                ckpt_path = os.path.join(
                    self.train_cfg.checkpoint_dir,
                    f"{self.cfg.name}_step{self._global_step}.pt",
                )
                self.algo.save(ckpt_path)
                print(f"[Trainer] Checkpoint saved → {ckpt_path}")

        # ---- Final save ----
        self._save_final_outputs()
        self.logger.close()
        print(f"[Trainer] Training complete in {(time.time()-start_time)/60:.1f} min.")

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def _end_episode(self) -> None:
        """Finalise per-episode stats and record them."""
        stats = self._current_episode_stats
        self.metrics.record_episode(stats)
        ep_dict = stats.to_dict()
        self.logger.log_episode(ep_dict, episode=self._episode)

        self._ep_rewards.append(stats.total_reward)
        self._ep_throughputs.append(stats.throughput())
        self._ep_collisions.append(stats.collision_rate())
        self._ep_travel_times.append(stats.mean_travel_time() if not
                                     (stats.mean_travel_time() != stats.mean_travel_time())
                                     else 0.0)
        self._episode += 1

    # ------------------------------------------------------------------
    # Final outputs
    # ------------------------------------------------------------------

    def _save_final_outputs(self) -> None:
        """Save final checkpoint, plots, and metrics JSON."""
        # Final checkpoint
        final_ckpt = os.path.join(
            self.train_cfg.checkpoint_dir, f"{self.cfg.name}_final.pt"
        )
        self.algo.save(final_ckpt)
        print(f"[Trainer] Final checkpoint → {final_ckpt}")

        plots_dir = self.train_cfg.plots_dir

        # Training curves
        if self._ep_rewards:
            plot_training_curves(
                rewards=self._ep_rewards,
                throughputs=self._ep_throughputs,
                collision_rates=self._ep_collisions,
                travel_times=self._ep_travel_times,
                save_path=os.path.join(plots_dir, f"{self.cfg.name}_training_curves.png"),
                title=f"{self.cfg.name} – Training Curves",
            )

        # Emergent patterns
        if self._entropy_history:
            plot_emergent_patterns(
                entropy_history=self._entropy_history,
                oscillation_history=self._oscillation_history,
                priority_emergence=self._priority_history,
                save_path=os.path.join(plots_dir, f"{self.cfg.name}_emergent_patterns.png"),
                title=f"{self.cfg.name} – Emergent Patterns",
            )

        # Intersection usage
        plot_intersection_usage(
            intersections=self.env._intersections,
            save_path=os.path.join(plots_dir, f"{self.cfg.name}_intersection_usage.png"),
            title=f"{self.cfg.name} – Intersection Usage",
        )

        # Metrics JSON
        import json
        results_path = os.path.join(
            self.train_cfg.results_dir, f"{self.cfg.name}_metrics.json"
        )
        with open(results_path, "w") as f:
            json.dump(self.metrics.export(), f, indent=2, default=str)
        print(f"[Trainer] Metrics saved → {results_path}")

        plt_path = os.path.join(plots_dir, f"{self.cfg.name}_training_curves.png")
        print(f"[Trainer] Plots saved in {plots_dir}/")


# ---------------------------------------------------------------------------
# Standalone evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    algo: MAPPO,
    env_cfg: EnvConfig,
    n_episodes: int = 5,
    aggressive_fraction: float = 0.0,
    seed: int = 999,
    render: bool = False,
) -> Dict[str, float]:
    """Run *n_episodes* greedy rollouts and return aggregate metrics.

    Parameters
    ----------
    algo:
        Trained MAPPO instance.
    env_cfg:
        Environment configuration.
    n_episodes:
        Number of evaluation episodes.
    aggressive_fraction:
        Fraction of aggressive agents in the evaluation environment.
    seed:
        RNG seed for the evaluation environment.
    render:
        If True, render each step (slow).

    Returns
    -------
    Dict of mean metric values across *n_episodes*.
    """
    eval_env = make_env(
        env_cfg,
        aggressive_fraction=aggressive_fraction,
        render_mode="human" if render else None,
    )
    metrics = MetricsTracker()

    for ep in range(n_episodes):
        obs_dict, _ = eval_env.reset(seed=seed + ep)
        global_state = eval_env.get_global_state()
        ep_stats = EpisodeStats()

        for _ in range(env_cfg.max_steps):
            if not obs_dict:
                break
            actions_dict = algo.select_actions(obs_dict, global_state, deterministic=True)
            obs_dict, rewards_dict, terminated, truncated, info = eval_env.step(actions_dict)
            global_state = eval_env.get_global_state()
            metrics.record_step(rewards_dict, info, ep_stats)

            if not obs_dict or all(truncated.get(a, False) for a in obs_dict):
                break

        metrics.record_episode(ep_stats)

    eval_env.close()
    return metrics.summary()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FlowMergence MAPPO")
    p.add_argument("--steps", type=int, default=None, help="Total training steps")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--vehicles", type=int, default=None, help="Max vehicles")
    p.add_argument("--grid", type=int, default=None, help="Grid size (N for NxN)")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    p.add_argument("--name", type=str, default="mappo_run", help="Run name")
    p.add_argument("--aggressive", type=float, default=0.0, help="Aggressive agent fraction")
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    cfg = make_default_config()
    cfg.name = args.name
    cfg.aggressive_fraction = args.aggressive

    if args.steps:
        cfg.training.total_steps = args.steps
    if args.seed is not None:
        cfg.training.seed = args.seed
        cfg.env.seed = args.seed
    if args.vehicles:
        cfg.env.max_vehicles = args.vehicles
    if args.grid:
        cfg.env.grid_size = args.grid

    trainer = Trainer(experiment_cfg=cfg, resume_from=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
