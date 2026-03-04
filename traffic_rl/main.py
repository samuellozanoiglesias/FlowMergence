"""
main.py – Top-level command dispatcher for FlowMergence.

Commands
--------
    flowmergence train          – run MAPPO training
    flowmergence eval           – evaluate a checkpoint
    flowmergence density        – run density sweep experiment
    flowmergence heterogeneity  – run heterogeneity experiment
    flowmergence ablation       – run ablation study
    flowmergence demo           – quick demo (100 steps, rendered)

Examples
--------
    # Train with defaults
    python -m traffic_rl.main train

    # Evaluate a checkpoint
    python -m traffic_rl.main eval --checkpoint checkpoints/mappo_run_final.pt

    # Run density sweep
    python -m traffic_rl.main density --steps 200000

    # Quick sanity-check demo
    python -m traffic_rl.main demo
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """Launch the training loop."""
    from traffic_rl.train import main as train_main
    # Patch sys.argv so the sub-module argument parser sees clean args
    sys.argv = [sys.argv[0]] + _remaining(args)
    train_main()


def cmd_eval(args: argparse.Namespace) -> None:
    """Launch checkpoint evaluation."""
    from traffic_rl.evaluate import main as eval_main
    sys.argv = [sys.argv[0]] + _remaining(args)
    eval_main()


def cmd_density(args: argparse.Namespace) -> None:
    """Launch density sweep experiment."""
    from traffic_rl.experiments.density_experiment import main as density_main
    sys.argv = [sys.argv[0]] + _remaining(args)
    density_main()


def cmd_heterogeneity(args: argparse.Namespace) -> None:
    """Launch heterogeneity experiment."""
    from traffic_rl.experiments.heterogeneity_experiment import main as hetero_main
    sys.argv = [sys.argv[0]] + _remaining(args)
    hetero_main()


def cmd_ablation(args: argparse.Namespace) -> None:
    """Launch ablation study."""
    from traffic_rl.experiments.ablation_study import main as ablation_main
    sys.argv = [sys.argv[0]] + _remaining(args)
    ablation_main()


def cmd_demo(args: argparse.Namespace) -> None:
    """Run a short interactive demo to verify the environment and policy."""
    import numpy as np
    from traffic_rl.config import EnvConfig
    from traffic_rl.env.traffic_env import TrafficEnv

    print("[Demo] Creating environment (2×2 grid, 20 vehicles)…")
    cfg = EnvConfig(max_vehicles=20, max_steps=200, seed=0)
    env = TrafficEnv(config=cfg, render_mode="rgb_array")

    obs_dict, info = env.reset()
    print(f"[Demo] Reset done.  Active agents: {env.n_agents}")
    print(f"[Demo] Obs dim: {env.observation_dim}, Global state dim: {env.global_state_dim}")

    # Random policy sanity check
    total_reward = 0.0
    for step in range(200):
        actions = {aid: env.action_space.sample() for aid in env.agent_ids}
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        total_reward += sum(rewards.values())
        if step % 50 == 0:
            print(
                f"  step={step:3d}  agents={info['n_vehicles']:3d}  "
                f"collisions={info['n_collisions']:2d}  "
                f"arrived={info['n_arrived']:2d}  "
                f"speed={info['mean_speed']:.2f}"
            )
        if not obs_dict:
            obs_dict, _ = env.reset()
    env.close()

    print(f"\n[Demo] Completed 200 steps with random policy.")
    print(f"[Demo] Total reward (random): {total_reward:.2f}")
    print(f"[Demo] Environment OK ✓")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remaining(args: argparse.Namespace):
    """Extract leftover CLI args after the sub-command."""
    return getattr(args, "remainder", [])


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowmergence",
        description="FlowMergence – Emergent Traffic Coordination via MARL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Sub-command to run")

    # Each sub-command passes remaining args through to the child parser
    for name in ("train", "eval", "density", "heterogeneity", "ablation", "demo"):
        sub_p = sub.add_parser(name, help=f"Run the '{name}' sub-command")
        sub_p.add_argument("remainder", nargs=argparse.REMAINDER)

    return parser


def main() -> None:
    """Entry point registered in setup.py."""
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "train": cmd_train,
        "eval": cmd_eval,
        "density": cmd_density,
        "heterogeneity": cmd_heterogeneity,
        "ablation": cmd_ablation,
        "demo": cmd_demo,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
