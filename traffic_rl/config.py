"""
config.py – Centralised, dataclass-based configuration for FlowMergence.

Every hyper-parameter that controls environment behaviour, the learning
algorithm or an experiment lives here.  Keeping all values in one place
makes experiments reproducible and easy to report in a paper.

Usage
-----
>>> from traffic_rl.config import EnvConfig, MAPPOConfig, TrainingConfig
>>> env_cfg = EnvConfig(grid_size=2, max_vehicles=80)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import json
import yaml


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    """Scalar weights that balance each reward component.

    Positive weights encourage a behaviour; negative (penalty) components
    should be kept positive here – they are subtracted in the reward formula.
    """
    progress: float = 1.0          # reward per unit distance toward destination
    waiting_time: float = 0.05     # penalty per step spent waiting
    collision: float = 50.0        # one-time penalty on collision
    blocking: float = 5.0          # penalty per step blocking an intersection
    sudden_brake: float = 1.0      # penalty for abrupt deceleration
    goal_reached: float = 20.0     # bonus on reaching destination


@dataclass
class EnvConfig:
    """Parameters that define the traffic simulation."""

    # Grid
    grid_size: int = 2                     # NxN intersection grid  (2 → 2×2 grid)
    road_length: float = 100.0             # metres between intersections
    lane_width: float = 3.5                # metres per lane (single lane each direction)

    # Vehicles
    max_vehicles: int = 50                 # hard cap on simultaneous agents
    min_vehicles: int = 10                 # minimum agents kept alive
    spawn_rate: float = 0.3               # probability of spawning a vehicle each step
    vehicle_length: float = 4.5           # metres
    vehicle_width: float = 2.0            # metres
    max_speed: float = 13.9               # m/s ≈ 50 km/h
    max_acceleration: float = 2.5         # m/s²
    max_deceleration: float = 4.5         # m/s² (comfortable braking)
    min_headway: float = 2.0              # minimum gap between vehicles (metres)

    # Observation / partial observability
    obs_radius: float = 50.0              # metres each vehicle can see
    obs_dim: int = 32                     # fixed size of observation vector
    comm_range: float = 30.0             # metres for communication channel

    # Intersection
    intersection_radius: float = 6.0     # conflict-zone radius (metres)
    max_queue_length: int = 10           # vehicles that can queue per approach

    # Simulation
    dt: float = 0.1                       # seconds per step
    max_steps: int = 2000                 # steps per episode
    seed: Optional[int] = 42

    # Reward
    reward: RewardWeights = field(default_factory=RewardWeights)

    # Action space size (fixed; see env/traffic_env.py for meanings)
    n_actions: int = 9


# ---------------------------------------------------------------------------
# MAPPO / Neural networks
# ---------------------------------------------------------------------------

@dataclass
class MAPPOConfig:
    """Hyper-parameters for the MAPPO algorithm."""

    # Optimisation
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99                   # discount factor
    gae_lambda: float = 0.95              # GAE λ
    clip_epsilon: float = 0.2             # PPO clipping ratio
    entropy_coef: float = 0.01            # entropy regularisation coefficient
    value_loss_coef: float = 0.5          # critic loss weight
    max_grad_norm: float = 0.5            # gradient clipping

    # Update schedule
    ppo_epochs: int = 10                  # optimisation epochs per rollout
    mini_batch_size: int = 256

    # Architecture
    hidden_dim: int = 256                 # width of hidden layers
    n_layers: int = 3                     # number of hidden layers (actor & critic)
    use_rnn: bool = False                 # use GRU recurrent actor
    rnn_hidden_dim: int = 128

    # Centralised critic
    use_centralised_critic: bool = True   # MAPPO: critic sees global state
    global_state_dim: int = 128           # dimension of the global state encoding

    # Misc
    use_gpu: bool = True
    share_policy: bool = True             # all agents share one policy network


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Controls the outer training loop."""

    total_steps: int = 5_000_000         # environment steps
    rollout_length: int = 256            # steps collected before each update
    eval_interval: int = 50_000          # steps between evaluations
    eval_episodes: int = 5               # episodes per evaluation
    checkpoint_interval: int = 100_000  # steps between checkpoint saves
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    results_dir: str = "results"
    plots_dir: str = "plots"
    seed: int = 42
    num_envs: int = 1                    # parallel environments (future extension)


# ---------------------------------------------------------------------------
# Experiment presets
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """High-level experiment parameters that override defaults."""

    name: str = "default"
    description: str = ""

    env: EnvConfig = field(default_factory=EnvConfig)
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Density sweep
    density_values: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])

    # Heterogeneity
    aggressive_fraction: float = 0.3     # fraction of aggressive agents
    cooperative_fraction: float = 0.7

    # Ablation
    ablation_targets: List[str] = field(
        default_factory=lambda: ["no_comm", "no_entropy", "no_centralised_critic"]
    )

    # Visibility sweep
    visibility_values: List[float] = field(
        default_factory=lambda: [10.0, 25.0, 50.0, 100.0]
    )

    def to_dict(self) -> Dict:
        """Serialise to plain Python dict (e.g. for JSON logging)."""
        return asdict(self)

    def save_json(self, path: str) -> None:
        """Persist configuration as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, path: str) -> None:
        """Persist configuration as YAML."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """Load an ExperimentConfig from a JSON file (partial override)."""
        with open(path) as f:
            data = json.load(f)
        cfg = cls()
        # Flat override – only top-level keys supported for brevity.
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Convenience factory for common presets
# ---------------------------------------------------------------------------

def make_default_config() -> ExperimentConfig:
    """Return the default experiment configuration."""
    return ExperimentConfig(name="default", description="Baseline MAPPO run.")


def make_density_config(density: float, seed: int = 42) -> ExperimentConfig:
    """Config for a single density-sweep point."""
    cfg = ExperimentConfig(name=f"density_{density:.2f}")
    cfg.env.spawn_rate = density
    cfg.training.seed = seed
    cfg.env.seed = seed
    return cfg


def make_heterogeneous_config(aggressive_frac: float = 0.5) -> ExperimentConfig:
    """Config for the agent-heterogeneity experiment."""
    cfg = ExperimentConfig(name=f"hetero_agg{int(aggressive_frac*100)}")
    cfg.aggressive_fraction = aggressive_frac
    cfg.cooperative_fraction = 1.0 - aggressive_frac
    return cfg
