"""
FlowMergence – Emergent Traffic Coordination via Multi-Agent Reinforcement Learning.

Top-level package initialiser. Exposes the most frequently used public symbols so
that user code only needs to import from ``traffic_rl``.
"""

from traffic_rl.config import EnvConfig, MAPPOConfig, TrainingConfig, ExperimentConfig

__version__ = "0.1.0"
__all__ = [
    "EnvConfig",
    "MAPPOConfig",
    "TrainingConfig",
    "ExperimentConfig",
]
