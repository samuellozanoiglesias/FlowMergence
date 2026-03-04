"""
mappo.py – Multi-Agent Proximal Policy Optimisation (MAPPO) trainer.

Implements the CTDE variant of PPO described in
    Yu et al. (2021) "The Surprising Effectiveness of MAPPO in Cooperative
    Multi-Agent Games". arXiv:2103.01955

Algorithm outline per iteration
--------------------------------
1. Collect a rollout of length T using the shared policy (decentralised).
2. Compute GAE advantages using the centralised critic.
3. Optimise actor with PPO-clip + entropy regularisation.
4. Optimise critic with MSE value loss.
5. Repeat for K epochs over mini-batches.

Extending to other algorithms
------------------------------
New algorithms (MADDPG, QMIX, …) should inherit from ``BaseAlgorithm`` and
implement ``select_actions``, ``update``, and ``save / load``.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from traffic_rl.config import EnvConfig, MAPPOConfig, TrainingConfig
from traffic_rl.marl.buffer import RolloutBuffer
from traffic_rl.marl.policy import SharedPolicy


# ---------------------------------------------------------------------------
# Abstract base (plug-in interface for alternative algorithms)
# ---------------------------------------------------------------------------

class BaseAlgorithm(ABC):
    """Abstract base class for multi-agent RL algorithms.

    Subclass this to add MADDPG, QMIX, IQL, or any other algorithm while
    reusing the existing training loop in ``train.py``.
    """

    @abstractmethod
    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        global_state: np.ndarray,
        deterministic: bool = False,
    ) -> Dict[str, int]:
        """Return a dict mapping agent_id → discrete action."""

    @abstractmethod
    def store_transition(
        self,
        obs_dict: Dict[str, np.ndarray],
        global_state: np.ndarray,
        actions_dict: Dict[str, int],
        rewards_dict: Dict[str, float],
        dones_dict: Dict[str, bool],
    ) -> None:
        """Store one transition in the internal buffer."""

    @abstractmethod
    def update(self, last_obs: Dict[str, np.ndarray], last_global: np.ndarray) -> Dict[str, float]:
        """Run a learning update and return a dict of training metrics."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the algorithm state to *path*."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore the algorithm state from *path*."""


# ---------------------------------------------------------------------------
# MAPPO
# ---------------------------------------------------------------------------

class MAPPO(BaseAlgorithm):
    """Multi-Agent PPO with a centralised critic (MAPPO / HAPPO variant).

    Parameters
    ----------
    obs_dim:
        Dimensionality of a single agent's observation.
    n_actions:
        Number of available discrete actions.
    global_state_dim:
        Dimensionality of the global state (for the centralised critic).
    mappo_cfg:
        MAPPO hyper-parameters.
    training_cfg:
        Training-loop configuration (rollout length, etc.).
    device:
        Torch device.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        global_state_dim: int,
        mappo_cfg: Optional[MAPPOConfig] = None,
        training_cfg: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = mappo_cfg or MAPPOConfig()
        self.train_cfg = training_cfg or TrainingConfig()

        self.device = device or torch.device(
            "cuda" if self.cfg.use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Shared policy (actor + centralised critic)
        self.policy = SharedPolicy(
            obs_dim=obs_dim,
            n_actions=n_actions,
            global_state_dim=global_state_dim,
            mappo_cfg=self.cfg,
            device=self.device,
        )

        # Optimisers
        self.actor_opt, self.critic_opt = self.policy.make_optimizers()

        # Rollout buffer
        self.buffer = RolloutBuffer(
            rollout_length=self.train_cfg.rollout_length,
            obs_dim=obs_dim,
            global_state_dim=global_state_dim,
            config=self.cfg,
            device=self.device,
        )

        # Tracking
        self.total_steps: int = 0
        self.update_count: int = 0
        self._last_obs: Dict[str, np.ndarray] = {}
        self._last_log_probs: Dict[str, float] = {}
        self._last_values: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # BaseAlgorithm interface
    # ------------------------------------------------------------------

    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        global_state: np.ndarray,
        deterministic: bool = False,
    ) -> Dict[str, int]:
        """Select actions for all active agents.

        Also caches log-probs and value estimates for the buffer insertion.

        Returns
        -------
        Dict[agent_id, action_int]
        """
        if not obs_dict:
            return {}

        agent_ids = list(obs_dict.keys())
        obs_mat = np.stack([obs_dict[a] for a in agent_ids], axis=0)  # (n_agents, obs_dim)
        n = len(agent_ids)

        # Actor (decentralised execution)
        actions_np, log_probs_np, _ = self.policy.act_batch(obs_mat, deterministic=deterministic)

        # Critic (centralised value estimation)
        gs_mat = np.tile(global_state[np.newaxis], (n, 1))  # replicate per agent
        values_np = self.policy.get_values_batch(gs_mat)

        # Cache for buffer
        self._last_obs = obs_dict
        self._last_log_probs = {a: float(log_probs_np[i]) for i, a in enumerate(agent_ids)}
        self._last_values = {a: float(values_np[i]) for i, a in enumerate(agent_ids)}

        return {a: int(actions_np[i]) for i, a in enumerate(agent_ids)}

    def store_transition(
        self,
        obs_dict: Dict[str, np.ndarray],
        global_state: np.ndarray,
        actions_dict: Dict[str, int],
        rewards_dict: Dict[str, float],
        dones_dict: Dict[str, bool],
    ) -> None:
        """Pack data from the current step and push to the rollout buffer."""
        agent_ids = list(obs_dict.keys())
        if not agent_ids:
            return

        obs_mat = np.stack([obs_dict[a] for a in agent_ids], axis=0)
        actions_arr = np.array([actions_dict.get(a, 0) for a in agent_ids], dtype=np.int64)
        rewards_arr = np.array([rewards_dict.get(a, 0.0) for a in agent_ids], dtype=np.float32)
        dones_arr = np.array([float(dones_dict.get(a, False)) for a in agent_ids], dtype=np.float32)
        log_probs_arr = np.array(
            [self._last_log_probs.get(a, 0.0) for a in agent_ids], dtype=np.float32
        )
        values_arr = np.array(
            [self._last_values.get(a, 0.0) for a in agent_ids], dtype=np.float32
        )

        self.buffer.add(
            obs=obs_mat,
            global_state=global_state,
            actions=actions_arr,
            rewards=rewards_arr,
            dones=dones_arr,
            log_probs=log_probs_arr,
            values=values_arr,
        )
        self.total_steps += len(agent_ids)

    def update(
        self,
        last_obs: Dict[str, np.ndarray],
        last_global: np.ndarray,
    ) -> Dict[str, float]:
        """Run PPO update over the filled rollout buffer.

        Parameters
        ----------
        last_obs:
            Observations at the *next* time step (used for bootstrap value).
        last_global:
            Global state at the next time step.

        Returns
        -------
        dict of training metrics (policy_loss, value_loss, entropy, …).
        """
        # Bootstrap value from last state
        if last_obs:
            agent_ids = list(last_obs.keys())
            gs_mat = np.tile(last_global[np.newaxis], (len(agent_ids), 1))
            last_values = self.policy.get_values_batch(gs_mat)
        else:
            last_values = np.zeros(1, dtype=np.float32)

        metrics: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        for _ in range(self.cfg.ppo_epochs):
            for batch in self.buffer.get_mini_batches(last_values):
                policy_loss, value_loss, entropy, approx_kl, clip_frac = (
                    self._ppo_update(batch)
                )
                metrics["policy_loss"].append(policy_loss)
                metrics["value_loss"].append(value_loss)
                metrics["entropy"].append(entropy)
                metrics["total_loss"].append(policy_loss + value_loss)
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_fraction"].append(clip_frac)

        self.buffer.reset()
        self.update_count += 1

        return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # PPO inner update
    # ------------------------------------------------------------------

    def _ppo_update(self, batch) -> Tuple[float, float, float, float, float]:
        """One mini-batch PPO update step.

        Returns
        -------
        policy_loss, value_loss, entropy_mean, approx_kl, clip_fraction
        """
        obs = batch.observations
        gs = batch.global_states
        actions = batch.actions
        old_log_probs = batch.old_log_probs
        advantages = batch.advantages
        returns = batch.returns

        # --- Actor update ---
        log_probs, entropy = self.policy.evaluate_actions(obs, actions)
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        clip_eps = self.cfg.clip_epsilon
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy regularisation (negative because we maximise entropy)
        entropy_loss = -self.cfg.entropy_coef * entropy.mean()

        actor_total = policy_loss + entropy_loss

        self.actor_opt.zero_grad()
        actor_total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()

        # --- Critic update ---
        value_loss_val = 0.0
        if self.critic_opt is not None and self.policy.critic is not None:
            values_pred = self.policy.compute_values(gs)
            value_loss = F.mse_loss(values_pred, returns)
            critic_total = self.cfg.value_loss_coef * value_loss

            self.critic_opt.zero_grad()
            critic_total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.cfg.max_grad_norm
            )
            self.critic_opt.step()
            value_loss_val = float(value_loss.item())

        # --- Diagnostics ---
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
            clip_frac = float(((ratio - 1.0).abs() > clip_eps).float().mean().item())

        return (
            float(policy_loss.item()),
            value_loss_val,
            float(entropy.mean().item()),
            approx_kl,
            clip_frac,
        )

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save policy weights and optimiser states."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "policy": {
                "actor": self.policy.actor.state_dict(),
            },
            "actor_opt": self.actor_opt.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
        }
        if self.policy.critic is not None:
            state["policy"]["critic"] = self.policy.critic.state_dict()
        if self.critic_opt is not None:
            state["critic_opt"] = self.critic_opt.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Restore weights and optimiser states from a checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.policy.actor.load_state_dict(state["policy"]["actor"])
        if "critic" in state["policy"] and self.policy.critic is not None:
            self.policy.critic.load_state_dict(state["policy"]["critic"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        if "critic_opt" in state and self.critic_opt is not None:
            self.critic_opt.load_state_dict(state["critic_opt"])
        self.total_steps = state.get("total_steps", 0)
        self.update_count = state.get("update_count", 0)

    # ------------------------------------------------------------------
    # Compatibility shim
    # ------------------------------------------------------------------

    def make_optimizers(self):
        """Alias for the policy optimiser factory."""
        return self.policy.make_optimizers()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MAPPO(steps={self.total_steps:,}, updates={self.update_count}, "
            f"device={self.device}, {self.policy})"
        )
