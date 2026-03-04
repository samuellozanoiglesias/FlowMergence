"""
policy.py – Shared policy wrapper for decentralised execution.

``SharedPolicy`` owns the actor (and optionally critic) networks and exposes
a clean interface consumed by the ``MAPPO`` trainer.  All agents share the
same weights, achieving parameter sharing.

Centralised-training / decentralised-execution (CTDE) is implemented by:
* Actors receiving only local observations.
* The critic receiving the global state during training (MAPPO).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from traffic_rl.config import EnvConfig, MAPPOConfig
from traffic_rl.marl.networks import ActorNetwork, CriticNetwork, RecurrentActorNetwork


class SharedPolicy(nn.Module):
    """Shared actor-critic policy for the MAPPO algorithm.

    Parameters
    ----------
    obs_dim:
        Dimensionality of a single agent's observation.
    n_actions:
        Number of discrete actions.
    global_state_dim:
        Dimensionality of the global state used by the centralised critic.
    mappo_cfg:
        MAPPO hyper-parameters.
    device:
        Torch device to allocate tensors on.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        global_state_dim: int,
        mappo_cfg: Optional[MAPPOConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.cfg = mappo_cfg or MAPPOConfig()
        self.device = device or torch.device(
            "cuda" if self.cfg.use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.global_state_dim = global_state_dim

        # Actor network
        if self.cfg.use_rnn:
            self.actor = RecurrentActorNetwork(obs_dim, n_actions, self.cfg)
        else:
            self.actor = ActorNetwork(obs_dim, n_actions, self.cfg)  # type: ignore[assignment]

        # Centralised critic
        self.critic: Optional[CriticNetwork] = None
        if self.cfg.use_centralised_critic:
            self.critic = CriticNetwork(global_state_dim, self.cfg)

        # Move to device
        self.to(self.device)

    # ------------------------------------------------------------------
    # Action inference  (decentralised)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act_batch(
        self,
        obs: np.ndarray,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
        """Select actions for a batch of agents.

        Parameters
        ----------
        obs:
            Observation matrix of shape ``(n_agents, obs_dim)``.
        hidden:
            RNN hidden state; only used if ``cfg.use_rnn=True``.
        deterministic:
            Argmax instead of sampling.

        Returns
        -------
        actions : np.ndarray  (n_agents,)
        log_probs : np.ndarray  (n_agents,)
        new_hidden : torch.Tensor or None
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

        if self.cfg.use_rnn:
            actions_t, lp_t, _, new_hidden = self.actor.act(
                obs_t, hidden, deterministic
            )
        else:
            actions_t, lp_t, _ = self.actor.act(obs_t, deterministic)
            new_hidden = None

        return (
            actions_t.cpu().numpy(),
            lp_t.cpu().numpy(),
            new_hidden,
        )

    # ------------------------------------------------------------------
    # Value estimation  (centralised)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_values_batch(
        self, global_states: np.ndarray
    ) -> np.ndarray:
        """Estimate V(s) for a batch of global states.

        Parameters
        ----------
        global_states:
            Array of shape ``(n_agents, global_state_dim)`` (the same global
            state is replicated for each agent so the critic produces one
            value per agent for bookkeeping).

        Returns
        -------
        np.ndarray of shape ``(n_agents,)``.
        """
        if self.critic is None:
            return np.zeros(len(global_states), dtype=np.float32)
        gs_t = torch.tensor(global_states, dtype=torch.float32, device=self.device)
        values = self.critic(gs_t)
        return values.cpu().numpy()

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log-probs and entropy for the given (obs, action) pairs.

        Used during the PPO update step.
        """
        if self.cfg.use_rnn:
            logits, _ = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
        else:
            dist = self.actor.get_action_distribution(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    def compute_values(self, global_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of the centralised critic."""
        if self.critic is None:
            return torch.zeros(global_states.shape[0], device=self.device)
        return self.critic(global_states)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save actor and critic weights to *path* (.pt file)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "actor": self.actor.state_dict(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "global_state_dim": self.global_state_dim,
        }
        if self.critic is not None:
            state["critic"] = self.critic.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load weights from a checkpoint file."""
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        if "critic" in state and self.critic is not None:
            self.critic.load_state_dict(state["critic"])

    # ------------------------------------------------------------------
    # Optimiser factory
    # ------------------------------------------------------------------

    def make_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.Optimizer]]:
        """Return (actor_optimizer, critic_optimizer)."""
        actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=self.cfg.lr_actor, eps=1e-5
        )
        if self.critic is not None:
            critic_opt = torch.optim.Adam(
                self.critic.parameters(), lr=self.cfg.lr_critic, eps=1e-5
            )
        else:
            critic_opt = None
        return actor_opt, critic_opt

    def parameter_count(self) -> Dict[str, int]:
        """Return dict of parameter counts for actor/critic."""
        actor_params = sum(p.numel() for p in self.actor.parameters())
        result = {"actor": actor_params}
        if self.critic is not None:
            critic_params = sum(p.numel() for p in self.critic.parameters())
            result["critic"] = critic_params
        return result

    def __repr__(self) -> str:  # pragma: no cover
        counts = self.parameter_count()
        return (
            f"SharedPolicy(obs={self.obs_dim}, actions={self.n_actions}, "
            f"actor_params={counts.get('actor', 0):,}, "
            f"critic_params={counts.get('critic', 0):,}, "
            f"device={self.device})"
        )
