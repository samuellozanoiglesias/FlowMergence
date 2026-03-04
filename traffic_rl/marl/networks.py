"""
networks.py – Neural network architectures for MAPPO.

Contains
--------
* ``MLP``          – generic multi-layer perceptron helper
* ``ActorNetwork`` – decentralised policy (maps local obs → action logits)
* ``CriticNetwork``– centralised value function (maps global state → V)
* ``RecurrentActorNetwork`` – optional GRU-based actor for partial observability
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from traffic_rl.config import MAPPOConfig


# ---------------------------------------------------------------------------
# Generic building block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Configurable multi-layer perceptron with layer normalisation and ReLU.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input vector.
    output_dim:
        Dimensionality of the output vector.
    hidden_dim:
        Width of each hidden layer.
    n_layers:
        Number of hidden layers (≥ 1).
    use_layer_norm:
        Whether to apply LayerNorm after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Actor network
# ---------------------------------------------------------------------------

class ActorNetwork(nn.Module):
    """Decentralised actor: local observation → discrete action distribution.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the per-agent observation.
    n_actions:
        Number of discrete actions.
    config:
        MAPPO hyper-parameters.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: Optional[MAPPOConfig] = None,
    ) -> None:
        super().__init__()
        cfg = config or MAPPOConfig()
        self.backbone = MLP(
            input_dim=obs_dim,
            output_dim=n_actions,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return raw action logits for a batch of observations.

        Parameters
        ----------
        obs:
            Float tensor of shape ``(batch, obs_dim)`` or ``(obs_dim,)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, n_actions)`` or ``(n_actions,)``.
        """
        return self.backbone(obs)

    def get_action_distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """Return a Categorical distribution over actions."""
        logits = self.forward(obs)
        return torch.distributions.Categorical(logits=logits)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) action and return action, log-prob, entropy.

        Parameters
        ----------
        obs:
            Float tensor of shape ``(batch, obs_dim)``.
        deterministic:
            If True, returns the argmax action without sampling.

        Returns
        -------
        action, log_prob, entropy  –  all of shape ``(batch,)``.
        """
        dist = self.get_action_distribution(obs)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log-probs and entropy for *actions* given *obs*.

        Used during the PPO optimisation phase.
        """
        dist = self.get_action_distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


# ---------------------------------------------------------------------------
# Recurrent actor (optional, for partial observability experiments)
# ---------------------------------------------------------------------------

class RecurrentActorNetwork(nn.Module):
    """GRU-based actor for partial observability.

    The observation is first encoded by an MLP, then processed by a GRU
    cell whose hidden state is carried across time-steps.

    Parameters
    ----------
    obs_dim, n_actions, config:
        Same as ``ActorNetwork``.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: Optional[MAPPOConfig] = None,
    ) -> None:
        super().__init__()
        cfg = config or MAPPOConfig()
        self.rnn_hidden_dim = cfg.rnn_hidden_dim

        # Encoder: obs → embedding
        self.encoder = MLP(obs_dim, cfg.rnn_hidden_dim, cfg.hidden_dim, n_layers=2)
        # GRU memory
        self.gru = nn.GRU(
            input_size=cfg.rnn_hidden_dim,
            hidden_size=cfg.rnn_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # Action head
        self.action_head = nn.Linear(cfg.rnn_hidden_dim, n_actions)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        obs:
            Shape ``(batch, seq_len, obs_dim)`` or ``(batch, obs_dim)`` for
            single-step inference.
        hidden:
            GRU hidden state ``(1, batch, rnn_hidden_dim)`` or None.

        Returns
        -------
        logits, new_hidden
        """
        # Handle both 2-D (batch, obs_dim) and 3-D (batch, seq, obs_dim) inputs
        squeeze = False
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)   # add seq dimension
            squeeze = True

        batch, seq_len, _ = obs.shape
        # Encode each step
        enc = self.encoder(obs.reshape(batch * seq_len, -1))
        enc = enc.reshape(batch, seq_len, -1)

        if hidden is None:
            hidden = torch.zeros(1, batch, self.rnn_hidden_dim, device=obs.device)

        rnn_out, new_hidden = self.gru(enc, hidden)  # (batch, seq, rnn_dim)
        logits = self.action_head(rnn_out)            # (batch, seq, n_actions)

        if squeeze:
            logits = logits.squeeze(1)

        return logits, new_hidden

    def act(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob, entropy, new_hidden)."""
        logits, new_hidden = self.forward(obs, hidden)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), new_hidden


# ---------------------------------------------------------------------------
# Centralised critic
# ---------------------------------------------------------------------------

class CriticNetwork(nn.Module):
    """Centralised value function: global state → scalar value estimate.

    In MAPPO each agent shares one critic that conditions on a joint/global
    observation to reduce variance.

    Parameters
    ----------
    global_state_dim:
        Dimensionality of the (concatenated) global state.
    config:
        MAPPO hyper-parameters.
    """

    def __init__(
        self,
        global_state_dim: int,
        config: Optional[MAPPOConfig] = None,
    ) -> None:
        super().__init__()
        cfg = config or MAPPOConfig()
        self.backbone = MLP(
            input_dim=global_state_dim,
            output_dim=1,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
        )
        # Small value-function output head (orthogonal init with small scale)
        nn.init.orthogonal_(
            list(self.backbone.net.children())[-1].weight, gain=0.01  # type: ignore[arg-type]
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return estimated value V(s).

        Parameters
        ----------
        global_state:
            Float tensor of shape ``(batch, global_state_dim)``.

        Returns
        -------
        torch.Tensor
            Value estimates of shape ``(batch, 1)`` or ``(batch,)`` if squeezed.
        """
        return self.backbone(global_state).squeeze(-1)
