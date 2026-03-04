"""
buffer.py – Rollout buffer for MAPPO experience collection.

Stores trajectories of ``(observation, global_state, action, reward,
done, log_prob, value)`` tuples and computes Generalised Advantage
Estimation (GAE) at the end of each rollout.

Design
------
* All tensors are stored in CPU memory during collection and moved to the
  target device only during mini-batch sampling.
* The buffer is multi-agent aware: each stored element is a list (one
  entry per active agent at that step).  Sequences are flattened across
  agents and time when sampling mini-batches.
* The buffer is **flat** (no episode boundary handling beyond the done
  flags) which is appropriate for the mostly non-terminating traffic
  episodes.
"""

from __future__ import annotations

from typing import Dict, Generator, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from traffic_rl.config import MAPPOConfig


# ---------------------------------------------------------------------------
# Named tuple for a single training sample
# ---------------------------------------------------------------------------

class ExperienceBatch(NamedTuple):
    """Mini-batch returned by the buffer iterator."""
    observations: torch.Tensor      # (batch, obs_dim)
    global_states: torch.Tensor     # (batch, global_state_dim)
    actions: torch.Tensor           # (batch,)  int64
    old_log_probs: torch.Tensor     # (batch,)
    advantages: torch.Tensor        # (batch,)
    returns: torch.Tensor           # (batch,)
    values: torch.Tensor            # (batch,)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-length rollout buffer for multi-agent trajectories.

    Parameters
    ----------
    rollout_length:
        Number of environment steps to collect before an update.
    obs_dim:
        Dimensionality of a single agent observation.
    global_state_dim:
        Dimensionality of the global state.
    config:
        MAPPO hyper-parameters (gamma, gae_lambda, mini_batch_size).
    device:
        PyTorch device for tensor operations during update.
    """

    def __init__(
        self,
        rollout_length: int,
        obs_dim: int,
        global_state_dim: int,
        config: Optional[MAPPOConfig] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.rollout_length = rollout_length
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        self.cfg = config or MAPPOConfig()
        self.device = device

        # Storage lists (ragged over agents per step)
        self._obs: List[np.ndarray] = []            # each entry: (n_agents, obs_dim)
        self._global_states: List[np.ndarray] = []  # each: (global_state_dim,)
        self._actions: List[np.ndarray] = []        # each: (n_agents,)
        self._rewards: List[np.ndarray] = []        # each: (n_agents,)
        self._dones: List[np.ndarray] = []          # each: (n_agents,)
        self._log_probs: List[np.ndarray] = []      # each: (n_agents,)
        self._values: List[np.ndarray] = []         # each: (n_agents,)

        self._step = 0

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,           # (n_agents, obs_dim)
        global_state: np.ndarray,  # (global_state_dim,)
        actions: np.ndarray,       # (n_agents,)
        rewards: np.ndarray,       # (n_agents,)
        dones: np.ndarray,         # (n_agents,)  bool / float
        log_probs: np.ndarray,     # (n_agents,)
        values: np.ndarray,        # (n_agents,)
    ) -> None:
        """Append one step of experience from all active agents."""
        self._obs.append(obs.copy())
        self._global_states.append(global_state.copy())
        self._actions.append(actions.copy())
        self._rewards.append(rewards.copy())
        self._dones.append(dones.astype(np.float32))
        self._log_probs.append(log_probs.copy())
        self._values.append(values.copy())
        self._step += 1

    def is_full(self) -> bool:
        """True once *rollout_length* steps have been collected."""
        return self._step >= self.rollout_length

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(
        self, last_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        last_values:
            Bootstrapped value estimates for the step *after* the buffer
            (shape ``(n_agents,)`` for the last active set of agents).

        Returns
        -------
        advantages, returns – both flattened over (steps × agents).
        """
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        # We process each time step backward; since agent counts may vary per
        # step we pad uniformly to a fixed width for simplicity.
        T = len(self._rewards)
        if T == 0:
            return np.array([]), np.array([])

        # Determine max agents across all steps
        max_agents = max(len(r) for r in self._rewards)

        def pad(arr: np.ndarray, width: int) -> np.ndarray:
            if len(arr) >= width:
                return arr[:width]
            return np.concatenate([arr, np.zeros(width - len(arr))])

        rewards_mat = np.array([pad(r, max_agents) for r in self._rewards])   # (T, A)
        dones_mat = np.array([pad(d, max_agents) for d in self._dones])        # (T, A)
        values_mat = np.array([pad(v, max_agents) for v in self._values])      # (T, A)

        last_val_padded = pad(last_values, max_agents)

        advantages = np.zeros_like(rewards_mat)
        gae = np.zeros(max_agents)

        for t in reversed(range(T)):
            next_val = last_val_padded if t == T - 1 else values_mat[t + 1]
            next_non_term = 1.0 - dones_mat[t]
            delta = rewards_mat[t] + gamma * next_val * next_non_term - values_mat[t]
            gae = delta + gamma * lam * next_non_term * gae
            advantages[t] = gae

        returns = advantages + values_mat

        # Flatten and return raw (un-normalised) advantages + returns
        adv_flat = advantages.flatten().astype(np.float32)
        ret_flat = returns.flatten().astype(np.float32)
        return adv_flat, ret_flat

    # ------------------------------------------------------------------
    # Mini-batch iterator
    # ------------------------------------------------------------------

    def get_mini_batches(
        self, last_values: np.ndarray
    ) -> Generator[ExperienceBatch, None, None]:
        """Compute advantages then yield random mini-batches.

        Parameters
        ----------
        last_values:
            Bootstrap values for the step following the rollout.

        Yields
        ------
        ExperienceBatch
        """
        adv_flat, ret_flat = self.compute_returns_and_advantages(last_values)

        if len(adv_flat) == 0:
            return

        # Normalise advantages
        adv_mean = adv_flat.mean()
        adv_std = adv_flat.std() + 1e-8
        adv_flat = (adv_flat - adv_mean) / adv_std

        # Flatten all fields to (total_samples, ...)
        T = len(self._rewards)
        max_agents = max(len(r) for r in self._rewards)

        def pad_to(arr: np.ndarray, width: int) -> np.ndarray:
            if len(arr) >= width:
                return arr[:width]
            return np.concatenate([arr, np.zeros(width - len(arr))])

        obs_list, gs_list, act_list, lp_list, val_list = [], [], [], [], []
        for t in range(T):
            n = max_agents
            # Observations: pad along agent axis
            obs_t = self._obs[t]
            if obs_t.shape[0] < n:
                pad_rows = np.zeros((n - obs_t.shape[0], self.obs_dim), dtype=np.float32)
                obs_t = np.concatenate([obs_t, pad_rows], axis=0)
            else:
                obs_t = obs_t[:n]
            obs_list.append(obs_t)

            gs_list.append(np.tile(self._global_states[t][np.newaxis], (n, 1)))
            act_list.append(pad_to(self._actions[t], n))
            lp_list.append(pad_to(self._log_probs[t], n))
            val_list.append(pad_to(self._values[t], n))

        obs_all = np.concatenate(obs_list, axis=0)                  # (T*A, obs_dim)
        gs_all = np.concatenate(gs_list, axis=0)                    # (T*A, gs_dim)
        act_all = np.concatenate(act_list).astype(np.int64)         # (T*A,)
        lp_all = np.concatenate(lp_list).astype(np.float32)         # (T*A,)
        val_all = np.concatenate(val_list).astype(np.float32)       # (T*A,)

        N = len(adv_flat)
        indices = np.arange(N)
        np.random.shuffle(indices)

        batch_size = self.cfg.mini_batch_size
        for start in range(0, N, batch_size):
            idx = indices[start: start + batch_size]
            yield ExperienceBatch(
                observations=torch.tensor(obs_all[idx], dtype=torch.float32).to(self.device),
                global_states=torch.tensor(gs_all[idx], dtype=torch.float32).to(self.device),
                actions=torch.tensor(act_all[idx], dtype=torch.int64).to(self.device),
                old_log_probs=torch.tensor(lp_all[idx], dtype=torch.float32).to(self.device),
                advantages=torch.tensor(adv_flat[idx], dtype=torch.float32).to(self.device),
                returns=torch.tensor(ret_flat[idx], dtype=torch.float32).to(self.device),
                values=torch.tensor(val_all[idx], dtype=torch.float32).to(self.device),
            )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored experience (called after each policy update)."""
        self._obs.clear()
        self._global_states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        self._log_probs.clear()
        self._values.clear()
        self._step = 0

    def __len__(self) -> int:
        return self._step
