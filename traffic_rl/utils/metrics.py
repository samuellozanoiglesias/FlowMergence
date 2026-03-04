"""
metrics.py – Research-grade metric tracking and emergent-behaviour detection.

This module is intentionally separated from the environment so that the
same metrics infrastructure can be reused across experiments.

Tracked metrics
---------------
* Traffic throughput         – vehicles completing trips per unit time
* Average travel time        – steps from spawn to destination
* Mean waiting time          – mean steps spent with speed < 1 m/s
* Collision rate             – collisions per vehicle per episode
* Gridlock probability       – fraction of steps where any intersection is gridlocked
* Intersection usage patterns – per-direction usage histograms
* Emergent priority patterns – entropy, oscillation, lane dominance

All per-episode values are collected in lists for downstream statistical
analysis (mean ± std, confidence intervals, etc.).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-episode snapshot
# ---------------------------------------------------------------------------

class EpisodeStats:
    """Accumulator for one training episode."""

    def __init__(self) -> None:
        self.total_reward: float = 0.0
        self.n_arrivals: int = 0
        self.n_collisions: int = 0
        self.travel_times: List[float] = []          # steps for each completed trip
        self.waiting_times: List[float] = []         # steps waiting per vehicle
        self.speeds: List[float] = []                # per-step mean speed
        self.gridlock_steps: int = 0
        self.total_steps: int = 0
        self.n_vehicles_spawned: int = 0
        self.episode_length: int = 0

    def throughput(self) -> float:
        """Vehicles per step that completed their journey."""
        if self.episode_length == 0:
            return 0.0
        return self.n_arrivals / self.episode_length

    def mean_travel_time(self) -> float:
        if not self.travel_times:
            return float("nan")
        return float(np.mean(self.travel_times))

    def mean_waiting_time(self) -> float:
        if not self.waiting_times:
            return float("nan")
        return float(np.mean(self.waiting_times))

    def collision_rate(self) -> float:
        if self.n_vehicles_spawned == 0:
            return 0.0
        return self.n_collisions / self.n_vehicles_spawned

    def gridlock_probability(self) -> float:
        if self.episode_length == 0:
            return 0.0
        return self.gridlock_steps / self.episode_length

    def mean_speed(self) -> float:
        if not self.speeds:
            return 0.0
        return float(np.mean(self.speeds))

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_reward": self.total_reward,
            "throughput": self.throughput(),
            "mean_travel_time": self.mean_travel_time(),
            "mean_waiting_time": self.mean_waiting_time(),
            "collision_rate": self.collision_rate(),
            "gridlock_probability": self.gridlock_probability(),
            "mean_speed": self.mean_speed(),
            "n_arrivals": float(self.n_arrivals),
            "n_collisions": float(self.n_collisions),
            "episode_length": float(self.episode_length),
        }


# ---------------------------------------------------------------------------
# Global metrics tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Persistent tracker across many episodes.

    Parameters
    ----------
    window:
        Size of the rolling window used for running averages.
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._episodes: List[EpisodeStats] = []

        # Rolling window buffers
        self._reward_window: Deque[float] = deque(maxlen=window)
        self._throughput_window: Deque[float] = deque(maxlen=window)
        self._travel_time_window: Deque[float] = deque(maxlen=window)
        self._waiting_window: Deque[float] = deque(maxlen=window)
        self._collision_window: Deque[float] = deque(maxlen=window)
        self._gridlock_window: Deque[float] = deque(maxlen=window)
        self._speed_window: Deque[float] = deque(maxlen=window)

        # Intersection analysis
        # Maps intersection_id → {'direction_usage': list, 'service_history': deque}
        self._intersection_data: Dict[int, Dict] = defaultdict(
            lambda: {
                "direction_counts": [0, 0, 0, 0],
                "service_history": deque(maxlen=1000),
                "entropy_history": [],
                "oscillation_history": [],
            }
        )

        # Emergent pattern scores (appended each episode)
        self._priority_entropy: List[float] = []
        self._oscillation_scores: List[float] = []
        self._lane_dominance: List[float] = []

        self.total_steps: int = 0

    # ------------------------------------------------------------------
    # Episode recording
    # ------------------------------------------------------------------

    def record_episode(self, stats: EpisodeStats) -> None:
        """Register a completed episode."""
        self._episodes.append(stats)
        self._reward_window.append(stats.total_reward)
        tp = stats.throughput()
        self._throughput_window.append(tp)
        if not math.isnan(stats.mean_travel_time()):
            self._travel_time_window.append(stats.mean_travel_time())
        if not math.isnan(stats.mean_waiting_time()):
            self._waiting_window.append(stats.mean_waiting_time())
        self._collision_window.append(stats.collision_rate())
        self._gridlock_window.append(stats.gridlock_probability())
        self._speed_window.append(stats.mean_speed())
        self.total_steps += stats.episode_length

    def record_step(
        self,
        rewards: Dict[str, float],
        info: Dict[str, Any],
        current_stats: EpisodeStats,
    ) -> None:
        """Update rolling state from a single environment step.

        Parameters
        ----------
        rewards:     Per-agent reward dict.
        info:        Environment info dict.
        current_stats:
            The EpisodeStats for the ongoing episode (mutated in-place).
        """
        current_stats.total_steps += 1
        current_stats.episode_length += 1
        current_stats.total_reward += float(sum(rewards.values()))
        current_stats.n_collisions += info.get("n_collisions", 0)
        current_stats.n_arrivals += info.get("n_arrived", 0)
        current_stats.speeds.append(info.get("mean_speed", 0.0))
        if info.get("gridlock_intersections", 0) > 0:
            current_stats.gridlock_steps += 1

    def update_intersection_stats(self, intersections: Dict) -> None:
        """Pull intersection usage data directly from Intersection objects."""
        for nid, inter in intersections.items():
            data = self._intersection_data[nid]
            for i in range(4):
                data["direction_counts"][i] = inter.direction_usage[i]
            data["entropy_history"].append(inter.direction_entropy())
            data["oscillation_history"].append(inter.oscillation_score())

    # ------------------------------------------------------------------
    # Running summaries
    # ------------------------------------------------------------------

    def running_mean(self, key: str) -> float:
        """Return the rolling mean for *key*.

        Valid keys: ``reward``, ``throughput``, ``travel_time``,
        ``waiting_time``, ``collision_rate``, ``gridlock``, ``speed``.
        """
        mapping = {
            "reward": self._reward_window,
            "throughput": self._throughput_window,
            "travel_time": self._travel_time_window,
            "waiting_time": self._waiting_window,
            "collision_rate": self._collision_window,
            "gridlock": self._gridlock_window,
            "speed": self._speed_window,
        }
        buf = mapping.get(key)
        if buf is None or len(buf) == 0:
            return float("nan")
        return float(np.mean(buf))

    def summary(self) -> Dict[str, float]:
        """Return dict of current running averages for all tracked metrics."""
        return {
            "mean_reward": self.running_mean("reward"),
            "mean_throughput": self.running_mean("throughput"),
            "mean_travel_time": self.running_mean("travel_time"),
            "mean_waiting_time": self.running_mean("waiting_time"),
            "mean_collision_rate": self.running_mean("collision_rate"),
            "mean_gridlock_prob": self.running_mean("gridlock"),
            "mean_speed": self.running_mean("speed"),
            "n_episodes": float(len(self._episodes)),
            "total_steps": float(self.total_steps),
        }

    # ------------------------------------------------------------------
    # Emergent pattern detection
    # ------------------------------------------------------------------

    def detect_priority_emergence(self, intersections: Dict) -> Dict[str, float]:
        """Analyse intersection usage to detect emergent priority rules.

        Returns
        -------
        dict with keys:
        * ``mean_entropy``:        Mean Shannon entropy across intersections.
          Low → one direction dominates → emergent priority.
        * ``mean_oscillation``:    Mean oscillation score (0=constant, 1=alternating).
          High ≈ 0.8 → turn-taking behaviour.
        * ``max_dominance``:       Maximum fraction of usage by the dominant direction.
        * ``priority_emergence``:  1 − normalised entropy (0 = no pattern, 1 = full dominance).
        """
        entropies: List[float] = []
        oscillations: List[float] = []
        dominances: List[float] = []

        for inter in intersections.values():
            entropies.append(inter.direction_entropy())
            oscillations.append(inter.oscillation_score())

            counts = np.array(inter.direction_usage[:4], dtype=float)
            total = counts.sum()
            if total > 0:
                dominances.append(float(counts.max() / total))

        max_entropy = 2.0  # log2(4) for 4 directions
        results = {
            "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "mean_oscillation": float(np.mean(oscillations)) if oscillations else 0.0,
            "max_dominance": float(np.max(dominances)) if dominances else 0.0,
            "priority_emergence": (
                1.0 - float(np.mean(entropies)) / max_entropy
                if entropies
                else 0.0
            ),
        }
        self._priority_entropy.append(results["mean_entropy"])
        self._oscillation_scores.append(results["mean_oscillation"])
        return results

    def detect_flow_oscillations(self, window_size: int = 20) -> Dict[str, float]:
        """Detect oscillating flow patterns across speed history.

        Looks for periodic speed waves in the trailing speed buffer.
        A dominant frequency with high spectral power indicates traffic waves.
        """
        buf = list(self._speed_window)
        if len(buf) < window_size:
            return {"wave_power": 0.0, "wave_frequency": 0.0}

        speeds = np.array(buf[-window_size:])
        # Detrend
        speeds = speeds - speeds.mean()
        # FFT
        freqs = np.abs(np.fft.rfft(speeds))
        dominant_idx = int(np.argmax(freqs[1:])) + 1   # exclude DC
        power = float(freqs[dominant_idx] ** 2 / (np.sum(freqs ** 2) + 1e-8))
        freq = float(dominant_idx / window_size)
        return {"wave_power": power, "wave_frequency": freq}

    def yield_sequence_analysis(
        self, intersections: Dict
    ) -> Dict[str, Any]:
        """Detect which direction yields at each intersection.

        A direction "yields" if its usage fraction is significantly lower
        than the others given that it had queue requests.  Returns a dict
        mapping intersection node IDs to their yielding pattern string.
        """
        result: Dict[int, str] = {}
        direction_names = ["NORTH", "SOUTH", "EAST", "WEST"]
        for inter in intersections.values():
            counts = np.array(inter.direction_usage[:4], dtype=float)
            total = counts.sum()
            if total < 10:
                result[inter.node_id] = "insufficient_data"
                continue
            fracs = counts / total
            sorted_dirs = sorted(range(4), key=lambda i: fracs[i])
            if fracs[sorted_dirs[0]] < 0.1:
                yielding = direction_names[sorted_dirs[0]]
                priority = direction_names[sorted_dirs[-1]]
                result[inter.node_id] = f"{priority}_priority__{yielding}_yields"
            else:
                result[inter.node_id] = "balanced"
        return {"intersection_patterns": result}

    # ------------------------------------------------------------------
    # Full history export
    # ------------------------------------------------------------------

    def export(self) -> Dict[str, Any]:
        """Export all accumulated metrics as a plain dict (JSON-serialisable)."""
        return {
            "summary": self.summary(),
            "episode_rewards": [e.total_reward for e in self._episodes],
            "episode_throughputs": [e.throughput() for e in self._episodes],
            "episode_travel_times": [e.mean_travel_time() for e in self._episodes],
            "episode_collisions": [e.collision_rate() for e in self._episodes],
            "priority_entropy_history": self._priority_entropy,
            "oscillation_history": self._oscillation_scores,
        }

    def reset_episode(self) -> EpisodeStats:
        """Return a fresh EpisodeStats object for the next episode."""
        return EpisodeStats()
