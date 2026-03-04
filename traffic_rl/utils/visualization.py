"""
visualization.py – Rendering and plotting utilities for FlowMergence.

Contains
--------
* ``TrafficRenderer``  – real-time / frame-by-frame environment renderer.
* ``plot_training_curves``  – episode-reward + auxiliary metrics over training.
* ``plot_intersection_usage``  – polar/bar chart of directional usage.
* ``plot_emergent_patterns``  – entropy and oscillation scores over time.
* ``animate_episode``  – produce an MP4 / GIF from a list of frames.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless-safe backend (switched to TkAgg on demand)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms
from matplotlib.patches import FancyArrow
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# ---------------------------------------------------------------------------
# Traffic renderer
# ---------------------------------------------------------------------------

class TrafficRenderer:
    """Matplotlib-based renderer for the traffic environment.

    Parameters
    ----------
    config:
        Environment configuration (for world dimensions).
    network:
        RoadNetwork object (provides node positions and road geometry).
    figsize:
        Matplotlib figure size in inches.
    """

    def __init__(self, config: Any, network: Any, figsize: Tuple[float, float] = (8, 8)) -> None:
        self.config = config
        self.network = network
        self.figsize = figsize
        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes] = None

    def _ensure_figure(self) -> None:
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=self.figsize)
            self._fig.tight_layout()

    def render(
        self,
        vehicles: Dict[int, Any],
        intersections: Dict[int, Any],
        step: int = 0,
        mode: str = "rgb_array",
    ) -> Optional[np.ndarray]:
        """Render a single frame.

        Parameters
        ----------
        vehicles:
            Dict of ``vehicle_id → Vehicle``.
        intersections:
            Dict of ``node_id → Intersection``.
        step:
            Current simulation step (displayed in title).
        mode:
            ``"human"`` to show the window; ``"rgb_array"`` to return pixels.

        Returns
        -------
        np.ndarray (H, W, 3) uint8  when mode == "rgb_array", else None.
        """
        if mode == "human":
            matplotlib.use("TkAgg")

        self._ensure_figure()
        ax = self._ax
        assert ax is not None
        ax.clear()

        self._draw_roads(ax)
        self._draw_intersections(ax, intersections)
        self._draw_vehicles(ax, vehicles)
        ax.set_title(f"FlowMergence – step {step}  |  agents: {len(vehicles)}", fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

        if mode == "human":
            plt.pause(0.001)
            return None

        # Render to RGB array
        self._fig.canvas.draw()
        buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = self._fig.canvas.get_width_height()
        return buf.reshape(h, w, 3)

    def _draw_roads(self, ax: Axes) -> None:
        """Draw road centre-lines."""
        for road in self.network.roads.values():
            x = [road.from_pos[0], road.to_pos[0]]
            y = [road.from_pos[1], road.to_pos[1]]
            ax.plot(x, y, color="#AAAAAA", linewidth=3, zorder=1)

    def _draw_intersections(self, ax: Axes, intersections: Dict[int, Any]) -> None:
        """Draw intersection conflict zones; colour indicates gridlock."""
        for inter in intersections.values():
            color = "#FF4444" if inter.is_gridlocked else "#DDDDDD"
            if inter.occupant_count > 0:
                color = "#FFD700"
            circle = plt.Circle(
                (inter.position[0], inter.position[1]),
                self.config.intersection_radius,
                color=color,
                zorder=2,
                alpha=0.7,
            )
            ax.add_patch(circle)
            # Queue depth text
            qlen = len(inter.entry_queue)
            if qlen > 0:
                ax.text(
                    inter.position[0], inter.position[1],
                    str(qlen), ha="center", va="center",
                    fontsize=6, color="#333333", zorder=5,
                )

    def _draw_vehicles(self, ax: Axes, vehicles: Dict[int, Any]) -> None:
        """Draw each vehicle as a small oriented rectangle."""
        coop_color = "#2196F3"   # blue
        aggr_color = "#FF5722"   # orange-red
        vl = self.config.vehicle_length
        vw = self.config.vehicle_width

        for veh in vehicles.values():
            cx, cy = float(veh.position[0]), float(veh.position[1])
            heading_deg = math.degrees(veh.heading)
            color = coop_color if int(veh.agent_type) == 0 else aggr_color

            rect = patches.Rectangle(
                (-vl / 2, -vw / 2), vl, vw,
                linewidth=0.5,
                edgecolor="#000000",
                facecolor=color,
                alpha=0.85,
                zorder=3,
            )
            transform = (
                matplotlib.transforms.Affine2D()
                .rotate_deg(heading_deg)
                .translate(cx, cy)
                + ax.transData
            )
            rect.set_transform(transform)
            ax.add_patch(rect)

    def save_frame(self, path: str, **kwargs: Any) -> None:
        """Save the current figure to *path*."""
        if self._fig is not None:
            self._fig.savefig(path, dpi=100, bbox_inches="tight", **kwargs)

    def close(self) -> None:
        """Close the matplotlib figure."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None


# ---------------------------------------------------------------------------
# Training curve plots
# ---------------------------------------------------------------------------

def plot_training_curves(
    rewards: List[float],
    throughputs: List[float],
    collision_rates: List[float],
    travel_times: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
    smoothing: int = 20,
) -> Figure:
    """Plot four training metrics on a 2×2 grid.

    Parameters
    ----------
    rewards, throughputs, collision_rates, travel_times:
        Lists of per-episode values.
    save_path:
        If provided, the figure is saved to this path.
    smoothing:
        Rolling-window width for the smoothed curve overlay.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13)

    data_rows = [
        (rewards, "Episode Reward", "royalblue"),
        (throughputs, "Throughput (vehicles/step)", "seagreen"),
        (collision_rates, "Collision Rate", "crimson"),
        (travel_times, "Mean Travel Time (steps)", "darkorange"),
    ]

    for ax, (values, ylabel, color) in zip(axes.flatten(), data_rows):
        xs = np.arange(len(values))
        ax.plot(xs, values, alpha=0.3, color=color, linewidth=0.8)
        if len(values) >= smoothing:
            smooth = np.convolve(values, np.ones(smoothing) / smoothing, mode="valid")
            ax.plot(np.arange(smoothing - 1, len(values)), smooth, color=color, linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_intersection_usage(
    intersections: Dict[int, Any],
    save_path: Optional[str] = None,
    title: str = "Intersection Directional Usage",
) -> Figure:
    """Bar charts of approach-direction usage at each intersection."""
    n = len(intersections)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No intersections", ha="center")
        return fig

    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    directions = ["North", "South", "East", "West"]
    colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]

    for idx, (nid, inter) in enumerate(intersections.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        counts = inter.direction_usage[:4]
        ax.bar(directions, counts, color=colors, edgecolor="white")
        ax.set_title(f"Intersection {nid}  (entropy={inter.direction_entropy():.2f})", fontsize=8)
        ax.set_ylabel("Usage count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_emergent_patterns(
    entropy_history: List[float],
    oscillation_history: List[float],
    priority_emergence: List[float],
    save_path: Optional[str] = None,
    title: str = "Emergent Traffic Patterns",
) -> Figure:
    """Three-panel plot showing emergent rule statistics over training."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=13)

    panels = [
        (entropy_history, "Directional Entropy (↓ = priority rule)", "purple"),
        (oscillation_history, "Oscillation Score (↑ = turn-taking)", "teal"),
        (priority_emergence, "Priority Emergence Score (↑ = stronger)", "darkorange"),
    ]

    for ax, (data, ylabel, color) in zip(axes, panels):
        if data:
            ax.plot(data, color=color, linewidth=1.5, alpha=0.9)
            # Rolling mean overlay
            if len(data) > 10:
                smooth = np.convolve(data, np.ones(10) / 10, mode="valid")
                ax.plot(
                    np.arange(9, len(data)), smooth,
                    color=color, linewidth=2.5, alpha=0.5, linestyle="--",
                )
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_density_sweep(
    densities: List[float],
    metrics_by_density: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Traffic Density Sweep",
) -> Figure:
    """Line plots of key metrics vs. spawn density."""
    keys = list(metrics_by_density.keys())
    n = len(keys)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle(title, fontsize=13)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, key in zip(axes_flat, keys):
        values = metrics_by_density[key]
        ax.plot(densities, values, marker="o", linewidth=2, color="steelblue")
        ax.fill_between(densities, values, alpha=0.15, color="steelblue")
        ax.set_xlabel("Spawn density")
        ax.set_ylabel(key.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[n:]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def animate_episode(
    frames: List[np.ndarray],
    save_path: str,
    fps: int = 10,
) -> None:
    """Export a list of RGB frames as an animated GIF or MP4.

    Parameters
    ----------
    frames:
        List of (H, W, 3) uint8 numpy arrays.
    save_path:
        Output path ending in ``.gif`` or ``.mp4``.
    fps:
        Frames per second.
    """
    if not frames:
        return

    ext = os.path.splitext(save_path)[-1].lower()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    if ext == ".gif":
        try:
            import imageio
            imageio.mimsave(save_path, frames, fps=fps)
            print(f"[Visualizer] Saved GIF → {save_path}")
        except ImportError:
            print("[Visualizer] imageio not installed – cannot save GIF.")
    elif ext == ".mp4":
        try:
            import imageio
            writer = imageio.get_writer(save_path, fps=fps, codec="libx264")
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"[Visualizer] Saved MP4 → {save_path}")
        except Exception as e:
            print(f"[Visualizer] Could not save MP4: {e}")
    else:
        print(f"[Visualizer] Unsupported format: {ext}")
