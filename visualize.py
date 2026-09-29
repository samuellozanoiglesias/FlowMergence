"""
visualize.py — Lattice animation from the snapshots stored by runner.py.

    python runner.py --set N=2048 exclusion=0.8 --snapshots 20
    python visualize.py <run folder>                      # -> <run folder>/animation.gif
    python visualize.py <run folder> --frame -1           # -> <run folder>/frame_last.png
    python visualize.py path/to/snapshots.npz --out my.gif

Colours: horizontal agents in blues (one shade per task), vertical agents in oranges,
overlapping crossings in purple, coordinator zones as red squares.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation, patches  # noqa: E402


def _rgb(h, v, m):
    L = h.shape[0]
    img = np.ones((L, L, 3))
    blues, oranges = plt.get_cmap("Blues"), plt.get_cmap("Oranges")
    for j in range(m):
        shade = 0.45 + 0.5 * (j // 2) / max((m - 1) // 2, 1)
        if j % 2 == 0:
            img[h == j] = blues(shade)[:3]
        else:
            img[v == j] = oranges(shade)[:3]
    img[(h >= 0) & (v >= 0)] = (0.45, 0.1, 0.55)
    return img


def _draw(ax, data, k, show_zones=True):
    ax.clear()
    m, r = int(data["m"]), int(data["radius"])
    ax.imshow(_rgb(data["h"][k], data["v"][k], m), origin="lower", interpolation="nearest")
    o = data["offsets"]
    cy, cx = data["cy"][o[k]:o[k + 1]], data["cx"][o[k]:o[k + 1]]
    if show_zones:
        for y, x in zip(cy, cx):
            ax.add_patch(patches.Rectangle((x - r - 0.5, y - r - 0.5), 2 * r + 1, 2 * r + 1,
                                           fill=False, edgecolor="red", lw=0.6, alpha=0.7))
    ax.scatter(cx, cy, s=6, c="red", marker="x", lw=0.8)
    nh, nv = int((data["h"][k] >= 0).sum()), int((data["v"][k] >= 0).sum())
    ax.set_title(f"t={int(data['t'][k])}   →{nh}  ↑{nv}   coordinators={len(cy)}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="run folder or snapshots.npz")
    ap.add_argument("--out", default=None, help="default: inside the run folder")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--frame", type=int, default=None, help="draw only this snapshot (png)")
    ap.add_argument("--no-zones", action="store_true")
    a = ap.parse_args()
    npz = a.source if a.source.endswith(".npz") else os.path.join(a.source, "snapshots.npz")
    run_dir = os.path.dirname(os.path.abspath(npz))
    data = dict(np.load(npz))
    n = len(data["t"])
    L = data["h"].shape[1]
    size = min(9, max(4, L / 40))
    fig, ax = plt.subplots(figsize=(size, size))
    if a.frame is not None:
        k = a.frame % n
        out = a.out or os.path.join(run_dir, "frame_last.png" if a.frame == -1 else f"frame_{k:04d}.png")
        _draw(ax, data, k, not a.no_zones)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    else:
        out = a.out or os.path.join(run_dir, "animation.gif")
        anim = animation.FuncAnimation(fig, lambda k: _draw(ax, data, k, not a.no_zones),
                                       frames=n, interval=1000 / a.fps)
        anim.save(out, writer=animation.PillowWriter(fps=a.fps))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
