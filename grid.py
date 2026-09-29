"""
grid.py — Operations on the periodic L×L lattice. All are O(L²) = O(N^a / rho0).

  count_grid(y, x)        -> number of points per cell (bincount, O(n + L²))
  box_sum_periodic(g, r)  -> sum of g over the (2r+1)² square centred on each cell,
                             with periodic boundaries (summed-area table, O(L²))
"""
from __future__ import annotations

from backend import get_array_module


def count_grid(xp, y, x, L):
    """Count how many points (y, x) fall in each cell. Returns an int64 L×L array."""
    flat = y.astype(xp.int64) * L + x.astype(xp.int64)
    return xp.bincount(flat, minlength=L * L).reshape(L, L)


def box_sum_periodic(g, r: int):
    """Sum of g over the Chebyshev neighbourhood of radius r (periodic).

    If r >= L/2 it is clipped to (L-1)//2 so that no cell is counted twice.
    """
    xp = get_array_module(g)
    L0, L1 = g.shape
    r = int(max(0, min(r, (min(L0, L1) - 1) // 2)))
    g = g.astype(xp.int64, copy=False)
    if r == 0:
        return g
    padded = xp.pad(g, r, mode="wrap")
    S = xp.zeros((L0 + 2 * r + 1, L1 + 2 * r + 1), dtype=xp.int64)
    S[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    w = 2 * r + 1
    # original cell i <-> rows i..i+2r of the padded array
    return S[w:w + L0, w:w + L1] - S[:L0, w:w + L1] - S[w:w + L0, :L1] + S[:L0, :L1]
