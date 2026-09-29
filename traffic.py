"""
traffic.py — Spatial substrate: BML-type traffic cellular automaton with two layers.

Periodic L×L lattice with TWO occupancy layers:
    occ[0, y, x] = id of the horizontal agent (moves right, +x) or −1
    occ[1, y, x] = id of the vertical agent   (moves up, +y) or −1

Rules:
  * Exclusion WITHIN a lane: always hard. If the target cell of its own layer is
    occupied AT THE START of the step, the agent does not move, even if the occupant
    leaves during the same step (as in BML).
  * Exclusion BETWEEN layers (crossing): if the target cell is occupied in the other
    layer, the agent is blocked with probability b(cell) = λ·(...) (see mechanism.py).
    λ = 1 reproduces BML Model I; with λ < 1 the flows may overlap, weakening
    exclusion as in BML Model III.
  * With traffic_light=True only one direction moves per step and all targets are
    distinct, so no conflict resolution is needed.

Cost per step: O(number of moving agents).
"""
from __future__ import annotations

from collections import namedtuple

from grid import box_sum_periodic, count_grid

MoveResult = namedtuple("MoveResult", "moved same_blocked cross_blocked y0 x0")


def move_layer(xp, rs, occ, d, cross_layer, movers, pos_y, pos_x, L, block_prob):
    """Try to move every agent in `movers` (ids) of layer d by one cell.

    cross_layer : occupancy of the other layer used to evaluate crossings.
    block_prob  : float (uniform λ) or L×L array with the per-cell block probability.
    Modifies occ, pos_y and pos_x in place. Returns masks over `movers` and the
    starting positions.
    """
    y0 = pos_y[movers]
    x0 = pos_x[movers]
    if d == 0:
        ty, tx = y0, (x0 + 1) % L
    else:
        ty, tx = (y0 + 1) % L, x0

    same_blocked = occ[d, ty, tx] >= 0
    cross_occ = cross_layer[ty, tx] >= 0
    if isinstance(block_prob, (float, int)):
        bp = float(block_prob)
    else:
        bp = block_prob[ty, tx]
    cross_blocked = cross_occ & (~same_blocked) & (rs.random_sample(int(movers.size)) < bp)
    moved = ~(same_blocked | cross_blocked)

    ids = movers[moved]
    if ids.size:
        # Targets were empty at the start of the step and sources were occupied: the
        # two sets are disjoint, so clearing then filling never overwrites anyone.
        occ[d, y0[moved], x0[moved]] = -1
        occ[d, ty[moved], tx[moved]] = ids
        pos_y[ids] = ty[moved]
        pos_x[ids] = tx[moved]
    return MoveResult(moved, same_blocked, cross_blocked, y0, x0)


def remove(occ, dirs, y, x):
    """Take agents off the lattice (job finished or abandoned)."""
    if y.size:
        occ[dirs, y, x] = -1


def try_enter(xp, rs, occ, cand_ids, cand_dirs, L):
    """Candidates try to enter a random cell of their layer.

    They only enter if the cell is free in their layer; ties for the same cell are
    broken at random. Returns (sel, y, x, n_failed), where sel are indices into
    cand_ids of those who entered. Cost O(k log k) for k candidates.
    """
    n = int(cand_ids.size)
    if n == 0:
        z = xp.zeros(0, dtype=xp.int64)
        return z, z.astype(xp.int32), z.astype(xp.int32), 0
    perm = rs.permutation(n)
    ids_p = cand_ids[perm]
    dirs_p = cand_dirs[perm]
    y = rs.randint(0, L, size=n).astype(xp.int32)
    x = rs.randint(0, L, size=n).astype(xp.int32)
    free = occ[dirs_p, y, x] < 0
    idx = xp.nonzero(free)[0]
    if idx.size == 0:
        z = xp.zeros(0, dtype=xp.int64)
        return z, z.astype(xp.int32), z.astype(xp.int32), n
    flat = (dirs_p[idx].astype(xp.int64) * L + y[idx]) * L + x[idx]
    _, first = xp.unique(flat, return_index=True)  # order is random thanks to the permutation
    ok = idx[first]
    occ[dirs_p[ok], y[ok], x[ok]] = ids_p[ok]
    return perm[ok], y[ok], x[ok], n - int(ok.size)


def same_task_counts(xp, ids, tasks, pos_y, pos_x, m, L, radius):
    """Number of OTHER agents doing the same task in the (2R+1)² square of each agent.

    Cost O(m·L² + n). Each task lives in one layer, with at most one agent per cell.
    """
    out = xp.zeros(int(ids.size), dtype=xp.float32)
    if ids.size == 0:
        return out
    y = pos_y[ids]
    x = pos_x[ids]
    for j in range(m):
        sel = xp.nonzero(tasks == j)[0]
        if sel.size == 0:
            continue
        box = box_sum_periodic(count_grid(xp, y[sel], x[sel], L), radius)
        out[sel] = (box[y[sel], x[sel]] - 1).astype(xp.float32)
    return out
