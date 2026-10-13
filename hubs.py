"""
hubs.py — Geometry of the hub lines (centralized delivery). Pure functions, no state.

Single hub (default, as in every experiment so far): the central column x = L//2 for
horizontal agents and the central row y = L//2 for vertical ones.

Polycentric delivery [NEW, experiment E9]: K evenly spaced hub lines,
    K = max(1, min(L, round((L / ell0) ** eta)))
    positions x_k = (L//2 + round(k·L/K)) mod L,   k = 0 … K−1
so K = 1 reproduces L//2 exactly. eta (hub_count_exponent) sets how the number of centres
grows with the lattice side: eta = 0 one headquarters at every size; eta = 1 a constant
spacing ell0 (hub_spacing_ref), i.e. a fully polycentric system. Since L ∝ N^(a/2),
K ∝ N^(eta·a/2).

A delivering agent keeps moving in its own direction and its job counts as soon as it
stands on ANY hub line of its direction, so it automatically delivers to the nearest
hub ahead. The mean trip length from a uniformly random cell is
    d̄ = Σ_k g_k (g_k − 1) / (2L),     g_k = gap (in cells) between consecutive lines
(an agent already on a line delivers at once). For K = 1, d̄ = (L − 1)/2.
"""
from __future__ import annotations

import numpy as np


def n_hubs(L: int, eta: float = 0.0, ell0: float = 16.0) -> int:
    """Number of hub lines per direction."""
    if eta <= 0.0:
        return 1
    return int(max(1, min(L, round((L / float(ell0)) ** eta))))


def hub_positions(L: int, eta: float = 0.0, ell0: float = 16.0) -> np.ndarray:
    """Sorted, unique hub-line coordinates in [0, L). K = 1 gives [L // 2]."""
    K = n_hubs(L, eta, ell0)
    pos = (L // 2 + np.round(np.arange(K) * L / K).astype(np.int64)) % L
    return np.unique(pos)


def hub_mask(L: int, eta: float = 0.0, ell0: float = 16.0) -> np.ndarray:
    """Boolean array of length L, True on hub coordinates."""
    m = np.zeros(L, dtype=bool)
    m[hub_positions(L, eta, ell0)] = True
    return m


def mean_hub_distance(L: int, eta: float = 0.0, ell0: float = 16.0) -> float:
    """Mean number of moves from a uniformly random coordinate to the next hub ahead
    (0 if already on a hub). Independent of the direction of motion."""
    pos = hub_positions(L, eta, ell0)
    gaps = np.diff(np.concatenate([pos, [pos[0] + L]]))
    return float(np.sum(gaps * (gaps - 1)) / (2.0 * L))


def distance_ahead(x: np.ndarray, L: int, mask: np.ndarray) -> np.ndarray:
    """Moves needed from each coordinate x (moving +1 per move, periodic) to reach the first
    hub coordinate, 0 if x is a hub. Used by the unit tests and for diagnostics."""
    x = np.asarray(x, dtype=np.int64) % L
    pos = np.flatnonzero(mask)
    ext = np.concatenate([pos, pos + L])
    idx = np.searchsorted(ext, x, side="left")
    return ext[idx] - x


def trip_exponent(a: float, eta: float) -> float:
    """Exponent of the mean trip length in N: d̄ ∝ L^(1−eta) ∝ N^(a(1−eta)/2)."""
    return 0.5 * a * (1.0 - eta)


def gamma_ff_exponent(a: float, eta: float, eps_tau: float = 1.0) -> float:
    """Exponent of the free-flow Γ in N: Γ_ff ∝ c · d̄ / τ, with d̄ ∝ N^(a(1−eta)/2) and the
    job duration τ ∝ ρ^(−eps_tau) ∝ N^(−eps_tau(1−a)).  eps_tau = 1 for b = 1 and κn ≫ g."""
    return trip_exponent(a, eta) + eps_tau * (1.0 - a)


def beta_asymptote(beta0: float, p: float, a: float, eta: float, eps_tau: float = 1.0) -> float:
    """Local exponent once Γ ≫ 1: β∞ = β0 − p · dlnΓ_ff/dlnN."""
    return beta0 - p * gamma_ff_exponent(a, eta, eps_tau)


if __name__ == "__main__":       # quick look:  python hubs.py
    for L in (41, 128, 921):
        for eta in (0.0, 0.5, 1.0):
            print(L, eta, n_hubs(L, eta), round(mean_hub_distance(L, eta), 2))
    print("asymptote a=2/3, eta=0, beta0=1.073, p=0.892:",
          round(beta_asymptote(1.073, 0.892, 2 / 3, 0.0), 3))
