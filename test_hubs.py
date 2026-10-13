"""Unit tests of hubs.py (pure). From the code directory:  python tests/test_hubs.py"""
import os, sys
sys.path.insert(0, os.environ.get("CODE_DIR", "."))
import numpy as np, hubs
for L in range(8, 400, 7):
    assert list(hubs.hub_positions(L, 0.0)) == [L // 2]
    assert abs(hubs.mean_hub_distance(L, 0.0) - (L - 1) / 2) < 1e-12
    for eta in (0.25, 0.5, 0.75, 1.0):
        for ell0 in (4.0, 16.0):
            K = hubs.n_hubs(L, eta, ell0); pos = hubs.hub_positions(L, eta, ell0)
            assert 1 <= K <= L and len(pos) == K and (L // 2) in pos
            mask = hubs.hub_mask(L, eta, ell0)
            bf = []
            for x in range(L):              # brute force: walk +1 until a hub line is hit
                k = 0
                while not mask[(x + k) % L]:
                    k += 1
                bf.append(k)
            assert np.array_equal(hubs.distance_ahead(np.arange(L), L, mask), bf)
            assert abs(np.mean(bf) - hubs.mean_hub_distance(L, eta, ell0)) < 1e-12
            gaps = np.diff(np.concatenate([pos, [pos[0] + L]]))
            assert gaps.max() - gaps.min() <= 1
assert abs(hubs.gamma_ff_exponent(2 / 3, 0.0) - 2 / 3) < 1e-12 and abs(hubs.gamma_ff_exponent(1.0, 1.0)) < 1e-12
print("test_hubs: PASSED")
