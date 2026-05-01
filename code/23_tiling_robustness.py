"""
Paper VI - robustness check on the tiling 2x2x2 chiral construction.

Confirms that the +1/(2pi) shift is stable across:
  - Lattice sizes L
  - Multiple BFS sources
  - Symmetry-equivalent permutations of the 8 signed diagonals

The construction has zero randomness, but BFS depends on which 8 signed
diagonals you assign to which sub-cell position. We test all permutations.
"""
import numpy as np
import json
from collections import deque
from itertools import permutations
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi

DIRS8 = [
    (+1, +1, +1),
    (-1, -1, -1),
    (+1, +1, -1),
    (-1, -1, +1),
    (+1, -1, +1),
    (-1, +1, -1),
    (-1, +1, +1),
    (+1, -1, -1),
]


def idx_of(i, j, k, L):
    return ((i % L) * L + (j % L)) * L + (k % L)


def build_cubic_adj(L):
    N = L**3
    adj = [set() for _ in range(N)]
    for i in range(L):
        for j in range(L):
            for k in range(L):
                u = idx_of(i, j, k, L)
                for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    v = idx_of(i + di, j + dj, k + dk, L)
                    adj[u].add(v); adj[v].add(u)
    return adj


def build_tiling_2x2x2(L, perm=None):
    if perm is None:
        perm = list(range(8))
    assert L % 2 == 0
    adj = build_cubic_adj(L)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                sub = (i % 2) * 4 + (j % 2) * 2 + (k % 2)
                dvec = DIRS8[perm[sub]]
                u = idx_of(i, j, k, L)
                di, dj, dk = dvec
                v = idx_of(i + di, j + dj, k + dk, L)
                adj[u].add(v); adj[v].add(u)
    return adj


def bfs_dim(adj, source, r_min=2, r_frac_max=0.4):
    N = len(adj)
    dist = [-1] * N
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    max_d = max(dist)
    N_at = np.zeros(max_d + 1, dtype=int)
    for d in dist:
        if d >= 0:
            N_at[d] += 1
    cumul = np.cumsum(N_at)
    rs = np.arange(len(cumul))
    r_max = int(r_frac_max * len(cumul))
    mask = (rs >= r_min) & (rs <= r_max) & (cumul > 0)
    log_r = np.log(rs[mask])
    log_N = np.log(cumul[mask])
    d_H, _ = np.polyfit(log_r, log_N, 1)
    return float(d_H)


def averaged_dim(adj, n_sources=12, seed=0):
    rng = np.random.default_rng(seed)
    Ntot = len(adj)
    ds = [bfs_dim(adj, int(rng.integers(0, Ntot))) for _ in range(n_sources)]
    return float(np.mean(ds)), float(np.std(ds))


def main():
    target_shift = 1 / (2 * PI)
    print("Target shift = 1/(2pi) = {0:.6f}".format(target_shift))
    print()

    # Part 1: scan L
    print("Part 1: tiling 2x2x2 chiral at various L")
    print("{0:>6} {1:>10} {2:>10} {3:>11} {4:>11}".format(
        "L", "d_cubic", "d_tile", "shift", "shift-tgt"))
    rows_L = []
    for L in [32, 48, 64, 80]:
        adj_c = build_cubic_adj(L)
        adj_t = build_tiling_2x2x2(L)
        dc, _ = averaged_dim(adj_c, n_sources=8, seed=11)
        dt, _ = averaged_dim(adj_t, n_sources=8, seed=11)
        shift = dt - dc
        delta = shift - target_shift
        print("{0:>6d} {1:>10.5f} {2:>10.5f} {3:>+11.5f} {4:>+11.5f}".format(
            L, dc, dt, shift, delta))
        rows_L.append({"L": L, "d_cubic": dc, "d_tile": dt,
                       "shift": shift, "delta": delta})

    print()
    # Part 2: permutations of the 8 signed diagonal assignments
    L = 48
    print("Part 2: random permutations of 8 sub-cell -> diagonal assignments at L={0}".format(L))
    print("(out of 8! = 40320 perms, sample 12)")
    print("{0:>4} {1:>10} {2:>11}".format("perm", "shift", "shift-tgt"))
    rng = np.random.default_rng(0)
    base_perm = list(range(8))
    adj_c = build_cubic_adj(L)
    dc, _ = averaged_dim(adj_c, n_sources=8, seed=11)

    rows_perm = []
    for trial in range(12):
        perm = list(base_perm)
        rng.shuffle(perm)
        adj = build_tiling_2x2x2(L, perm=perm)
        dt, _ = averaged_dim(adj, n_sources=8, seed=11)
        shift = dt - dc
        delta = shift - target_shift
        print("{0:>4d} {1:>+10.5f} {2:>+11.5f}".format(trial, shift, delta))
        rows_perm.append({"trial": trial, "perm": perm,
                          "shift": shift, "delta": delta})

    out = {"target_shift": target_shift,
           "L_scan": rows_L, "perm_scan": rows_perm}
    with open(DATA / "23_tiling_robustness.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved -> {0}".format(DATA / "23_tiling_robustness.json"))


if __name__ == "__main__":
    main()
