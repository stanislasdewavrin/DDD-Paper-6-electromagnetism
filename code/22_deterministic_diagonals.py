"""
Paper VI - deterministic body-diagonal substrates.

Tests three deterministic constructions (no Poisson randomness) to see
whether any reproduces d_BFS = 3 + 1/(2*pi) ~ 3.1592.
"""
import numpy as np
import json
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi

DIRS4 = [
    (+1, +1, +1),
    (+1, +1, -1),
    (+1, -1, +1),
    (-1, +1, +1),
]

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
                    adj[u].add(v)
                    adj[v].add(u)
    return adj


def add_diag(adj, i, j, k, dvec, L):
    u = idx_of(i, j, k, L)
    di, dj, dk = dvec
    v = idx_of(i + di, j + dj, k + dk, L)
    adj[u].add(v)
    adj[v].add(u)


def build_fixed(L, dvec=(1, 1, 1)):
    adj = build_cubic_adj(L)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                add_diag(adj, i, j, k, dvec, L)
    return adj


def build_tiling_2x2x2(L):
    """(a) 8 sub-positions of the 2x2x2 super-cell -> 8 signed diagonals."""
    assert L % 2 == 0
    adj = build_cubic_adj(L)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                sub = (i % 2) * 4 + (j % 2) * 2 + (k % 2)
                dvec = DIRS8[sub]
                add_diag(adj, i, j, k, dvec, L)
    return adj


def build_bloch(L):
    """(b) Direction at (i,j,k) determined by (i+j+k) mod 4."""
    adj = build_cubic_adj(L)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                idx_dir = (i + j + k) % 4
                dvec = DIRS4[idx_dir]
                add_diag(adj, i, j, k, dvec, L)
    return adj


def build_all4(L):
    """(c) All four (unsigned) diagonals at every node."""
    adj = build_cubic_adj(L)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for dvec in DIRS4:
                    add_diag(adj, i, j, k, dvec, L)
    return adj


def build_poisson(L, mu=1.0, seed=42):
    rng = np.random.default_rng(seed)
    adj = build_cubic_adj(L)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                k_short = rng.poisson(mu)
                for _ in range(k_short):
                    dvec = DIRS4[rng.integers(0, 4)]
                    sign = 1 if rng.random() < 0.5 else -1
                    dvec = tuple(sign * d for d in dvec)
                    add_diag(adj, i, j, k, dvec, L)
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
    ds = []
    for _ in range(n_sources):
        src = int(rng.integers(0, Ntot))
        ds.append(bfs_dim(adj, src))
    return float(np.mean(ds)), float(np.std(ds))


def main():
    L = 64
    target = 3 + 1 / (2 * PI)
    target_shift = 1 / (2 * PI)
    print("L = {0}, target d_eff = 3 + 1/(2pi) = {1:.6f}".format(L, target))
    print("target shift over cubic = +{0:.5f}".format(target_shift))
    print()

    constructions = [
        ("pure cubic (control)",      build_cubic_adj(L)),
        ("fixed (1,1,1) diagonal",    build_fixed(L, (1, 1, 1))),
        ("(a) tiling 2x2x2 chiral",   build_tiling_2x2x2(L)),
        ("(b) Bloch-twist (i+j+k)%4", build_bloch(L)),
        ("(c) all 4 diagonals/node",  build_all4(L)),
        ("Poisson(mu=1) reference",   build_poisson(L, mu=1.0, seed=42)),
    ]

    results = []
    for name, adj in constructions:
        d_mean, d_std = averaged_dim(adj, n_sources=12, seed=11)
        results.append((name, d_mean, d_std))

    d_cubic = results[0][1]

    print("{0:<32} {1:>10} {2:>7} {3:>10} {4:>11}".format(
        "Construction", "d_BFS", "sigma", "shift", "shift-tgt"))
    print("-" * 76)
    rows = []
    for name, d_mean, d_std in results:
        shift = d_mean - d_cubic
        delta = shift - target_shift
        is_ctrl = (name == "pure cubic (control)")
        marker = " ok" if (not is_ctrl and abs(delta) < 0.02) else ""
        print("{0:<32} {1:>10.5f} {2:>7.4f} {3:>+10.5f} {4:>+11.5f}{5}".format(
            name, d_mean, d_std, shift, delta, marker))
        rows.append({
            "name": name,
            "d_BFS_mean": d_mean,
            "d_BFS_std": d_std,
            "shift_over_cubic": shift,
            "delta_to_target_shift": delta,
            "L": L,
        })

    print()
    print("Notes:")
    print(" - shift = d_BFS - d_cubic (removes finite-size offset).")
    print(" - target_shift = 1/(2pi) = 0.15916.")
    print(" - shift-tgt close to zero means matches the small-world target.")

    out = {"L": L, "target": target, "target_shift": target_shift,
           "d_cubic": d_cubic, "results": rows}
    with open(DATA / "22_deterministic_diagonals.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved -> {0}".format(DATA / "22_deterministic_diagonals.json"))


if __name__ == "__main__":
    main()
