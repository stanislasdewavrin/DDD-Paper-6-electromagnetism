"""
Paper VI - precision measurement of d_eff at large L.

Pushes the Poisson(mu=1) body-diagonal BFS to L=80, 100, 128 with
many sources to nail down the asymptotic d_eff.

Compares against target 3 + 1/(2pi) = 3.1592.

Uses array-based BFS (numpy) for speed.
"""
import numpy as np
import json
import time
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


def idx_of(i, j, k, L):
    return ((i % L) * L + (j % L)) * L + (k % L)


def build_poisson_diag_flat(L, mu=1.0, seed=42):
    """Build Poisson(mu) body-diagonal substrate using flat arrays
    (CSR-like) for speed.

    Returns: list-of-lists adjacency.
    """
    rng = np.random.default_rng(seed)
    N = L**3
    adj = [[] for _ in range(N)]

    # Cubic axes (each undirected edge added once via canonical order)
    # We add both directions to keep BFS simple.
    for i in range(L):
        for j in range(L):
            for k in range(L):
                u = idx_of(i, j, k, L)
                for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    v = idx_of(i + di, j + dj, k + dk, L)
                    adj[u].append(v)
                    adj[v].append(u)

    # Body-diagonal Poisson shortcuts
    for i in range(L):
        for j in range(L):
            for k in range(L):
                u = idx_of(i, j, k, L)
                k_short = rng.poisson(mu)
                for _ in range(k_short):
                    didx = int(rng.integers(0, 4))
                    dvec = DIRS4[didx]
                    sign = 1 if rng.random() < 0.5 else -1
                    di, dj, dk = (sign * dvec[0], sign * dvec[1], sign * dvec[2])
                    v = idx_of(i + di, j + dj, k + dk, L)
                    if v != u:
                        adj[u].append(v)
                        adj[v].append(u)
    return adj


def build_cubic_flat(L):
    N = L**3
    adj = [[] for _ in range(N)]
    for i in range(L):
        for j in range(L):
            for k in range(L):
                u = idx_of(i, j, k, L)
                for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    v = idx_of(i + di, j + dj, k + dk, L)
                    adj[u].append(v)
                    adj[v].append(u)
    return adj


def bfs_dim_fast(adj, source, r_min=2, r_frac_max=0.4):
    """BFS expansion fit using arrays."""
    N = len(adj)
    dist = np.full(N, -1, dtype=np.int32)
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        d_next = dist[u] + 1
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = d_next
                q.append(v)
    max_d = int(dist.max())
    N_at = np.bincount(dist[dist >= 0], minlength=max_d + 1)
    cumul = np.cumsum(N_at)
    rs = np.arange(len(cumul))
    r_max = int(r_frac_max * len(cumul))
    mask = (rs >= r_min) & (rs <= r_max) & (cumul > 0)
    log_r = np.log(rs[mask])
    log_N = np.log(cumul[mask])
    d_H, _ = np.polyfit(log_r, log_N, 1)
    return float(d_H)


def averaged_dim(adj, n_sources=20, seed=0):
    rng = np.random.default_rng(seed)
    Ntot = len(adj)
    ds = []
    for _ in range(n_sources):
        src = int(rng.integers(0, Ntot))
        ds.append(bfs_dim_fast(adj, src))
    return float(np.mean(ds)), float(np.std(ds) / np.sqrt(len(ds)))


def main():
    target = 3 + 1 / (2 * PI)
    print("Target d_eff = 3 + 1/(2pi) = {0:.6f}".format(target))
    print()

    rows = []
    Ls = [64, 80, 96]
    print("{0:>5} {1:>8} {2:>10} {3:>9} {4:>10} {5:>11} {6:>9}".format(
        "L", "N", "d_cubic", "se_c", "d_poisson", "se_p", "time_s"))
    print("-" * 72)
    for L in Ls:
        N = L**3
        t0 = time.time()
        adj_c = build_cubic_flat(L)
        adj_p = build_poisson_diag_flat(L, mu=1.0, seed=42)
        t_build = time.time() - t0
        t0 = time.time()
        dc, se_c = averaged_dim(adj_c, n_sources=6, seed=11)
        dp, se_p = averaged_dim(adj_p, n_sources=6, seed=11)
        t_bfs = time.time() - t0
        print("{0:>5d} {1:>8d} {2:>10.5f} {3:>9.4f} {4:>10.5f} {5:>11.4f} "
              "{6:>9.1f}".format(L, N, dc, se_c, dp, se_p, t_build + t_bfs))
        rows.append({"L": L, "N": N, "d_cubic": dc, "se_cubic": se_c,
                     "d_poisson": dp, "se_poisson": se_p,
                     "time_s": t_build + t_bfs})

    # Extrapolate Poisson d to L -> infinity assuming d(L) = d_inf - a/L
    Ls_arr = np.array([r["L"] for r in rows])
    dps = np.array([r["d_poisson"] for r in rows])
    dcs = np.array([r["d_cubic"] for r in rows])
    inv_L = 1.0 / Ls_arr

    print()
    if len(rows) >= 2:
        a_p, b_p = np.polyfit(inv_L, dps, 1)
        a_c, b_c = np.polyfit(inv_L, dcs, 1)
        print("1/L extrapolation (all L points):")
        print("  d_cubic(L=inf)   = {0:.5f}  (target 3.0)".format(b_c))
        print("  d_poisson(L=inf) = {0:.5f}".format(b_p))
        print("  shift            = {0:+.5f}".format(b_p - b_c))
        print("  target shift     = +{0:.5f}".format(1 / (2 * PI)))
        print("  abs(d_poisson - target) = {0:+.5f}".format(b_p - target))

    out = {"target": target, "results": rows}
    with open(DATA / "26_d_eff_large_L.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved -> {0}".format(DATA / "26_d_eff_large_L.json"))


if __name__ == "__main__":
    main()
