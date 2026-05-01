"""
Extrapolation of d_BFS for the tiling 2x2x2 chiral construction
to the L -> infinity limit.

Tests whether the asymptotic dimension is 3 + 1/(2pi) (paper claim)
or just 3 (no real shift).
"""
import numpy as np
import json
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi

DIRS8 = [
    (+1, +1, +1), (-1, -1, -1),
    (+1, +1, -1), (-1, -1, +1),
    (+1, -1, +1), (-1, +1, -1),
    (-1, +1, +1), (+1, -1, -1),
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


def build_tiling_2x2x2(L):
    assert L % 2 == 0
    adj = build_cubic_adj(L)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                sub = (i % 2) * 4 + (j % 2) * 2 + (k % 2)
                dvec = DIRS8[sub]
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


def averaged_dim(adj, n_sources=6, seed=0):
    rng = np.random.default_rng(seed)
    Ntot = len(adj)
    ds = [bfs_dim(adj, int(rng.integers(0, Ntot))) for _ in range(n_sources)]
    return float(np.mean(ds)), float(np.std(ds))


def main():
    target_eff = 3 + 1 / (2 * PI)
    target_shift = 1 / (2 * PI)
    print("Target: d_eff = 3 + 1/(2pi) = {0:.6f}".format(target_eff))
    print()

    Ls = [32, 48, 64, 80, 96]
    print("{0:>6} {1:>10} {2:>10} {3:>10} {4:>11}".format(
        "L", "d_cubic", "d_tile", "shift", "shift-tgt"))
    rows = []
    for L in Ls:
        adj_c = build_cubic_adj(L)
        adj_t = build_tiling_2x2x2(L)
        dc, _ = averaged_dim(adj_c, n_sources=6, seed=11)
        dt, _ = averaged_dim(adj_t, n_sources=6, seed=11)
        shift = dt - dc
        delta = shift - target_shift
        print("{0:>6d} {1:>10.5f} {2:>10.5f} {3:>+10.5f} {4:>+11.5f}".format(
            L, dc, dt, shift, delta))
        rows.append({"L": L, "d_cubic": dc, "d_tile": dt,
                     "shift": shift, "delta": delta})

    print()
    # Extrapolate d_tile to L=infinity assuming 1/L correction
    Ls_arr = np.array([r["L"] for r in rows])
    dts = np.array([r["d_tile"] for r in rows])
    dcs = np.array([r["d_cubic"] for r in rows])
    inv_L = 1.0 / Ls_arr

    # Fit d(L) = d_inf + a/L over the largest 3 points
    fit_mask = Ls_arr >= 64
    a_t, b_t = np.polyfit(inv_L[fit_mask], dts[fit_mask], 1)
    a_c, b_c = np.polyfit(inv_L[fit_mask], dcs[fit_mask], 1)
    print("1/L extrapolation (largest 4 L values):")
    print("  d_cubic(L=inf) = {0:.5f} (target 3.0)".format(b_c))
    print("  d_tile (L=inf) = {0:.5f}".format(b_t))
    print("  asymptotic shift = {0:+.5f}".format(b_t - b_c))
    print("  target shift     = {0:+.5f}".format(target_shift))

    out = {"target_eff": target_eff, "target_shift": target_shift,
           "L_scan": rows,
           "d_cubic_inf": b_c, "d_tile_inf": b_t,
           "asymptotic_shift": b_t - b_c}
    with open(DATA / "24_tiling_extrapolation.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved -> {0}".format(DATA / "24_tiling_extrapolation.json"))


if __name__ == "__main__":
    main()
