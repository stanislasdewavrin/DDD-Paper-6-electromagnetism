"""
Test the EUCLIDEAN-weighted BFS dimension vs the topological BFS.

Standard BFS: each edge = 1 step.
Euclidean BFS: cubic edge = 1, body-diagonal edge = sqrt(3).

If d_BFS(topo) != d_BFS(Eucl), we may resolve the alpha_EM mismatch.
"""
import numpy as np
import json
import time
from pathlib import Path
import heapq

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

DIRS4 = np.array([
    [+1, +1, +1],
    [+1, +1, -1],
    [+1, -1, +1],
    [-1, +1, +1],
], dtype=np.int32)

SQRT3 = np.sqrt(3.0)

def idx_of(i, j, k, L):
    return ((i % L) * L + (j % L)) * L + (k % L)


def dijkstra_weighted(L, seed=42, weight_cubic=1.0, weight_diag=SQRT3):
    """Dijkstra with Euclidean weights. Returns dist (float)."""
    N = L**3
    rng = np.random.default_rng(seed)
    theta0 = rng.integers(0, 4, size=N).astype(np.int8)
    dist = np.full(N, np.inf, dtype=np.float32)
    src = idx_of(L // 2, L // 2, L // 2, L)
    dist[src] = 0.0
    heap = [(0.0, src, 0)]  # (dist, node_idx, tick)
    cubic_dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while heap:
        d_u, u, t_u = heapq.heappop(heap)
        if d_u > dist[u]:
            continue
        i = u // (L * L)
        j = (u // L) % L
        k = u % L
        # Cubic neighbors
        for di, dj, dk in cubic_dirs:
            v = idx_of(i + di, j + dj, k + dk, L)
            new_d = d_u + weight_cubic
            if new_d < dist[v]:
                dist[v] = new_d
                heapq.heappush(heap, (new_d, v, t_u + 1))
        # Diagonal neighbors gated by phase
        for line_idx in range(4):
            d_vec = DIRS4[line_idx]
            for sign in (1, -1):
                v = idx_of(i + sign * d_vec[0], j + sign * d_vec[1], k + sign * d_vec[2], L)
                phi_v_t = (theta0[v] + t_u + 1) % 4
                if phi_v_t == line_idx:
                    new_d = d_u + weight_diag
                    if new_d < dist[v]:
                        dist[v] = new_d
                        heapq.heappush(heap, (new_d, v, t_u + 1))
    return dist


def fit_dim_continuous(dist, n_bins=50, r_min=2.0, r_frac_max=0.4):
    """Fit dimension from N(r) ~ r^d in continuous distance."""
    finite = dist[np.isfinite(dist)]
    max_r = finite.max()
    r_max = r_frac_max * max_r
    bins = np.linspace(0, max_r, n_bins+1)
    cumul = np.zeros(n_bins)
    for i in range(n_bins):
        cumul[i] = np.sum(finite <= bins[i+1])
    rs_centers = (bins[:-1] + bins[1:]) / 2
    mask = (rs_centers >= r_min) & (rs_centers <= r_max) & (cumul > 0)
    log_r = np.log(rs_centers[mask])
    log_N = np.log(cumul[mask])
    d, _ = np.polyfit(log_r, log_N, 1)
    return float(d), float(max_r)


def main():
    L = 80  # Smaller because Dijkstra is slower than BFS
    print(f"L = {L}")
    print()

    rows = []
    for seed in range(3):
        t0 = time.time()
        # Topological BFS (weight = 1 for all edges)
        dist_topo = dijkstra_weighted(L, seed=seed, weight_cubic=1.0, weight_diag=1.0)
        d_topo, _ = fit_dim_continuous(dist_topo)
        # Euclidean weighted (cubic=1, diag=sqrt(3))
        dist_eucl = dijkstra_weighted(L, seed=seed, weight_cubic=1.0, weight_diag=SQRT3)
        d_eucl, max_r_eucl = fit_dim_continuous(dist_eucl)
        elapsed = time.time() - t0
        print(f"seed={seed}: d_topo = {d_topo:.5f},  d_Eucl = {d_eucl:.5f}  ({elapsed:.1f}s)")
        rows.append({"seed": seed, "d_topo": d_topo, "d_Eucl": d_eucl,
                     "max_r_Eucl": max_r_eucl, "time_s": elapsed})

    print()
    d_topos = [r["d_topo"] for r in rows]
    d_eucls = [r["d_Eucl"] for r in rows]
    print(f"Topological BFS (all edges = 1):  d = {np.mean(d_topos):.5f} ± {np.std(d_topos)/np.sqrt(len(rows)):.5f}")
    print(f"Euclidean weighted (diag = sqrt(3)): d = {np.mean(d_eucls):.5f} ± {np.std(d_eucls)/np.sqrt(len(rows)):.5f}")
    print()
    print("Comparison to candidates:")
    print(f"  alpha_EM exact target          : 3.131")
    print(f"  L=400 topological mean         : 3.106")
    print(f"  1/(3pi) ~ 3.106                : matches topo")
    print(f"  1/(2pi) ~ 3.159                : ?")

    out = {"L": L, "rows": rows}
    with open(DATA / "38_bfs_euclidean.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
