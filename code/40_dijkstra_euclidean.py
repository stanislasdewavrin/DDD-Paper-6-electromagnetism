"""
Compute the dimension via Euclidean-weighted Dijkstra:
  - Cubic edge weight = 1
  - Body-diagonal edge weight = sqrt(3)

Compare to topological BFS dimension (all weights = 1).

Uses scipy.sparse.csgraph.dijkstra for speed.
"""
import numpy as np
import json
import time
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

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


def build_graph(L, seed=42, weight_cubic=1.0, weight_diag=SQRT3):
    """Build sparse adjacency for variant B asynchronous static sample.

    Note: since Dijkstra runs on a static graph, we use a static realisation
    of the substrate where each node's body-diagonal line is fixed by
    theta_x(0). This is equivalent to averaging over time, capturing the
    asymptotic random-substrate behaviour.
    """
    rng = np.random.default_rng(seed)
    N = L**3
    theta0 = rng.integers(0, 4, size=N).astype(np.int32)

    rows = []
    cols = []
    weights = []

    # Cubic edges
    idx_array = np.arange(N, dtype=np.int64).reshape(L, L, L)
    for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        shifted = np.roll(idx_array, shift=(-di, -dj, -dk), axis=(0, 1, 2))
        rows.append(idx_array.ravel())
        cols.append(shifted.ravel())
        rows.append(shifted.ravel())
        cols.append(idx_array.ravel())
        weights.extend([weight_cubic] * (2 * N))

    # Diagonal edges: each node has only the diag line indexed by theta0[node]
    # active. Edge to both endpoints (sign +/- of d_line).
    for line_idx in range(4):
        d = DIRS4[line_idx]
        active = np.where(theta0 == line_idx)[0]
        if len(active) == 0:
            continue
        ai = active // (L * L)
        aj = (active // L) % L
        ak = active % L
        for sign in (1, -1):
            ni = (ai + sign * d[0]) % L
            nj = (aj + sign * d[1]) % L
            nk = (ak + sign * d[2]) % L
            ny = (ni * L + nj) * L + nk
            rows.append(active)
            cols.append(ny)
            rows.append(ny)
            cols.append(active)
            n_edges = 2 * len(active)
            weights.extend([weight_diag] * n_edges)

    rows_all = np.concatenate(rows)
    cols_all = np.concatenate(cols)
    weights_all = np.array(weights, dtype=np.float32)
    G = csr_matrix((weights_all, (rows_all, cols_all)), shape=(N, N))
    return G


def fit_dim_continuous(dist, n_bins=50, r_min=2.0, r_frac_max=0.4):
    finite = dist[np.isfinite(dist)]
    max_r = float(finite.max())
    r_max = r_frac_max * max_r
    bins = np.linspace(0, max_r, n_bins + 1)
    cumul = np.array([np.sum(finite <= b) for b in bins[1:]], dtype=float)
    rs_centers = (bins[:-1] + bins[1:]) / 2
    mask = (rs_centers >= r_min) & (rs_centers <= r_max) & (cumul > 0)
    log_r = np.log(rs_centers[mask])
    log_N = np.log(cumul[mask])
    d, _ = np.polyfit(log_r, log_N, 1)
    return float(d), max_r


def main():
    L = 80
    print(f"L = {L}")
    print()
    print(f"{'seed':<5} {'d_topo':<10} {'d_Eucl':<10} {'time_topo':<12} {'time_Eucl':<10}")
    print("-" * 60)
    rows = []
    for seed in range(3):
        # Topological (all weights = 1)
        t0 = time.time()
        G_topo = build_graph(L, seed=seed, weight_cubic=1.0, weight_diag=1.0)
        src = idx_of(L // 2, L // 2, L // 2, L)
        dist_topo = dijkstra(G_topo, indices=src, directed=False)
        d_topo, _ = fit_dim_continuous(dist_topo)
        t_topo = time.time() - t0
        # Euclidean weights
        t0 = time.time()
        G_eucl = build_graph(L, seed=seed, weight_cubic=1.0, weight_diag=SQRT3)
        dist_eucl = dijkstra(G_eucl, indices=src, directed=False)
        d_eucl, max_r_e = fit_dim_continuous(dist_eucl)
        t_eucl = time.time() - t0
        print(f"{seed:<5} {d_topo:<10.5f} {d_eucl:<10.5f} {t_topo:<12.1f} {t_eucl:<10.1f}")
        rows.append({"seed": seed, "d_topo": d_topo, "d_Eucl": d_eucl,
                     "time_topo": t_topo, "time_eucl": t_eucl})

    print()
    d_topos = [r["d_topo"] for r in rows]
    d_eucls = [r["d_Eucl"] for r in rows]
    print("=" * 60)
    print(f"Mean d_topo = {np.mean(d_topos):.5f} ± {np.std(d_topos)/np.sqrt(len(rows)):.5f}")
    print(f"Mean d_Eucl = {np.mean(d_eucls):.5f} ± {np.std(d_eucls)/np.sqrt(len(rows)):.5f}")
    print()
    print("Difference d_Eucl - d_topo:")
    delta = np.array(d_eucls) - np.array(d_topos)
    print(f"  Mean = {np.mean(delta):+.5f} ± {np.std(delta)/np.sqrt(len(rows)):.5f}")
    print()
    print("Comparison to physical targets:")
    print(f"  d_alpha (alpha_EM exact) = 3.1310")
    print(f"  d_BFS (L=400, 20 seeds)  = 3.1058")

    with open(DATA / "40_dijkstra_euclidean.json", "w") as f:
        json.dump({"L": L, "rows": rows,
                   "d_topo_mean": float(np.mean(d_topos)),
                   "d_eucl_mean": float(np.mean(d_eucls))}, f, indent=2)


if __name__ == "__main__":
    main()
