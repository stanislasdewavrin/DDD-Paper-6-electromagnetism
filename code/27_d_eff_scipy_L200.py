"""
Paper VI - precision d_eff at very large L using scipy sparse BFS.

Targets L=128 and L=200 (8M nodes) for the Poisson(mu=1) body-diagonal
substrate, to nail down d_eff vs target 3 + 1/(2pi).
"""
import numpy as np
import json
import time
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi

DIRS4 = np.array([
    [+1, +1, +1],
    [+1, +1, -1],
    [+1, -1, +1],
    [-1, +1, +1],
], dtype=np.int32)


def build_edges_cubic(L):
    """Cubic-axis edges as flat (rows, cols) arrays.

    Uses periodic boundaries.
    """
    idx = np.arange(L**3, dtype=np.int64).reshape(L, L, L)
    rows = []
    cols = []
    for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        shifted = np.roll(idx, shift=(-di, -dj, -dk), axis=(0, 1, 2))
        rows.append(idx.ravel())
        cols.append(shifted.ravel())
    return (np.concatenate(rows), np.concatenate(cols))


def build_edges_poisson_diag(L, mu=1.0, seed=42):
    """Body-diagonal Poisson(mu) shortcuts."""
    rng = np.random.default_rng(seed)
    N = L**3
    # For each node, draw Poisson(mu) shortcuts
    k_short = rng.poisson(mu, size=N)
    total = int(k_short.sum())
    if total == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    # Source nodes (each repeated k_short[u] times)
    src = np.repeat(np.arange(N, dtype=np.int64), k_short)
    # Choose direction index 0..3 and sign +/-1
    didx = rng.integers(0, 4, size=total)
    sign = np.where(rng.random(total) < 0.5, 1, -1).astype(np.int32)
    dvecs = DIRS4[didx] * sign[:, None]
    # Source coords
    src_i = src // (L * L)
    src_j = (src // L) % L
    src_k = src % L
    dst_i = (src_i + dvecs[:, 0]) % L
    dst_j = (src_j + dvecs[:, 1]) % L
    dst_k = (src_k + dvecs[:, 2]) % L
    dst = (dst_i * L + dst_j) * L + dst_k
    return (src.astype(np.int64), dst.astype(np.int64))


def build_csr(L, with_poisson=True, mu=1.0, seed=42):
    N = L**3
    rows_c, cols_c = build_edges_cubic(L)
    if with_poisson:
        rows_p, cols_p = build_edges_poisson_diag(L, mu=mu, seed=seed)
        rows = np.concatenate([rows_c, cols_c, rows_p, cols_p])
        cols = np.concatenate([cols_c, rows_c, cols_p, rows_p])
    else:
        rows = np.concatenate([rows_c, cols_c])
        cols = np.concatenate([cols_c, rows_c])
    data = np.ones(len(rows), dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(N, N))


def bfs_dim_scipy(graph, source, r_min=2, r_frac_max=0.4):
    """BFS dimension via scipy unweighted shortest path."""
    dist = dijkstra(graph, indices=source, unweighted=True,
                    directed=False, return_predecessors=False)
    finite = dist[np.isfinite(dist)].astype(np.int32)
    max_d = int(finite.max())
    N_at = np.bincount(finite, minlength=max_d + 1)
    cumul = np.cumsum(N_at)
    rs = np.arange(len(cumul))
    r_max = int(r_frac_max * len(cumul))
    mask = (rs >= r_min) & (rs <= r_max) & (cumul > 0)
    log_r = np.log(rs[mask])
    log_N = np.log(cumul[mask])
    d_H, _ = np.polyfit(log_r, log_N, 1)
    return float(d_H)


def averaged_dim(graph, n_sources=4, seed=0):
    rng = np.random.default_rng(seed)
    N = graph.shape[0]
    ds = []
    for _ in range(n_sources):
        src = int(rng.integers(0, N))
        ds.append(bfs_dim_scipy(graph, src))
    return float(np.mean(ds)), float(np.std(ds) / np.sqrt(len(ds)))


def main():
    target = 3 + 1 / (2 * PI)
    print("Target d_eff = 3 + 1/(2pi) = {0:.6f}".format(target))
    print()

    rows = []
    Ls = [80, 128, 200]
    print("{0:>5} {1:>9} {2:>10} {3:>10} {4:>10} {5:>10} {6:>9}".format(
        "L", "N", "d_cubic", "se_c", "d_poisson", "se_p", "time_s"))
    print("-" * 72)
    for L in Ls:
        N = L**3
        t0 = time.time()
        g_c = build_csr(L, with_poisson=False)
        g_p = build_csr(L, with_poisson=True, mu=1.0, seed=42)
        t_build = time.time() - t0
        t0 = time.time()
        n_src = 4 if L <= 128 else 2
        dc, se_c = averaged_dim(g_c, n_sources=n_src, seed=11)
        dp, se_p = averaged_dim(g_p, n_sources=n_src, seed=11)
        t_bfs = time.time() - t0
        ttot = t_build + t_bfs
        print("{0:>5d} {1:>9d} {2:>10.5f} {3:>10.4f} {4:>10.5f} "
              "{5:>10.4f} {6:>9.1f}".format(L, N, dc, se_c, dp, se_p, ttot))
        rows.append({"L": L, "N": N, "n_sources": n_src,
                     "d_cubic": dc, "se_cubic": se_c,
                     "d_poisson": dp, "se_poisson": se_p,
                     "time_s": ttot})

    # Extrapolation
    Ls_arr = np.array([r["L"] for r in rows])
    dps = np.array([r["d_poisson"] for r in rows])
    dcs = np.array([r["d_cubic"] for r in rows])
    inv_L = 1.0 / Ls_arr

    print()
    if len(rows) >= 2:
        a_p, b_p = np.polyfit(inv_L, dps, 1)
        a_c, b_c = np.polyfit(inv_L, dcs, 1)
        # Bias-corrected: assume d_cubic(L=inf) = 3.0 exactly
        offset = 3.0 - b_c
        b_p_corr = b_p + offset
        print("1/L extrapolation:")
        print("  d_cubic(L=inf)   = {0:.5f}  (analytic 3.0)".format(b_c))
        print("  d_poisson(L=inf) = {0:.5f}  (raw)".format(b_p))
        print("  d_poisson(L=inf) = {0:.5f}  (bias-corrected)".format(b_p_corr))
        print("  target           = {0:.5f}".format(target))
        print("  abs(d_poisson_corr - target) = {0:+.5f}  ({1:.2f}%)".format(
            b_p_corr - target, 100 * (b_p_corr - target) / target))

    out = {"target": target, "results": rows}
    with open(DATA / "27_d_eff_scipy_L200.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved -> {0}".format(DATA / "27_d_eff_scipy_L200.json"))


if __name__ == "__main__":
    main()
