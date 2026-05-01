"""
Paper VI - large-L scan of asynchronous phase rotation BFS.

Pushes the dynamic asynchronous BFS (variant B from script 28) to
L=128, 160, 200 to confirm asymptotic d_eff convergence toward
3 + 1/(2pi).

Uses scipy sparse for cubic adjacency, plus per-tick diagonal gating.
For BFS we still need a custom level-by-level traversal because the
diagonal rules vary with t.
"""
import numpy as np
import json
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi

DIRS4 = np.array([
    [+1, +1, +1],
    [+1, +1, -1],
    [+1, -1, +1],
    [-1, +1, +1],
], dtype=np.int32)


def idx_of(i, j, k, L):
    return ((i % L) * L + (j % L)) * L + (k % L)


def coords_of(idx, L):
    k = idx % L
    j = (idx // L) % L
    i = idx // (L * L)
    return i, j, k


def bfs_async_fast(L, seed=42, max_ticks=None):
    """Fast custom BFS for variant B with vectorised neighbour expansion."""
    N = L**3
    if max_ticks is None:
        max_ticks = 2 * L
    rng = np.random.default_rng(seed)
    theta0 = rng.integers(0, 4, size=N).astype(np.int8)
    dist = np.full(N, -1, dtype=np.int32)
    src = idx_of(L // 2, L // 2, L // 2, L)
    dist[src] = 0
    frontier = np.array([src], dtype=np.int64)
    t = 0

    while len(frontier) > 0 and t < max_ticks:
        t += 1
        # Frontier coordinates
        fi = (frontier // (L * L)) % L
        fj = (frontier // L) % L
        fk = frontier % L
        next_frontier = []

        # 6 cubic axis moves (always)
        for di, dj, dk in [(1, 0, 0), (-1, 0, 0),
                            (0, 1, 0), (0, -1, 0),
                            (0, 0, 1), (0, 0, -1)]:
            ni = (fi + di) % L
            nj = (fj + dj) % L
            nk = (fk + dk) % L
            ny = (ni * L + nj) * L + nk
            mask = dist[ny] < 0
            ny_new = ny[mask]
            if len(ny_new) > 0:
                dist[ny_new] = t
                next_frontier.append(ny_new)

        # 4 diagonal lines, both signs; gate by destination phase
        for line_idx in range(4):
            d = DIRS4[line_idx]
            for sign in (1, -1):
                ni = (fi + sign * d[0]) % L
                nj = (fj + sign * d[1]) % L
                nk = (fk + sign * d[2]) % L
                ny = (ni * L + nj) * L + nk
                # Destination phase at tick t = (theta0[ny] + t) mod 4
                phi_y_t = (theta0[ny].astype(np.int32) + t) % 4
                mask = (phi_y_t == line_idx) & (dist[ny] < 0)
                ny_new = ny[mask]
                if len(ny_new) > 0:
                    dist[ny_new] = t
                    next_frontier.append(ny_new)

        if next_frontier:
            frontier = np.unique(np.concatenate(next_frontier))
        else:
            frontier = np.array([], dtype=np.int64)
    return dist


def bfs_static_cubic_fast(L):
    N = L**3
    dist = np.full(N, -1, dtype=np.int32)
    src = idx_of(L // 2, L // 2, L // 2, L)
    dist[src] = 0
    frontier = np.array([src], dtype=np.int64)
    t = 0
    while len(frontier) > 0:
        t += 1
        fi = (frontier // (L * L)) % L
        fj = (frontier // L) % L
        fk = frontier % L
        next_frontier = []
        for di, dj, dk in [(1, 0, 0), (-1, 0, 0),
                            (0, 1, 0), (0, -1, 0),
                            (0, 0, 1), (0, 0, -1)]:
            ni = (fi + di) % L
            nj = (fj + dj) % L
            nk = (fk + dk) % L
            ny = (ni * L + nj) * L + nk
            mask = dist[ny] < 0
            ny_new = ny[mask]
            if len(ny_new) > 0:
                dist[ny_new] = t
                next_frontier.append(ny_new)
        if next_frontier:
            frontier = np.unique(np.concatenate(next_frontier))
        else:
            frontier = np.array([], dtype=np.int64)
    return dist


def fit_dim(dist, r_min=2, r_frac_max=0.4):
    finite = dist[dist >= 0]
    max_d = int(finite.max())
    N_at = np.bincount(finite, minlength=max_d + 1)
    cumul = np.cumsum(N_at)
    rs = np.arange(len(cumul))
    r_max = int(r_frac_max * len(cumul))
    mask = (rs >= r_min) & (rs <= r_max) & (cumul > 0)
    log_r = np.log(rs[mask])
    log_N = np.log(cumul[mask])
    d, _ = np.polyfit(log_r, log_N, 1)
    return float(d)


def main():
    target = 3 + 1 / (2 * PI)
    target_shift = 1 / (2 * PI)
    print("Target d_eff = 3 + 1/(2pi) = {0:.6f}".format(target))
    print("Target shift over cubic = +{0:.5f}".format(target_shift))
    print()

    print("{0:>5} {1:>9} {2:>10} {3:>10} {4:>11} {5:>10}".format(
        "L", "N", "d_cubic", "d_async", "shift", "time_s"))
    print("-" * 64)
    rows = []
    for L in [80, 128, 200]:
        N = L**3
        t0 = time.time()
        dc = fit_dim(bfs_static_cubic_fast(L))
        # 2 seeds for L=200, 3 for smaller, to fit in time budget
        n_seeds = 3 if L <= 128 else 2
        d_asyncs = [fit_dim(bfs_async_fast(L, seed=s)) for s in range(n_seeds)]
        d_async = float(np.mean(d_asyncs))
        d_async_se = float(np.std(d_asyncs) / np.sqrt(n_seeds))
        shift = d_async - dc
        ttot = time.time() - t0
        print("{0:>5d} {1:>9d} {2:>10.5f} {3:>10.5f} {4:>+11.5f} "
              "{5:>10.1f}".format(L, N, dc, d_async, shift, ttot))
        rows.append({"L": L, "N": N, "d_cubic": dc,
                     "d_async": d_async, "d_async_se": d_async_se,
                     "shift": shift, "time_s": ttot,
                     "n_seeds": n_seeds, "seeds_d": d_asyncs})

    # 1/L extrapolation
    Ls = np.array([r["L"] for r in rows])
    das = np.array([r["d_async"] for r in rows])
    dcs = np.array([r["d_cubic"] for r in rows])
    inv_L = 1.0 / Ls
    print()
    if len(rows) >= 2:
        a_a, b_a = np.polyfit(inv_L, das, 1)
        a_c, b_c = np.polyfit(inv_L, dcs, 1)
        offset = 3.0 - b_c
        b_a_corr = b_a + offset
        print("1/L extrapolation:")
        print("  d_cubic(L=inf)   = {0:.5f}  (analytic 3.0)".format(b_c))
        print("  d_async(L=inf)   = {0:.5f}  (raw)".format(b_a))
        print("  d_async(L=inf)   = {0:.5f}  (bias-corrected)".format(b_a_corr))
        print("  target           = {0:.5f}".format(target))
        print("  abs(d_async_corr - target) = {0:+.5f}  ({1:.2f}%)".format(
            b_a_corr - target, 100 * (b_a_corr - target) / target))

    out = {"target": target, "target_shift": target_shift, "results": rows}
    with open(DATA / "29_bfs_async_large_L.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved -> {0}".format(DATA / "29_bfs_async_large_L.json"))


if __name__ == "__main__":
    main()
