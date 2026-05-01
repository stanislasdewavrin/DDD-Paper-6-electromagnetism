"""
L=600 or L=800 BFS measurement, one seed at a time.

Usage: python3 37_L_large_one_seed.py <L> <seed>
"""
import sys
import numpy as np
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

DIRS4 = np.array([
    [+1, +1, +1],
    [+1, +1, -1],
    [+1, -1, +1],
    [-1, +1, +1],
], dtype=np.int32)


def idx_of(i, j, k, L):
    return ((i % L) * L + (j % L)) * L + (k % L)


def bfs_async_fast(L, seed=42, max_ticks=None):
    N = L**3
    if max_ticks is None:
        # Just enough to cover fit range (we fit up to 0.4 * max_dist)
        max_ticks = int(0.7 * L)
    rng = np.random.default_rng(seed)
    theta0 = rng.integers(0, 4, size=N).astype(np.int8)
    dist = np.full(N, -1, dtype=np.int32)
    src = idx_of(L // 2, L // 2, L // 2, L)
    dist[src] = 0
    frontier = np.array([src], dtype=np.int64)
    t = 0
    while len(frontier) > 0 and t < max_ticks:
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
        for line_idx in range(4):
            d = DIRS4[line_idx]
            for sign in (1, -1):
                ni = (fi + sign * d[0]) % L
                nj = (fj + sign * d[1]) % L
                nk = (fk + sign * d[2]) % L
                ny = (ni * L + nj) * L + nk
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
    return float(d), max_d, len(rs[mask])


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    OUT_FILE = DATA / f"37_L{L}_seeds.json"
    print(f"Running L={L}, seed={seed}...")
    t0 = time.time()
    dist = bfs_async_fast(L, seed=seed)
    d, max_d, n_fit_pts = fit_dim(dist)
    elapsed = time.time() - t0
    print(f"  d_BFS = {d:.5f}  (max_d={max_d}, fit_pts={n_fit_pts}, time={elapsed:.1f}s)")

    if OUT_FILE.exists():
        data = json.loads(OUT_FILE.read_text())
    else:
        data = {"L": L, "seeds": []}
    data["seeds"].append({"seed": seed, "d_BFS": d, "max_d": max_d,
                          "n_fit_pts": n_fit_pts, "time_s": elapsed})
    OUT_FILE.write_text(json.dumps(data, indent=2))
    if len(data['seeds']) >= 2:
        ds = [s['d_BFS'] for s in data['seeds']]
        mean = np.mean(ds)
        se = np.std(ds, ddof=1) / np.sqrt(len(ds))
        print(f"  Mean (n={len(ds)}): d_BFS = {mean:.5f} ± {se:.5f}")


if __name__ == "__main__":
    main()
