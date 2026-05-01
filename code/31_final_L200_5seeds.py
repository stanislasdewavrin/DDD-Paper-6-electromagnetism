"""
Final precision measurement: variant B asynchronous at L=200, 5 seeds.

Locks down the asymptotic d_eff with low statistical uncertainty.
"""
import numpy as np
import json
import time
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


def bfs_async_fast(L, seed=42, max_ticks=None):
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
    return float(d)


def main():
    target = 3 + 1 / (2 * PI)
    L = 200
    print("Final test: L = {0}, variant B asynchronous, 5 seeds".format(L))
    print("Target d_eff = 3 + 1/(2pi) = {0:.6f}".format(target))
    print()

    ds = []
    for s in range(5):
        t0 = time.time()
        d = fit_dim(bfs_async_fast(L, seed=s))
        ds.append(d)
        ttot = time.time() - t0
        print("  seed={0}: d_BFS = {1:.5f}  ({2:.1f}s)".format(s, d, ttot))

    d_mean = float(np.mean(ds))
    d_std = float(np.std(ds))
    d_se = d_std / np.sqrt(len(ds))
    print()
    print("d_BFS = {0:.5f} +- {1:.5f} (1 sigma)".format(d_mean, d_std))
    print("       = {0:.5f} +- {1:.5f} (SE)".format(d_mean, d_se))
    print()
    print("Target = {0:.5f}".format(target))
    print("Diff   = {0:+.5f}  ({1:+.2f}%)".format(
        d_mean - target, 100 * (d_mean - target) / target))

    out = {"L": L, "target": target, "n_seeds": 5,
           "d_BFS_seeds": ds, "d_BFS_mean": d_mean,
           "d_BFS_std": d_std, "d_BFS_se": d_se,
           "diff_target": d_mean - target}
    with open(DATA / "31_final_L200_5seeds.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved -> {0}".format(DATA / "31_final_L200_5seeds.json"))


if __name__ == "__main__":
    main()
