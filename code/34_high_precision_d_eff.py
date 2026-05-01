"""
Paper VI - high precision d_eff to discriminate 1/(2pi) vs 1/6.

Targets:
  3 + 1/(2pi) = 3.15915
  3 + 1/6     = 3.16667
  diff       = 0.00752

We want SE < 0.004 on the extrapolated d to distinguish at ~2 sigma.

Strategy: run variant B asynchronous (cubic always + 4 body-diagonals
gated 1/4) at L=128, 200, 300 with many seeds. Extrapolate 1/L.
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


def bfs_async(L, seed=42, max_ticks=None):
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


def bfs_cubic(L):
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
    print("HIGH-PRECISION d_eff measurement")
    print("Targets: 3 + 1/(2pi) = {0:.5f}".format(3 + 1/(2*PI)))
    print("         3 + 1/6     = {0:.5f}".format(3 + 1/6))
    print()

    plan = [(128, 12), (200, 8)]   # (L, n_seeds)
    rows = []
    for L, n_seeds in plan:
        t0 = time.time()
        dc = fit_dim(bfs_cubic(L))
        ds = []
        for s in range(n_seeds):
            d = fit_dim(bfs_async(L, seed=s))
            ds.append(d)
        d_mean = float(np.mean(ds))
        d_se = float(np.std(ds, ddof=1) / np.sqrt(n_seeds))
        ttot = time.time() - t0
        print("L={0:4d}: n={1:3d}  d_BFS = {2:.5f} +- {3:.5f}  d_cubic = {4:.5f}  ({5:.1f}s)".format(
            L, n_seeds, d_mean, d_se, dc, ttot))
        rows.append({"L": L, "n_seeds": n_seeds, "d_cubic": dc,
                     "d_mean": d_mean, "d_se": d_se,
                     "d_seeds": ds, "time_s": ttot})

    print()
    # Extrapolation 1/L (raw)
    Ls = np.array([r["L"] for r in rows])
    dms = np.array([r["d_mean"] for r in rows])
    dcs = np.array([r["d_cubic"] for r in rows])
    inv_L = 1.0 / Ls

    if len(rows) >= 2:
        # Weighted fit
        ses = np.array([r["d_se"] for r in rows])
        weights = 1.0 / (ses**2)
        a, b = np.polyfit(inv_L, dms, 1, w=weights)
        # Bias-correction: assume d_cubic_inf = 3.0 exact
        a_c, b_c = np.polyfit(inv_L, dcs, 1)
        offset = 3.0 - b_c
        b_corr = b + offset
        # Estimate uncertainty on extrapolation
        # Simple: use largest L SE as proxy
        se_extrap = ses[-1] * 1.5  # rough estimate
        print("1/L extrapolation:")
        print("  d_async(L=inf, raw)        = {0:.5f}".format(b))
        print("  d_cubic(L=inf, raw)        = {0:.5f}  (analytic 3.0)".format(b_c))
        print("  d_async(L=inf, corrected)  = {0:.5f} +- {1:.5f}".format(
            b_corr, se_extrap))
        print()
        print("Comparison:")
        print("  vs 3 + 1/(2pi) = 3.15915 : diff = {0:+.5f} ({1:.1f}sigma)".format(
            b_corr - (3 + 1/(2*PI)), abs(b_corr - (3 + 1/(2*PI))) / se_extrap))
        print("  vs 3 + 1/6     = 3.16667 : diff = {0:+.5f} ({1:.1f}sigma)".format(
            b_corr - (3 + 1/6), abs(b_corr - (3 + 1/6)) / se_extrap))

    out = {"results": rows,
           "target_1_2pi": 3 + 1/(2*PI),
           "target_1_6": 3 + 1/6}
    with open(DATA / "34_high_precision_d_eff.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved -> {0}".format(DATA / "34_high_precision_d_eff.json"))


if __name__ == "__main__":
    main()
