"""
Paper VI - test if 1/(2pi) is intrinsic or m-dependent.

Vary the cardinality m of the phase Z/m Z. Two parameterisations:

  Option A: line = phase mod 4 (4 lines cycled, m controls cycle length)
            -> long-time fraction of each line = 1/4 regardless of m

  Option B: line = phase if phase < 4 else NO_DIAG
            -> active fraction = 4/m, scales with m

If shift is constant under Option A but varies under Option B,
the shift is determined by the static activation distribution,
not by the cycle structure.

If shift varies under both, m is intrinsically relevant.
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


def bfs_async_phase_m(L, m, option='A', seed=42, max_ticks=None):
    """Variant B asynchronous with phase in Z/m Z."""
    N = L**3
    if max_ticks is None:
        max_ticks = 2 * L
    rng = np.random.default_rng(seed)
    theta0 = rng.integers(0, m, size=N).astype(np.int32)
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
        # Cubic axes always
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
        # Diagonal lines gated by phase
        for line_idx in range(4):
            d = DIRS4[line_idx]
            for sign in (1, -1):
                ni = (fi + sign * d[0]) % L
                nj = (fj + sign * d[1]) % L
                nk = (fk + sign * d[2]) % L
                ny = (ni * L + nj) * L + nk
                phi_y_t = (theta0[ny] + t) % m
                if option == 'A':
                    active_line = phi_y_t % 4
                else:  # option B
                    active_line = np.where(phi_y_t < 4, phi_y_t, -1)
                mask = (active_line == line_idx) & (dist[ny] < 0)
                ny_new = ny[mask]
                if len(ny_new) > 0:
                    dist[ny_new] = t
                    next_frontier.append(ny_new)
        if next_frontier:
            frontier = np.unique(np.concatenate(next_frontier))
        else:
            frontier = np.array([], dtype=np.int64)
    return dist


def bfs_static_cubic(L):
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
    L = 64
    print("L = {0}, target shift = 1/(2pi) = {1:.5f}".format(L, target_shift))
    print()

    d_cubic = fit_dim(bfs_static_cubic(L))
    print("d_cubic = {0:.5f}".format(d_cubic))
    print()

    print("=" * 65)
    print("OPTION A: line = phase mod 4 (each line active 1/4 of time)")
    print("=" * 65)
    print("{0:>3} {1:>10} {2:>9} {3:>11}".format(
        "m", "d_BFS_mean", "se", "shift"))
    rowsA = []
    for m in [4, 8, 12, 16]:
        d_seeds = []
        for s in range(3):
            d = fit_dim(bfs_async_phase_m(L, m, option='A', seed=s))
            d_seeds.append(d)
        d_mean = float(np.mean(d_seeds))
        d_se = float(np.std(d_seeds) / np.sqrt(len(d_seeds)))
        shift = d_mean - d_cubic
        print("{0:>3d} {1:>10.5f} {2:>9.4f} {3:>+11.5f}".format(
            m, d_mean, d_se, shift))
        rowsA.append({"m": m, "d_BFS": d_mean, "se": d_se, "shift": shift})

    print()
    print("=" * 65)
    print("OPTION B: active fraction = 4/m (NO_DIAG when phase >= 4)")
    print("=" * 65)
    print("{0:>3} {1:>8} {2:>10} {3:>9} {4:>11}".format(
        "m", "frac", "d_BFS_mean", "se", "shift"))
    rowsB = []
    for m in [4, 6, 8, 12, 16]:
        d_seeds = []
        for s in range(3):
            d = fit_dim(bfs_async_phase_m(L, m, option='B', seed=s))
            d_seeds.append(d)
        d_mean = float(np.mean(d_seeds))
        d_se = float(np.std(d_seeds) / np.sqrt(len(d_seeds)))
        shift = d_mean - d_cubic
        frac = 4.0 / m
        print("{0:>3d} {1:>8.3f} {2:>10.5f} {3:>9.4f} {4:>+11.5f}".format(
            m, frac, d_mean, d_se, shift))
        rowsB.append({"m": m, "active_frac": frac, "d_BFS": d_mean,
                      "se": d_se, "shift": shift})

    print()
    print("Reference: target shift 1/(2pi) = {0:.5f}".format(target_shift))

    out = {"L": L, "target_shift": target_shift, "d_cubic": d_cubic,
           "option_A": rowsA, "option_B": rowsB}
    with open(DATA / "32_phase_cardinality_scan.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved -> {0}".format(DATA / "32_phase_cardinality_scan.json"))


if __name__ == "__main__":
    main()
