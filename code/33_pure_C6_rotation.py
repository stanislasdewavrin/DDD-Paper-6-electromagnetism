"""
Paper VI - test of pure C_6 cubic-port rotation.

Stan's proposal: each node has 6 cubic ports, phase theta in Z/6Z,
rotation theta(t+1) = theta(t)+1 mod 6, asynchronous initial phases.
No body-diagonals at all.

Three variants tested:

  (P1) Strict 1-port: at each tick, only ONE cubic port is open as
       receiver (the one indexed by theta(t)).

  (P2) Bidirectional: at each tick, the LINE indexed by theta(t)//2
       is open (both +/- of one axis). 3 lines x 2 directions = 6 ports
       grouped as 3 axes; phase Z/6Z splits into 3 lines x 2 phases.

  (P3) Cubic permanent + phase as bonus: 6 cubic axes always open,
       phase encodes a 7th "shortcut bonus" that activates 1/6 of time.
       (closer to our previous variant B but with C_6 instead of C_4)

Goal: measure d_BFS for each, compare to 3 + 1/6 = 3.16667 and
3 + 1/(2pi) = 3.15915.
"""
import numpy as np
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi

# 6 cubic directions, indexed 0..5
CUBIC = [
    (+1, 0, 0), (-1, 0, 0),
    (0, +1, 0), (0, -1, 0),
    (0, 0, +1), (0, 0, -1),
]


def idx_of(i, j, k, L):
    return ((i % L) * L + (j % L)) * L + (k % L)


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
        for di, dj, dk in CUBIC:
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


def bfs_P1_strict(L, seed=42, max_ticks=None):
    """Strict 1-cubic-port rotating per tick.

    At node y at tick t, only cubic port indexed by theta_y(t) is open.
    Move x -> y allowed iff CUBIC[theta_y(t)] = direction from y to x.
    """
    N = L**3
    if max_ticks is None:
        max_ticks = 6 * L  # may need many ticks since restricted
    rng = np.random.default_rng(seed)
    theta0 = rng.integers(0, 6, size=N).astype(np.int8)
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
        for port_idx in range(6):
            di, dj, dk = CUBIC[port_idx]
            # If y's port at tick t is direction (di, dj, dk), then y can
            # receive from y + (di, dj, dk). So from frontier x, the
            # destination y = x - (di, dj, dk).
            ni = (fi - di) % L
            nj = (fj - dj) % L
            nk = (fk - dk) % L
            ny = (ni * L + nj) * L + nk
            phi_y_t = (theta0[ny].astype(np.int32) + t) % 6
            mask = (phi_y_t == port_idx) & (dist[ny] < 0)
            ny_new = ny[mask]
            if len(ny_new) > 0:
                dist[ny_new] = t
                next_frontier.append(ny_new)
        if next_frontier:
            frontier = np.unique(np.concatenate(next_frontier))
        else:
            frontier = np.array([], dtype=np.int64)
    return dist


def bfs_P2_bidir(L, seed=42, max_ticks=None):
    """Phase Z/6Z, but open BOTH +/- ports of an axis per tick.

    At each tick, theta_y(t) mod 3 = axis. The line on that axis is open
    (both +/- directions).

    So at tick t, line x is open at y iff theta_y(t) mod 3 = 0
                       y axis     iff theta_y(t) mod 3 = 1
                       z axis     iff theta_y(t) mod 3 = 2
    Each axis is active 1/3 of time. Each line has 2 endpoints.
    Effective: 2 ports active out of 6 per tick = 1/3 active.
    """
    N = L**3
    if max_ticks is None:
        max_ticks = 4 * L
    rng = np.random.default_rng(seed)
    theta0 = rng.integers(0, 6, size=N).astype(np.int8)
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
        for axis_idx in range(3):
            for sign in (1, -1):
                di = sign if axis_idx == 0 else 0
                dj = sign if axis_idx == 1 else 0
                dk = sign if axis_idx == 2 else 0
                ni = (fi + di) % L
                nj = (fj + dj) % L
                nk = (fk + dk) % L
                ny = (ni * L + nj) * L + nk
                phi_y_t = (theta0[ny].astype(np.int32) + t) % 6
                axis_active = phi_y_t % 3
                mask = (axis_active == axis_idx) & (dist[ny] < 0)
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
    if len(finite) == 0:
        return float('nan')
    max_d = int(finite.max())
    if max_d < 2:
        return float('nan')
    N_at = np.bincount(finite, minlength=max_d + 1)
    cumul = np.cumsum(N_at)
    rs = np.arange(len(cumul))
    r_max = int(r_frac_max * len(cumul))
    mask = (rs >= r_min) & (rs <= r_max) & (cumul > 0)
    if mask.sum() < 3:
        return float('nan')
    log_r = np.log(rs[mask])
    log_N = np.log(cumul[mask])
    d, _ = np.polyfit(log_r, log_N, 1)
    return float(d)


def main():
    L = 64
    print("L = {0}".format(L))
    print("Targets: 3 + 1/6  = {0:.5f}".format(3 + 1/6))
    print("         3 + 1/(2pi) = {0:.5f}".format(3 + 1/(2*PI)))
    print()

    d_cubic = fit_dim(bfs_static_cubic(L))
    print("Reference d_static_cubic = {0:.5f} (finite-size)".format(d_cubic))
    print("(asymptotically should be 3.0)")
    print()

    print("=" * 65)
    print("P1: STRICT 1-port rotating (1/6 active fraction)")
    print("=" * 65)
    print("  Note: this severely restricts BFS expansion since only 1")
    print("  of 6 cubic edges is open per tick at each node.")
    print()
    d_P1s = []
    for s in range(3):
        t0 = time.time()
        d = fit_dim(bfs_P1_strict(L, seed=s, max_ticks=8*L))
        ttot = time.time() - t0
        if not np.isnan(d):
            d_P1s.append(d)
            print("  seed={0}: d_BFS = {1:.5f}  ({2:.1f}s)".format(s, d, ttot))
        else:
            print("  seed={0}: BFS too restricted to fit  ({1:.1f}s)".format(
                s, ttot))
    if d_P1s:
        d_P1_mean = np.mean(d_P1s)
        d_P1_std = np.std(d_P1s)
        print("  Mean d_BFS = {0:.5f} +- {1:.5f}".format(d_P1_mean, d_P1_std))
        print("  Shift over cubic = {0:+.5f}".format(d_P1_mean - d_cubic))
    print()

    print("=" * 65)
    print("P2: BIDIRECTIONAL (line active per axis, 1/3 active fraction)")
    print("=" * 65)
    print()
    d_P2s = []
    for s in range(3):
        t0 = time.time()
        d = fit_dim(bfs_P2_bidir(L, seed=s, max_ticks=4*L))
        ttot = time.time() - t0
        if not np.isnan(d):
            d_P2s.append(d)
            print("  seed={0}: d_BFS = {1:.5f}  ({2:.1f}s)".format(s, d, ttot))
        else:
            print("  seed={0}: BFS too restricted to fit  ({1:.1f}s)".format(
                s, ttot))
    if d_P2s:
        d_P2_mean = np.mean(d_P2s)
        d_P2_std = np.std(d_P2s)
        print("  Mean d_BFS = {0:.5f} +- {1:.5f}".format(d_P2_mean, d_P2_std))
        print("  Shift over cubic = {0:+.5f}".format(d_P2_mean - d_cubic))
    print()

    out = {
        "L": L,
        "d_static_cubic": d_cubic,
        "P1_strict": {"d_seeds": d_P1s,
                      "d_mean": float(np.mean(d_P1s)) if d_P1s else None},
        "P2_bidir":  {"d_seeds": d_P2s,
                      "d_mean": float(np.mean(d_P2s)) if d_P2s else None},
        "target_1_over_6": 3 + 1/6,
        "target_1_over_2pi": 3 + 1/(2*PI),
    }
    with open(DATA / "33_pure_C6_rotation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved -> {0}".format(DATA / "33_pure_C6_rotation.json"))


if __name__ == "__main__":
    main()
