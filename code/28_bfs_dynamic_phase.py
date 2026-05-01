"""
Paper VI - dynamic BFS under the "phase = port rotator" rule.

At each node x and tick t, theta_x(t) selects which body-diagonal
line (out of 4) is currently open at x. The 6 cubic axes are always
open. A diagonal move from x to y at BFS distance t+1 is allowed iff
y's phase at tick t+1 matches the line connecting x and y.

Two variants:
  (A) synchronous: theta_x(t) = t mod 4 (same for all x)
  (B) asynchronous: theta_x(t) = (theta_x(0) + t) mod 4
      with theta_x(0) random per node

Reference for comparison:
  - static cubic (no diagonal at all)
  - static Poisson(mu=1) body-diagonal (paper's claim)
"""
import numpy as np
import json
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi

# 4 unsigned body-diagonal lines, represented by one positive endpoint each
DIRS4 = np.array([
    [+1, +1, +1],   # line L0
    [+1, +1, -1],   # line L1
    [+1, -1, +1],   # line L2
    [-1, +1, +1],   # line L3
], dtype=np.int32)


def idx_of(i, j, k, L):
    return ((i % L) * L + (j % L)) * L + (k % L)


def coords_of(idx, L):
    k = idx % L
    j = (idx // L) % L
    i = idx // (L * L)
    return i, j, k


def line_index_of_displacement(dx, dy, dz):
    """Return line index (0..3) for a body-diagonal displacement,
    or -1 if not a body diagonal."""
    if abs(dx) != 1 or abs(dy) != 1 or abs(dz) != 1:
        return -1
    # Normalize so dx > 0; if dx < 0, flip all signs
    if dx < 0:
        dx, dy, dz = -dx, -dy, -dz
    # Now dx = +1, find which line
    for i, d in enumerate(DIRS4):
        if d[0] == dx and d[1] == dy and d[2] == dz:
            return i
    return -1


def bfs_dynamic_synchronous(L, max_ticks=None):
    """Variant A: theta(t) = t mod 4 same everywhere."""
    N = L**3
    if max_ticks is None:
        max_ticks = 3 * L
    dist = np.full(N, -1, dtype=np.int32)
    src = idx_of(L // 2, L // 2, L // 2, L)
    dist[src] = 0
    frontier = [src]
    t = 0
    cubic_steps = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while frontier and t < max_ticks:
        t += 1
        active_line = (t - 1) % 4   # diagonal line active at tick t
        d_act = DIRS4[active_line]
        diag_steps = [tuple(d_act), tuple(-d_act)]  # both signs along the line
        next_frontier = []
        for x_idx in frontier:
            i, j, k = coords_of(x_idx, L)
            # Cubic moves (always)
            for di, dj, dk in cubic_steps:
                y_idx = idx_of(i + di, j + dj, k + dk, L)
                if dist[y_idx] < 0:
                    dist[y_idx] = t
                    next_frontier.append(y_idx)
            # Diagonal moves (only along the active line at this tick)
            for di, dj, dk in diag_steps:
                y_idx = idx_of(i + di, j + dj, k + dk, L)
                if dist[y_idx] < 0:
                    dist[y_idx] = t
                    next_frontier.append(y_idx)
        frontier = next_frontier
    return dist


def bfs_dynamic_asynchronous(L, seed=42, max_ticks=None):
    """Variant B: theta_x(0) random per node, theta_x(t) = (theta_x(0)+t) mod 4."""
    N = L**3
    if max_ticks is None:
        max_ticks = 3 * L
    rng = np.random.default_rng(seed)
    theta0 = rng.integers(0, 4, size=N).astype(np.int8)
    dist = np.full(N, -1, dtype=np.int32)
    src = idx_of(L // 2, L // 2, L // 2, L)
    dist[src] = 0
    frontier = [src]
    t = 0
    cubic_steps = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while frontier and t < max_ticks:
        t += 1
        next_frontier = []
        for x_idx in frontier:
            i, j, k = coords_of(x_idx, L)
            # Cubic moves (always)
            for di, dj, dk in cubic_steps:
                y_idx = idx_of(i + di, j + dj, k + dk, L)
                if dist[y_idx] < 0:
                    dist[y_idx] = t
                    next_frontier.append(y_idx)
            # Diagonal moves: for each line index, check if destination phase matches
            for line_idx in range(4):
                d = DIRS4[line_idx]
                for sign in (1, -1):
                    di, dj, dk = sign * d[0], sign * d[1], sign * d[2]
                    y_idx = idx_of(i + di, j + dj, k + dk, L)
                    # destination phase at tick t = (theta0[y] + t) mod 4
                    phi_y_t = (theta0[y_idx] + t) % 4
                    if phi_y_t == line_idx and dist[y_idx] < 0:
                        dist[y_idx] = t
                        next_frontier.append(y_idx)
        frontier = next_frontier
    return dist


def bfs_static_cubic(L):
    """Plain cubic BFS, for comparison."""
    N = L**3
    dist = np.full(N, -1, dtype=np.int32)
    src = idx_of(L // 2, L // 2, L // 2, L)
    dist[src] = 0
    frontier = [src]
    cubic_steps = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    t = 0
    while frontier:
        t += 1
        next_frontier = []
        for x_idx in frontier:
            i, j, k = coords_of(x_idx, L)
            for di, dj, dk in cubic_steps:
                y_idx = idx_of(i + di, j + dj, k + dk, L)
                if dist[y_idx] < 0:
                    dist[y_idx] = t
                    next_frontier.append(y_idx)
        frontier = next_frontier
    return dist


def fit_dim(dist, r_min=2, r_frac_max=0.4):
    """Fit d_BFS from N(t) ~ t^d."""
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
    return float(d), cumul.tolist()


def main():
    target = 3 + 1 / (2 * PI)
    print("Target d_eff = 3 + 1/(2pi) = {0:.6f}".format(target))
    print()

    print("L scan: variant B asynchronous (3 seeds per L)")
    print("{0:>5} {1:>10} {2:>9} {3:>11}".format(
        "L", "d_B_mean", "se", "shift_vs_cubic"))
    print("-" * 50)
    rows = []
    for L in [48, 64, 80]:
        d_c, _ = fit_dim(bfs_static_cubic(L))
        d_Bs = []
        for s in range(3):
            d, _ = fit_dim(bfs_dynamic_asynchronous(L, seed=s))
            d_Bs.append(d)
        d_mean = float(np.mean(d_Bs))
        d_se = float(np.std(d_Bs) / np.sqrt(len(d_Bs)))
        shift = d_mean - d_c
        print("{0:>5d} {1:>10.5f} {2:>9.4f} {3:>+11.5f}".format(
            L, d_mean, d_se, shift))
        rows.append({"L": L, "d_cubic": d_c, "d_B_mean": d_mean,
                     "d_B_se": d_se, "shift": shift})

    d_B_mean = rows[-1]["d_B_mean"]
    d_B_std = rows[-1]["d_B_se"]
    d_c = rows[-1]["d_cubic"]
    L = rows[-1]["L"]
    d_A = 0.0  # not run in this version
    d_Bs = []

    out = {
        "L": L,
        "target": target,
        "target_shift": 1 / (2 * PI),
        "d_static_cubic": d_c,
        "d_synchronous_A": d_A,
        "d_asynchronous_B_mean": d_B_mean,
        "d_asynchronous_B_std": d_B_std,
        "d_asynchronous_B_seeds": d_Bs,
    }
    with open(DATA / "28_bfs_dynamic_phase.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved -> {0}".format(DATA / "28_bfs_dynamic_phase.json"))


if __name__ == "__main__":
    main()
