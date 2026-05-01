"""
Paper VI - test of the STRICT routing variant (variant C-strict).

In this version, each node has theta_x in Z/10Z selecting one of its
10 incident ports (6 cubic + 4 body-diagonal lines). At each tick,
ONLY that single port is open as the receiver.

This is the strict version of "energy enters through exactly one port
per tick".

Asynchronous initial phases, deterministic rotation.

Compared to variant B (where 6 cubic axes are always open and only
the 4 diagonals are gated).
"""
import numpy as np
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi

# 10 ports: 6 cubic + 4 body-diagonal lines
PORTS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
    # Diagonal lines: each line has both signs, but we use one port idx per line
    # We'll use 8 endpoints to keep it simple? Or just 4 line-indices?
]
# Actually let me use 14 ports: 6 cubic + 8 diagonal-corner. Each port is unique.
PORTS_14 = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
    (1, 1, 1), (-1, -1, -1),
    (1, 1, -1), (-1, -1, 1),
    (1, -1, 1), (-1, 1, -1),
    (-1, 1, 1), (1, -1, -1),
]
PORTS_14 = np.array(PORTS_14, dtype=np.int32)
NUM_PORTS = len(PORTS_14)


def idx_of(i, j, k, L):
    return ((i % L) * L + (j % L)) * L + (k % L)


def bfs_strict(L, seed=42, max_ticks=None):
    """At each tick, each node has ONE active in-port chosen by theta(t)."""
    N = L**3
    if max_ticks is None:
        max_ticks = 4 * L  # may need more time since restricted
    rng = np.random.default_rng(seed)
    theta0 = rng.integers(0, NUM_PORTS, size=N).astype(np.int8)
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

        # For each port, check if any neighbour gated by that port can be
        # reached from the frontier
        for port_idx in range(NUM_PORTS):
            d = PORTS_14[port_idx]
            # Source goes to y in direction d; y receives from -d at port_idx
            # i.e., y has its in-port direction = -d at tick t.
            # But we encoded port_idx as a direction. Let's say port_idx
            # encodes the direction the in-port points to (= direction TO
            # the source from y).
            # So if y's in-port at tick t is direction port_idx, then the
            # neighbour to y in that direction (= y + PORTS_14[port_idx])
            # is the source.
            # Equivalent: from x in frontier, the destination y = x - d.
            ni = (fi - d[0]) % L
            nj = (fj - d[1]) % L
            nk = (fk - d[2]) % L
            ny = (ni * L + nj) * L + nk
            phi_y_t = (theta0[ny].astype(np.int32) + t) % NUM_PORTS
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
    print("Target d_eff = 3 + 1/(2pi) = {0:.6f}".format(target))
    print()
    print("Strict routing: each node has ONE active in-port per tick")
    print("among 14 ports (6 cubic + 8 diagonal corners).")
    print()

    print("{0:>5} {1:>10} {2:>9} {3:>10}".format(
        "L", "d_strict", "se", "time_s"))
    print("-" * 40)
    rows = []
    for L in [48, 64, 80]:
        t0 = time.time()
        d_strict = []
        for s in range(3):
            d = fit_dim(bfs_strict(L, seed=s, max_ticks=4*L))
            d_strict.append(d)
        d_mean = float(np.mean(d_strict))
        d_se = float(np.std(d_strict) / np.sqrt(len(d_strict)))
        ttot = time.time() - t0
        print("{0:>5d} {1:>10.5f} {2:>9.4f} {3:>10.1f}".format(
            L, d_mean, d_se, ttot))
        rows.append({"L": L, "d_strict_mean": d_mean,
                     "d_strict_se": d_se, "time_s": ttot})

    out = {"target": target, "results": rows}
    with open(DATA / "30_bfs_strict_routing.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved -> {0}".format(DATA / "30_bfs_strict_routing.json"))


if __name__ == "__main__":
    main()
