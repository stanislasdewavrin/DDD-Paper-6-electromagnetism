"""
Paper VI - shortcuts at arbitrary distance (small-world variants).

Tests several long-range shortcut distributions to see how d_BFS scales.

Models tested:
  (NW) Newman-Watts: each node has Poisson(mu) shortcuts to a UNIFORMLY
       random node anywhere on the lattice. The classical 3D small-world.

  (PL-alpha) Power-law distance: each node has 1 shortcut to a random
       direction at random Euclidean distance d, sampled with PDF
       P(d) ~ d^{-alpha} on the lattice. alpha=0 -> uniform, alpha large
       -> short shortcuts.

  (FixedR) Fixed Euclidean radius: each node has 1 shortcut to a random
       lattice point at exactly radius R (modulo discretisation).

Reference: Poisson(mu=1) body-diagonal (the paper's claim).

Goal: see whether any of these gives a clean d_BFS = 3 + 1/(2pi),
or whether the value is sensitive to the distance distribution.
"""
import numpy as np
import json
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi

DIRS4 = [
    (+1, +1, +1),
    (+1, +1, -1),
    (+1, -1, +1),
    (-1, +1, +1),
]


def idx_of(i, j, k, L):
    return ((i % L) * L + (j % L)) * L + (k % L)


def coords_of(idx, L):
    k = idx % L
    j = (idx // L) % L
    i = idx // (L * L)
    return i, j, k


def build_cubic_adj(L):
    N = L**3
    adj = [set() for _ in range(N)]
    for i in range(L):
        for j in range(L):
            for k in range(L):
                u = idx_of(i, j, k, L)
                for di, dj, dk in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    v = idx_of(i + di, j + dj, k + dk, L)
                    adj[u].add(v); adj[v].add(u)
    return adj


def add_edge(adj, u, v):
    if u != v:
        adj[u].add(v); adj[v].add(u)


def build_NW(L, mu=1.0, seed=42):
    """(NW) Newman-Watts: Poisson(mu) shortcuts to uniform random node."""
    rng = np.random.default_rng(seed)
    adj = build_cubic_adj(L)
    N = L**3
    for u in range(N):
        k_short = rng.poisson(mu)
        for _ in range(k_short):
            v = int(rng.integers(0, N))
            add_edge(adj, u, v)
    return adj


def build_powerlaw(L, alpha=2.0, mu=1.0, seed=42):
    """(PL) Power-law distance: shortcut probability ~ d^{-alpha}."""
    rng = np.random.default_rng(seed)
    adj = build_cubic_adj(L)
    N = L**3
    # Precompute shortcut probability per Euclidean distance
    # Use rejection sampling: pick random offsets, accept with prob ~ d^{-alpha}
    max_offset = L // 2
    for u in range(N):
        k_short = rng.poisson(mu)
        i, j, k = coords_of(u, L)
        attempts = 0
        added = 0
        while added < k_short and attempts < 1000:
            di = int(rng.integers(-max_offset, max_offset + 1))
            dj = int(rng.integers(-max_offset, max_offset + 1))
            dk = int(rng.integers(-max_offset, max_offset + 1))
            attempts += 1
            d2 = di * di + dj * dj + dk * dk
            if d2 == 0:
                continue
            d = np.sqrt(d2)
            # Accept with probability proportional to d^{-alpha}
            # Normalize: max accept prob = 1 at d=1 (smallest non-zero d=1)
            p_accept = d ** (-alpha)
            if rng.random() < p_accept:
                v = idx_of(i + di, j + dj, k + dk, L)
                add_edge(adj, u, v)
                added += 1
    return adj


def build_fixedR(L, R=10.0, mu=1.0, seed=42):
    """(FixedR) Shortcut to random lattice point at approx radius R."""
    rng = np.random.default_rng(seed)
    adj = build_cubic_adj(L)
    N = L**3
    # Build set of offsets at Euclidean distance approx R
    R2_low = (R - 0.5) ** 2
    R2_high = (R + 0.5) ** 2
    valid_offsets = []
    rmax = int(R + 1)
    for di in range(-rmax, rmax + 1):
        for dj in range(-rmax, rmax + 1):
            for dk in range(-rmax, rmax + 1):
                d2 = di*di + dj*dj + dk*dk
                if R2_low <= d2 <= R2_high:
                    valid_offsets.append((di, dj, dk))
    if not valid_offsets:
        return adj
    valid_offsets = np.array(valid_offsets)
    n_valid = len(valid_offsets)
    for u in range(N):
        k_short = rng.poisson(mu)
        i, j, k = coords_of(u, L)
        for _ in range(k_short):
            choice = int(rng.integers(0, n_valid))
            di, dj, dk = valid_offsets[choice]
            v = idx_of(i + di, j + dj, k + dk, L)
            add_edge(adj, u, v)
    return adj


def build_poisson_diag(L, mu=1.0, seed=42):
    """Reference: Poisson(mu) body-diagonal shortcuts (paper's setup)."""
    rng = np.random.default_rng(seed)
    adj = build_cubic_adj(L)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                u = idx_of(i, j, k, L)
                k_short = rng.poisson(mu)
                for _ in range(k_short):
                    dvec = DIRS4[rng.integers(0, 4)]
                    sign = 1 if rng.random() < 0.5 else -1
                    dvec = tuple(sign * d for d in dvec)
                    di, dj, dk = dvec
                    v = idx_of(i + di, j + dj, k + dk, L)
                    add_edge(adj, u, v)
    return adj


def bfs_dim(adj, source, r_min=2, r_frac_max=0.4):
    N = len(adj)
    dist = [-1] * N
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    max_d = max(dist)
    N_at = np.zeros(max_d + 1, dtype=int)
    for d in dist:
        if d >= 0:
            N_at[d] += 1
    cumul = np.cumsum(N_at)
    rs = np.arange(len(cumul))
    r_max = int(r_frac_max * len(cumul))
    mask = (rs >= r_min) & (rs <= r_max) & (cumul > 0)
    log_r = np.log(rs[mask])
    log_N = np.log(cumul[mask])
    d_H, _ = np.polyfit(log_r, log_N, 1)
    return float(d_H)


def averaged_dim(adj, n_sources=4, seed=0):
    rng = np.random.default_rng(seed)
    Ntot = len(adj)
    ds = [bfs_dim(adj, int(rng.integers(0, Ntot))) for _ in range(n_sources)]
    return float(np.mean(ds)), float(np.std(ds))


def main():
    L = 32  # smaller for speed, multiple constructions tested
    target = 3 + 1 / (2 * PI)
    target_shift = 1 / (2 * PI)
    print("L = {0}, target = 3 + 1/(2pi) = {1:.5f}".format(L, target))
    print("(target shift over cubic = {0:.5f})".format(target_shift))
    print()

    constructions = []
    constructions.append(("pure cubic (control)",
                          build_cubic_adj(L)))
    constructions.append(("Poisson(1) body-diag (paper)",
                          build_poisson_diag(L, mu=1.0, seed=42)))
    constructions.append(("(NW) Poisson(1) random global",
                          build_NW(L, mu=1.0, seed=42)))
    constructions.append(("(NW) Poisson(0.5) random global",
                          build_NW(L, mu=0.5, seed=42)))
    constructions.append(("(NW) Poisson(0.1) random global",
                          build_NW(L, mu=0.1, seed=42)))
    constructions.append(("(NW) Poisson(0.01) random global",
                          build_NW(L, mu=0.01, seed=42)))
    constructions.append(("(FixedR) R=5, mu=1",
                          build_fixedR(L, R=5.0, mu=1.0, seed=42)))
    constructions.append(("(FixedR) R=10, mu=1",
                          build_fixedR(L, R=10.0, mu=1.0, seed=42)))
    constructions.append(("(FixedR) R=15, mu=1",
                          build_fixedR(L, R=15.0, mu=1.0, seed=42)))

    d_cubic = None
    print("{0:<35} {1:>10} {2:>7} {3:>11} {4:>11}".format(
        "Construction", "d_BFS", "sigma", "shift", "shift-tgt"))
    print("-" * 78)
    rows = []
    for name, adj in constructions:
        d_mean, d_std = averaged_dim(adj, n_sources=6, seed=11)
        if d_cubic is None:
            d_cubic = d_mean
        shift = d_mean - d_cubic
        delta_shift = shift - target_shift
        marker = " ok" if (name != "pure cubic (control)"
                           and abs(delta_shift) < 0.02) else ""
        print("{0:<35} {1:>10.5f} {2:>7.4f} {3:>+11.5f} {4:>+11.5f}{5}".format(
            name, d_mean, d_std, shift, delta_shift, marker))
        rows.append({"name": name, "d_BFS": d_mean, "sigma": d_std,
                     "shift": shift, "delta_to_target_shift": delta_shift})

    print()
    print("Notes:")
    print(" - d_cubic baseline (finite-size at L={0}): {1:.5f}".format(L, d_cubic))
    print(" - Newman-Watts (NW) with mu=1 has very large d_BFS (small-world deep regime).")
    print(" - Power-law with large alpha approaches local shortcuts (low shift).")
    print(" - Look for the (mu, alpha) combination that gives shift = 1/(2pi).")

    out = {"L": L, "target": target, "target_shift": target_shift,
           "d_cubic": d_cubic, "results": rows}
    with open(DATA / "25_arbitrary_distance_shortcuts.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved -> {0}".format(DATA / "25_arbitrary_distance_shortcuts.json"))


if __name__ == "__main__":
    main()
