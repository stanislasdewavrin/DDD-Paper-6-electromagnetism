"""
Paper VI — d_BFS as a function of shortcut radius R.

Each node gets ONE random shortcut to a node within Euclidean radius R
(but not the immediate axis neighbors).

Goal: find R* such that d_BFS = 3 + 1/(2π) ≈ 3.159, and check if R*
is structurally meaningful.
"""
import numpy as np
import json
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi


def build_lattice_radius(L, R, p_link=1.0, seed=42):
    """Build cubic lattice + ONE random shortcut per node within radius R.
    
    p_link: probability a node gets a shortcut (default 1.0 = all nodes).
    """
    rng = np.random.default_rng(seed)
    N = L**3
    
    def idx(i, j, k):
        return ((i % L) * L + (j % L)) * L + (k % L)
    
    adj = [set() for _ in range(N)]
    
    # Axis edges
    for i in range(L):
        for j in range(L):
            for k in range(L):
                u = idx(i, j, k)
                for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1)]:
                    v = idx(i+di, j+dj, k+dk)
                    adj[u].add(v); adj[v].add(u)
    
    # Generate offsets within radius R, excluding axis neighbors
    offsets = []
    R_int = int(np.ceil(R))
    for di in range(-R_int, R_int+1):
        for dj in range(-R_int, R_int+1):
            for dk in range(-R_int, R_int+1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                d = np.sqrt(di**2 + dj**2 + dk**2)
                if d > R:
                    continue
                # Exclude axis neighbors (length 1)
                if d <= 1.001:
                    continue
                offsets.append((di, dj, dk, d))
    
    # Add one shortcut per node with probability p_link
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if rng.random() > p_link:
                    continue
                u = idx(i, j, k)
                # Pick a random offset
                if not offsets: continue
                di, dj, dk, d = offsets[rng.integers(len(offsets))]
                v = idx(i+di, j+dj, k+dk)
                adj[u].add(v); adj[v].add(u)
    
    return [list(s) for s in adj]


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


def main():
    L = 50
    target = 3 + 1/(2*PI)
    print(f"L = {L}, target d_BFS = 3 + 1/(2π) = {target:.6f}")
    print()
    
    # Test at fixed p = 1 (all nodes get a shortcut)
    print(f"{'R':>6} {'d_BFS':>10} {'Δ target':>12}")
    for R in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0]:
        adj = build_lattice_radius(L, R, p_link=1.0)
        center = ((L//2)*L + L//2)*L + L//2
        d_H = bfs_dim(adj, center)
        delta = d_H - target
        print(f"{R:6.2f} {d_H:10.4f} {delta:+12.4f}")
    
    print()
    print("--- Now vary p_link at fixed R = 1.8 (around body-diagonal length) ---")
    print(f"{'p_link':>8} {'d_BFS':>10} {'Δ target':>12}")
    for p in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        adj = build_lattice_radius(L, 1.8, p_link=p)
        center = ((L//2)*L + L//2)*L + L//2
        d_H = bfs_dim(adj, center)
        delta = d_H - target
        print(f"{p:8.3f} {d_H:10.4f} {delta:+12.4f}")
    
    print()
    print("--- Special: R = √3 + ε (just body-diagonal-like) ---")
    for R in [np.sqrt(3) - 0.01, np.sqrt(3) + 0.01, np.sqrt(2) + 0.01, np.sqrt(5) + 0.01]:
        adj = build_lattice_radius(L, R, p_link=1.0)
        center = ((L//2)*L + L//2)*L + L//2
        d_H = bfs_dim(adj, center)
        print(f"R = {R:.4f} (exclus < 1, includes √3): d_BFS = {d_H:.4f}")

    print()
    print(f"CODATA target d_eff = {target:.6f}")


if __name__ == "__main__":
    main()
