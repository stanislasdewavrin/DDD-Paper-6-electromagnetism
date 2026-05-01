"""
Paper VI — discrete shell-based shortcuts.

Each cube has shortcuts to specific n-th nearest neighbor SHELLS (not
random radius). For each n, the shortcut is to ALL points in shell n
(or one chosen direction).

Goal: identify a discrete shell topology that gives d_BFS = 3 + 1/(2π).

Specific tests:
  - N points in BFS ball: count exactly
  - d_BFS extracted from log fit
"""
import numpy as np
import json
from collections import deque
from pathlib import Path
from itertools import product

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi


def get_shell_offsets(shell_n, max_dist=10):
    """Return offsets at the shell_n-th Euclidean distance shell.
    Excludes (0,0,0)."""
    offsets_by_dist = {}
    for di in range(-max_dist, max_dist+1):
        for dj in range(-max_dist, max_dist+1):
            for dk in range(-max_dist, max_dist+1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                d2 = di*di + dj*dj + dk*dk
                if d2 > max_dist**2:
                    continue
                offsets_by_dist.setdefault(d2, []).append((di, dj, dk))
    distances = sorted(offsets_by_dist.keys())
    if shell_n - 1 >= len(distances):
        return [], 0
    d2 = distances[shell_n - 1]
    return offsets_by_dist[d2], np.sqrt(d2)


def build_lattice_shell(L, shells, p_link_per_shell=1.0, seed=42):
    """Build cubic lattice + shortcuts to specified shells."""
    rng = np.random.default_rng(seed)
    N = L**3
    
    def idx(i, j, k):
        return ((i % L) * L + (j % L)) * L + (k % L)
    
    adj = [set() for _ in range(N)]
    
    # Axis edges (shell 1)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                u = idx(i, j, k)
                for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1)]:
                    v = idx(i+di, j+dj, k+dk)
                    adj[u].add(v); adj[v].add(u)
    
    # Add shortcuts at specified shells
    for shell_n in shells:
        offsets, dist = get_shell_offsets(shell_n)
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    u = idx(i, j, k)
                    for di, dj, dk in offsets:
                        if rng.random() < p_link_per_shell:
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
    return float(d_H), cumul


def main():
    L = 50
    target = 3 + 1/(2*PI)
    print(f"L = {L}, target d_eff = {target:.6f}")
    print()
    
    # First, characterize each shell
    print("Shell characterisation:")
    for n in range(1, 9):
        offsets, dist = get_shell_offsets(n)
        print(f"  Shell {n}: distance = {dist:.4f}, count = {len(offsets)}")
    print()
    
    # Test shortcuts at each shell (full inclusion)
    print(f"{'Shells':<25} {'p_link':>8} {'d_BFS':>10} {'Δ target':>12}")
    rows = []
    
    # No shortcuts - pure cubic
    adj = build_lattice_shell(L, [], p_link_per_shell=0.0)
    center = ((L//2)*L + L//2)*L + L//2
    d_H, _ = bfs_dim(adj, center)
    print(f"{'(none)':<25} {0.0:>8.3f} {d_H:>10.4f} {d_H-target:+12.4f}")
    
    # Single shells, all included
    for shell_n in [2, 3, 4, 5, 6]:
        adj = build_lattice_shell(L, [shell_n], p_link_per_shell=1.0)
        d_H, _ = bfs_dim(adj, center)
        offsets, dist = get_shell_offsets(shell_n)
        label = f"shell {shell_n} (d={dist:.2f}, n={len(offsets)})"
        print(f"{label:<25} {1.0:>8.3f} {d_H:>10.4f} {d_H-target:+12.4f}")
        rows.append({"shells": [shell_n], "p_link": 1.0, "d_H": d_H,
                     "delta_target": d_H - target})
    
    print()
    # Combined shells (small fraction at higher shells = "small-world")
    print("--- Combined shell topologies ---")
    test_combos = [
        ([3], 1.0, "body diag (all)"),
        ([2, 3], 1.0, "face + body diag (all)"),
        ([3, 4], 1.0, "body + axis-2"),
        ([3], 0.5, "body diag (half)"),
        ([3, 6], 1.0, "body + 2-2-1"),
        ([2, 3, 4], 1.0, "face + body + axis-2"),
    ]
    print(f"{'Combo':<35} {'p_link':>8} {'d_BFS':>10} {'Δ target':>12}")
    for shells, p, label in test_combos:
        adj = build_lattice_shell(L, shells, p_link_per_shell=p)
        d_H, _ = bfs_dim(adj, center)
        delta = d_H - target
        print(f"{label:<35} {p:>8.3f} {d_H:>10.4f} {delta:+12.4f}")
        rows.append({"label": label, "shells": shells, "p_link": p,
                     "d_H": d_H, "delta_target": delta})
    
    print()
    # Find p that gives target with body diagonals only
    print("--- Find p for body diagonals to give d_eff = 3 + 1/(2π) ---")
    print(f"{'p':>8} {'d_BFS':>10} {'Δ target':>12}")
    p_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    for p in p_values:
        adj = build_lattice_shell(L, [3], p_link_per_shell=p)
        d_H, _ = bfs_dim(adj, center)
        delta = d_H - target
        marker = " ✓" if abs(delta) < 0.02 else ""
        print(f"{p:>8.3f} {d_H:>10.4f} {delta:+12.4f}{marker}")

    out = {"L": L, "target": target, "results": rows}
    with open(DATA / "19_discrete_shells.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
