"""
Paper VI — variable number of shortcuts per node.

Each node gets a variable number of shortcuts, drawn from a
distribution. Goal: find which distribution gives d_BFS = 3 + 1/(2π).

Parameters:
  - mean number of shortcuts per node μ
  - distribution shape (Poisson, fixed, uniform, etc.)
  - shell selection (which shells eligible)
"""
import numpy as np
import json
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi


def get_shell_offsets(shell_n, max_dist=10):
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


def build_lattice_variable(L, mu, dist_type="poisson", shells=[3], seed=42):
    """Build cubic lattice with variable # shortcuts per node.
    
    mu: mean shortcuts per node
    dist_type: 'poisson', 'fixed', 'uniform_0_2mu', 'binomial'
    shells: which shells eligible (default body diagonal only)
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
    
    # Collect all eligible offsets
    eligible_offsets = []
    for shell_n in shells:
        offs, _ = get_shell_offsets(shell_n)
        eligible_offsets.extend(offs)
    n_eligible = len(eligible_offsets)
    
    # Add shortcuts per node
    for i in range(L):
        for j in range(L):
            for k in range(L):
                u = idx(i, j, k)
                # Sample number of shortcuts
                if dist_type == "poisson":
                    n_short = rng.poisson(mu)
                elif dist_type == "fixed":
                    n_short = int(round(mu))
                elif dist_type == "uniform_0_2mu":
                    n_short = rng.integers(0, int(2*mu) + 1)
                elif dist_type == "binomial":
                    n_short = rng.binomial(n_eligible, mu/n_eligible) if n_eligible else 0
                else:
                    raise ValueError(dist_type)
                
                # Cap at n_eligible
                n_short = min(n_short, n_eligible)
                
                # Pick n_short random offsets without replacement
                if n_short > 0:
                    chosen = rng.choice(n_eligible, size=n_short, replace=False)
                    for ci in chosen:
                        di, dj, dk = eligible_offsets[ci]
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
    print(f"L = {L}, target d_eff = 3 + 1/(2π) = {target:.6f}")
    print()
    
    # Test 1: Poisson distribution, body diagonals only
    print("=== Test 1: Poisson(μ) shortcuts per node, body diagonals only ===")
    print(f"{'μ':>6} {'d_BFS':>10} {'Δ target':>12}")
    for mu in [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
        adj = build_lattice_variable(L, mu, dist_type="poisson", shells=[3])
        center = ((L//2)*L + L//2)*L + L//2
        d_H = bfs_dim(adj, center)
        delta = d_H - target
        marker = " ✓" if abs(delta) < 0.02 else ""
        print(f"{mu:>6.2f} {d_H:>10.4f} {delta:+12.4f}{marker}")
    
    print()
    print("=== Test 2: Fixed N, body diagonals only ===")
    print(f"{'N':>6} {'d_BFS':>10} {'Δ target':>12}")
    for N in [0, 1, 2, 3, 4]:
        adj = build_lattice_variable(L, N, dist_type="fixed", shells=[3])
        center = ((L//2)*L + L//2)*L + L//2
        d_H = bfs_dim(adj, center)
        delta = d_H - target
        marker = " ✓" if abs(delta) < 0.02 else ""
        print(f"{N:>6} {d_H:>10.4f} {delta:+12.4f}{marker}")
    
    print()
    print("=== Test 3: Poisson with shells {2, 3, 4} (face + body + axis-2) ===")
    print(f"{'μ':>6} {'d_BFS':>10} {'Δ target':>12}")
    for mu in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        adj = build_lattice_variable(L, mu, dist_type="poisson", shells=[2,3,4])
        center = ((L//2)*L + L//2)*L + L//2
        d_H = bfs_dim(adj, center)
        delta = d_H - target
        marker = " ✓" if abs(delta) < 0.02 else ""
        print(f"{mu:>6.2f} {d_H:>10.4f} {delta:+12.4f}{marker}")
    
    print()
    print("=== Test 4: Uniform [0, 2μ] count ===")
    print(f"{'μ':>6} {'d_BFS':>10} {'Δ target':>12}")
    for mu in [0.5, 1.0, 1.5, 2.0]:
        adj = build_lattice_variable(L, mu, dist_type="uniform_0_2mu", shells=[3])
        center = ((L//2)*L + L//2)*L + L//2
        d_H = bfs_dim(adj, center)
        delta = d_H - target
        marker = " ✓" if abs(delta) < 0.02 else ""
        print(f"{mu:>6.2f} {d_H:>10.4f} {delta:+12.4f}{marker}")
    
    print()
    print("=== Special: μ = 1/(2π) (twist density-like) ===")
    mu_special = 1/(2*PI)
    for shells_set, label in [([3], "body diag"), ([2,3], "face+body"), ([3,4], "body+axis2")]:
        adj = build_lattice_variable(L, mu_special, dist_type="poisson", shells=shells_set)
        center = ((L//2)*L + L//2)*L + L//2
        d_H = bfs_dim(adj, center)
        delta = d_H - target
        print(f"  μ = 1/(2π) = {mu_special:.4f}, shells={shells_set} ({label}): d_BFS = {d_H:.4f}, Δ = {delta:+.4f}")


if __name__ == "__main__":
    main()
