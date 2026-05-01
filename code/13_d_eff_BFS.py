"""
Paper VI — numerical verification: d_eff from BFS expansion.

Build a cubic lattice with L^3 nodes:
  - 6 axis-neighbour links per node (standard cubic)
  - p_diag fraction of body-diagonal links present (each cube has 4)

Compute BFS distances from a central node. Count N(r) = number of
nodes at BFS distance ≤ r. Fit N(r) ~ r^d_H over a clean window.

Goal: verify that d_H = 3 + 1/(2π) ≈ 3.159 emerges for some natural
choice of body-diagonal density. Compare with the candidate values:
  p = 0:   pure cubic, d_H = 3
  p = 1/4: one diagonal direction per cube, d_H ≈ ?
  p = 1:   all 4 diagonals per cube, d_H ≈ ?
"""
import numpy as np
import json
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi


def build_lattice(L, p_diag, seed=42):
    """Build cubic lattice + body-diagonal shortcuts.
    
    L: side length
    p_diag: fraction of body diagonals to include (0 to 1).
       Diagonal links go between corners (i,j,k) and (i+1,j+1,k+1).
       (Other 3 directions: (i+1,j-1,k+1), etc.) — we sample from all 4.
    """
    rng = np.random.default_rng(seed)
    N = L**3
    
    def idx(i, j, k):
        return ((i % L) * L + (j % L)) * L + (k % L)
    
    adj = [[] for _ in range(N)]
    
    # Axis-aligned edges (6 per node)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                u = idx(i, j, k)
                for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1)]:
                    v = idx(i+di, j+dj, k+dk)
                    adj[u].append(v); adj[v].append(u)
    
    # Body diagonals (4 per cube): (1,1,1), (1,1,-1), (1,-1,1), (-1,1,1)
    diag_dirs = [(1,1,1), (1,1,-1), (1,-1,1), (-1,1,1)]
    for i in range(L):
        for j in range(L):
            for k in range(L):
                u = idx(i, j, k)
                for di, dj, dk in diag_dirs:
                    if rng.random() < p_diag:
                        v = idx(i+di, j+dj, k+dk)
                        adj[u].append(v); adj[v].append(u)
    
    return adj


def bfs_expansion(adj, source, max_r=None):
    """Return array N where N[r] = number of nodes at BFS distance ≤ r."""
    N_total = len(adj)
    dist = [-1] * N_total
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        if max_r is not None and dist[u] >= max_r:
            continue
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    
    max_d = max(dist)
    N_at_r = np.zeros(max_d + 1, dtype=int)
    for d in dist:
        if d >= 0:
            N_at_r[d] += 1
    cumul = np.cumsum(N_at_r)
    return cumul  # cumul[r] = number of nodes at distance ≤ r


def fit_d_H(N_cumul, r_min=2, r_max=None):
    """Fit log N ~ d_H · log r over [r_min, r_max]."""
    rs = np.arange(len(N_cumul))
    if r_max is None:
        r_max = min(len(N_cumul) - 1, int(0.4 * len(N_cumul)))
    mask = (rs >= r_min) & (rs <= r_max) & (N_cumul > 0)
    log_r = np.log(rs[mask])
    log_N = np.log(N_cumul[mask])
    d_H, intercept = np.polyfit(log_r, log_N, 1)
    return float(d_H), float(intercept), int(rs[mask].min()), int(rs[mask].max())


def main():
    L = 60
    N = L**3
    print(f"Cubic lattice L = {L}, total {N} nodes")
    print()
    
    target_d_eff = 3 + 1/(2*PI)
    print(f"Target: d_eff = 3 + 1/(2π) = {target_d_eff:.6f}")
    print()
    
    # Test various shortcut densities
    p_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]
    
    print(f"{'p_diag':>8} {'d_H_fit':>10} {'fit window':>15} {'Δ from d_eff':>14}")
    rows = []
    for p in p_values:
        adj = build_lattice(L, p)
        # BFS from center
        center = ((L//2)*L + L//2)*L + L//2
        cumul = bfs_expansion(adj, center)
        d_H, intercept, r_lo, r_hi = fit_d_H(cumul)
        delta = d_H - target_d_eff
        print(f"{p:8.3f} {d_H:10.4f} [{r_lo}, {r_hi}]{'':>3} {delta:+13.4f}")
        rows.append({"p_diag": p, "d_H": d_H,
                     "r_min": r_lo, "r_max": r_hi,
                     "delta_from_d_eff": delta})
    
    print()
    
    # Find the p that gives d_H = target via interpolation
    p_arr = np.array([r["p_diag"] for r in rows])
    d_arr = np.array([r["d_H"] for r in rows])
    p_target = np.interp(target_d_eff, d_arr, p_arr)
    print(f"Interpolated p_diag for d_H = {target_d_eff:.4f}: p* = {p_target:.4f}")
    print()
    
    # Specific structural candidates
    print("Structural candidates:")
    candidates = [(1.0, "all 4 body diagonals"),
                  (0.25, "1 of 4 diagonal directions"),
                  (1/(2*PI), "λ = 1/(2π) (twist density)"),
                  (1/(4*PI), "1/(4π)"),
                  (p_target, "interpolated for d_H = 3 + 1/(2π)")]
    for p, label in candidates:
        if 0 <= p <= 1:
            adj = build_lattice(L, p)
            center = ((L//2)*L + L//2)*L + L//2
            cumul = bfs_expansion(adj, center)
            d_H, _, r_lo, r_hi = fit_d_H(cumul)
            print(f"  p = {p:.4f} ({label}): d_H = {d_H:.4f}")
    
    out = {"L": L, "N": N, "target_d_eff": target_d_eff,
           "results": rows, "p_target_for_d_eff": p_target}
    with open(DATA / "13_d_eff_BFS.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
