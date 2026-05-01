"""
Paper VI — test specific Weyl topologies for d_eff = 3 + 1/(2π).

Different ways to add diagonal links systematically (deterministic):
  - "all4": all 4 body diagonals per cube (saturated)
  - "one_dir_111": only (1,1,1) direction
  - "pair_chiral": (1,1,1) and (1,1,-1) — opposite z-chirality
  - "single_z_axion": only diagonals with positive z-component
  - "every_other_cube_111": (1,1,1) on alternate cubes (checkerboard)
  - "single_axis_aligned_to_z": only z-tilted diagonals
  
Goal: identify a STRUCTURAL topology (deterministic, not random density)
that gives d_H ≈ 3 + 1/(2π) ≈ 3.159.
"""
import numpy as np
import json
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi


def build_lattice(L, mode):
    """Build cubic lattice with axis edges + various diagonal selection modes."""
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
    
    # All 4 body diagonal directions
    all4 = [(1,1,1), (1,1,-1), (1,-1,1), (-1,1,1)]
    
    if mode == "cubic_only":
        diag_list = []
    elif mode == "all4":
        diag_list = all4
    elif mode == "one_dir_111":
        diag_list = [(1,1,1)]
    elif mode == "pair_chiral":
        diag_list = [(1,1,1), (1,1,-1)]
    elif mode == "two_z_positive":
        diag_list = [(1,1,1), (1,-1,1)]   # both have +z, different x or y
    elif mode == "every_other_111":
        diag_list = [(1,1,1)]  # but only on cells with (i+j+k) even
    elif mode == "single_axis_z":
        diag_list = [(1,1,1), (-1,1,1)]   # both +z, +y; differ in x
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if mode == "every_other_111":
                    if (i + j + k) % 2 != 0:
                        continue
                u = idx(i, j, k)
                for di, dj, dk in diag_list:
                    v = idx(i+di, j+dj, k+dk)
                    adj[u].add(v); adj[v].add(u)
    
    # Convert to lists
    return [list(s) for s in adj]


def bfs_expansion(adj, source):
    N_total = len(adj)
    dist = [-1] * N_total
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    max_d = max(dist)
    N_at_r = np.zeros(max_d + 1, dtype=int)
    for d in dist:
        if d >= 0:
            N_at_r[d] += 1
    return np.cumsum(N_at_r)


def fit_d_H(N_cumul, r_min=2, r_frac_max=0.4):
    rs = np.arange(len(N_cumul))
    r_max = int(r_frac_max * len(N_cumul))
    mask = (rs >= r_min) & (rs <= r_max) & (N_cumul > 0)
    log_r = np.log(rs[mask])
    log_N = np.log(N_cumul[mask])
    d_H, _ = np.polyfit(log_r, log_N, 1)
    return float(d_H), int(rs[mask].min()), int(rs[mask].max())


def main():
    L = 60
    target = 3 + 1/(2*PI)
    print(f"L = {L}, target d_eff = 3 + 1/(2π) = {target:.6f}")
    print()
    
    modes = [
        ("cubic_only", "Pure cubic (no diagonal)", 0),
        ("one_dir_111", "ONE direction (1,1,1)", 1),
        ("pair_chiral", "PAIR (1,1,1) and (1,1,-1)", 2),
        ("two_z_positive", "Both +z diagonals (1,1,1) and (1,-1,1)", 2),
        ("single_axis_z", "Both +z+y diagonals (1,1,1) and (-1,1,1)", 2),
        ("every_other_111", "(1,1,1) on alternate cubes (checkerboard)", 0.5),
        ("all4", "ALL 4 diagonals", 4),
    ]
    
    print(f"{'Mode':<48} {'count':>6} {'d_H':>8} {'Δ d_eff':>10}")
    print("-" * 80)
    rows = []
    for mode, label, count in modes:
        adj = build_lattice(L, mode)
        center = ((L//2)*L + L//2)*L + L//2
        cumul = bfs_expansion(adj, center)
        d_H, r_lo, r_hi = fit_d_H(cumul)
        delta = d_H - target
        marker = "  ✓" if abs(delta) < 0.02 else ""
        print(f"{label:<48} {count:>6.1f} {d_H:>8.4f} {delta:+10.4f}{marker}")
        rows.append({"mode": mode, "label": label, "diag_per_cube": count,
                     "d_H": d_H, "delta_from_target": delta})
    
    print()
    print(f"Target: d_eff = 3 + 1/(2π) = {target:.6f}")
    
    out = {"L": L, "target_d_eff": target, "results": rows}
    with open(DATA / "14_d_eff_Weyl_topologies.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
