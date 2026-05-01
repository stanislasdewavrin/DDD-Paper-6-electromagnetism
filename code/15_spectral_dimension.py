"""
Paper VI — spectral dimension d_s of the diagonal-twist substrate.

The spectral dimension of a graph is defined via random walk return
probability:
    p_return(t) ~ t^(-d_s/2)

This dimension governs how the Coulomb-like Green function scales,
which is what enters the Wilson loop self-energy. NOT the BFS
expansion dimension.

Test: for our diagonal-twist substrate, does d_s = 3 + 1/(2π)?
"""
import numpy as np
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi


def build_lattice(L, mode):
    """Build cubic lattice + diagonal links."""
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
    
    if mode == "all4":
        diag_list = [(1,1,1), (1,1,-1), (1,-1,1), (-1,1,1)]
    elif mode == "one_111":
        diag_list = [(1,1,1)]
    elif mode == "pair_z":
        diag_list = [(1,1,1), (1,-1,1)]  # both +z direction
    elif mode == "cubic_only":
        diag_list = []
    else:
        raise ValueError(mode)
    
    for i in range(L):
        for j in range(L):
            for k in range(L):
                u = idx(i, j, k)
                for di, dj, dk in diag_list:
                    v = idx(i+di, j+dj, k+dk)
                    adj[u].add(v); adj[v].add(u)
    
    return [list(s) for s in adj]


def random_walk_return(adj, source, n_steps, n_walks, rng):
    """Estimate return probability p(t) for t up to n_steps."""
    N = len(adj)
    return_count = np.zeros(n_steps + 1)
    for w in range(n_walks):
        u = source
        return_count[0] += 1.0  # by definition
        for t in range(1, n_steps + 1):
            neighbors = adj[u]
            u = neighbors[rng.integers(len(neighbors))]
            if u == source:
                return_count[t] += 1.0
    return return_count / n_walks


def main():
    L = 50
    target = 3 + 1/(2*PI)
    print(f"L = {L}, target spectral dim d_s = 3 + 1/(2π) = {target:.6f}")
    print()
    
    modes = [
        ("cubic_only", "Pure cubic"),
        ("one_111", "Single direction (1,1,1)"),
        ("pair_z", "Both +z diagonals"),
        ("all4", "All 4 diagonals"),
    ]
    
    rng = np.random.default_rng(42)
    n_steps = 200
    n_walks = 5000
    
    print(f"{'Mode':<35} {'d_s_fit':>10} {'fit window':>15}")
    rows = []
    for mode, label in modes:
        adj = build_lattice(L, mode)
        center = ((L//2)*L + L//2)*L + L//2
        p_ret = random_walk_return(adj, center, n_steps, n_walks, rng)
        # p_return(t) ~ t^(-d_s/2). Fit on log scale.
        ts = np.arange(1, n_steps + 1)
        ps = p_ret[1:]  # skip t=0
        # Use only times where p > 0 and t in middle range (avoid finite-size)
        # Take t in [10, 100] for clean power law
        t_lo, t_hi = 8, 80
        mask = (ts >= t_lo) & (ts <= t_hi) & (ps > 1e-6)
        if mask.sum() < 5:
            print(f"{label:<35} {'(insufficient data)':>10}")
            continue
        log_t = np.log(ts[mask])
        log_p = np.log(ps[mask])
        slope, _ = np.polyfit(log_t, log_p, 1)
        d_s = -2 * slope
        print(f"{label:<35} {d_s:>10.4f} [{t_lo}, {t_hi}]")
        rows.append({"mode": mode, "label": label, "d_s": d_s})
    
    print()
    print(f"Target d_eff = {target:.6f}")
    
    out = {"L": L, "n_walks": n_walks, "n_steps": n_steps,
           "target_d_eff": target, "results": rows}
    with open(DATA / "15_spectral_dimension.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
