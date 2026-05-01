"""
Test of the SPECTRAL dimension d_s vs the BFS expansion dimension.

In a small-world graph, these can differ. The spectral dimension is
defined by the random walk return probability:
    P_t(0,0) ~ t^(-d_s/2)  for large t.

If d_s != d_BFS, we may resolve the alpha_EM mismatch.
"""
import numpy as np
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

DIRS4 = np.array([
    [+1, +1, +1],
    [+1, +1, -1],
    [+1, -1, +1],
    [-1, +1, +1],
], dtype=np.int32)


def idx_of(i, j, k, L):
    return ((i % L) * L + (j % L)) * L + (k % L)


def random_walk_return(L, seed=42, n_walks=10000, max_steps=200):
    """Run many random walks from origin, count return rate at each tick."""
    rng = np.random.default_rng(seed)
    N = L**3
    src = idx_of(L // 2, L // 2, L // 2, L)
    theta0 = rng.integers(0, 4, size=N).astype(np.int8)

    # Track position evolution of each walker
    # Position stored as flat index
    positions = np.full(n_walks, src, dtype=np.int64)
    return_count = np.zeros(max_steps + 1, dtype=np.float64)
    return_count[0] = n_walks  # initially all at source

    cubic_dirs = np.array([(1, 0, 0), (-1, 0, 0),
                            (0, 1, 0), (0, -1, 0),
                            (0, 0, 1), (0, 0, -1)], dtype=np.int32)

    for t in range(1, max_steps + 1):
        # Each walker moves
        # Walker at position p has N+1 = 6 cubic + 4 diag (with phase) = 7 active edges
        # We choose uniformly among active edges
        new_positions = np.zeros(n_walks, dtype=np.int64)
        for w in range(n_walks):
            p = positions[w]
            i = p // (L * L)
            j = (p // L) % L
            k = p % L
            phi = (theta0[p] + t) % 4  # phase at this node at this tick
            # Possible moves: 6 cubic + 2 diag (line_idx == phi, both signs)
            # Pick uniformly among 8
            choice = rng.integers(0, 8)
            if choice < 6:
                di, dj, dk = cubic_dirs[choice]
            else:
                d = DIRS4[phi]
                sign = 1 if choice == 6 else -1
                di, dj, dk = sign * d[0], sign * d[1], sign * d[2]
            new_p = idx_of(i + di, j + dj, k + dk, L)
            new_positions[w] = new_p
        positions = new_positions
        # Count returns to origin
        return_count[t] = np.sum(positions == src)
    return_prob = return_count / n_walks
    return return_prob


def fit_d_spectral(return_prob, t_min=10, t_frac_max=0.6):
    """Fit P_t(0,0) ~ t^(-d_s/2) -> log P = -d_s/2 * log t + c."""
    ts = np.arange(len(return_prob))
    t_max = int(t_frac_max * len(return_prob))
    mask = (ts >= t_min) & (ts <= t_max) & (return_prob > 0)
    if mask.sum() < 5:
        return float('nan')
    log_t = np.log(ts[mask])
    log_P = np.log(return_prob[mask])
    slope, intercept = np.polyfit(log_t, log_P, 1)
    d_s = -2 * slope
    return float(d_s)


def main():
    L = 50
    print(f"L = {L}")
    print()

    print("Random walk return probabilities:")
    print(f"{'seed':<6} {'n_walks':<10} {'max_steps':<12} {'d_s':<10}")
    print("-" * 40)
    rows = []
    for seed in range(3):
        n_walks = 5000
        max_steps = 60
        t0 = time.time()
        return_prob = random_walk_return(L, seed=seed, n_walks=n_walks, max_steps=max_steps)
        d_s = fit_d_spectral(return_prob, t_min=8, t_frac_max=0.6)
        elapsed = time.time() - t0
        print(f"{seed:<6} {n_walks:<10} {max_steps:<12} {d_s:<10.4f}  ({elapsed:.1f}s)")
        rows.append({"seed": seed, "d_s": d_s,
                     "return_prob_t1": float(return_prob[1]),
                     "return_prob_t10": float(return_prob[10]),
                     "time_s": elapsed})

    d_ss = [r["d_s"] for r in rows]
    print()
    print(f"Mean d_s = {np.mean(d_ss):.5f} ± {np.std(d_ss)/np.sqrt(len(rows)):.5f}")
    print()
    print("Comparison:")
    print(f"  d_BFS  (L=400) = 3.106 ± 0.005")
    print(f"  d_alpha (target) = 3.131")
    print(f"  d_s    (this test) = {np.mean(d_ss):.4f}")

    with open(DATA / "39_spectral_dimension.json", "w") as f:
        json.dump({"L": L, "rows": rows, "d_s_mean": float(np.mean(d_ss))}, f, indent=2)


if __name__ == "__main__":
    main()
