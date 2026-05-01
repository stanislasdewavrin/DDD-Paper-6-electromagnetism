#!/usr/bin/env python3
"""
2-loop coefficient extraction for the body-diagonal Wilson loop
in compact U(1) lattice gauge theory with diagonal twist.

Setup
-----
- 3D cubic lattice of size L^3 with periodic boundaries
- 3 cubic link variables per site: psi_x, psi_y, psi_z (each in [0, 2*pi))
- 4 body-diagonal link variables per site (lines along (+,+,+), (+,+,-), (+,-,+), (-,+,+))
- Action: S = beta * sum_p [1 - cos(Phi_p)]
  where p runs over all plaquettes (cubic + body-diagonal)
- Phi_p is the gauge-invariant phase around plaquette p

Body-diagonal Wilson loop (the central object of paper VI):
  W_diag(C) = closed loop through (0,0,0) -> (1,0,0) -> (1,1,0) -> (1,1,1) -> (0,0,0)
  i.e., 3 cubic steps + 1 body-diagonal step (tetrahedral cycle)

Procedure
---------
1. Initialise random link phases
2. Heat-bath thermalisation (n_therm sweeps)
3. Measure <W_diag> at various beta = 1/alpha
4. Fit polynomial: 1 - <W_diag> = a_1 * alpha + a_2 * alpha^2 + ...
5. Extract c_2 = 2 * a_2 / a_1^2 (relative to abelian-exp prediction c_2 = 1)

Configurable parameters
----------------------
L : lattice size (8 to 16 reasonable for laptop)
n_therm : thermalisation sweeps (1000-5000)
n_meas : measurement sweeps (5000-20000)
betas : list of inverse couplings to scan
seed : random seed

Estimated runtime
-----------------
L=12, n_therm=2000, n_meas=10000, 7 betas : ~2-4 hours on a single CPU
For faster preliminary results: L=8, n_therm=500, n_meas=2000, 5 betas : ~10-20 min

Usage
-----
$ python 43_2loop_extraction_local.py
   (uses default fast settings; takes ~15 min)

To customize:
$ python 43_2loop_extraction_local.py --L 16 --n_therm 3000 --n_meas 15000

Output
------
Saved JSON file with per-beta <W>, statistical errors, and c_2 fit result.
"""

import numpy as np
import json
import time
import argparse
from pathlib import Path


# Body-diagonal directions (4 lines, sign convention: positive endpoint)
DIAG_DIRS = np.array([
    [+1, +1, +1],
    [+1, +1, -1],
    [+1, -1, +1],
    [-1, +1, +1],
], dtype=np.int32)


def initialize_links(L, rng):
    """
    Initialize link phases on the lattice.

    Returns
    -------
    psi_cubic : array of shape (L, L, L, 3)
        Cubic link phases (3 directions per site: x, y, z)
    psi_diag : array of shape (L, L, L, 4)
        Body-diagonal link phases (4 directions per site)
    """
    psi_cubic = rng.uniform(0, 2*np.pi, size=(L, L, L, 3))
    psi_diag = rng.uniform(0, 2*np.pi, size=(L, L, L, 4))
    return psi_cubic, psi_diag


def cubic_plaquette_phase(psi_cubic, i, j, k, mu, nu, L):
    """
    Compute gauge-invariant phase around a cubic (mu,nu) plaquette at site (i,j,k).

    Plaquette: link[i,j,k,mu] + link[i+e_mu, j+e_mu, k+e_mu, nu]
              - link[i+e_nu, j+e_nu, k+e_nu, mu] - link[i,j,k,nu]
    """
    # Define cubic shift: e_x = (1,0,0), e_y = (0,1,0), e_z = (0,0,1)
    if mu == 0: i_mu, j_mu, k_mu = (i+1)%L, j, k
    elif mu == 1: i_mu, j_mu, k_mu = i, (j+1)%L, k
    else: i_mu, j_mu, k_mu = i, j, (k+1)%L
    if nu == 0: i_nu, j_nu, k_nu = (i+1)%L, j, k
    elif nu == 1: i_nu, j_nu, k_nu = i, (j+1)%L, k
    else: i_nu, j_nu, k_nu = i, j, (k+1)%L
    return (psi_cubic[i,j,k,mu]
            + psi_cubic[i_mu,j_mu,k_mu,nu]
            - psi_cubic[i_nu,j_nu,k_nu,mu]
            - psi_cubic[i,j,k,nu])


def total_action(psi_cubic, psi_diag, beta, L):
    """Compute total Wilson action."""
    S = 0.0
    # Cubic plaquettes
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for mu in range(3):
                    for nu in range(mu+1, 3):
                        phi = cubic_plaquette_phase(psi_cubic, i, j, k, mu, nu, L)
                        S += beta * (1 - np.cos(phi))
    # Body-diagonal plaquettes (tetrahedral closures)
    # For each (cubic dir, body-diag dir) pair, there's a closure
    # Simplified: for each site, each body-diagonal contributes one tetrahedral plaquette
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for d in range(4):
                    di, dj, dk = DIAG_DIRS[d]
                    # Closed cycle: (i,j,k) -> (i+di,j,k) -> (i+di,j+dj,k) -> (i+di,j+dj,k+dk)
                    #                -> diagonal back to (i,j,k)
                    # Phases:
                    s_x = +1 if di > 0 else -1
                    s_y = +1 if dj > 0 else -1
                    s_z = +1 if dk > 0 else -1
                    # Step 1: x in direction s_x
                    if s_x > 0:
                        phi1 = psi_cubic[i, j, k, 0]
                    else:
                        phi1 = -psi_cubic[(i-1)%L, j, k, 0]
                    # Step 2: y in direction s_y at (i+di,j,k)
                    iA = (i+di)%L
                    if s_y > 0:
                        phi2 = psi_cubic[iA, j, k, 1]
                    else:
                        phi2 = -psi_cubic[iA, (j-1)%L, k, 1]
                    # Step 3: z in direction s_z at (i+di,j+dj,k)
                    jB = (j+dj)%L
                    if s_z > 0:
                        phi3 = psi_cubic[iA, jB, k, 2]
                    else:
                        phi3 = -psi_cubic[iA, jB, (k-1)%L, 2]
                    # Step 4: diagonal back from (i+di,j+dj,k+dk) to (i,j,k)
                    phi4 = -psi_diag[i, j, k, d]
                    Phi = phi1 + phi2 + phi3 + phi4
                    S += beta * (1 - np.cos(Phi))
    return S


def heat_bath_sweep_simple(psi_cubic, psi_diag, beta, L, rng):
    """
    Metropolis update of all links. Simple but slow.

    Returns
    -------
    acc_rate : float
        Acceptance rate this sweep.
    """
    n_acc = 0
    n_tot = 0
    delta = 0.5  # update step size

    # Cubic links
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for mu in range(3):
                    old = psi_cubic[i,j,k,mu]
                    new = old + rng.uniform(-delta, delta)
                    # Compute action contribution from plaquettes containing this link
                    # Approximation: count only the 4 cubic plaquettes (2 in each (mu,nu) plane)
                    S_old = 0.0
                    S_new = 0.0
                    for nu in range(3):
                        if nu == mu: continue
                        # Plaquette at (i,j,k) in (mu,nu) plane
                        phi_old = cubic_plaquette_phase(psi_cubic, i, j, k, mu, nu, L)
                        S_old += 1 - np.cos(phi_old)
                        # Update temporarily and recompute
                        psi_cubic[i,j,k,mu] = new
                        phi_new = cubic_plaquette_phase(psi_cubic, i, j, k, mu, nu, L)
                        S_new += 1 - np.cos(phi_new)
                        psi_cubic[i,j,k,mu] = old
                        # Plaquette at (i-e_nu, j-e_nu, k-e_nu) in (mu,nu) plane
                        if nu == 0: i_b, j_b, k_b = (i-1)%L, j, k
                        elif nu == 1: i_b, j_b, k_b = i, (j-1)%L, k
                        else: i_b, j_b, k_b = i, j, (k-1)%L
                        phi_old_b = cubic_plaquette_phase(psi_cubic, i_b, j_b, k_b, mu, nu, L)
                        S_old += 1 - np.cos(phi_old_b)
                        psi_cubic[i,j,k,mu] = new
                        phi_new_b = cubic_plaquette_phase(psi_cubic, i_b, j_b, k_b, mu, nu, L)
                        S_new += 1 - np.cos(phi_new_b)
                        psi_cubic[i,j,k,mu] = old
                    delta_S = beta * (S_new - S_old)
                    if delta_S < 0 or rng.random() < np.exp(-delta_S):
                        psi_cubic[i,j,k,mu] = new
                        n_acc += 1
                    n_tot += 1

    # Body-diagonal links
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for d in range(4):
                    old = psi_diag[i,j,k,d]
                    new = old + rng.uniform(-delta, delta)
                    # Single tetrahedral plaquette per body-diag link
                    di, dj, dk = DIAG_DIRS[d]
                    s_x = +1 if di > 0 else -1
                    s_y = +1 if dj > 0 else -1
                    s_z = +1 if dk > 0 else -1
                    iA = (i+di)%L
                    jB = (j+dj)%L
                    if s_x > 0: phi1 = psi_cubic[i, j, k, 0]
                    else: phi1 = -psi_cubic[(i-1)%L, j, k, 0]
                    if s_y > 0: phi2 = psi_cubic[iA, j, k, 1]
                    else: phi2 = -psi_cubic[iA, (j-1)%L, k, 1]
                    if s_z > 0: phi3 = psi_cubic[iA, jB, k, 2]
                    else: phi3 = -psi_cubic[iA, jB, (k-1)%L, 2]
                    Phi_old = phi1 + phi2 + phi3 - old
                    Phi_new = phi1 + phi2 + phi3 - new
                    delta_S = beta * ((1 - np.cos(Phi_new)) - (1 - np.cos(Phi_old)))
                    if delta_S < 0 or rng.random() < np.exp(-delta_S):
                        psi_diag[i,j,k,d] = new
                        n_acc += 1
                    n_tot += 1
    return n_acc / n_tot


def measure_wilson_diag_avg(psi_cubic, psi_diag, L):
    """
    Average over all body-diagonal Wilson loops in the lattice.

    Each loop = 3 cubic steps + 1 diagonal closure.
    Returns <cos(Phi_loop)>.
    """
    W_total = 0.0
    n = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for d in range(4):
                    di, dj, dk = DIAG_DIRS[d]
                    s_x = +1 if di > 0 else -1
                    s_y = +1 if dj > 0 else -1
                    s_z = +1 if dk > 0 else -1
                    iA = (i+di)%L
                    jB = (j+dj)%L
                    if s_x > 0: phi1 = psi_cubic[i, j, k, 0]
                    else: phi1 = -psi_cubic[(i-1)%L, j, k, 0]
                    if s_y > 0: phi2 = psi_cubic[iA, j, k, 1]
                    else: phi2 = -psi_cubic[iA, (j-1)%L, k, 1]
                    if s_z > 0: phi3 = psi_cubic[iA, jB, k, 2]
                    else: phi3 = -psi_cubic[iA, jB, (k-1)%L, 2]
                    Phi = phi1 + phi2 + phi3 - psi_diag[i,j,k,d]
                    W_total += np.cos(Phi)
                    n += 1
    return W_total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--n_therm", type=int, default=500)
    parser.add_argument("--n_meas", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="2loop_local.json")
    args = parser.parse_args()

    L = args.L
    n_therm = args.n_therm
    n_meas = args.n_meas
    seed = args.seed
    rng = np.random.default_rng(seed)

    # Beta values to scan: small alpha (large beta) to slightly larger
    betas = [40.0, 30.0, 20.0, 15.0, 10.0, 7.0, 5.0]

    print(f"Compact U(1) lattice gauge theory with body-diagonal twist")
    print(f"L={L}, n_therm={n_therm}, n_meas={n_meas}, seed={seed}")
    print(f"Estimated time per beta: ~{0.001 * L**3 * (n_therm + n_meas) / 60:.1f} min")
    print(f"Total estimated: ~{0.001 * L**3 * (n_therm + n_meas) * len(betas) / 60:.1f} min")
    print()
    print(f"{'beta':<8} {'alpha':<10} {'<W_diag>':<14} {'sigma':<14} {'acc':<8} {'time(s)':<10}")
    print("-" * 80)

    results = []
    for beta in betas:
        rng_local = np.random.default_rng(seed + int(beta))
        psi_cubic, psi_diag = initialize_links(L, rng_local)
        t0 = time.time()
        # Thermalization
        for sw in range(n_therm):
            heat_bath_sweep_simple(psi_cubic, psi_diag, beta, L, rng_local)
            if (sw + 1) % max(1, n_therm // 5) == 0:
                # Print progress occasionally
                pass
        # Measurement
        Ws = []
        acc_rates = []
        for sw in range(n_meas):
            acc = heat_bath_sweep_simple(psi_cubic, psi_diag, beta, L, rng_local)
            acc_rates.append(acc)
            W = measure_wilson_diag_avg(psi_cubic, psi_diag, L)
            Ws.append(W)
        Ws = np.array(Ws)
        # Block average to estimate error
        n_blocks = 10
        block_size = len(Ws) // n_blocks
        block_means = np.array([np.mean(Ws[i*block_size:(i+1)*block_size])
                                 for i in range(n_blocks)])
        W_mean = np.mean(block_means)
        W_se = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
        elapsed = time.time() - t0
        alpha = 1.0 / beta
        print(f"{beta:<8.1f} {alpha:<10.5f} {W_mean:<14.7f} {W_se:<14.7f} "
              f"{np.mean(acc_rates):<8.3f} {elapsed:<10.1f}")
        results.append({
            "beta": beta, "alpha": alpha,
            "W_mean": float(W_mean), "W_se": float(W_se),
            "acc_rate": float(np.mean(acc_rates)),
            "time_s": elapsed,
        })

    # Fit polynomial
    print()
    alphas = np.array([r["alpha"] for r in results])
    Ws = np.array([r["W_mean"] for r in results])
    Ws_se = np.array([r["W_se"] for r in results])

    # Fit (1 - W) vs alpha as polynomial degree 2
    y = 1.0 - Ws
    weights = 1.0 / Ws_se
    coeffs = np.polyfit(alphas, y, 2, w=weights)
    a_2, a_1, a_0 = coeffs
    print(f"Polynomial fit: 1 - <W> = {a_0:.5f} + ({a_1:.5f})*alpha + ({a_2:.5f})*alpha^2")
    if a_1 != 0:
        c_2_relative = 2 * a_2 / a_1**2
        print(f"c_2 (relative to abelian exp = 1): {c_2_relative:.4f}")
        print(f"  (Continuum exp: c_2_relative = 1.0; lattice expected: 0.3-1.0)")

    # Save
    out = {
        "L": L, "n_therm": n_therm, "n_meas": n_meas, "seed": seed,
        "betas": list(betas),
        "results": results,
        "fit": {"a_0": float(a_0), "a_1": float(a_1), "a_2": float(a_2),
                "c_2_relative": float(c_2_relative) if a_1 != 0 else None}
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
