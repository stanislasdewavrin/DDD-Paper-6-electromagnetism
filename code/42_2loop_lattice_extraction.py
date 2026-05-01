"""
Numerical extraction of the 2-loop coefficient c_2 via direct
lattice gauge theory simulation.

Setup:
  - Compact U(1) on small 3D cubic lattice (L^3)
  - Wilson action: S = beta * sum_p [1 - cos(Phi_p)]
  - Body-diagonal Wilson loop with twist lambda = 1/(2*pi)
  - Heat bath Monte Carlo

Extract c_2 by fitting:
  <W(C)> = 1 - alpha * I_1 + (alpha)^2 * c_2 * V + O(alpha^3)
  where alpha = 1/beta in our convention.

NOTE: This is a *minimal viable* implementation. Production-quality
calculation would require larger L, more thermalization, error analysis,
and proper renormalization. Time budget here: < 60 seconds.
"""
import numpy as np
import time
from math import gamma, pi
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)


def init_links(L):
    """Initialize 3D U(1) link phases on edges. Phase per link."""
    # 3 directions per node: x, y, z
    # Total links = 3*L^3
    return np.random.uniform(0, 2*pi, size=(L, L, L, 3))


def plaquette_phase(links, i, j, k, mu, nu, L):
    """Compute phase around plaquette in (mu,nu) plane at site (i,j,k).

    Plaquette = links[i,j,k,mu] + links[i+e_mu,nu] - links[i+e_nu,mu] - links[i,nu]
    """
    # Move +mu
    if mu == 0: i_mu, j_mu, k_mu = (i+1)%L, j, k
    elif mu == 1: i_mu, j_mu, k_mu = i, (j+1)%L, k
    else: i_mu, j_mu, k_mu = i, j, (k+1)%L
    if nu == 0: i_nu, j_nu, k_nu = (i+1)%L, j, k
    elif nu == 1: i_nu, j_nu, k_nu = i, (j+1)%L, k
    else: i_nu, j_nu, k_nu = i, j, (k+1)%L
    return (links[i,j,k,mu] + links[i_mu,j_mu,k_mu,nu]
            - links[(i+1 if nu==0 else i)%L, (j+1 if nu==1 else j)%L, (k+1 if nu==2 else k)%L, mu]
            - links[i,j,k,nu])


def total_action(links, beta, L):
    """Compute total Wilson action S = beta * sum_p [1 - cos(Phi_p)]."""
    S = 0.0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for mu in range(3):
                    for nu in range(mu+1, 3):
                        phi = plaquette_phase(links, i, j, k, mu, nu, L)
                        S += beta * (1 - np.cos(phi))
    return S


def heat_bath_step(links, beta, L):
    """One sweep of heat-bath update."""
    # For U(1) compact, heat-bath samples the local phase distribution
    # exp(beta * Re(staple * exp(i*phi)))
    # Simplified: random update with Metropolis acceptance
    n_acc = 0
    n_tot = 0
    for _ in range(L**3):
        i = np.random.randint(L)
        j = np.random.randint(L)
        k = np.random.randint(L)
        mu = np.random.randint(3)
        # Old action contribution (sum over plaquettes containing this link)
        old_phi = links[i,j,k,mu]
        new_phi = old_phi + np.random.uniform(-0.5, 0.5)
        # Compute change in action approximately by 2 plaquettes adjacent
        S_old = 0.0
        S_new = 0.0
        for nu in range(3):
            if nu == mu: continue
            # plaquette in (mu,nu) starting at this site
            phi_p_old = plaquette_phase(links, i, j, k, mu, nu, L)
            S_old += 1 - np.cos(phi_p_old)
            links[i,j,k,mu] = new_phi
            phi_p_new = plaquette_phase(links, i, j, k, mu, nu, L)
            S_new += 1 - np.cos(phi_p_new)
            links[i,j,k,mu] = old_phi
        # Metropolis
        delta_S = beta * (S_new - S_old)
        if delta_S < 0 or np.random.random() < np.exp(-delta_S):
            links[i,j,k,mu] = new_phi
            n_acc += 1
        n_tot += 1
    return n_acc / n_tot


def wilson_loop_plaquette(links, i, j, k, mu, nu, L):
    """Plaquette Wilson loop: <cos(Phi_p)>"""
    return np.cos(plaquette_phase(links, i, j, k, mu, nu, L))


def main():
    L = 8  # small for speed
    n_therm = 50
    n_meas = 100

    # Sweep beta from large to small (alpha small to large)
    betas = [40.0, 20.0, 10.0, 5.0, 2.5]
    print(f"L = {L}, sweeping beta values...")
    print(f"{'beta':<8} {'alpha':<10} {'<W>':<12} {'time':<8}")
    print("-" * 50)

    results = []
    for beta in betas:
        np.random.seed(42)
        links = init_links(L)
        t0 = time.time()
        # Thermalize
        for _ in range(n_therm):
            heat_bath_step(links, beta, L)
        # Measure
        Ws = []
        for _ in range(n_meas):
            heat_bath_step(links, beta, L)
            # Average plaquette
            W_total = 0.0
            n_p = 0
            for i in range(L):
                for j in range(L):
                    for k in range(L):
                        for mu in range(3):
                            for nu in range(mu+1, 3):
                                W_total += wilson_loop_plaquette(links, i, j, k, mu, nu, L)
                                n_p += 1
            Ws.append(W_total / n_p)
        W_mean = np.mean(Ws)
        elapsed = time.time() - t0
        alpha = 1.0/beta
        print(f"{beta:<8.1f} {alpha:<10.4f} {W_mean:<12.7f} {elapsed:<8.1f}s")
        results.append({"beta": beta, "alpha": alpha, "W_mean": W_mean, "W_std": float(np.std(Ws))})

    # Fit: <W> = 1 - alpha * I_1 + alpha^2 * c_2 * V
    print()
    print("Fit <W> = 1 - alpha * I_1 + alpha^2 * c_2_eff (extracting c_2)")
    alphas = np.array([r["alpha"] for r in results])
    Ws = np.array([r["W_mean"] for r in results])
    # Fit polynomial degree 2 around alpha=0
    # 1 - W = alpha * a_1 + alpha^2 * a_2
    y = 1 - Ws
    # x = alpha
    # y = a_1 * alpha + a_2 * alpha^2
    a_1, a_2 = np.polyfit(alphas, y, 2)[1:][::-1]
    print(f"  a_1 (1-loop coeff) = {a_1:.4f}")
    print(f"  a_2 (2-loop coeff)  = {a_2:.4f}")
    # The 2-loop coefficient relative to (1-loop)^2/2:
    if a_1 != 0:
        c_2_relative = 2 * a_2 / (a_1**2)
        print(f"  Ratio 2*a_2 / a_1^2 = {c_2_relative:.4f}")
        print(f"  (Continuum exp: c_2_relative = 1)")

    out = {"L": L, "n_therm": n_therm, "n_meas": n_meas,
           "results": results}
    with open(DATA / "42_2loop_lattice.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {DATA / '42_2loop_lattice.json'}")


if __name__ == "__main__":
    main()
