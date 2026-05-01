"""
Paper VI — direct numerical computation of the lattice Wilson loop
self-energy coefficient.

Setup:
  - 3D cubic lattice L^3 with periodic boundary
  - Optional diagonal twist links (modifies the Laplacian)
  - Compute the lattice Green function G(x - y) = (-Δ)^(-1)(x, y)
  - Integrate G along a body diagonal of length √3 (i.e., from
    (0,0,0) to (1,1,1), with appropriate lattice positions)
  - The "Wilson loop self-energy coefficient" is given by the
    relevant double integral

Key observable:
  C_lattice = (1/L_diag) · Σ_{x on diagonal, y on diagonal} G(x, y) / a²

Compare with:
  - V_3 = 4π/3 ≈ 4.189 (Euclidean unit ball volume)
  - V_{d_eff} ≈ 4.332 (volume in fractional dimension)
  - Or some other lattice-specific number
"""
import numpy as np
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi


def laplacian_eigenvalues(L, mode="cubic"):
    """Return diagonal of Laplacian in momentum space (per Fourier mode)."""
    k = 2*PI*np.fft.fftfreq(L)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    
    # Standard cubic Laplacian: L(k) = 2(3 - cos kx - cos ky - cos kz)
    L_cubic = 2*(3 - np.cos(KX) - np.cos(KY) - np.cos(KZ))
    
    if mode == "cubic":
        return L_cubic
    elif mode == "diag_111":
        # Add hopping in (1,1,1) direction: -2cos(kx+ky+kz)
        # so contribution to Laplacian: 2(1 - cos(kx+ky+kz))
        return L_cubic + 2*(1 - np.cos(KX + KY + KZ))
    elif mode == "diag_all4":
        return (L_cubic
                + 2*(1 - np.cos(KX + KY + KZ))
                + 2*(1 - np.cos(KX + KY - KZ))
                + 2*(1 - np.cos(KX - KY + KZ))
                + 2*(1 - np.cos(-KX + KY + KZ)))
    elif mode == "twist_111":
        # Twist on (1,1,1): hopping with phase λ·(kx+ky+kz)/(2π)
        # The Laplacian gets a -2cos(kx+ky+kz) term times (1+phase factor?)
        # For our purposes: same kinetic term as diag_111
        return L_cubic + 2*(1 - np.cos(KX + KY + KZ))
    else:
        raise ValueError(mode)


def propagator_real_space(L, mode="cubic"):
    """Compute G(x) = (1/L^3) Σ_k exp(ik·x) / Λ(k) for the lattice mode."""
    Lambda_k = laplacian_eigenvalues(L, mode)
    # Replace zero mode with infinity (or use a regularization)
    # For periodic lattice, k=0 is exactly zero and gives a uniform shift.
    # We can either subtract the zero mode or use a small mass regulator.
    # Here we just zero it out, giving G(x) up to a constant.
    Lambda_k_inv = np.where(Lambda_k > 1e-10, 1.0/Lambda_k, 0.0)
    # G(x) = inverse FFT of 1/Λ(k)
    G = np.fft.ifftn(Lambda_k_inv).real
    return G


def diagonal_double_sum(G, L, n_steps=10):
    """Sum G(x_i, y_j) over points along the body diagonal.
    The body diagonal goes from (0,0,0) to (n,n,n) for some n.
    We use n = n_steps lattice steps."""
    N = L
    total = 0.0
    points = [(i, i, i) for i in range(n_steps + 1)]
    L_diag = n_steps * np.sqrt(3)  # length in lattice units
    
    for (xi, yi, zi) in points:
        for (xj, yj, zj) in points:
            # G(x-y) is G at displacement (xi-xj, yi-yj, zi-zj)
            di = (xi - xj) % N
            dj = (yi - yj) % N
            dk = (zi - zj) % N
            total += G[di, dj, dk]
    
    return total, L_diag


def main():
    L = 32
    print(f"Lattice L = {L}")
    print()
    
    target_V3 = 4*PI/3
    from scipy.special import gammaln
    d_e = 3 + 1/(2*PI)
    target_Vdeff = PI**(d_e/2) / np.exp(gammaln(d_e/2 + 1))
    
    print(f"Reference values:")
    print(f"  V_3       = 4π/3      = {target_V3:.6f}")
    print(f"  V_d_eff                = {target_Vdeff:.6f}")
    print()
    
    print(f"{'Mode':<20} {'sum/L_diag':>14} {'sum/L_diag/(4π)':>20}")
    print("-" * 60)
    rows = []
    for mode in ["cubic", "diag_111", "diag_all4"]:
        G = propagator_real_space(L, mode)
        for n in [3, 5, 10]:
            total, L_diag = diagonal_double_sum(G, L, n_steps=n)
            # Coefficient when normalized by perimeter
            coef = total / L_diag if L_diag > 0 else 0
            # Often Coulomb propagator has factor 1/(4π); de-normalize
            coef_4pi = coef * (4*PI)
            print(f"{mode}_n={n:<11} {coef:14.6f} {coef_4pi:20.6f}")
            rows.append({"mode": mode, "n_steps": n, "L_diag": L_diag,
                         "total": float(total), "coef_normalized": float(coef),
                         "coef_4pi": float(coef_4pi)})
    
    print()
    print("Note: this is a rough calculation. Lattice Green function is")
    print("not the continuum 1/(4π·r); it has specific lattice corrections")
    print("(Madelung constants in 3D).")
    
    # Compute the difference between cubic and diag_111
    # which isolates the diagonal-twist contribution
    G_cubic = propagator_real_space(L, "cubic")
    G_diag = propagator_real_space(L, "diag_111")
    
    # Sum at typical body diagonal endpoints
    n = 5
    G_diff = G_diag - G_cubic
    # Display G(x) for x along the body diagonal
    print()
    print("Green function along body diagonal:")
    print(f"{'r/a':>6} {'G_cubic(r)':>14} {'G_diag111(r)':>14} {'difference':>14}")
    for i in range(0, 8):
        if i < L:
            r = i * np.sqrt(3)
            print(f"{r:6.2f} {G_cubic[i,i,i]:14.6e} {G_diag[i,i,i]:14.6e} {G_diff[i,i,i]:14.6e}")

    out = {"L": L, "V_3": target_V3, "V_d_eff": float(target_Vdeff),
           "results": rows}
    with open(DATA / "16_propagator_integral.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
