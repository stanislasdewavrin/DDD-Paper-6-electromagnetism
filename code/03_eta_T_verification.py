"""
Paper VI — numerical verification of the eta = T identity.

Setup:
  - 2D Floquet 2-step rule from Paper III (T = tau_A + tau_B = 2)
  - Modified to include a varying R(x, y)/R_0 background
  - Each sub-tick (A and B) is bandwidth-modulated by R/R_0 at the
    relevant lattice site
  - We track a wavepacket near the gap (massive Dirac, slow propagation)
    or away from it (less massive, faster) and measure its effective
    coordinate speed

The prediction is:
    c_eff(r) = c_eff_vacuum * (R(r)/R_0)^T
with T = 2 for the 2-step rule. We test this by comparing measured
group velocity in modulated vs uniform backgrounds.

Note: this is the simplest test we can do without the full 3D Weyl
spinor sector. It uses a 2D Floquet-Dirac wavepacket as proxy for
the photon mode; the test is on the EXPONENT eta, not on the absolute
gapless dispersion (which requires 3D).
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi
LAMBDA = 1.0 / (2 * PI)
TAU_A = 1.0
TAU_B = 1.0
T_PERIOD = TAU_A + TAU_B   # = 2


def step_A_bandwidth(psi, R_field, tau_A, L):
    """Matter step with bandwidth modulation: rate scaled by R(x)/R_0."""
    # We use FFT for efficiency, but the bandwidth field is local.
    # The rate of update is modulated locally: psi(t+tau_A) = exp(-i H_A * tau_A * R/R_0) psi
    # For uniform R/R_0 = chi_0: equivalent to step_A with tau_A_eff = tau_A * chi_0.
    # For varying R/R_0(x): we approximate by Trotter with local rates.
    # In real space: apply hopping with local rate factor.
    L_x, L_y = R_field.shape
    a_k = np.fft.fft2(psi[..., 0]); b_k = np.fft.fft2(psi[..., 1])
    kx = 2*PI*np.fft.fftfreq(L_x); ky = 2*PI*np.fft.fftfreq(L_y)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    dx = np.sin(KX); dy = np.sin(KY); dz = np.cos(KX) + np.cos(KY)
    dn = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-15)
    # Compute uniform-bandwidth U_A first
    c = np.cos(dn * tau_A); s = np.sin(dn * tau_A)
    U00 = c - 1j*s*dz/dn
    U01 = -1j*s*(dx - 1j*dy)/dn
    U10 = -1j*s*(dx + 1j*dy)/dn
    U11 = c + 1j*s*dz/dn
    new_a_k = U00 * a_k + U01 * b_k
    new_b_k = U10 * a_k + U11 * b_k
    new_a = np.fft.ifft2(new_a_k); new_b = np.fft.ifft2(new_b_k)
    # Bandwidth modulation: blend uniform-bandwidth with no-update according to R(x)/R_0
    # When R/R_0 = 1: full update. When R/R_0 = 0: no update (psi unchanged).
    # Linear interpolation in update amplitude:
    chi = R_field
    new_a_mod = chi * new_a + (1 - chi) * psi[..., 0]
    new_b_mod = chi * new_b + (1 - chi) * psi[..., 1]
    return np.stack([new_a_mod, new_b_mod], axis=-1)


def step_B_bandwidth(psi, R_field_link, tau_B, L, lam=LAMBDA):
    """Gauge step (link update) with bandwidth modulation by link-midpoint R/R_0."""
    L_x, L_y = R_field_link.shape
    a_k = np.fft.fft2(psi[..., 0]); b_k = np.fft.fft2(psi[..., 1])
    kx = 2*PI*np.fft.fftfreq(L_x); ky = 2*PI*np.fft.fftfreq(L_y)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    dz = lam * np.cos(KX + KY)
    phase = np.exp(-1j * dz * tau_B)
    new_a_k = phase * a_k
    new_b_k = np.conj(phase) * b_k
    new_a = np.fft.ifft2(new_a_k); new_b = np.fft.ifft2(new_b_k)
    # Bandwidth modulation: same recipe, blend with original by link bandwidth
    chi = R_field_link
    new_a_mod = chi * new_a + (1 - chi) * psi[..., 0]
    new_b_mod = chi * new_b + (1 - chi) * psi[..., 1]
    return np.stack([new_a_mod, new_b_mod], axis=-1)


def make_R_field(L, beta=0.0, chi0=1.0):
    """R(x, y) / R_0 background.
    For beta > 0: spherical R(r) = sqrt(1 - beta/r) outside the horizon.
    For beta = 0: uniform chi0 (test uniform-bandwidth scaling).
    """
    if beta == 0.0:
        return np.full((L, L), chi0, dtype=np.float64)
    cx, cy = L // 2, L // 2
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    r = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    chi = np.where(r > beta + 0.5,
                   np.sqrt(np.maximum(1 - beta / np.maximum(r, 1.0), 0.01)),
                   0.01)
    return chi


def make_wavepacket(L, k0, sigma, x0):
    """Gaussian wavepacket at offset position x0."""
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    env = np.exp(-((xs - x0[0])**2 + (ys - x0[1])**2) / (2 * sigma**2))
    plane = np.exp(1j * (k0[0] * xs + k0[1] * ys))
    psi = np.zeros((L, L, 2), dtype=complex)
    psi[..., 0] = env * plane
    psi[..., 1] = 0.0
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
    return psi


def centroid(psi, L):
    rho = np.abs(psi[..., 0])**2 + np.abs(psi[..., 1])**2
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    ax = np.angle(np.sum(rho * np.exp(2j * PI * xs / L)))
    ay = np.angle(np.sum(rho * np.exp(2j * PI * ys / L)))
    return float(ax / (2*PI) * L % L), float(ay / (2*PI) * L % L)


def measure_vg_uniform(L, chi_0, k0, sigma, n_cycles):
    """Run wavepacket in uniform R/R_0 = chi_0 and measure group velocity."""
    R_field = make_R_field(L, beta=0.0, chi0=chi_0)
    psi = make_wavepacket(L, k0, sigma, x0=(L/2, L/2))
    cx0, _ = centroid(psi, L)
    for it in range(n_cycles):
        psi = step_A_bandwidth(psi, R_field, TAU_A, L)
        psi = step_B_bandwidth(psi, R_field, TAU_B, L)
    cxf, _ = centroid(psi, L)
    # Account for periodic wrap
    drift = cxf - cx0
    if drift > L/2: drift -= L
    if drift < -L/2: drift += L
    v_meas = drift / (n_cycles * T_PERIOD)
    return v_meas


def main():
    L = 64
    sigma = 6.0
    n_cycles = 30
    # Wavepacket in lower band, away from the gap (so it has finite v_g)
    delta = 0.4
    k0 = (PI - delta, 0.0)

    print(f"L={L} sigma={sigma} n_cycles={n_cycles}")
    print(f"Wavepacket at k0=(pi - {delta}, 0)")
    print(f"Floquet period T = {T_PERIOD}")
    print()

    # Reference vacuum velocity (chi0 = 1)
    v_vacuum = measure_vg_uniform(L, 1.0, k0, sigma, n_cycles)
    print(f"v_g(vacuum) = {v_vacuum:.6f}")
    print()

    # Uniform-bandwidth test: scan chi_0 in (0, 1] and measure v_g(chi_0)
    chi_values = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
    print(f"Uniform-bandwidth scan (chi_0 = R/R_0 uniform):")
    print(f"{'chi_0':>8} {'v_meas':>10} {'v_vac*chi^1':>14} {'v_vac*chi^2':>14} "
          f"{'eta_fit':>10}")
    rows = []
    for chi_0 in chi_values:
        v_meas = measure_vg_uniform(L, chi_0, k0, sigma, n_cycles)
        v_pred_eta1 = v_vacuum * chi_0
        v_pred_eta2 = v_vacuum * chi_0**2
        # eta_fit = log(v_meas / v_vacuum) / log(chi_0)
        if chi_0 < 1.0 and v_meas > 0 and v_vacuum > 0:
            eta_fit = np.log(v_meas / v_vacuum) / np.log(chi_0)
        else:
            eta_fit = float('nan')
        print(f"{chi_0:8.2f} {v_meas:10.6f} {v_pred_eta1:14.6f} "
              f"{v_pred_eta2:14.6f} {eta_fit:10.4f}")
        rows.append({"chi_0": chi_0, "v_meas": float(v_meas),
                     "v_pred_eta1": float(v_pred_eta1),
                     "v_pred_eta2": float(v_pred_eta2),
                     "eta_fit": float(eta_fit) if not np.isnan(eta_fit) else None})

    out = {
        "L": L, "sigma": sigma, "n_cycles": n_cycles,
        "T_PERIOD": T_PERIOD, "delta": delta,
        "v_vacuum": float(v_vacuum),
        "results": rows,
    }
    with open(DATA / "03_eta_T_verification.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {DATA / '03_eta_T_verification.json'}")


if __name__ == "__main__":
    main()
