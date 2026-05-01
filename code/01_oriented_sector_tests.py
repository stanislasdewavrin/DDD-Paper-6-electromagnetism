"""
Paper VI — Numerical validation of the oriented sector
========================================================

Three tests on a 2D XY-like lattice with the local update rule
    alpha_i(t+1) = alpha_i(t) + (1/tau_align) * sum_{j~i} J_{ij} * sin(alpha_j - alpha_i)

(Link phases psi_ij set to zero throughout for these baseline tests;
the gauge-invariance test introduces non-trivial psi_ij later.)

(1) Wave propagation: a localized perturbation in alpha propagates at
    a speed v_propagation = c determined by sqrt(J/tau_align), in the
    linear regime |Delta| << pi/2.

(2) Vortex stability: an alpha configuration with winding number +1 is
    a stable topological defect; we evolve it for many ticks and verify
    that the winding number is conserved.

(3) Vortex-antivortex interaction: two opposite-charge vortices feel a
    Coulomb-like attractive interaction with energy E ~ -ln(r) in 2D.
    We measure the static energy of two isolated vortices vs separation.

Outputs:
    data/em_results.json
    figures/fig01_em_tests.pdf
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

J        = 1.0      # link coupling
TAU      = 1.0      # alignment timescale (so c = sqrt(J/tau) = 1)
DT       = 0.1      # integration timestep
SEED     = 2024
np.random.seed(SEED)

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def step(alpha):
    """Periodic 2D XY-like update.

    d alpha_i / dt = (J/tau) * sum_{j~i} sin(alpha_j - alpha_i)
    Integrated by forward Euler with timestep DT.
    """
    da = np.zeros_like(alpha)
    for sh, ax in [(+1, 0), (-1, 0), (+1, 1), (-1, 1)]:
        rolled = np.roll(alpha, sh, axis=ax)
        da += np.sin(rolled - alpha)
    return alpha + (J / TAU) * da * DT


def step_2nd_order(alpha, alpha_prev):
    """Second-order (wave) version: d^2 alpha / dt^2 = c^2 * Laplacian sin.

    This is the linearised wave-equation regime if we identify the
    sin coupling with phase-elastic restoring force. Use leapfrog:
    alpha_new = 2*alpha - alpha_prev + (c^2 dt^2) * sum sin(alpha_j - alpha_i)
    """
    da = np.zeros_like(alpha)
    for sh, ax in [(+1,0), (-1,0), (+1,1), (-1,1)]:
        rolled = np.roll(alpha, sh, axis=ax)
        da += np.sin(rolled - alpha)
    c2_dt2 = (J / TAU) * (DT ** 2)
    return 2 * alpha - alpha_prev + c2_dt2 * da


# ----------------------------------------- Test 1: wave propagation
def test_wave():
    print("\n--- Test 1: wave propagation ---")
    N = 80
    alpha = np.zeros((N, N))
    # Localized perturbation at the centre, small amplitude
    cx = cy = N // 2
    sigma = 2.0
    xs, ys = np.indices((N, N))
    pulse = 0.05 * np.exp(-((xs - cx)**2 + (ys - cy)**2) / sigma**2)
    alpha = pulse.copy()
    alpha_prev = pulse.copy()  # zero initial velocity

    n_steps = 400
    history_alpha = []
    history_radii = []
    for t in range(n_steps):
        alpha_new = step_2nd_order(alpha, alpha_prev)
        alpha_prev = alpha
        alpha = alpha_new
        if t % 20 == 0:
            history_alpha.append(alpha.copy())
        # Measure radial extent of perturbation: argmax of |alpha| along central row
        row = np.abs(alpha[cx, :])
        if row.max() > 1e-3:
            # Find rightmost peak
            r_peak = np.argmax(row[cx:]) + 0.0
            history_radii.append({"t": t * DT, "r": float(r_peak)})

    # Linear fit r(t) = v * t (after the wavefront has separated from origin)
    if len(history_radii) > 20:
        ts = np.array([h["t"] for h in history_radii[10:]])
        rs = np.array([h["r"] for h in history_radii[10:]])
        # Take only times where the wavefront is moving forward
        valid = (rs > 5) & (rs < N//2 - 5)
        if valid.sum() >= 5:
            v_meas, _ = np.polyfit(ts[valid], rs[valid], 1)
        else:
            v_meas = None
    else:
        v_meas = None

    c_predicted = np.sqrt(J / TAU)
    print(f"Predicted speed c = sqrt(J/tau) = {c_predicted:.3f}")
    if v_meas is not None:
        print(f"Measured wave speed = {v_meas:.3f}")
        print(f"Ratio v_meas/c_pred = {v_meas/c_predicted:.3f}")

    return {
        "c_predicted":     float(c_predicted),
        "v_measured":      float(v_meas) if v_meas is not None else None,
        "history_radii":   history_radii,
        "snapshots":       [a.tolist() for a in history_alpha[:5]],
    }


# ----------------------------------------- Test 2: vortex stability
def make_vortex(N, charge=+1):
    """Create alpha field with winding number = charge around lattice centre."""
    cx = cy = (N - 1) / 2
    xs, ys = np.indices((N, N))
    return charge * np.arctan2(ys - cy, xs - cx)


def winding_number(alpha, contour_radius):
    """Compute winding of alpha along a circular contour around the lattice centre."""
    N = alpha.shape[0]
    cx = cy = (N - 1) / 2
    n_pts = 64
    thetas = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    samples = []
    for th in thetas:
        i = int(round(cx + contour_radius * np.cos(th)))
        j = int(round(cy + contour_radius * np.sin(th)))
        if 0 <= i < N and 0 <= j < N:
            samples.append(alpha[i, j])
    if len(samples) < 8:
        return None
    samples = np.array(samples)
    # Sum of phase differences (mod 2pi)
    diffs = np.diff(samples)
    diffs = np.mod(diffs + np.pi, 2*np.pi) - np.pi
    last = np.mod(samples[0] - samples[-1] + np.pi, 2*np.pi) - np.pi
    return float((diffs.sum() + last) / (2*np.pi))


def test_vortex():
    print("\n--- Test 2: vortex stability ---")
    N = 40
    alpha = make_vortex(N, charge=+1)
    n_steps = 800

    initial_w = winding_number(alpha, 10)
    print(f"Initial winding number: {initial_w}")

    history_w = []
    for t in range(n_steps):
        alpha = step(alpha)
        if t % 50 == 0:
            w = winding_number(alpha, 10)
            history_w.append({"t": t, "w": w})

    final_w = winding_number(alpha, 10)
    print(f"Final winding (after {n_steps} steps): {final_w}")
    print(f"Conservation residual: {abs(final_w - initial_w):.4f}")

    return {
        "initial_winding":      initial_w,
        "final_winding":        final_w,
        "n_steps":              n_steps,
        "conservation_resid":   float(abs(final_w - initial_w)),
        "history":              history_w,
    }


# ---------------------- Test 3: vortex-antivortex interaction (energy vs r)
def make_vortex_pair(N, x1, y1, x2, y2, c1=+1, c2=-1):
    """Create alpha with two vortices of charges c1, c2."""
    xs, ys = np.indices((N, N))
    a = c1 * np.arctan2(ys - y1, xs - x1) + c2 * np.arctan2(ys - y2, xs - x2)
    return a


def total_energy(alpha):
    """E = sum_links J [1 - cos(alpha_j - alpha_i)]."""
    e = 0.0
    for sh, ax in [(+1, 0), (+1, 1)]:
        rolled = np.roll(alpha, sh, axis=ax)
        e += float((1.0 - np.cos(rolled - alpha)).sum())
    return J * e


def test_coulomb():
    print("\n--- Test 3: vortex-antivortex Coulomb-like interaction ---")
    N = 60
    cy = N // 2
    energies = []
    distances = [4, 6, 8, 10, 14, 18, 22, 28]
    for d in distances:
        x1 = N // 2 - d // 2
        x2 = N // 2 + d // 2
        alpha = make_vortex_pair(N, x1, cy, x2, cy, c1=+1, c2=-1)
        # Brief relaxation to stabilize the configuration
        for _ in range(100):
            alpha = step(alpha)
        E = total_energy(alpha)
        energies.append((d, E))
        print(f"  d={d}: E = {E:.3f}")

    # Fit E(d) = -A ln(d) + B  (Coulomb in 2D)
    ds = np.array([d for d, e in energies])
    es = np.array([e for d, e in energies])
    log_ds = np.log(ds)
    A_fit, B_fit = np.polyfit(log_ds, es, 1)
    print(f"Fit: E = {A_fit:.3f} * ln(d) + {B_fit:.3f}")
    print(f"  Coulomb-like behavior expected: E ~ +k*ln(d)+const for opposite charges (binding)")
    # Residuals
    predicted = A_fit * log_ds + B_fit
    rss = float(np.sum((predicted - es)**2))

    return {
        "distances":   [int(d) for d in ds],
        "energies":    [float(e) for e in es],
        "log_fit_slope": float(A_fit),
        "log_fit_intercept": float(B_fit),
        "rss":          rss,
    }


# ----------------------------------------- main
if __name__ == "__main__":
    print("Paper VI — Oriented sector tests")
    print("=" * 60)

    results = {}
    results["wave"]    = test_wave()
    results["vortex"]  = test_vortex()
    results["coulomb"] = test_coulomb()

    out_path = DATA_DIR / "em_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Combined figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Test 1: wave radius vs time
    ax = axes[0]
    h = results["wave"]["history_radii"]
    if h:
        ts = [x["t"] for x in h]
        rs = [x["r"] for x in h]
        ax.plot(ts, rs, "o-", markersize=3, label="measured wavefront")
        c = results["wave"]["c_predicted"]
        v = results["wave"]["v_measured"]
        if v is not None:
            ts_th = np.linspace(0, max(ts), 100)
            ax.plot(ts_th, c * ts_th, "k--", lw=1, label=fr"$c=\sqrt{{J/\tau}}={c:.2f}$")
            ax.plot(ts_th, v * ts_th, "r:", lw=1, label=fr"fit $v={v:.3f}$")
    ax.set_xlabel("t")
    ax.set_ylabel("wavefront radius")
    ax.set_title("Test 1: wave propagation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test 2: winding history
    ax = axes[1]
    h = results["vortex"]["history"]
    ts = [x["t"] for x in h]
    ws = [x["w"] for x in h]
    ax.plot(ts, ws, "o-", markersize=4, color="C2")
    ax.axhline(results["vortex"]["initial_winding"], color="grey", lw=0.5, ls="--",
               label="initial w")
    ax.set_xlabel("tick")
    ax.set_ylabel("winding number")
    ax.set_title("Test 2: vortex stability\n"
                 fr"residual = {results['vortex']['conservation_resid']:.4f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Test 3: energy vs ln(d)
    ax = axes[2]
    ds = np.array(results["coulomb"]["distances"])
    es = np.array(results["coulomb"]["energies"])
    ax.semilogx(ds, es, "o", markersize=8, color="C3", label="measured")
    A, B = results["coulomb"]["log_fit_slope"], results["coulomb"]["log_fit_intercept"]
    ds_th = np.logspace(np.log10(min(ds)), np.log10(max(ds)), 50)
    ax.semilogx(ds_th, A * np.log(ds_th) + B, "k--", lw=1,
                label=fr"$E = {A:.2f}\ln d + {B:.2f}$")
    ax.set_xlabel(r"separation $d$")
    ax.set_ylabel("total energy")
    ax.set_title("Test 3: vortex-antivortex (Coulomb-like)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Paper VI — Oriented sector validation", fontsize=12, y=1.02)
    fig.tight_layout()
    fig_path = FIG_DIR / "fig01_em_tests.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf",".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
