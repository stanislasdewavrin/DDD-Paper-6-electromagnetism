"""
Paper VI — Test extension to 3D
=================================

Three 3D tests of the oriented sector:

(1) 3D wave propagation: localized perturbation propagates spherically
    at speed c = sqrt(J/tau).

(2) Straight vortex line along z-axis: stable filament with conserved
    winding at every z-slice over many ticks.

(3) Two parallel vortex lines (one +1, one -1) along z-axis at
    transverse separation d_perp: total energy per unit length scales
    logarithmically with d_perp (same as 2D, since physics is per
    transverse plane).

Outputs:
    data/em_3d_results.json
    figures/fig02_em_3d.pdf
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

J        = 1.0
TAU      = 1.0
DT       = 0.1
SEED     = 2024
np.random.seed(SEED)

HERE     = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
FIG_DIR  = HERE / "figures"


def step3d(alpha):
    """3D update with 6 neighbors."""
    da = np.zeros_like(alpha)
    for sh, ax in [(+1, 0), (-1, 0), (+1, 1), (-1, 1), (+1, 2), (-1, 2)]:
        rolled = np.roll(alpha, sh, axis=ax)
        da += np.sin(rolled - alpha)
    return alpha + (J / TAU) * da * DT


def step3d_2nd(alpha, alpha_prev):
    """Second-order leapfrog for wave equation."""
    da = np.zeros_like(alpha)
    for sh, ax in [(+1,0),(-1,0),(+1,1),(-1,1),(+1,2),(-1,2)]:
        rolled = np.roll(alpha, sh, axis=ax)
        da += np.sin(rolled - alpha)
    return 2*alpha - alpha_prev + (J/TAU) * (DT**2) * da


def total_energy_3d(alpha):
    e = 0.0
    for sh, ax in [(+1,0),(+1,1),(+1,2)]:
        rolled = np.roll(alpha, sh, axis=ax)
        e += float((1.0 - np.cos(rolled - alpha)).sum())
    return J * e


# --------------------- Test 1: 3D wave propagation
def test_wave_3d():
    print("\n--- Test 1 (3D): wave propagation ---")
    N = 40
    cx = N // 2
    xs, ys, zs = np.indices((N, N, N))
    sigma = 1.5
    pulse = 0.05 * np.exp(-((xs-cx)**2 + (ys-cx)**2 + (zs-cx)**2) / sigma**2)
    alpha = pulse.copy()
    alpha_prev = pulse.copy()

    n_steps = 200
    history = []
    for t in range(n_steps):
        new = step3d_2nd(alpha, alpha_prev)
        alpha_prev = alpha
        alpha = new
        # Track wavefront radius along +x axis
        row = np.abs(alpha[cx:, cx, cx])
        if row.max() > 1e-3:
            r_peak = float(np.argmax(row[1:]) + 1)
            history.append({"t": t * DT, "r": r_peak})

    if len(history) > 30:
        ts = np.array([h["t"] for h in history[10:]])
        rs = np.array([h["r"] for h in history[10:]])
        valid = (rs > 3) & (rs < N//2 - 3)
        if valid.sum() >= 5:
            v_meas, _ = np.polyfit(ts[valid], rs[valid], 1)
        else:
            v_meas = None
    else:
        v_meas = None

    c_pred = np.sqrt(J / TAU)
    print(f"  Predicted: c = {c_pred:.3f}")
    if v_meas is not None:
        print(f"  Measured:  v = {v_meas:.3f}")
        print(f"  Ratio: {v_meas/c_pred:.3f}")
    return {
        "c_predicted": float(c_pred),
        "v_measured":  float(v_meas) if v_meas is not None else None,
        "history":     history,
    }


# --------------------- Test 2: vortex line stability
def make_vortex_line(N, x0, y0, charge=+1):
    """Vortex line along z-axis at (x0, y0). Same alpha for every z-slice."""
    xs, ys, zs = np.indices((N, N, N))
    return charge * np.arctan2(ys - y0, xs - x0)


def winding_at_slice(alpha, z, x0, y0, radius):
    """Compute winding number of alpha[:,:,z] around (x0, y0)."""
    n_pts = 48
    thetas = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    samples = []
    N = alpha.shape[0]
    for th in thetas:
        i = int(round(x0 + radius * np.cos(th)))
        j = int(round(y0 + radius * np.sin(th)))
        if 0 <= i < N and 0 <= j < N:
            samples.append(alpha[i, j, z])
    samples = np.array(samples)
    diffs = np.diff(samples)
    diffs = np.mod(diffs + np.pi, 2*np.pi) - np.pi
    last = np.mod(samples[0] - samples[-1] + np.pi, 2*np.pi) - np.pi
    return float((diffs.sum() + last) / (2*np.pi))


def test_vortex_line_3d():
    print("\n--- Test 2 (3D): vortex line stability ---")
    N = 30
    x0 = y0 = N // 2
    alpha = make_vortex_line(N, x0, y0, charge=+1)

    # Initial windings at three different z slices
    w_initial = [winding_at_slice(alpha, z, x0, y0, 7) for z in [5, 15, 25]]
    print(f"  Initial windings at z=5,15,25: {[f'{w:.4f}' for w in w_initial]}")

    n_steps = 300
    for t in range(n_steps):
        alpha = step3d(alpha)

    w_final = [winding_at_slice(alpha, z, x0, y0, 7) for z in [5, 15, 25]]
    print(f"  Final windings (after {n_steps} steps): {[f'{w:.4f}' for w in w_final]}")
    residuals = [abs(wf - wi) for wf, wi in zip(w_final, w_initial)]
    print(f"  Residuals: {[f'{r:.2e}' for r in residuals]}")
    return {
        "n_steps":           n_steps,
        "initial_windings":  w_initial,
        "final_windings":    w_final,
        "max_residual":      float(max(residuals)),
    }


# --------------------- Test 3: parallel vortex-antivortex lines
def make_pair_lines(N, x1, x2, y0, c1=+1, c2=-1):
    xs, ys, zs = np.indices((N, N, N))
    a = c1 * np.arctan2(ys - y0, xs - x1) + c2 * np.arctan2(ys - y0, xs - x2)
    return a


def test_pair_lines_3d():
    print("\n--- Test 3 (3D): parallel vortex-antivortex lines ---")
    N = 36
    y0 = N // 2
    distances = [4, 6, 8, 10, 14, 18, 22]
    energies_per_length = []
    for d in distances:
        x1 = N // 2 - d // 2
        x2 = N // 2 + d // 2
        alpha = make_pair_lines(N, x1, x2, y0)
        for _ in range(80):
            alpha = step3d(alpha)
        E = total_energy_3d(alpha)
        E_per_L = E / N
        energies_per_length.append((d, E_per_L))
        print(f"  d={d}: E/L = {E_per_L:.4f}")

    ds = np.array([d for d,_ in energies_per_length])
    es = np.array([e for _,e in energies_per_length])
    log_d = np.log(ds)
    A_fit, B_fit = np.polyfit(log_d, es, 1)
    rss = float(np.sum((A_fit*log_d + B_fit - es)**2))
    print(f"  Fit: E/L = {A_fit:.3f} * ln(d) + {B_fit:.3f}")
    print(f"  RSS: {rss:.2e}")

    return {
        "distances":         [int(d) for d in ds],
        "energy_per_length": [float(e) for e in es],
        "log_fit_slope":     float(A_fit),
        "log_fit_intercept": float(B_fit),
        "rss":               rss,
    }


# ------------------- main
if __name__ == "__main__":
    print("Paper VI — 3D oriented sector tests")
    print("=" * 60)
    results = {}
    results["wave_3d"]   = test_wave_3d()
    results["vortex_3d"] = test_vortex_line_3d()
    results["pair_3d"]   = test_pair_lines_3d()

    out_path = DATA_DIR / "em_3d_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Wave 3D
    ax = axes[0]
    h = results["wave_3d"]["history"]
    if h:
        ts = [x["t"] for x in h]
        rs = [x["r"] for x in h]
        ax.plot(ts, rs, "o", markersize=3, label="3D wavefront")
        c = results["wave_3d"]["c_predicted"]
        v = results["wave_3d"]["v_measured"]
        ts_th = np.linspace(0, max(ts), 100)
        ax.plot(ts_th, c*ts_th, "k--", lw=1, label=fr"$c={c:.2f}$")
        if v is not None:
            ax.plot(ts_th, v*ts_th, "r:", lw=1, label=fr"fit $v={v:.2f}$")
    ax.set_xlabel("t")
    ax.set_ylabel("wavefront radius")
    ax.set_title("Test 1 (3D): wave propagation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Vortex line
    ax = axes[1]
    zs = [5, 15, 25]
    wi = results["vortex_3d"]["initial_windings"]
    wf = results["vortex_3d"]["final_windings"]
    ax.plot(zs, wi, "o", markersize=10, label="initial")
    ax.plot(zs, wf, "x", markersize=12, label=f"after {results['vortex_3d']['n_steps']} steps")
    ax.axhline(1.0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("z slice")
    ax.set_ylabel("winding number")
    ax.set_ylim(0.5, 1.5)
    ax.set_title("Test 2 (3D): vortex line stability\n"
                 fr"max residual = {results['vortex_3d']['max_residual']:.2e}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pair lines
    ax = axes[2]
    ds = np.array(results["pair_3d"]["distances"])
    es = np.array(results["pair_3d"]["energy_per_length"])
    ax.semilogx(ds, es, "o", markersize=8, color="C3")
    A = results["pair_3d"]["log_fit_slope"]
    B = results["pair_3d"]["log_fit_intercept"]
    ds_th = np.logspace(np.log10(min(ds)), np.log10(max(ds)), 50)
    ax.semilogx(ds_th, A*np.log(ds_th) + B, "k--", lw=1,
                label=fr"$E/L = {A:.2f}\ln d + {B:.2f}$")
    ax.set_xlabel(r"$d_\perp$")
    ax.set_ylabel("E / L (energy per unit length)")
    ax.set_title("Test 3 (3D): parallel vortex lines")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Paper VI — 3D validation of the oriented sector", fontsize=12, y=1.02)
    fig.tight_layout()
    fig_path = FIG_DIR / "fig02_em_3d.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf",".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
