"""
Paper XVII — Attempt at deriving alpha_EM from substrate parameters
=======================================================================

The fine-structure constant alpha_EM = e^2 / (4 pi epsilon_0 hbar c)
~ 1/137.036 has the dimensionless value
    alpha_EM = 7.2973525693e-3

Question: can we get this number out of the local rule?

In DDD, the electromagnetic coupling sits in the oriented sector.
The XY/Kuramoto term J cos(psi_i - psi_j) gives the photon-vortex
interaction strength. We have:

  alpha_EM_attempt = J / (2 pi v_F^2)
where v_F is the Fermi-like velocity (the front speed measured in
Paper XIV) and J = ALPHA_F is the XY coupling.

Numerical inputs:
  ALPHA_F = 0.15
  v_F     = 0.169 (Paper XIV measurement)
  alpha_EM_naive = 0.15 / (2 pi * 0.169^2) = 0.835

This is two orders of magnitude away from 1/137. We must therefore
include the screening from the random graph: in d_H ~ 2.5, the
effective coupling at distance r runs as J(r) ~ J / log(r/a).
At the Compton scale of the electron, r/a ~ 1/(m_e a) where
m_e a in lattice units = some small number.

For the Compton wavelength of the electron, lambda_e = h/(m_e c)
= 2.426e-12 m. At the Planck scale a = 1.616e-35 m, the ratio
lambda_e / a = 1.5e23, so log(lambda_e/a) ~ 53.5.

  alpha_EM_screened = 0.835 / 53.5 = 0.0156
  ratio to truth     = 0.0156 / 0.0073 = 2.14

We are now at factor ~2 of the right value. This is an
order-of-magnitude / structural success, not a precise derivation,
but it is the best DDD can do without an explicit two-loop
calculation of the running coupling.

Output:
    data/alpha_attempt.json
    figures/fig01_alpha_running.pdf
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)

ALPHA_F = 0.15
V_F     = 0.169  # cells/tick from Paper XIV
ALPHA_EM_TRUE = 7.2973525693e-3

ELL_PLANCK = 1.616e-35  # m
LAMBDA_E   = 2.426e-12  # m, electron Compton wavelength

print("Paper XVII -- alpha_EM derivation attempt")
print("=" * 60)

# Step 1: bare coupling from substrate parameters
alpha_naive = ALPHA_F / (2 * np.pi * V_F ** 2)
print(f"  Bare coupling J/(2 pi v_F^2) = {alpha_naive:.4f}")

# Step 2: log-screening at the electron Compton scale
log_factor = np.log(LAMBDA_E / ELL_PLANCK)
print(f"  log(lambda_e / l_P) = {log_factor:.2f}")

alpha_screened = alpha_naive / log_factor
print(f"  alpha_EM after log-screening = {alpha_screened:.5f}")
print(f"  alpha_EM (CODATA)            = {ALPHA_EM_TRUE:.5f}")
print(f"  ratio                        = {alpha_screened / ALPHA_EM_TRUE:.2f}")
print(f"  log10 deviation              = {np.log10(alpha_screened / ALPHA_EM_TRUE):.2f}")

# Step 3: running coupling alpha(r/a) over many decades
ratios_log = np.linspace(0, 60, 200)
ratios = 10 ** ratios_log
alpha_run = alpha_naive / np.maximum(np.log(ratios), 1.0)

# Save
results = {
    "ALPHA_F":            ALPHA_F,
    "v_F":                V_F,
    "alpha_EM_true":      ALPHA_EM_TRUE,
    "alpha_naive":        float(alpha_naive),
    "log_factor":         float(log_factor),
    "alpha_screened":     float(alpha_screened),
    "ratio_to_truth":     float(alpha_screened / ALPHA_EM_TRUE),
    "log10_deviation":    float(np.log10(alpha_screened / ALPHA_EM_TRUE)),
    "verdict":            "Order-of-magnitude; not a precision derivation. "
                          "Factor ~2 within the structural estimate.",
}
with open(DATA / "alpha_attempt.json", "w") as f:
    json.dump(results, f, indent=2)

# Figure
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.semilogx(ratios, alpha_run, "b-", lw=2,
            label=r"$\alpha_{\rm EM}(r) = J / (2\pi v_F^2 \log(r/a))$")
ax.axhline(ALPHA_EM_TRUE, color="red", ls="--", lw=1.2,
           label=fr"$\alpha_{{\rm EM}} = 1/137 = {ALPHA_EM_TRUE:.4f}$")
ax.axvline(LAMBDA_E / ELL_PLANCK, color="green", ls=":", lw=1.2,
           label=r"$\lambda_e/\ell_P$")
ax.scatter([LAMBDA_E / ELL_PLANCK], [alpha_screened], color="black",
           s=80, marker="*", zorder=5,
           label=f"DDD estimate = {alpha_screened:.4f}")
ax.set_xlabel(r"distance ratio $r / \ell_P$")
ax.set_ylabel(r"effective coupling $\alpha(r)$")
ax.set_title("DDD running coupling vs CODATA $\\alpha_{\\rm EM}$\n"
             f"Estimate: $\\alpha_{{\\rm DDD}} = {alpha_screened:.4f}$, "
             f"factor {alpha_screened/ALPHA_EM_TRUE:.2f} of truth")
ax.legend(fontsize=9, loc="upper right")
ax.set_yscale("log")
ax.set_ylim(1e-3, 2)
ax.grid(True, alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(FIG / "fig01_alpha_running.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig01_alpha_running.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: {FIG / 'fig01_alpha_running.pdf'}")
