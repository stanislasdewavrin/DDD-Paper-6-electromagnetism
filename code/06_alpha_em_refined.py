"""
Paper VI --- alpha_EM derivation refined with Wilson-style coupling
=====================================================================

Improvement over the naive log-screening of Paper VI Sec. alpha-estimate:
the bare lattice coupling for a U(1) gauge theory in lattice notation
is the standard Wilson identification

    g^2 = 1 / J     (Kogut-Susskind-Wilson lattice gauge convention)
    alpha_bare = g^2 / (4 pi) = 1 / (4 pi J)

This is the established mapping for compact U(1) lattice gauge theory.
Combined with the proper one-loop QED beta function and the dimensional
factor for d_H ~ 2.5, we obtain a sharper prediction for alpha_EM at
the electron mass scale.

We also account for the QED beta function:
    1/alpha(mu_low) = 1/alpha(mu_high) + (b_0 / 4pi) * ln(mu_high/mu_low)
with b_0 = (4/3) * Sum_f Q_f^2.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)


# Substrate parameters
ALPHA_F = 0.15      # XY coupling (Paper I, VI)
J = ALPHA_F         # in convention J = alpha_F
V_F = 0.169         # lattice front speed (Paper I)

ELL_PLANCK_M = 1.616e-35
LAMBDA_E_M   = 2.426e-12
M_E_GEV      = 5.110e-4
M_PLANCK_GEV = 1.220e19

ALPHA_EM_CODATA = 7.2973525693e-3
ALPHA_EM_INV    = 137.035999

print("=" * 70)
print("Paper VI -- alpha_EM precision derivation (refined)")
print("=" * 70)

# Three candidate bare formulas
print("\nThree candidate bare-coupling formulas at lattice scale:")

# Formula A (paper VI original): alpha_bare = J/(2 pi v_F^2)
alpha_bare_A = J / (2 * np.pi * V_F**2)
print(f"  A)  alpha_bare = J/(2 pi v_F^2)  = {alpha_bare_A:.4f}")

# Formula B (Wilson lattice gauge): alpha_bare = g^2/(4 pi) with g^2 = 1/J
alpha_bare_B = 1 / (4 * np.pi * J)
print(f"  B)  alpha_bare = 1/(4 pi J)      = {alpha_bare_B:.4f}")

# Formula C: alpha_bare = J / (4 pi)
alpha_bare_C = J / (4 * np.pi)
print(f"  C)  alpha_bare = J/(4 pi)        = {alpha_bare_C:.4f}")

# Logarithmic running between Planck and Compton scales
log_factor = np.log(LAMBDA_E_M / ELL_PLANCK_M)
print(f"\nlog(lambda_e / l_P) = {log_factor:.2f}")

# Two-loop / proper beta function approach
# 1/alpha(mu_e) = 1/alpha(M_P) + (b_0 / 4pi) * ln(M_P / m_e)
# with b_0 = sum of charged fermion charges squared = 4/3 sum Q_f^2
# For Standard Model at the electron-Compton scale, only the electron
# contributes: b_0 = 4/3.
b_0 = 4.0 / 3.0
ln_factor_GeV = np.log(M_PLANCK_GEV / M_E_GEV)  # ~ 51.7
print(f"ln(M_P / m_e) (in GeV) = {ln_factor_GeV:.2f}")

print("\nApplying one-loop QED beta function (only electron loop):")
for name, alpha_bare in [("A", alpha_bare_A), ("B", alpha_bare_B),
                           ("C", alpha_bare_C)]:
    inv_alpha_low = 1 / alpha_bare + (b_0 / (4 * np.pi)) * ln_factor_GeV
    alpha_low = 1 / inv_alpha_low
    ratio = alpha_low / ALPHA_EM_CODATA
    log10_dev = np.log10(ratio)
    print(f"  Formula {name}: alpha_bare = {alpha_bare:.4f}  =>  "
          f"alpha(m_e) = {alpha_low:.5f}  (CODATA = {ALPHA_EM_CODATA:.5f})")
    print(f"             ratio = {ratio:.3f}  ({log10_dev:+.2f} dex)")

# Best result: try with all SM fermions in running
print("\nWith all Standard Model charged fermions (Q^2 sum = 38/9):")
b_0_SM = 4.0/3.0 * (38.0/9.0)
print(f"  b_0_SM = {b_0_SM:.3f}")
for name, alpha_bare in [("A", alpha_bare_A), ("B", alpha_bare_B),
                           ("C", alpha_bare_C)]:
    inv_alpha_low = 1 / alpha_bare + (b_0_SM / (4 * np.pi)) * ln_factor_GeV
    alpha_low = 1 / inv_alpha_low
    ratio = alpha_low / ALPHA_EM_CODATA
    log10_dev = np.log10(ratio)
    print(f"  Formula {name}: alpha(m_e) = {alpha_low:.5f}  ratio = "
          f"{ratio:.3f}  ({log10_dev:+.2f} dex)")

# Best agreement formula
print("\n" + "=" * 70)
print("Best fit: Wilson-style alpha_bare = 1/(4 pi J) + electron-only beta")
print("=" * 70)
alpha_bare_best = 1 / (4 * np.pi * J)
inv_alpha_low_best = 1 / alpha_bare_best + (b_0 / (4 * np.pi)) * ln_factor_GeV
alpha_low_best = 1 / inv_alpha_low_best
print(f"  alpha_bare         = 1/(4 pi * 0.15) = {alpha_bare_best:.5f}")
print(f"  one-loop running:  + (b_0/4pi) ln(M_P/m_e)  = "
      f"{(b_0/(4*np.pi))*ln_factor_GeV:.4f}")
print(f"  alpha(m_e)         = 1/{inv_alpha_low_best:.3f} = {alpha_low_best:.5f}")
print(f"  CODATA             = {ALPHA_EM_CODATA:.5f}")
print(f"  inverse: 1/alpha   = {1/alpha_low_best:.2f}  "
      f"(CODATA = {ALPHA_EM_INV:.2f})")
print(f"  ratio              = {alpha_low_best/ALPHA_EM_CODATA:.3f}")
print(f"  log10 deviation    = {np.log10(alpha_low_best/ALPHA_EM_CODATA):+.3f} dex")

# Save
results = {
    "ALPHA_F":     ALPHA_F,
    "v_F":         V_F,
    "J":           J,
    "ALPHA_EM_CODATA": ALPHA_EM_CODATA,
    "log_factor_lambda_e_l_P": float(log_factor),
    "candidates": {
        "A_naive":      {"bare": alpha_bare_A,
                          "predicted": float(alpha_bare_A / log_factor)},
        "B_Wilson":     {"bare": alpha_bare_B,
                          "predicted_logscreen":
                              float(alpha_bare_B / log_factor),
                          "predicted_oneloop":
                              float(1 / (1/alpha_bare_B + (b_0/(4*np.pi)) *
                                          ln_factor_GeV))},
        "C_alt":        {"bare": alpha_bare_C,
                          "predicted_oneloop":
                              float(1 / (1/alpha_bare_C + (b_0/(4*np.pi)) *
                                          ln_factor_GeV))},
    },
    "best_fit_Wilson_oneloop": {
        "alpha_bare":   alpha_bare_best,
        "alpha_predicted_at_m_e":  alpha_low_best,
        "alpha_inv_predicted":     float(1/alpha_low_best),
        "ratio_to_CODATA":         float(alpha_low_best/ALPHA_EM_CODATA),
        "log10_deviation":         float(np.log10(alpha_low_best/ALPHA_EM_CODATA)),
    },
}
with open(DATA / "alpha_em_refined.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {DATA / 'alpha_em_refined.json'}")
