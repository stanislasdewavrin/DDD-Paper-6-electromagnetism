"""
Paper VI — Angle A: systematic test of structural identifications for J.

Hypothesis: alpha_EM = (kappa/J) * F_struct, where F_struct is a structural
factor (1/(4π·C), 1/S(d_eff), b_tot/(2π·something), etc.).

Under Paper II Planck constraint kappa/(alpha_D R_0) = 4π, and with
R_0 = alpha_D = 1 normalization: kappa = 4π.

Free parameter: J. Free structure: F_struct.

We try various combinations and check if any matches CODATA
alpha_EM = 7.2974e-3 = 1/137.036 cleanly.

Structural quantities available:
  λ = 1/(2π) (twist density)
  T = 2 (Floquet period)
  m = λ/T = 1/(4π) (gap mass at Weyl point)
  c_eff = 1/T = 1/2 (effective speed of light)
  b_tot = π (axion vector magnitude, from 4 Weyl points at ±π/2)
  C = 1 (Chern number of central slice)
  d_eff = 3 + 1/(2π) (effective dimension)
  S(d) = 2π^(d/2) / Γ(d/2) (surface of unit (d-1)-sphere)
"""
import numpy as np
from math import gamma

PI = np.pi
ALPHA_EM_CODATA = 7.2973525693e-3  # CODATA 2018
INV_ALPHA = 137.036

# Substrate structural numbers
LAMBDA = 1.0 / (2 * PI)
T_PERIOD = 2.0
m_gap = LAMBDA / T_PERIOD
c_eff = 1.0 / T_PERIOD
b_tot = PI
C_chern = 1
d_eff = 3 + 1 / (2 * PI)


def S(d):
    """Surface of unit (d-1)-sphere in d dimensions."""
    return 2 * PI**(d/2) / gamma(d/2)


# Planck-anchored: kappa = 4π * alpha_D * R_0 with alpha_D = R_0 = 1
KAPPA = 4 * PI

# Test candidates for J
J_candidates = [
    ("J = λ = 1/(2π)",                   LAMBDA),
    ("J = m = 1/(4π)",                    m_gap),
    ("J = c_eff = 1/2",                   c_eff),
    ("J = T = 2",                         T_PERIOD),
    ("J = 1",                             1.0),
    ("J = 2",                             2.0),
    ("J = 4π",                            4*PI),
    ("J = 8π² = 2T·(2π)",                 8*PI**2),
    ("J = 16π² = (4π)²",                  16*PI**2),
    ("J = 137 (manual = 1/α_EM)",         137.0),
    ("J = π",                             PI),
    ("J = 2π = 1/λ",                      2*PI),
    ("J = T·(2π) = 4π",                   T_PERIOD*2*PI),
    ("J = T·b_tot = 2π",                  T_PERIOD*b_tot),
    ("J = 4π·(2π) = 8π²",                 4*PI*2*PI),
    ("J = (4π)² · (1/(2π)) = 8π",         16*PI**2 * 1/(2*PI)),
]

# Test candidates for F_struct (the structural factor multiplying kappa/J)
F_candidates = [
    ("F = 1/(4π·C)",                  1.0 / (4 * PI * C_chern)),
    ("F = 1/(8π)",                    1.0 / (8 * PI)),
    ("F = 1/(4π²)",                   1.0 / (4 * PI**2)),
    ("F = 1/S(d_eff)",                1.0 / S(d_eff)),
    ("F = b_tot/(2π·4π)",             b_tot / (2 * PI * 4 * PI)),
    ("F = 1/(2π·b_tot)",              1.0 / (2 * PI * b_tot)),
    ("F = (b_tot/π) / (4π)",          (b_tot/PI) / (4*PI)),
    ("F = 1",                         1.0),
    ("F = 1/(4π)·(b_tot/π) = 1/(4π)", 1.0/(4*PI) * (b_tot/PI)),
]

print(f"CODATA alpha_EM = {ALPHA_EM_CODATA:.6e} = 1/{INV_ALPHA:.3f}")
print(f"Substrate: kappa = 4π·α_D·R_0 = {KAPPA:.4f} (Planck-anchored, α_D=R_0=1)")
print()
print(f"S(d_eff = 3 + 1/(2π) = {d_eff:.4f}) = {S(d_eff):.4f}")
print(f"S(3) = {S(3.0):.4f} = 4π")
print()

# Compute alpha_EM for all combinations
print(f"{'J candidate':>40} {'kappa/J':>12} | {'F candidate':>40} | {'alpha_EM':>12} {'inv':>10} {'%CODATA':>10}")
print("-" * 150)

results = []
best_match = None
best_diff = 1e10
for J_label, J_val in J_candidates:
    if J_val == 0: continue
    kJ = KAPPA / J_val
    for F_label, F_val in F_candidates:
        alpha_pred = kJ * F_val
        if alpha_pred <= 0: continue
        inv_alpha_pred = 1.0 / alpha_pred
        rel_diff = abs(alpha_pred - ALPHA_EM_CODATA) / ALPHA_EM_CODATA
        results.append((rel_diff, alpha_pred, J_label, F_label, J_val, F_val))
        if rel_diff < best_diff:
            best_diff = rel_diff
            best_match = (alpha_pred, J_label, F_label)

# Sort by closeness
results.sort()
print("Top 15 closest combinations to CODATA alpha_EM:")
print()
for rel_diff, alpha_pred, J_label, F_label, J_val, F_val in results[:15]:
    inv = 1.0 / alpha_pred
    pct = 100 * (alpha_pred / ALPHA_EM_CODATA - 1)
    print(f"  {J_label:>32} | {F_label:>32} | α = {alpha_pred:.5e} = 1/{inv:6.3f}, {pct:+7.2f}%")

print()
print(f"Best match: alpha = {best_match[0]:.5e} = 1/{1/best_match[0]:.3f}")
print(f"  via J = {best_match[1]}")
print(f"  and F = {best_match[2]}")
print(f"  rel_diff = {best_diff*100:.3f}%")
