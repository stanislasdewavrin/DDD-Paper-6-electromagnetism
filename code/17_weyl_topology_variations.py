"""
Paper VI — explore Weyl topology variations for cleanest α_EM match.

For each Weyl topology variation, the "structural one-loop coefficient"
might be different. Test:
  - Number of Weyl pairs (1, 2, 4, 8)
  - Magnitude of axion vector (π, 2π, π/2)
  - Multi-Weyl charges (Chern ±1, ±2, ±3)
  - Effective dimensions consistent with each topology

For each, solve the self-consistent equation:
  α_EM = 1/(N_geom · √x)
  x = 3 + α_EM · L_diag · (1 - α_EM · K)

with N_geom, L_diag, K depending on topology.

Find which topology gives clean structural numbers AND match to CODATA.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

PI = np.pi
ALPHA_CODATA = 7.2973525693e-3
INV_ALPHA_CODATA = 137.035999084


def find_fixed_point(N_geom, L_diag, K_coef):
    """Solve x = 3 + δ(α(x)) with α = 1/(N_geom·√x)."""
    def f(x):
        a = 1.0/(N_geom * np.sqrt(x))
        return x - 3 - a * L_diag * (1 - a * K_coef)
    try:
        return brentq(f, 2.5, 3.5, xtol=1e-15)
    except ValueError:
        return float('nan')


def V_d(d):
    return PI**(d/2) / np.exp(gammaln(d/2 + 1))


# ---- DDD baseline ----
print(f"{'='*80}")
print(f"DDD baseline (4 Weyl points, 2 pairs, |b| = π, single-charge):")
print(f"  N_geom = 8π² = 4 (body diagonals/cube) × 2π² (S^3 phase space)")
print(f"  L_diag = √3 (body diagonal length)")
print(f"  K_coef ∈ {{V_3, V_d_eff}} candidates")
print(f"{'='*80}\n")

baseline = (8*PI**2, np.sqrt(3))

candidates = [
    ("V_3 = 4π/3 (Euclidean)",                  4*PI/3),
    ("V_{d=3+1/(2π)}",                          V_d(3 + 1/(2*PI))),
    ("V_{d=3+1/π}",                             V_d(3 + 1/PI)),
    ("V_{d=3+2/(2π)} = V_{d=3+1/π}",            V_d(3 + 1/PI)),
    ("V_{d=3+1/(4π)}",                          V_d(3 + 1/(4*PI))),
    ("V_{d=π}",                                 V_d(PI)),
]
print(f"{'K coefficient':<35} {'value':>10} {'1/α*':>14} {'Δ ppm':>10}")
for label, K in candidates:
    x = find_fixed_point(*baseline, K)
    a = 1.0/(8*PI**2 * np.sqrt(x))
    inv = 1/a
    ppm = 1e6*(inv - INV_ALPHA_CODATA)/INV_ALPHA_CODATA
    print(f"{label:<35} {K:10.5f} {inv:14.6f} {ppm:+10.3f}")

print()

# ---- Topology variations ----
print(f"{'='*80}")
print("Topology variations: change N_geom (combinatorial factor):")
print(f"{'='*80}\n")

print(f"{'Topology':<45} {'N_geom':>12} {'L_diag':>8} {'1/α* (V_3)':>14} {'Δ ppm':>10}")

variations = [
    ("DDD: 4 BD/cube × 2π² (S³)",                4*2*PI**2, np.sqrt(3)),
    ("Var1: 2 BD/cube × 2π²",                    2*2*PI**2, np.sqrt(3)),
    ("Var2: 8 BD (Dirac)/cube × 2π²",            8*2*PI**2, np.sqrt(3)),
    ("Var3: 4 BD × 4π² (Vol(S^3) doubled)",     4*4*PI**2, np.sqrt(3)),
    ("Var4: 4 BD × 2π² with L = 2",              4*2*PI**2, 2.0),
    ("Var5: 4 FD × 2π² (face diag, L = √2)",    4*2*PI**2, np.sqrt(2)),
    ("Var6: 6 edges × 2π² (axis edge, L = 1)",   6*2*PI**2, 1.0),
]

for label, N, L in variations:
    x = find_fixed_point(N, L, 4*PI/3)
    if not np.isfinite(x): continue
    a = 1.0/(N * np.sqrt(x))
    inv = 1/a
    ppm = 1e6*(inv - INV_ALPHA_CODATA)/INV_ALPHA_CODATA
    print(f"{label:<45} {N:12.4f} {L:8.4f} {inv:14.6f} {ppm:+10.3f}")

print()

# ---- Different axion magnitudes ----
print(f"{'='*80}")
print("Axion magnitude variations (changes effective V_d_eff):")
print(f"{'='*80}\n")

axion_mags = [PI/2, PI, 3*PI/2, 2*PI, PI/4]
print(f"{'b_axion':<15} {'(1/(2π)) × b':>20} {'d_eff':>10} {'V_{d_eff}':>14} {'Δ ppm':>10}")
for b in axion_mags:
    deff = 3 + b/(2*PI)
    Vdeff = V_d(deff)
    x = find_fixed_point(8*PI**2, np.sqrt(3), Vdeff)
    a = 1.0/(8*PI**2 * np.sqrt(x))
    inv = 1/a
    ppm = 1e6*(inv - INV_ALPHA_CODATA)/INV_ALPHA_CODATA
    print(f"{b:<15.4f} {b/(2*PI):>20.4f} {deff:10.5f} {Vdeff:14.6f} {ppm:+10.3f}")

print()
print(f"CODATA: 1/α = {INV_ALPHA_CODATA:.6f}")
