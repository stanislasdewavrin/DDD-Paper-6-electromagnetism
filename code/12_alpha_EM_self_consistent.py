"""
Paper VI — self-consistent derivation of alpha_EM.

Key insight (this paper): a Wilson loop around a body diagonal of the
DDD lattice is bandwidth-modulated by its own EM coupling. The
back-reaction self-consistently shifts the effective length squared
of the diagonal.

Tree-level (perimeter law):
    delta_1 = alpha · perimeter = alpha · sqrt(3)

One-loop (volume integral over effective dimension):
    delta_2 = -alpha^2 · sqrt(3) · V_{d_eff}
where V_d = pi^(d/2)/Gamma(d/2+1) is the volume of unit ball in d
dimensions, and d_eff = 3 + 1/(2*pi) is the substrate's effective
dimension (Paper VI Sec. on Weyl substrate).

Combining:
    delta = alpha · sqrt(3) · (1 - alpha · V_{d_eff})

Self-consistent equation:
    x = 3 + delta(alpha(x))
    alpha(x) = 1/(8*pi^2 * sqrt(x))

Solving the fixed point gives:
    1/alpha* = 137.03596
    CODATA = 137.03600
    Match at 0.38 ppm.

This script verifies the fixed point computation and reports the
decomposition.
"""
import json
import numpy as np
from scipy.optimize import brentq
from math import gamma
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi
ALPHA_CODATA = 7.2973525693e-3
INV_ALPHA_CODATA = 137.035999084
D_EFF = 3 + 1/(2*PI)


def V_unit_ball(d):
    """Volume of unit ball in d dimensions."""
    return PI**(d/2) / gamma(d/2 + 1)


def delta(alpha, c2):
    """Back-reaction shift. delta = alpha*sqrt(3)*(1 - alpha*c2)."""
    return alpha * np.sqrt(3) * (1 - alpha * c2)


def find_fixed_point(c2):
    """Solve x = 3 + delta(alpha(x), c2) where alpha(x) = 1/(8*pi^2*sqrt(x))."""
    def f(x):
        a = 1.0/(8*PI**2*np.sqrt(x))
        return x - 3 - delta(a, c2)
    return brentq(f, 2.5, 3.5, xtol=1e-15)


def main():
    print("="*65)
    print("Paper VI: self-consistent fine-structure constant")
    print("="*65)
    print()
    V_3 = 4*PI/3
    V_de = V_unit_ball(D_EFF)
    print(f"Substrate parameters:")
    print(f"  d_eff   = 3 + 1/(2π)   = {D_EFF:.10f}")
    print(f"  V_3     = 4π/3         = {V_3:.10f}")
    print(f"  V_d_eff               = {V_de:.10f}")
    print()

    print("Three levels of approximation:")
    print()

    # Tree-level only
    x_tree = find_fixed_point(0.0)
    a_tree = 1.0/(8*PI**2*np.sqrt(x_tree))
    print(f"(1) Tree-level only (no back-reaction):")
    print(f"      δ = α·√3")
    print(f"      x* = {x_tree:.10f}")
    print(f"      1/α* = {1/a_tree:.6f}")
    print(f"      CODATA = {INV_ALPHA_CODATA:.6f}")
    print(f"      Δ = {1e6*(1/a_tree - INV_ALPHA_CODATA)/INV_ALPHA_CODATA:+.2f} ppm")
    print()

    # One-loop with V_3 (Euclidean)
    x_v3 = find_fixed_point(V_3)
    a_v3 = 1.0/(8*PI**2*np.sqrt(x_v3))
    print(f"(2) One-loop with Euclidean V_3 = 4π/3:")
    print(f"      δ = α·√3·(1 − α·4π/3)")
    print(f"      x* = {x_v3:.10f}")
    print(f"      1/α* = {1/a_v3:.6f}")
    print(f"      Δ = {1e6*(1/a_v3 - INV_ALPHA_CODATA)/INV_ALPHA_CODATA:+.2f} ppm")
    print()

    # One-loop with V_d_eff (fractional dimension)
    x_de = find_fixed_point(V_de)
    a_de = 1.0/(8*PI**2*np.sqrt(x_de))
    print(f"(3) One-loop with V_d_eff (fractional dimension):")
    print(f"      δ = α·√3·(1 − α·V_d_eff)")
    print(f"      x* = {x_de:.10f}")
    print(f"      1/α* = {1/a_de:.10f}")
    print(f"      CODATA = {INV_ALPHA_CODATA:.10f}")
    print(f"      Δ = {1e6*(1/a_de - INV_ALPHA_CODATA)/INV_ALPHA_CODATA:+.4f} ppm")
    print(f"        = {1e9*(1/a_de - INV_ALPHA_CODATA)/INV_ALPHA_CODATA:+.2f} ppb")
    print()

    # Decomposition at the V_d_eff fixed point
    a = a_de
    d1 = a*np.sqrt(3)
    d2 = -a**2*V_de*np.sqrt(3)
    print(f"Decomposition at V_d_eff fixed point:")
    print(f"  α* = {a:.14e}")
    print(f"  δ₁ = α*·√3              = {d1:.10e}")
    print(f"  δ₂ = −α*²·√3·V_d_eff    = {d2:.10e}")
    print(f"  δ_tot                    = {d1+d2:.10e}")
    print(f"  x* − 3                   = {x_de - 3:.10e}")
    print(f"  |δ₂/δ₁| = α·V_d_eff     = {abs(d2/d1):.6f}")
    print()

    # Save result
    out = {
        "constants": {
            "alpha_EM_CODATA": ALPHA_CODATA,
            "inv_alpha_CODATA": INV_ALPHA_CODATA,
            "d_eff": D_EFF,
            "V_3 (Euclidean)": V_3,
            "V_d_eff (fractional)": V_de,
        },
        "results": {
            "tree_level": {
                "x_star": x_tree, "alpha_star": a_tree,
                "inv_alpha": 1/a_tree, "delta_ppm": 1e6*(1/a_tree-INV_ALPHA_CODATA)/INV_ALPHA_CODATA},
            "one_loop_V3": {
                "x_star": x_v3, "alpha_star": a_v3,
                "inv_alpha": 1/a_v3, "delta_ppm": 1e6*(1/a_v3-INV_ALPHA_CODATA)/INV_ALPHA_CODATA},
            "one_loop_V_d_eff": {
                "x_star": x_de, "alpha_star": a_de,
                "inv_alpha": 1/a_de, "delta_ppm": 1e6*(1/a_de-INV_ALPHA_CODATA)/INV_ALPHA_CODATA},
        },
    }
    with open(DATA / "12_alpha_EM_self_consistent.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {DATA / '12_alpha_EM_self_consistent.json'}")


if __name__ == "__main__":
    main()
