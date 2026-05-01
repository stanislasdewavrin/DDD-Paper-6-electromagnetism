"""
Paper VI - test multiple forms of the back-reaction.

In abelian U(1) gauge theory, the Wilson loop exponentiates:
  <W(C)> = exp(-alpha * I(C))
where I(C) is the perimeter integral of the photon propagator.

This means the back-reaction has a specific functional form, not just a
truncated polynomial. We test:

  Form 1: Truncated polynomial (paper's current)
          delta = a*sqrt(3) * (1 - a*V)

  Form 2: Geometric Dyson
          delta = a*sqrt(3) / (1 + a*V)

  Form 3: Abelian exponentiation
          delta = a*sqrt(3) * exp(-a*V)

  Form 4: Truncated 2-loop with V2 = c*V1 (calibrated coefficient)
          delta = a*sqrt(3) * (1 - a*V + (a*V)^2 * c)
"""
import math
from math import gamma, pi
import scipy.optimize as opt
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

CODATA = 137.0359991
PI = pi

def V_of_d(d):
    return PI**(d/2) / gamma(d/2 + 1)


def alpha_at_x(x):
    return 1.0 / (8 * PI**2 * math.sqrt(x))


def fixedpoint(delta_func, V):
    """Find x* such that x* = 3 + delta(alpha(x*), V)."""
    def fn(x):
        a = alpha_at_x(x)
        return x - 3 - delta_func(a, V)
    sol = opt.brentq(fn, 3.0, 3.1)
    return sol, 1.0 / alpha_at_x(sol)


# Form 1: truncated polynomial
def delta_poly(a, V):
    return a * math.sqrt(3) * (1 - a * V)


# Form 2: geometric Dyson resummation
def delta_geom(a, V):
    return a * math.sqrt(3) / (1 + a * V)


# Form 3: abelian exponentiation
def delta_exp(a, V):
    return a * math.sqrt(3) * math.exp(-a * V)


# Form 4: truncated with V2 coefficient
def make_delta_2loop(c):
    def f(a, V):
        return a*math.sqrt(3) * (1 - a*V + (a*V)**2 * c)
    return f


def main():
    print("Comparing forms of back-reaction self-consistency.")
    print(f"Target CODATA: 1/alpha = {CODATA}")
    print()

    results = {}
    for d_label, d_eff in [("3 + 1/(2pi)", 3 + 1/(2*PI)),
                            ("3 + 1/6", 3 + 1/6),
                            ("3 (Euclidean)", 3.0)]:
        V = V_of_d(d_eff)
        print(f"=== d_eff = {d_label}, V = {V:.5f} ===")
        for label, dfn in [("Poly truncated", delta_poly),
                           ("Geometric Dyson", delta_geom),
                           ("Abelian exp", delta_exp)]:
            x, inv_a = fixedpoint(dfn, V)
            ppm = (inv_a - CODATA) / CODATA * 1e6
            print(f"  {label:18s}: 1/alpha = {inv_a:.7f}, ppm = {ppm:+.3f}")
        # Find the c coefficient that gives 0 ppm for this V
        def err(c):
            x, inv_a = fixedpoint(make_delta_2loop(c), V)
            return inv_a - CODATA
        c_opt = opt.brentq(err, -5, 5)
        x, inv_a = fixedpoint(make_delta_2loop(c_opt), V)
        ppm = (inv_a - CODATA) / CODATA * 1e6
        print(f"  Best 2-loop coef c = {c_opt:.5f}: ppm = {ppm:+.5f}")
        # Some structural candidate values for c
        for c_label, c in [("c = 0", 0), ("c = 1/2", 0.5), ("c = 1", 1),
                            ("c = -1/2", -0.5)]:
            x, inv_a = fixedpoint(make_delta_2loop(c), V)
            ppm = (inv_a - CODATA) / CODATA * 1e6
            print(f"  Truncated, c={c_label:8s}: ppm = {ppm:+.4f}")
        print()
        results[d_label] = {
            "V": V,
            "c_opt_for_zero_ppm": c_opt,
        }

    # Most striking: for d_eff = 3 + 1/6, what's c_opt vs 1/2?
    # And does (1/2) have any structural meaning (Taylor series exp expansion)?
    print()
    print("Key observation: in the abelian exponentiation,")
    print("  exp(-aV) = 1 - aV + (aV)^2/2 - ...")
    print("So c = +1/2 corresponds exactly to the exp series at 2nd order.")
    print()
    print("Comparing 2-loop coef vs 1/2 (true exp series):")
    for d_label in results:
        V = results[d_label]["V"]
        c_opt = results[d_label]["c_opt_for_zero_ppm"]
        print(f"  {d_label}: c_opt = {c_opt:.4f}, 1/2 = 0.5, diff = {c_opt - 0.5:+.4f}")

    # Save
    out = {"target": CODATA, "results": {k: {"V": v["V"],
                                              "c_opt": v["c_opt_for_zero_ppm"]}
                                          for k, v in results.items()}}
    with open(DATA / "35_two_loop_forms.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Saved.")


if __name__ == "__main__":
    main()
