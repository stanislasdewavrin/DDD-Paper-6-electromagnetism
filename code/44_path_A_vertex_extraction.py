#!/usr/bin/env python3
"""
Path A: extract the 2-loop vertex coefficient c_v from existing
lattice U(1) Wilson loop data.

Method
------
The Wilson loop expectation in compact U(1) lattice gauge theory at small
alpha = 1/beta admits the abelian-exponentiation expansion

  <W(C)> = exp(- alpha * L_eff * F(alpha))

with

  F(alpha) = c_0 + c_1 * alpha + c_2 * alpha^2 + O(alpha^3),

where L_eff is an effective perimeter (combinatorial constant).
For our compact U(1) on the cubic+body-diagonal substrate:

  c_0 -- tree-level coefficient (Coulomb-like, perimeter law)
  c_1 -- 1-loop self-energy ("bubble"); contains V_{d_eff}
  c_2 -- 2-loop, includes the vertex correction we want

We fit
  y(alpha) = -log<W>/(L_eff * alpha) = c_0 + c_1*alpha + c_2*alpha^2
in the small-alpha regime and read off c_2.

Comparison with prediction
--------------------------
For Paper VI's self-consistent fixed point with V_{d_eff}, exploration
(see /tmp/explore_paths.py) gives:
  c_v(needed) = -0.038 (eps = 1/(3*pi))
              = +0.043 (eps = 1/(2*pi))
              = +0.054 (eps = 1/6)
to close the residual exactly to CODATA.

This is the lattice-extracted 2-loop coefficient c_2 (relative to the
loop's natural normalization). Comparing the two answers tests whether
the residual ~0.4 ppm is indeed due to vertex 2-loop content.
"""
import json
import math
from math import log, sqrt, pi
from pathlib import Path
import numpy as np


HERE = Path(__file__).resolve().parent
data_path = HERE / "2loop_local.json"
out_path = HERE.parent / "data" / "44_path_A_vertex_extraction.json"
out_path.parent.mkdir(exist_ok=True)


def main():
    with open(data_path) as f:
        d = json.load(f)

    L = d["L"]
    print(f"Analysis of existing data: L={L}, n_meas={d['n_meas']}")
    print()

    # The loop in 43_*.py is a tetrahedral path: 3 cubic + 1 body-diagonal
    # Effective perimeter (squared length): 3 cubic edges of length 1
    # plus 1 body-diagonal of length sqrt(3); total Euclidean length is
    # 3 + sqrt(3) ~ 4.732. For perimeter-law fitting we use this.
    L_eff = 3.0 + sqrt(3.0)  # 4.732

    # Collect (alpha, W, W_se)
    rows = []
    for r in d["results"]:
        rows.append((r["alpha"], r["W_mean"], r["W_se"]))
    rows.sort()
    rows = np.array(rows)
    alpha_arr = rows[:, 0]
    W_arr = rows[:, 1]
    W_se = rows[:, 2]

    # Compute y = -log<W> / (L_eff * alpha)
    minus_logW = -np.log(W_arr)
    y = minus_logW / (L_eff * alpha_arr)
    y_se = W_se / W_arr / (L_eff * alpha_arr)  # error propagation

    print(f"{'alpha':>8} {'<W>':>10} {'-log<W>':>10} {'y':>10} {'y_se':>10}")
    for a, w, mlw, yv, ys in zip(alpha_arr, W_arr, minus_logW, y, y_se):
        print(f"{a:>8.4f} {w:>10.6f} {mlw:>10.6f} {yv:>10.6f} {ys:>10.2e}")

    # Fit y(alpha) = c_0 + c_1*alpha + c_2*alpha^2
    # Use only the small-alpha points for cleanest extraction
    masks = {
        "all 7 points":  np.ones(len(alpha_arr), dtype=bool),
        "alpha <= 0.1":  alpha_arr <= 0.10,
        "alpha <= 0.07": alpha_arr <= 0.07,
        "alpha <= 0.05": alpha_arr <= 0.05,
    }

    print()
    print(f"{'mask':<18} {'n':>3} {'c_0':>10} {'c_1':>10} {'c_2':>12}")
    fits = []
    for label, mask in masks.items():
        if mask.sum() < 3:
            print(f"{label:<18} (only {mask.sum()} points, skip)")
            continue
        a_sel = alpha_arr[mask]
        y_sel = y[mask]
        ye_sel = y_se[mask]
        # Weighted polynomial fit
        deg = 2
        coef = np.polyfit(a_sel, y_sel, deg, w=1.0/np.maximum(ye_sel, 1e-9))
        c2, c1, c0 = coef[0], coef[1], coef[2]
        print(f"{label:<18} {int(mask.sum()):>3d} {c0:>+10.5f} {c1:>+10.5f} {c2:>+12.5f}")
        fits.append({"mask": label, "n": int(mask.sum()),
                     "c0": float(c0), "c1": float(c1), "c2": float(c2)})

    # Compare to Path A predictions
    print()
    print("Predicted c_v (from /tmp/explore_paths.py) to close CODATA:")
    print("   eps = 1/(3*pi) -> c_v = -0.038")
    print("   eps = 1/(2*pi) -> c_v = +0.043")
    print("   eps = 1/6      -> c_v = +0.054")
    print()
    print("Note: the lattice c_2 above is in 'normalized loop' units.")
    print("Direct comparison requires translating between:")
    print("   c_2 (lattice, in y = -log<W>/(L_eff alpha))")
    print("   c_v (Paper VI fixed-point fudge term in x_*)")
    print("These differ by a combinatorial conversion factor specific to")
    print("the Wilson-loop topology and the choice of L_eff.")
    print()
    print("What we can read off directly:")
    print("  - sign of c_2 (should match sign of c_v)")
    print("  - magnitude (c_2 ~ O(c_v) up to combinatorial factor of order unity)")

    # Sign test
    print()
    print("=== SIGN TEST ===")
    for f in fits:
        sign_str = "POSITIVE" if f["c2"] > 0 else "NEGATIVE"
        print(f"  {f['mask']:<18} c_2 = {f['c2']:+.4f}   ({sign_str})")
        if f["c2"] > 0:
            print(f"     -> compatible with eps = 1/(2*pi) or eps = 1/6")
        else:
            print(f"     -> compatible with eps = 1/(3*pi)")

    # Magnitude
    print()
    print("=== MAGNITUDE COMPARISON ===")
    for f in fits:
        ratio_3pi = abs(f["c2"]) / 0.038
        ratio_2pi = abs(f["c2"]) / 0.043
        ratio_6   = abs(f["c2"]) / 0.054
        print(f"  {f['mask']:<18} |c_2|/0.038 = {ratio_3pi:.2f}  "
              f"|c_2|/0.043 = {ratio_2pi:.2f}  |c_2|/0.054 = {ratio_6:.2f}")

    # Save
    summary = {
        "L_lattice": L,
        "L_eff": L_eff,
        "alpha": alpha_arr.tolist(),
        "W_mean": W_arr.tolist(),
        "W_se": W_se.tolist(),
        "y": y.tolist(),
        "y_se": y_se.tolist(),
        "fits": fits,
        "predicted_c_v": {
            "eps_1_3pi": -0.038,
            "eps_1_2pi": +0.043,
            "eps_1_6":   +0.054,
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
