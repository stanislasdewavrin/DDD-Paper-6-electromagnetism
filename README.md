# Paper VI — Maxwell-Like Dynamics and a Candidate Fixed-Point Formula for α_EM

The oriented sector of Discrete Drainage Dynamics (DDD) recovers
Maxwell-like electromagnetism with no additional postulate beyond the
local energy `E_orient` and the topological structure of the graph.
A self-consistent Wilson-loop back-reaction along the body diagonal of
the unit cube yields a candidate fixed-point formula for the
fine-structure constant.

## Headline results

- Two homogeneous Maxwell equations from the discrete Bianchi identity
- Two inhomogeneous equations from Euler-Lagrange of `E_orient`
- U(1) gauge invariance is structural
- Integer charge sectors as topological winding of the orientation field

## Self-consistent α_EM formula

```
x_*       = 3 + α_EM · √3 · (1 − α_EM · V_{d_eff})
α_EM      = 1 / (8π² √x_*)
V_{d_eff} = π^(d_eff/2) / Γ(d_eff/2 + 1)
d_eff     = 3 + ε,   ε ∈ {1/(2π), 1/6}
```

Result: `1/α_EM* ∈ [137.0359336, 137.0359473]`, matching CODATA
(137.0359990840 ± 0.0000000210) to within 0.5 ppm. The residual is
consistent in magnitude with expected two-loop corrections.

## Companion gravitational Wilson loop

The same back-reaction structure applied to the cube-surface Wilson
loop (area 6 instead of body-diagonal length² 3) gives at the
substrate scale:

```
α_G_lat / α_EM_lat = √(3/6) = 1/√2 ≈ 0.707
```

This fixes the substrate's natural mass scale at m_lat ≈ 0.072 M_P,
i.e. lattice spacing ≈ 14 ℓ_Planck. Photon dispersion deviation at
all experimentally accessible energies is below 10⁻⁷, compatible with
all current Lorentz-invariance-violation limits by a factor 10⁶.

## Numerical validations

- BFS expansion dimension at L=400 (20 seeds, 64M nodes):
  **d_BFS = 3.106 ± 0.005** (0.15% precision)
- Compatible with ε = 1/(3π) at 0.07σ
- Determ​inistic 2×2×2 chiral tilings tested: do not reproduce 3+ε
  asymptotically (the small-world correction is intrinsically
  statistical)
- Two-loop Wilson loop residual ~0.4 ppm consistent with expected
  α² × V correction (open problem b)

## Reproducibility

```bash
# Compile the paper
make all
# or:
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

```bash
# Reproduce the BFS measurements
cd code
python 31_final_L200_5seeds.py
python 36_L400_one_seed.py 0    # one seed at L=400
python 36_L400_one_seed.py 1
# ...etc, to accumulate seeds
```

```bash
# (optional) Run the local 2-loop extraction
# Quick test (~10 min): L=8, n_therm=500, n_meas=2000
python code/43_2loop_extraction_local.py

# Production (~2-4 hours): L=12, n_therm=3000, n_meas=15000
python code/43_2loop_extraction_local.py --L 12 --n_therm 3000 --n_meas 15000
```

## Open questions

(a) Analytic derivation of the small-world correction: is
`d_BFS = 3 + 1/(3π)` exact, or 3 + 1/(2π), or something else?

(b) Explicit one-loop Wilson-loop self-energy on the diagonal-twist
compact U(1) lattice, yielding the coefficient V_{d_eff} from first
principles (currently a heuristic identification).

(c) Two-loop coefficient c_2 in the back-reaction expansion. Continuum
abelian exponentiation gives c_2 = 1/2; lattice cutoff effects
modify this. A preliminary L=6 simulation (10 min run) gives
c_2_relative ≈ 2.4, consistent with lattice partial-exponentiation
but not precise.

## Repository

https://github.com/stanislasdewavrin/DDD-Paper-6-electromagnetism

## Contact

Stanislas Dewavrin — sdewavrin@ohbibi.com
