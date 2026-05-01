"""
Paper VI --- alpha_EM from substrate effective dimension with shortcuts
==========================================================================

Tests Stan's intuition (April 2026):
  alpha_EM = (kappa/J) / S(d_eff)
where S(d) is the surface of the unit sphere in dimension d, and
d_eff is the substrate's effective dimension including shortcuts.

The prediction d_eff = 3 + 1/(2*pi) ~ 3.1592 (with 1/(2*pi) being the
shortcut contribution per U(1) phase cycle) gives 1/alpha = 136.85
at the Planck scale, which the QED running brings to ~ 137.04 (CODATA)
at the electron Compton scale.

Test: build a random geometric graph (mean degree ~6) and add p_short
fraction of random long-range shortcuts. Measure d_H by BFS expansion
N(r) ~ r^d, find the p_short for which d_H = 3 + 1/(2pi).

Outputs:
    data/alpha_from_dimension.json
    figures/d_H_vs_pshort.pdf
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.special import gamma

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
FIG  = HERE / "figures"; FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# parameters
N_NODES   = 5000        # graph size
BOX       = 20.0        # density n/V = 0.625
R_LINK    = 1.32        # for mean degree ~6: R^3 = 6/(0.625 * 4pi/3)
N_SAMPLES = 80          # BFS sources for d_H measurement
SEED      = 2026

# DDD parameters (independent of alpha_EM)
KAPPA = 0.015
J     = 0.15
ALPHA_EM_CODATA = 7.2973525693e-3
D_EFF_TARGET  = 3 + 1/(2*np.pi)


def S_d(d):
    return 2 * np.pi**(d/2) / gamma(d/2)


def alpha_pred(d):
    return KAPPA / (S_d(d) * J)


def build_rgg(rng, n=N_NODES, r_link=R_LINK):
    """Random geometric graph in 3D box."""
    pos = rng.uniform(0, BOX, size=(n, 3))
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r_link)
    adj = [[] for _ in range(n)]
    for i, j in pairs:
        adj[i].append(j); adj[j].append(i)
    return pos, adj


def add_shortcuts(adj, rng, p_short):
    """Add roughly p_short * n shortcuts between random pairs."""
    n = len(adj)
    n_short = int(p_short * n)
    added = 0
    for _ in range(n_short * 3):  # try a bit more for robustness
        i, j = rng.integers(0, n, size=2)
        if i == j: continue
        if j in adj[i]: continue
        adj[i].append(j); adj[j].append(i)
        added += 1
        if added >= n_short:
            break
    return adj


def measure_d_H(adj, rng, n_samples=N_SAMPLES, r_max=8):
    """For each of n_samples random sources, BFS up to r_max, record
    N(r) = number of nodes within graph distance r. Fit log N = d log r
    over r in [r_min, r_max-1]."""
    n = len(adj)
    sources = rng.choice(n, size=min(n_samples, n), replace=False)
    counts = np.zeros((len(sources), r_max + 1), dtype=int)
    for s_idx, src in enumerate(sources):
        visited = {int(src): 0}
        frontier = [int(src)]
        for r in range(1, r_max + 1):
            new_frontier = []
            for v in frontier:
                for u in adj[v]:
                    if u not in visited:
                        visited[u] = r
                        new_frontier.append(u)
            frontier = new_frontier
            counts[s_idx, r] = len(visited)
        counts[s_idx, 0] = 1
    # Fit on bulk regime: r in [2, r_max - 1]
    mean_counts = counts.mean(axis=0)
    rs = np.arange(r_max + 1)
    fit_lo, fit_hi = 2, r_max - 1
    log_r = np.log(rs[fit_lo:fit_hi + 1])
    log_n = np.log(np.maximum(mean_counts[fit_lo:fit_hi + 1], 1))
    slope, intercept = np.polyfit(log_r, log_n, 1)
    return float(slope), mean_counts


# ---------------------------------------------------------------------
# main: sweep p_short
# ---------------------------------------------------------------------
print("=" * 70)
print("Paper VI: d_H vs shortcut density")
print("=" * 70)
print(f"Target d_eff = 3 + 1/(2 pi) = {D_EFF_TARGET:.4f}")
print(f"Target prediction alpha_EM = {alpha_pred(D_EFF_TARGET):.6f}")
print(f"CODATA alpha_EM           = {ALPHA_EM_CODATA:.6f}")
print()

rng = np.random.default_rng(SEED)
pos, adj_base = build_rgg(rng)
deg_init = np.array([len(a) for a in adj_base])
print(f"Initial RGG: N={N_NODES}, mean deg = {deg_init.mean():.2f}")

# Sweep
ps = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]
results = []
for p in ps:
    rng2 = np.random.default_rng(SEED + 1)
    adj = [list(a) for a in adj_base]
    adj = add_shortcuts(adj, rng2, p)
    deg = np.array([len(a) for a in adj])
    d_H, _ = measure_d_H(adj, rng2)
    a_pred = alpha_pred(d_H)
    print(f"  p_short = {p:5.3f}  mean deg = {deg.mean():5.2f}  "
          f"d_H = {d_H:5.4f}  alpha_pred = {a_pred:.5f}  "
          f"ratio = {a_pred/ALPHA_EM_CODATA:.3f}")
    results.append({
        "p_short":    p,
        "mean_deg":   float(deg.mean()),
        "d_H":        float(d_H),
        "alpha_pred": float(a_pred),
        "ratio_codata": float(a_pred/ALPHA_EM_CODATA),
    })

# Find p_short that matches d_H = 3 + 1/(2 pi)
ds   = np.array([r["d_H"] for r in results])
psv  = np.array([r["p_short"] for r in results])
# interpolate
if ds.max() > D_EFF_TARGET > ds.min():
    p_match = float(np.interp(D_EFF_TARGET, ds, psv))
    print(f"\np_short matching d_eff = {D_EFF_TARGET:.4f}: "
          f"p_short ~ {p_match:.4f}")
else:
    p_match = None
    print(f"\nd_H sweep does not reach target {D_EFF_TARGET:.4f}; "
          f"range [{ds.min():.3f}, {ds.max():.3f}]")

# Save
summary = {
    "params":      {"N_NODES": N_NODES, "BOX": BOX, "R_LINK": R_LINK,
                    "KAPPA": KAPPA, "J": J},
    "target_d_eff":     D_EFF_TARGET,
    "alpha_target":     float(alpha_pred(D_EFF_TARGET)),
    "alpha_CODATA":     ALPHA_EM_CODATA,
    "p_short_match":    p_match,
    "sweep":            results,
}
with open(DATA / "alpha_from_dimension.json", "w") as f:
    json.dump(summary, f, indent=2)

# Figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(psv, ds, "bo-", markersize=8)
ax.axhline(D_EFF_TARGET, color="red", ls="--", lw=1,
            label=fr"$3+1/(2\pi) = {D_EFF_TARGET:.4f}$")
ax.axhline(np.sqrt(10), color="orange", ls=":", lw=1,
            label=fr"$\sqrt{{10}} = {np.sqrt(10):.4f}$")
if p_match is not None:
    ax.axvline(p_match, color="green", ls=":", lw=0.7,
                label=fr"$p^* \approx {p_match:.3f}$")
ax.set_xlabel("shortcut density $p_{\\rm short}$")
ax.set_ylabel(r"effective dimension $d_H$")
ax.set_title("$d_H$ vs shortcut density")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
alphas = np.array([r["alpha_pred"] for r in results])
ax.plot(psv, alphas, "bo-", markersize=8, label="DDD prediction")
ax.axhline(ALPHA_EM_CODATA, color="red", ls="--", lw=1,
            label=fr"CODATA $\alpha_{{\rm EM}} = 0.00730$")
ax.set_xlabel("shortcut density $p_{\\rm short}$")
ax.set_ylabel(r"$\alpha_{\rm EM}$ predicted")
ax.set_title(r"DDD-predicted $\alpha_{\rm EM}$ vs $p_{\rm short}$")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle("Test: shortcuts in DDD substrate produce alpha_EM",
              fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(FIG / "d_H_vs_pshort.pdf", bbox_inches="tight")
fig.savefig(FIG / "d_H_vs_pshort.png", dpi=150, bbox_inches="tight")
print("Done.")
