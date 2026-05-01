"""Paper VI v7 — figures for the α_EM derivation."""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import gammaln
from pathlib import Path

HERE = Path("/sessions/sweet-determined-tesla/mnt/Physique/paperVI_electromagnetism")
DATA = HERE / "data"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

PI = np.pi
ALPHA_CODATA = 7.2973525693e-3
INV_ALPHA_CODATA = 137.035999084
D_EFF = 3 + 1/(2*PI)
V_3 = 4*PI/3
V_DEFF = PI**(D_EFF/2) / np.exp(gammaln(D_EFF/2 + 1))


# ============================================================
# Figure 1: Substrate schematic (cube with body diagonal twists)
# ============================================================
print("Fig 1: substrate schematic...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: 3D-ish projection of cubic lattice + body diagonals
ax = axes[0]
ax.set_aspect('equal')
# Draw a 3x3x3 chunk of cubic lattice
for x in range(4):
    for y in range(4):
        # Project to 2D: simple isometric
        xp = x + 0.4*y; yp = y*0.7
        ax.plot(xp, yp, 'ko', ms=4)

# Axis edges (a few)
def draw_edge(x1, y1, x2, y2, **kwargs):
    ax.plot([x1, x2], [y1, y2], **kwargs)

# Body diagonals shown as dashed red arrows in a few cubes
# Body diagonal in projection from (0,0) to (1,1) within an isometric cube
for cx in range(3):
    for cy in range(3):
        # Start corner
        x0 = cx + 0.4*cy; y0 = cy*0.7
        # Body diagonal end (shifted by (1,1,1) → in projection (1+0.4, 0.7) + (0.4, 0.7) shift)
        x1 = (cx+1) + 0.4*(cy+1); y1 = (cy+1)*0.7 + 0.5  # +0.5 for z component
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.4, lw=1))

# Mark example node with multiple body-diagonal shortcuts (Poisson distribution)
center = (1.5 + 0.4*1.5, 1.5*0.7 + 0.4)
ax.plot(*center, 'go', ms=10, label='node with multiple shortcuts')

ax.set_title('Cubic substrate + body diagonal shortcuts (shown as red arrows)\n'
             r'Poisson($\mu=1$): each node has 1 shortcut on average',
             fontsize=10)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.5)

# Right panel: shell structure in a unit cube
ax = axes[1]
ax.set_aspect('equal')

# Draw a unit cube in 3D-ish projection
verts_2d = []
for x in [0, 1]:
    for y in [0, 1]:
        for z in [0, 1]:
            xp = x + 0.4*z; yp = y + 0.5*z
            verts_2d.append((xp, yp, x, y, z))

# Draw edges
edges = []
for i, (xp1, yp1, x1, y1, z1) in enumerate(verts_2d):
    for j, (xp2, yp2, x2, y2, z2) in enumerate(verts_2d):
        if i >= j: continue
        d2 = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
        if d2 == 1:  # axis edge
            ax.plot([xp1, xp2], [yp1, yp2], 'k-', lw=1.5, alpha=0.7)

# Draw 4 body diagonals
for i, (xp1, yp1, x1, y1, z1) in enumerate(verts_2d):
    for j, (xp2, yp2, x2, y2, z2) in enumerate(verts_2d):
        if i >= j: continue
        d2 = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
        if d2 == 3:  # body diagonal
            ax.annotate('', xy=(xp2, yp2), xytext=(xp1, yp1),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7))

# Mark vertices
for xp, yp, x, y, z in verts_2d:
    ax.plot(xp, yp, 'ko', ms=8)

# Labels
ax.text(0.5, -0.2, 'edge: length 1', fontsize=10, ha='center', color='black')
ax.text(0.7, 1.7, 'body diagonal: length √3', fontsize=10, ha='center', color='red')
ax.text(0.7, 1.85, '(carries twist λ = 1/(2π))', fontsize=9, ha='center', color='red')

ax.set_title(r'Unit cube: 12 axis edges + 4 body diagonals'
             '\n' r'(structural: $4 \times 2\pi^2 = 8\pi^2$, body diag length = $\sqrt{3}$)',
             fontsize=10)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-0.4, 1.7); ax.set_ylim(-0.4, 2.0)

fig.tight_layout()
fig.savefig(FIG / "fig_substrate_schematic.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig_substrate_schematic.png", dpi=150, bbox_inches='tight')
plt.close(fig)


# ============================================================
# Figure 2: Self-consistent α_EM convergence
# ============================================================
print("Fig 2: α_EM convergence...")

def find_fp(K):
    def f(x):
        a = 1.0/(8*PI**2 * np.sqrt(x))
        return x - 3 - a*np.sqrt(3)*(1 - a*K)
    return brentq(f, 2.5, 3.5, xtol=1e-15)

# Three approximations
levels = [
    ("Tree-level only\n(no back-reaction)", 0, "blue"),
    (r"+ One-loop, Euclidean $V_3 = 4\pi/3$", V_3, "orange"),
    (r"+ One-loop, fractional $V_{d_{\rm eff}}$", V_DEFF, "green"),
]
labels = []
inv_alphas = []
deltas_ppm = []
colors_list = []
for label, K, color in levels:
    x = find_fp(K)
    a = 1.0/(8*PI**2*np.sqrt(x))
    inv = 1/a
    delta = 1e6*(inv - INV_ALPHA_CODATA)/INV_ALPHA_CODATA
    labels.append(label); inv_alphas.append(inv); deltas_ppm.append(delta); colors_list.append(color)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: bar chart of 1/α* vs CODATA
ax = axes[0]
y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, inv_alphas, color=colors_list, alpha=0.7, edgecolor='black')
ax.axvline(INV_ALPHA_CODATA, color='red', linestyle='--', lw=2, label=f'CODATA: 1/α = {INV_ALPHA_CODATA:.4f}')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel(r'$1/\alpha_{\rm EM}^*$')
ax.set_xlim(136.5, 137.2)
for bar, inv, delta in zip(bars, inv_alphas, deltas_ppm):
    ax.text(inv + 0.01, bar.get_y() + bar.get_height()/2,
            f'  {inv:.4f}  ({delta:+.2f} ppm)',
            va='center', fontsize=9)
ax.legend(loc='lower right', fontsize=10)
ax.set_title(r'Self-consistent fixed point of $\alpha_{\rm EM}$', fontsize=11)
ax.grid(axis='x', alpha=0.3)

# Right: log-scale ppm offset
ax = axes[1]
ppm_abs = [abs(d) for d in deltas_ppm]
short_labels = ['Tree-level\n(0 corr)',
                r'+ Euclidean $V_3$',
                r'+ Fractional $V_{d_{\rm eff}}$']
ax.bar(short_labels, ppm_abs, color=colors_list, alpha=0.7, edgecolor='black')
ax.set_yscale('log')
ax.set_ylabel(r'$|$Δ from CODATA$|$ [ppm]')
ax.set_title('Convergence with successive corrections', fontsize=11)
for i, (lbl, ppm) in enumerate(zip(short_labels, ppm_abs)):
    ax.text(i, ppm * 1.15, f'{ppm:.2f}', ha='center', fontsize=10)
ax.grid(axis='y', alpha=0.3, which='both')

fig.tight_layout()
fig.savefig(FIG / "fig_alpha_convergence.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig_alpha_convergence.png", dpi=150, bbox_inches='tight')
plt.close(fig)


# ============================================================
# Figure 3: BFS dimension vs μ — small-world emergence of d_eff
# ============================================================
print("Fig 3: d_BFS vs μ...")

# Data from script 20 (Poisson Test 1)
mus_test1 = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
d_bfs_test1 = [2.7721, 2.9927, 3.0317, 3.0322, 3.1449, 3.1813, 3.2046, 3.0679, 3.0627]

# Data from script 20 (Test 3, shells 2,3,4)
mus_test3 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
d_bfs_test3 = [2.7721, 2.9747, 3.1485, 3.1341, 3.1608, 3.0534]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(mus_test1, d_bfs_test1, 'o-', color='blue', ms=8, lw=2,
        label='Poisson(μ) → body diagonals')
ax.plot(mus_test3, d_bfs_test3, 's-', color='green', ms=8, lw=2,
        label='Poisson(μ) → shells {2, 3, 4}')
ax.axhline(D_EFF, color='red', linestyle='--', lw=2,
           label=f'Target: $d_{{\\rm eff}} = 3 + 1/(2\\pi) = {D_EFF:.4f}$')
ax.axhline(3.0, color='gray', linestyle=':', alpha=0.6,
           label='Pure cubic dim = 3')
ax.axvline(1.0, color='black', linestyle=':', alpha=0.4)
ax.text(1.02, 2.85, r'$\mu = 1$', fontsize=11)
ax.set_xlabel(r'Mean shortcuts per node $\mu$', fontsize=12)
ax.set_ylabel(r'$d_{\rm BFS}$ (BFS expansion exponent)', fontsize=12)
ax.set_title(r'Emergence of $d_{\rm eff} = 3 + 1/(2\pi)$ from statistical small-world',
             fontsize=11)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(-0.1, 3.2)
ax.set_ylim(2.7, 3.3)

fig.tight_layout()
fig.savefig(FIG / "fig_d_BFS_emergence.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig_d_BFS_emergence.png", dpi=150, bbox_inches='tight')
plt.close(fig)


# ============================================================
# Figure 4: Self-consistency loop diagram
# ============================================================
print("Fig 4: self-consistency loop...")
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_aspect('equal')
ax.set_xlim(-1, 8)
ax.set_ylim(-1, 6)

# Draw boxes
def box(ax, x, y, w, h, text, color='lightblue', fontsize=10):
    rect = plt.Rectangle((x, y), w, h, fc=color, ec='black', lw=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize)

box(ax, 0, 4.5, 3, 1, r'$\alpha_{\rm EM} = \frac{1}{8\pi^2 \sqrt{x}}$', 'lightblue', 12)
box(ax, 5, 4.5, 3, 1, r'$x = 3 + \delta(\alpha_{\rm EM})$', 'lightgreen', 12)
box(ax, 0, 0, 3, 1.4, r'$\delta_1 = \alpha_{\rm EM} \sqrt{3}$' + '\n(tree-level Wilson loop\nself-energy)', 'lightyellow', 9)
box(ax, 5, 0, 3, 1.4, r'$\delta_2 = -\alpha_{\rm EM}^2 \sqrt{3}\, V_{d_{\rm eff}}$' + '\n(one-loop volume\nintegration)', 'lightyellow', 9)
box(ax, 2.5, 2.3, 3, 1.4, r'$\delta = \delta_1 + \delta_2$', 'lightcoral', 11)

# Arrows
ax.annotate('', xy=(5, 5), xytext=(3, 5), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(1.5, 4.5), xytext=(1.5, 3.7), arrowprops=dict(arrowstyle='<-', lw=1.5))
ax.annotate('', xy=(6.5, 4.5), xytext=(6.5, 3.7), arrowprops=dict(arrowstyle='<-', lw=1.5))
ax.annotate('', xy=(4, 2.3), xytext=(1.5, 1.4), arrowprops=dict(arrowstyle='->', lw=1.2))
ax.annotate('', xy=(4, 2.3), xytext=(6.5, 1.4), arrowprops=dict(arrowstyle='->', lw=1.2))
ax.text(4, 5.3, 'feeds back into', ha='center', fontsize=10, color='blue')
ax.text(4, 2, 'both contribute', ha='center', fontsize=9, color='red')

# Title and result
ax.text(4, 5.9, r'Self-consistent equation for $\alpha_{\rm EM}$',
        ha='center', fontsize=13, fontweight='bold')
ax.text(4, -0.7,
        r'Fixed point: $\alpha_{\rm EM}^* = 7.297\,355 \times 10^{-3}$  $\Leftrightarrow$  '
        r'$1/\alpha^* = 137.035947$  (CODATA: $137.035999$, $\Delta = -0.38$ ppm)',
        ha='center', fontsize=10, fontweight='bold')

ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

fig.tight_layout()
fig.savefig(FIG / "fig_self_consistency_loop.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig_self_consistency_loop.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("All figures saved to:", FIG)
