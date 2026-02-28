import matplotlib as mpl

LABEL_SIZE = 16
TICK_SIZE = 12
mpl.rcParams.update({
    "text.usetex"           : True,
    "text.latex.preamble"   : r"\usepackage{amsmath,amssymb}",
    "font.family"           : "serif",
    "font.serif"            : ["Computer Modern Roman"],
    "font.size"             : LABEL_SIZE,
    "axes.labelsize"        : LABEL_SIZE,
    "axes.titlesize"        : LABEL_SIZE,
    "legend.fontsize"       : TICK_SIZE,
    "xtick.labelsize"       : TICK_SIZE,
    "ytick.labelsize"       : TICK_SIZE,
    "axes.linewidth"        : 1.2,
    "lines.linewidth"       : 2.0,
    "xtick.direction"       : "in",
    "ytick.direction"       : "in",
    "xtick.major.size"      : 4,
    "ytick.major.size"      : 4,
    "xtick.major.width"     : 1.0,
    "ytick.major.width"     : 1.0,
    "legend.frameon"        : False,
})
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations

# ----------------------------
# Figure + axis
# ----------------------------
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Use orthographic projection (much cleaner for geometry)
ax.set_proj_type('ortho')

# ----------------------------
# Simplex vertices
# ----------------------------
S = 1.0
V = np.array([
    [0, 0, 0],
    [S, 0, 0],
    [0, S, 0],
    [0, 0, S]
])

faces = [
    [V[0], V[1], V[2]],
    [V[0], V[1], V[3]],
    [V[0], V[2], V[3]],
    [V[1], V[2], V[3]]
]

# ----------------------------
# Translucent faces
# ----------------------------
poly = Poly3DCollection(
    faces,
    facecolor=(0.4, 0.6, 0.9, 0.15),   # soft blue w/ low alpha
    edgecolor='none'
)
# poly.set_zsort('min')  # reduces face-over-point artifacts
ax.add_collection3d(poly)
elev = 20
azim = 45
ax.view_init(elev, azim)
# ----------------------------
# Crisp edges (always visible)
# ----------------------------
for i, j in combinations(range(4), 2):
    ls = None
    c = 'black'
    if i == 0:
        ls = '--'
        c = [0, 0, 0, 0.5]
    ax.plot(
        [V[i,0], V[j,0]],
        [V[i,1], V[j,1]],
        [V[i,2], V[j,2]],
        color=c,
        lw=1.5,
        ls=ls
    )

# ----------------------------
# Vertices
# ----------------------------
ax.scatter(
    V[1:,0], V[1:,1], V[1:,2],
    s=80,
    c='black',
    depthshade=False
)

# ----------------------------
# Dirichlet samples
# ----------------------------
rng = np.random.default_rng(95)
K = 200
sampled = rng.dirichlet(alpha=[1,1,1], size=K)

ax.scatter(
    sampled[:,0],
    sampled[:,1],
    sampled[:,2],
    s=16,
    c='red',#(0.8, 0.1, 0.1, 1.0),
    depthshade=False
)

# ----------------------------
# Formatting
# ----------------------------
ax.set_box_aspect([1,1,1])
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.invert_yaxis()
ax.view_init(elev=20, azim=-35)

# Remove panes & grid (important for papers)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.xaxis.labelpad = -10
ax.yaxis.labelpad = -10
ax.zaxis.labelpad = -10
ax.set_xlabel("Joint Accuracy")
ax.set_ylabel("Swing Arms")
ax.set_zlabel("")  # remove default

ax.text2D(
    0.06, 0.5, "Smoothness",
    transform=ax.transAxes,
    rotation=90,
    va='center',
    ha='center'
)
ax.zaxis.line.set_visible(False)
# ax.xaxis.pane.set_visible(False)
# ax.yaxis.pane.set_visible(False)
# ax.zaxis.pane.set_visible(False)
ax.set_box_aspect([1,1,1])

# plt.tight_layout()
fig.set_size_inches((6, 5))
plt.savefig('ral/images/simplex.svg')