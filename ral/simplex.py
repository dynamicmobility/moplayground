import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Vertices of the canonical 3-simplex
S = 0.99
V = np.array([
    [0, 0, 0],
    [S, 0, 0],
    [0, S, 0],
    [0, 0, S]
])

# Faces (triangles)
faces = [
    [V[0], V[1], V[2]],
    [V[0], V[1], V[3]],
    [V[0], V[2], V[3]],
    [V[1], V[2], V[3]]
]

# Add translucent faces
poly = Poly3DCollection(
    faces,
    alpha=0.5,
    facecolor='lightblue',
    edgecolor='none'
)
poly.set_zsort('min')  # <---- key line
ax.add_collection3d(poly)
plt.draw()
# Scatter vertices
ax.scatter(V[:,0], V[:,1], V[:,2], s=60, c='black')

K = 100
rng = np.random.default_rng(95)
sampled = rng.dirichlet(alpha=[1, 1, 1], size=(K,)) * 1.01
ax.scatter(
    *(sampled.T),
    s=10,
    c='red',
    alpha=1.0,
    depthshade=False  # <---- important
)
print(np.sum(sampled, axis=1))

# for point in sampled:
#     print(point)
#     x, y, z = point
#     ax.plot(
#         [0, x],
#         [0, y],
#         [0, z],
#         color='red',
#         # linestyle='--',
#         lw=1
#     )

# ---- Add edges ----
for i, j in combinations(range(4), 2):
    ax.plot(
        [V[i,0], V[j,0]],
        [V[i,1], V[j,1]],
        [V[i,2], V[j,2]],
        color='black',
        lw=2,
        # linestyle='--'
    )

# Formatting
ax.set_box_aspect([1,1,1])
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
elev = 6
azim = -20
ax.view_init(elev, azim)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.show()