import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.integrate import solve_ivp
import matplotlib.cm as cm
np.set_printoptions(precision=2, suppress=True)


fig, ax = plt.subplots()
fig.set_facecolor('k')
ax.set_facecolor('k')
plt.subplots_adjust(0.01, 0.01, 0.99, 0.99)
plt.gca().set_aspect('equal')
plt.xticks([])
plt.yticks([])
ax.spines[:].set_visible(False)


G = 0.1
bounces = 10
N = 10
xx = np.linspace(-1, 1, 2 * N)
yy = np.linspace(0, -1, N)
X, Y = np.meshgrid(xx, yy)
Z = np.zeros([N, 2 * N, 4])
points = zip(np.ravel(X), np.ravel(Y))


for i, (x, y) in enumerate(points):
    print(i)

    if x**2 + y**2 - 1 < 0:
        posn, veln = (x, y), (0, 0)
        final_pos = 0

        def ev(t, x):
            return x[0] ** 2 + x[1] ** 2 - 1
        ev.terminal = True

        def f(t, x):
            return x[2], x[3], 0, -G

        # Path calculation
        for j in range(bounces):
            # Find path by solving a projectile motion
            Y = solve_ivp(f, [0, 100], [*posn, *veln], events=ev)
            final_pos = Y.y[:, -1]

            # Adjust the velocity vector at point of collision
            surf_norm = final_pos[:2]
            surf_norm = surf_norm / norm(surf_norm)
            proj_matrix = np.eye(2) - 2 * np.outer(surf_norm, surf_norm)
            veln = proj_matrix @ final_pos[2:]
            posn = final_pos[:2] + veln * 1e-6 - (1 / 2) * G * 1e-12

        j, k = np.unravel_index(i, (N, 2 * N))
        col = (1/2) * (final_pos[0] + 1)
        Z[j, k] = cm.jet(col)

print(Z.shape)

plt.imshow(Z, extent=(-1, 1, 1, 0), interpolation='nearest')


# Draw a parabola on which balls fall
# xx = np.linspace(-1, 1, 10000)
# yy = -np.sqrt(1 - xx ** 2)
# cc = cm.jet_r(abs(np.linspace(-0.9, 0.9, 10000)))
# plt.scatter(1.02 * xx, 1.02 * yy, lw=0.02, c=cc)

plt.get_current_fig_manager().window.showMaximized()
plt.show()


