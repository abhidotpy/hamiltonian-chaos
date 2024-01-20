import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation


DT = 0.1                                    # Timestamps
rad = 0.02                                  # Radius of ball
del_0 = 1e-10                               # Initial separation of position
G = 0.1                                     # Acceleration due to gravity
pos1 = np.array([0.6, 0])                   # Initial position of blue ball
pos2 = np.array([pos1[0] + del_0, 0])       # Initial position of red ball
vel1 = np.zeros_like(pos1)                  # Initial velocity of blue ball
vel2 = np.zeros_like(pos2)                  # Initial velocity of red ball
lya_exp = np.array([[0, np.log10(del_0)]])  # Lyapunov exponent graph points


# Formatting the axes
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
ax1.set_ylim([-1.2, 0.8])
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_ylim([np.log10(del_0), 0])


# Create the falling balls
C1 = Circle(pos1, radius=rad, edgecolor='k', facecolor='turquoise', animated=True)
C2 = Circle(pos2, radius=rad, edgecolor='k', facecolor='orangered', animated=True)
ax1.add_patch(C1)
ax1.add_patch(C2)

# Draw the graph for Lyapunov exponent
lm, = ax2.plot(lya_exp[:, 0], lya_exp[:, 1], 'g', animated=True)

# Draw a parabola on which balls fall
xx = np.linspace(-1, 1, 10000)
yy = -np.sqrt(1 - xx ** 2)
ax1.plot(1.02 * xx, 1.02 * yy, 'k')


def find_path(pos_b, pos_r, vel_b, vel_r):
    Xp_b, Yp_b = [], []
    Xp_r, Yp_r = [], []
    bounces = 40

    def ev(t, x):
        return x[0] ** 2 + x[1] ** 2 - 1

    ev.terminal = True

    def f(t, x):
        return x[2], x[3], 0, -G

    # Path calculation for blue ball
    for i in range(bounces):

        # Find path by solving a projectile motion
        Y = solve_ivp(f, [0, 100], [*pos_b, *vel_b], events=ev, dense_output=True)
        Np = int(Y.t[-1] / DT)
        T = np.linspace(0, Y.t[-1], Np)
        Z = Y.sol(T)
        Xp_b.extend(Z[0])
        Yp_b.extend(Z[1])

        # Adjust the velocity vector at point of collision
        surf_norm = Z[:2, -1]
        surf_norm = surf_norm / norm(surf_norm)
        proj_matrix = np.eye(2) - 2 * np.outer(surf_norm, surf_norm)
        vel_b = proj_matrix @ Z[2:, -1]
        pos_b = Z[:2, -1] + vel_b * 1e-6 - (1 / 2) * G * 1e-12

    # Path calculation for red ball
    for i in range(bounces):

        # Find path by solving a projectile motion
        Y = solve_ivp(f, [0, 100], [*pos_r, *vel_r], events=ev, dense_output=True)
        Np = int(Y.t[-1] / DT)
        T = np.linspace(0, Y.t[-1], Np)
        Z = Y.sol(T)
        Xp_r.extend(Z[0])
        Yp_r.extend(Z[1])

        # Adjust the velocity vector at point of collision
        surf_norm = Z[:2, -1]
        surf_norm = surf_norm / norm(surf_norm)
        proj_matrix = np.eye(2) - 2 * np.outer(surf_norm, surf_norm)
        vel_r = proj_matrix @ Z[2:, -1]
        pos_r = Z[:2, -1] + vel_r * 1e-6 - (1 / 2) * G * 1e-12

    # Blue and red ball points length may be different
    return np.asarray(list(zip(Xp_b, Yp_b, Xp_r, Yp_r)))


H = find_path(pos1, pos2, vel1, vel2)
tot_frames = H.shape[0]
ax2.set_xlim([0, tot_frames * DT])
print(tot_frames)

def animate(frame):
    global lya_exp
    print(frame)

    posn_b = H[frame, :2]
    posn_r = H[frame, 2:]
    C1.set_center(posn_b)
    C2.set_center(posn_r)

    x1 = lya_exp[-1][0] + DT
    y1 = np.log10(norm(posn_b - posn_r))
    lya_exp = np.vstack([lya_exp, [x1, y1]])
    lm.set_data(lya_exp[:, 0], lya_exp[:, 1])

    return C1, C2, lm


aaaa = FuncAnimation(fig, animate, frames=tot_frames, interval=30, repeat=False, blit=True)

plt.show()
