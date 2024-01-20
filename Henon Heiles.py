import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def hhl(xx, t):         # The time evolution function
    x, y, px, py = xx
    dx = px
    dy = py
    dpx = - x - 2 * x * y
    dpy = - y - x * x + y * y
    return np.array([dx, dy, dpx, dpy])

N = 100000
Nr = 15
T = np.linspace(0, 5000, N)
E = 0.12
P = np.linspace(0, np.sqrt(2 * E) * 0.99, Nr)

# Axes creation
fig, ax = plt.subplots()
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
plt.gca().set_aspect('equal')

# Check at index 11
for j in range(Nr):
    print(j)

    # Generate the initial conditions
    y0, px0 = 0, 0
    py0 = P[j]
    x0 = np.sqrt(2 * E - py0 * py0)
    x_init = x0, y0, px0, py0

    # Solve the differential equations
    Y = odeint(hhl, x_init, T)
    xn, yn, pxn, pyn = Y.T

    # Find the intersection points on Poincare plane using linear interpolation
    div = abs(np.diff(np.sign(xn)))
    div_idx = [i for i in range(len(div)) if div[i] == 2]
    yg, pyg = np.zeros([2, len(div_idx)])

    for i, d in enumerate(div_idx):
        delta = -xn[d] / (xn[d+1] - xn[d])
        yg[i] = yn[d] + delta * (yn[d+1] - yn[d])
        pyg[i] = pyn[d] + delta * (pyn[d+1] - pyn[d])

    # Plot the Poincare section for these initial conditions
    plt.plot(yg, pyg, '.', markersize=7)


plt.title('E = {:.3f}'.format(E))
leg = [''.join('{}: {:.5f}'.format(k, i)) for k, i in enumerate(P)]
plt.legend(leg)

plt.show()
