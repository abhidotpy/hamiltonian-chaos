import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pendulum(xx, t, _m=1, _l=1, _g=1):  # The time evolution function
    ths1, ths2, ws1, ws2 = xx

    del_th = ths2 - ths1
    k1 = _m * _l * _l * ws2 * ws2 * np.sin(del_th)
    k2 = 2 * _m * _l * _g * np.sin(ths1)
    k3 = _m * _l * _l * ws1 * ws1 * np.sin(del_th)
    k4 = _m * _l * _g * np.sin(ths2)

    wk1 = (k1 - k2 + k3 * np.cos(del_th) + k4 * np.cos(del_th)) / (_m * _l * _l * (2 - np.cos(del_th) ** 2))
    wk2 = (k1 * np.cos(del_th) - k2 * np.cos(del_th) + k3 * 2 + k4 * 2) / (_m * _l * _l * (np.cos(del_th) ** 2 - 2))

    return np.array([ws1, ws2, wk1, wk2])


N = 10000
T = np.linspace(0, 1000, N)
Nr = 10
Mass = 1
Length = 1
G_acc = 1
E = 0.5
kn = np.sqrt(E / Mass / Length ** 2) * 0.99
W = np.linspace(-kn, kn, Nr)

# Axes creation
fig, ax = plt.subplots()
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
plt.gca().set_aspect('equal')


for j in range(Nr):
    print(j)

    # Generate the initial conditions
    wm1 = W[j]
    wm2 = -wm1 + np.sqrt((2 * E / Mass / Length**2) - wm1 ** 2)
    thm1 = 0
    thm2 = 0

    x_init = thm1, thm2, wm1, wm2

    # Solve the differential equations
    Y = odeint(pendulum, x_init, T, args=(Mass, Length, G_acc))
    thp1, thp2, wp1, wp2 = Y.T

    # Find the intersection points on Poincare plane
    div = np.diff(np.sign(thp1))
    div_idx = [i for i in range(len(div)) if div[i] == 2]
    thg, wg = np.zeros([2, len(div_idx)])

    for i, d in enumerate(div_idx):
        delta = -thp1[d] / (thp1[d + 1] - thp1[d])
        thg[i] = thp2[d] + delta * (thp2[d + 1] - thp2[d])
        wg[i] = wp2[d] + delta * (wp2[d + 1] - wp2[d])

    # Plot the Poincare section for these initial conditions
    plt.plot(thg, wg, 'o', markersize=1)

plt.title('E = {:.3f}'.format(E))
leg = [''.join('{}: {:.2f}'.format(k, i)) for k, i in enumerate(W)]
plt.legend(leg)

plt.show()
