import numpy as np
from numpy import pi
from numpy.random import uniform
import matplotlib.pyplot as plt


N = 1000
Nr = 10
K = 0.8
xx = np.linspace(-pi, pi, Nr)
yy = np.linspace(0, 2 * pi, Nr)
J, Q = np.meshgrid(xx, yy)

J = np.ravel(J)
Q = np.ravel(Q)
L = list(range(len(J)))

fig, ax = plt.subplots(1)
plt.subplots_adjust(0.03, 0.05, 0.99, 0.95)
plt.xlim([0, 2 * pi])
plt.ylim([-pi, pi])


for jj, j_ini, q_ini in zip(L, J, Q):
    print(jj)
    # j_ini = uniform(-pi, pi)
    # q_ini = uniform(0, 2 * pi)

    Z = np.array([[j_ini, q_ini]])

    for n in range(N):
        j0, th0 = Z[-1]
        j1 = j0 + K * np.sin(th0)
        th1 = th0 + j1
        Z = np.vstack((Z, [np.mod(j1 + pi, 2 * pi) - pi, np.mod(th1, 2 * pi)]))

    X1, Y1 = Z[:, 1], Z[:, 0]
    plt.plot(X1, Y1, '.', markersize=0.3)

plt.title('K = {:.3f}'.format(K))

plt.show()
