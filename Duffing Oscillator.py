import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation


def duffing(t, xx, a, b, c, d, w):         # The time evolution function
    x, v, th = xx
    dx = v
    dth = w
    dv = -b * v + a * x - c * x**3 + d * np.cos(th)
    return np.array([dx, dv, dth])


# 0.1, 1, 1, 0.39, 1.4 is default
# (0.1, 1, 1, 3, 0.5) is interesting !!
damp = 0.1  # 0.1
lin_stiffness = 1
cubic_stiffness = 1
forced_amp = 0.39
forced_freq = 1.4


fig, ax = plt.subplots()
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)


period = (2 * pi) / forced_freq
T_max = 10000
step = 100              
dt = period / step 
T = np.arange(0, T_max * period, dt)
T_ev = [period * i for i in range(T_max)]


# Solve the differential equation
x_init = np.zeros(3)
Z = solve_ivp(duffing, [0, T_max * period], x_init, t_eval=T_ev, args=(lin_stiffness, damp, cubic_stiffness, forced_amp, forced_freq))
xn, vn, thn = Z.y

plt.plot(xn, vn, '.k')
print(len(xn))

plt.show()
