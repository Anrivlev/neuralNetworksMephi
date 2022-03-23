import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from celluloid import Camera

alpha = 1
w0 = np.array([1, 1, 1])
sample = np.array([[1, 1.1, 0.2],
                   [1, 0.8, -0.1],
                   [1, 0.7, 0.1],
                   [1, 0.2, 1],
                   [1, -0.2, 1.2],
                   [1, 0.1, 0.7]])
sample_y = np.array([0, 0, 0, 1, 1, 1])
N = len(sample_y)
w = np.zeros(sample.shape)
w[0] = w0

for i in range(1, N):
    h = w[i - 1][0] * sample[i - 1][0] + w[i - 1][1] * sample[i - 1][1] + w[i - 1][2] * sample[i - 1][2]
    if h > 0:
        y = 1
    else:
        y = 0
    if sample_y[i - 1] < y:
        w[i] = w[i - 1] - alpha * sample[i - 1]
    elif sample_y[i - 1] > y:
        w[i] = w[i - 1] + alpha * sample[i - 1]
    else:
        w[i] = w[i - 1]

x1_min = -2
x1_max = 2
x1_n = 100
x1_range = np.linspace(x1_min, x1_max, x1_n)

x2_min = x1_min
x2_max = x1_max

fig = plt.figure()
ax = plt.axes(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))
line, = ax.plot([], [], lw=3)
plt.grid()

camera = Camera(fig)
for i in range(N):
    x2_range = - (w[i][0] + w[i][1] * x1_range) / (w[i][2])
    plt.plot(x1_range, x2_range)
    plt.plot(sample[:, 1], sample[:, 2], linestyle='None', marker='o')
    plt.legend(str(i))
    camera.snap()

animation = camera.animate()

animation.save('solution.gif', writer='imagemagick')
