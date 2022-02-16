import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def calculate_sync(coeffs, activation_funcs, x, y0, n):
    # assume len(y0) = len(activation_funcs) = len(coeffs[i;:]) for i in range(n)
    y = np.zeros((n, len(y0)))
    y[0, :] = y0
    for t in range(1, n):
        B = np.zeros(4)
        B[0] = x(t)
        B[1:4] = y[t - 1, :]
        y[t, :] = coeffs.dot(B)
        y[t, :] = [activation_funcs[i](y[t, i]) for i in range(len(y0))]
    return y


def calculate_async(coeffs, activation_funcs, x, y0, n):
    # assume len(y0) = len(activation_funcs) = len(coeffs[i;:]) for i in range(n)
    y = np.zeros((n, len(y0)))
    y[0, :] = y0
    for t in range(1, n):
        for i in range(len(y0)):
            tail = 0
            for j in range(len(y0)):
                if j < i:
                    tail += y[t, i] * coeffs[i, j]
                else:
                    tail += y[t - 1, i] * coeffs[i, j]
            y[t, i] = x(t) * coeffs[i, 0] + tail
            y[t, i] = activation_funcs[i](y[t, i])
    return y


def show_result(y):
    print(y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y[:n - m, 0], y[:n - m, 1], y[:n - m, 2], color='blue', label='parametric curve')
    ax.plot(y[n - m:, 0], y[n - m:, 1], y[n - m:, 2], color='pink', label='parametric curve')
    ax.scatter(y[0, 0], y[0, 1], y[0, 2], s=[20], color='red')

    plt.show()

def stable_example():
    y0 = np.array([0.1, 0.1, 0.1])  # initial condition
    # [X, 1, 2, 3], i - i'th neuron
    coeffs = np.array([[1, -0.5, 0.2, 0.7],
                      [1, 0.2, 0.1, 1.2],
                      [0, -0.3, -0.5, 0.5]])
    x = x1
    activation_funcs = np.array([a_func1, a_func2, a_func1])

    y = calculate_sync(coeffs, activation_funcs, x, y0, n)
    show_result(y)

def cycle_example():
    y0 = np.array([0.1, 0.1, 0.1])  # initial condition
    # [X, 1, 2, 3], i - i'th neuron
    coeffs = np.array([[1, -0.5, 0.2, 0.7],
                      [1, 0.2, 0.9, 1.2],
                      [0, -0.3, -0.5, 0.5]])
    x = x1
    activation_funcs = np.array([a_func1, a_func2, a_func1])

    y = calculate_async(coeffs, activation_funcs, x, y0, n)

    show_result(y)


n = 1000  # number of iterations
m = 10 # number of final iterations which are shown as pink
# Various X(t) functions:
x1 = lambda t: 0
x2 = lambda t: 1./t
x3 = lambda t: np.sin((np.pi * t)/10)
# Various activation functions:
a_func1 = lambda h: np.tanh(h)
a_func2 = lambda h: np.exp(-h**2)

cycle_example()
