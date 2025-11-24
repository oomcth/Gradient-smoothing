import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng()
sigma = 0.1
t = 0.1


def coulomb_friction(x):
    return x**2 + (-1 if x < 0 else 1)


def smoothed_coulomb_friction(x, t):
    assert t > 0
    return x**2 + (-1 if x < -t / 2 else 1 if x > t / 2 else 2 * x / t)


def ReLU(x):
    return np.maximum(x, 0)


def visualize(f):
    xs = np.linspace(-1, 1, 1000)
    y = np.array([f(x) for x in xs])

    plt.plot(xs, y, label="f")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f")
    plt.title("y = f(x)")
    plt.grid(True)
    plt.show()


def gaussian_pdf(x, mu=0.0, sigma=1.0):
    return 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_cdf(x, mu=0.0, sigma=1.0):
    return norm.cdf(x, loc=mu, scale=sigma)


def zo_gradient(f, x, M=100):
    x = np.array([x], dtype=np.float64)
    omegas = np.random.normal(loc=0, scale=sigma, size=M)
    xs = x + omegas
    fx = np.array([f(xi) for xi in xs])
    glp = -omegas / (sigma**2)
    return -(fx * glp).sum(axis=0) / M


def fo_gradient(f, x, M=100):
    x = np.array([x], dtype=np.float64)
    omegas = np.random.normal(loc=0, scale=sigma, size=M)
    xs = x + omegas
    if f == coulomb_friction:
        grads = 2 * xs
        return grads.mean() + 2 * gaussian_pdf(-x, mu=0, sigma=sigma)
    elif f == ReLU:
        grads = xs >= 0
        return grads.mean()


def bi_gradient(f, x, M=100):
    x = np.array([x], dtype=np.float64)
    if f == coulomb_friction:
        grads = 2 * x
        return grads.mean() + 2 * gaussian_pdf(-x, mu=0, sigma=sigma)
    elif f == ReLU:
        grads = x >= 0
        return grads.mean()


def continuous_bi_gradient(f, x, M=100):
    x = np.array([x], dtype=np.float64)
    if f == coulomb_friction:
        grads = 2 * x
        return grads.mean() + 2 * gaussian_pdf(-x, mu=0, sigma=sigma)
    elif f == ReLU:
        grads = gaussian_cdf(x)
        return grads.mean()


def visualize_derivatives(f):
    xs = np.linspace(-1, 1, 1000)

    y1 = np.array([zo_gradient(f, x, M=100) for x in xs])
    y2 = np.array([fo_gradient(f, x, M=100) for x in xs])
    y3 = np.array([bi_gradient(f, x, M=100) for x in xs])
    y4 = np.array([continuous_bi_gradient(f, x, M=100) for x in xs])

    plt.plot(xs, y1, label="ZoBG derivative estimate")
    plt.plot(xs, y2, label="FoBG derivative estimate")
    plt.plot(xs, y3, label="biBG derivative estimate")
    plt.plot(xs, y4, label="continuous biBG derivative estimate")
    plt.legend()
    plt.grid(True)
    plt.show()


visualize(coulomb_friction)
visualize(ReLU)
visualize_derivatives(ReLU)
visualize_derivatives(coulomb_friction)
