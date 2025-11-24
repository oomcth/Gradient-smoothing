import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
sigma = 0.1
t = 0.1


def coulomb_friction(x):
    return x**2 + (-1 if x < 0 else 1)


def smoothed_coulomb_friction(x, t):
    assert t > 0
    return -1 if x < -t / 2 else 1 if x > t / 2 else 2 * x / t


def visualize_coulomb(t):
    x = np.linspace(-1, 1, 1000)
    y1 = np.array([coulomb_friction(l) for l in x])
    y2 = np.array([smoothed_coulomb_friction(l, t) for l in x])

    plt.plot(x, y1, label="Coulomb friction")
    plt.plot(x, y2, label="Smoothed Coulomb friction")

    plt.legend()
    plt.xlabel("Input")
    plt.ylabel("Friction")
    plt.title("Coulomb vs Smoothed Coulomb friction")
    plt.grid(True)
    plt.show()


def gaussian_pdf(x, mu=0.0, sigma=1.0):
    return 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


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
    grads = 2 * xs
    return grads.mean() + 2 * gaussian_pdf(-x, mu=0, sigma=sigma)


def visualize_coulomb_derivatives():
    x = np.linspace(-1, 1, 1000)

    y1 = np.array([zo_gradient(coulomb_friction, l, M=100) for l in x])
    y2 = np.array([fo_gradient(coulomb_friction, l, M=100) for l in x])

    plt.plot(x, y1, label="ZoBG derivative estimate")
    plt.plot(x, y2, label="FoBG derivative estimate")
    plt.legend()
    plt.grid(True)
    plt.show()


visualize_coulomb(t)
visualize_coulomb_derivatives()
