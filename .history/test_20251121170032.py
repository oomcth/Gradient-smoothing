import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
sigma = 0.1
t = 0.1


def coulomb_friction(x):
    return -1 if x < 0 else 1


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


def zo_gradient(f, x, M=1000):
    x = np.array([x], dtype=float)
    omegas = rng.normal(0.0, sigma)
    xs = x + omegas  # broadcast

    f_vals = np.array([f(float(xi)) for xi in xs])  # list → float
    glp = grad_log_p(omegas)

    integrand = -(f_vals[:, None] * glp)
    return integrand.mean(axis=0).squeeze()  # return scalar if d=1


def visualize_coulomb_derivatives(t):
    sigma = t
    x = np.linspace(-1, 1, 1000)
    sampler = gaussian_sampler_factory(sigma, 1)
    grad_log_p = gaussian_grad_logp_factory(sigma)

    y = np.array([zo_gradient(coulomb_friction, l, sampler, grad_log_p) for l in x])

    plt.plot(x, y, label="ZoBG derivative estimate")
    plt.legend()
    plt.grid(True)
    plt.show()


# Run tests
visualize_coulomb_derivatives(t)
visualize_coulomb(t)
