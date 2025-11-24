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


def zo_gradient(f, x, sigma=0.1, M=100):
    x = np.array([x], dtype=float)
    omegas = rng.normal(loc=0.0, scale=sigma, size=int(M))

    xs = x + omegas
    f_vals = np.array([f(float(xi)) for xi in xs])

    grad_log_p = -omegas / sigma**2
    print(f_vals.shape)
    print(grad_log_p.shape)
    input()
    grad_estimate = (f_vals * grad_log_p).mean(axis=0)
    return grad_estimate


def visualize_coulomb_derivatives(M=100):
    x = np.linspace(-1, 1, 1000)
    y = np.array([zo_gradient(coulomb_friction, l, M) for l in x])
    plt.plot(x, y, label="ZoBG derivative estimate")
    plt.legend()
    plt.grid(True)
    plt.show()


# visualize_estimator_variance(0.1, M=100)
# visualize_coulomb(t)
visualize_coulomb_derivatives()
