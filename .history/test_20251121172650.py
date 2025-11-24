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


def g_prime(x_minus_u, mu=0.0, sigma=1.0):
    g = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -((x_minus_u - mu) ** 2) / (2 * sigma**2)
    )
    # Dérivée g'(x_minus_u) = - ((x_minus_u - mu)/sigma^2) * g(x_minus_u)
    g_prime = -((x_minus_u - mu) / sigma**2) * g
    return g_prime


def zo_gradient(f, x, M=100):
    x = np.array([x], dtype=np.float64)
    omegas = rng.normal(loc=0, scale=sigma, size=M)
    xs = x + omegas
    fx = np.array([f(xi) for xi in xs])
    glp = omegas / (sigma**2)
    return (fx * glp).mean()


def visualize_coulomb_derivatives():
    x = np.linspace(-1, 1, 1000)

    y1 = np.array(
        [
            zo_gradient(
                coulomb_friction,
                l,
            )
            for l in x
        ]
    )
    y2 = np.array(
        [
            zo_gradient(
                coulomb_friction,
                l,
            )
            for l in x
        ]
    )

    plt.plot(x, y1, label="ZoBG derivative estimate")
    plt.plot(x, y2, label="ZoBG2 derivative estimate")
    plt.legend()
    plt.grid(True)
    plt.show()


print(zo_gradient(coulomb_friction, 10, 1000))
input()
visualize_coulomb_derivatives()
visualize_coulomb(t)
