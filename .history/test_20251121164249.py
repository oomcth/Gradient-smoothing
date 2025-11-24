import numpy as np
import matplotlib.pyplot as plt


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


def gaussian_sampler_factory(sigma, d):
    def sampler(M, rng):
        return rng.normal(loc=0.0, scale=sigma, size=(M, d))

    return sampler


def gaussian_grad_logp_factory(sigma):
    def grad_log_p(omegas):
        return -omegas / (sigma**2)

    return grad_log_p


def zo_gradient(f, x, sample_omega, grad_log_p, M=1000):
    x = np.array([x], dtype=float)
    omegas = sample_omega(M)
    xs = x + omegas

    f_vals = np.array([f(float(xi)) for xi in xs])
    glp = grad_log_p(omegas)

    integrand = -(f_vals[:, None] * glp)
    return integrand.mean(axis=0).squeeze()


def visualize_coulomb_derivatives(t):
    x = np.linspace(-1, 1, 1000)
    y = np.array([zo_gradient(coulomb_friction, l) for l in x])

    plt.plot(x, y, label="ZoBG derivative estimate")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_estimator_variance(t, M=500, K=50):
    sigma = t
    x = np.linspace(-1, 1, 200)

    sampler = gaussian_sampler_factory(sigma, 1)
    grad_log_p = gaussian_grad_logp_factory(sigma)

    variances = []

    for l in x:
        estimates = []
        for _ in range(K):
            rng = np.random.default_rng()  # Re-crée une graine différente
            est = zo_gradient(coulomb_friction, l, sampler, grad_log_p, M=M, rng=rng)
            estimates.append(est)
        variances.append(np.var(estimates, ddof=1))

    plt.plot(x, variances, label="Variance de l'estimateur ZoBG")
    plt.xlabel("x")
    plt.ylabel("Variance")
    plt.title(f"Variance pour σ={sigma}")
    plt.legend()
    plt.grid()
    plt.show()


visualize_estimator_variance(0.1, M=100)
visualize_coulomb_derivatives(0.1)
visualize_coulomb(0.1)
