import numpy as np
import matplotlib.pyplot as plt


def coulomb_friction(x):
    return -1 if x < 0 else 1


def smoothed_coulomb_friction(x, t):
    assert t>0
    return -1 if x < -t/2 else 1 if x > t/2 else 2*x/t


def visualize_coulomb(t):
    x = np.linspace(-1, 1, 1000)
    y1 = np.array([coulomb_friction(l) for l in x])
    y2 = np.array([smoothed_coulomb_friction(l, t) for l in x])
    
    plt.plot(x, y1, label='Coulomb friction')
    plt.plot(x, y2, label='Smoothed Coulomb friction')
    
    plt.legend()
    plt.xlabel("Input")
    plt.ylabel("Friction")
    plt.title("Coulomb vs Smoothed Coulomb friction")
    plt.grid(True)
    plt.show()


def visualize_coulomb_derivatives(t):
    x = np.linspace(-1, 1, 1000)
    y1 = np.array([zo_gradient(coulomb_friction, l,) for l in x])
    pass


def gaussian_sampler_factory(sigma, d):
    def sampler(M, rng):
        return rng.normal(scale=sigma, size=(M, d))
    return sampler

def gaussian_grad_logp(omegas, sigma):
    # omegas shape (M, d) or (d,)
    return -omegas / (sigma**2)


def zo_gradient(f, x, sample_omega, grad_log_p, M=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    omegas = sample_omega(M, rng)
    xs = x.reshape(1, -1) + omegas
    f_vals = np.array([f(xi) for xi in xs])
    glp = grad_log_p(omegas)
    integrand = - (f_vals[:, None] * glp)
    return integrand.mean(axis=0)

def FoGB():
    pass

visualize_coulomb(0.1)