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


def zo_gradient(f, x, sample_omega, grad_log_p, M=1000, rng=None):
    """
    Monte-Carlo estimator of the zeroth-order bundled gradient:
      ∇ f_mu(x) = - E_{ω ~ p}[ f(x+ω) * ∇ log p(ω) ]
    
    Args:
      f: callable x -> scalar
      x: numpy array, point where gradient is estimated
      sample_omega: callable (M, rng) -> array shape (M, d) of samples ω
      grad_log_p: callable ω -> array shape (d,) or (M, d) of ∇_ω log p(ω)
      M: number of samples
      rng: np.random.Generator or None (will create one)
    Returns:
      grad_est: numpy array shape (d,)
    """
    if rng is None:
        rng = np.random.default_rng()
    # draw samples
    omegas = sample_omega(M, rng)   # shape (M, d)
    xs = x.reshape(1, -1) + omegas
    f_vals = np.array([f(xi) for xi in xs])
    glp = grad_log_p(omegas)
    integrand = - (f_vals[:, None] * glp)
    return integrand.mean(axis=0)

def FoGB():
    pass

visualize_coulomb(0.1)