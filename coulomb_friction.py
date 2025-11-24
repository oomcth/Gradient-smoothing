import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng()
sigma = 0.1
t = 0.1
M = 100


def coulomb_friction(x):
    return -1 if x < 0 else 1


def smoothed_coulomb_friction(x, t=t):
    assert t > 0
    return -1 if x < -t / 2 else 1 if x > t / 2 else 2 * x / t


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
    grads = (
        0 * xs
        if f == coulomb_friction
        else (2 / (t)) * ((-t / 2 < xs) & (xs < t / 2)).astype(int)
    )
    return grads.mean() + (2 if f == coulomb_friction else 0) * gaussian_pdf(
        -x, mu=0, sigma=sigma
    )


def bi_gradient(f, x, M=100):
    x = np.array([x], dtype=np.float64)
    grads = (
        0 * x
        if f == coulomb_friction
        else (2 / t) * ((-t / 2 < x) & (x < t / 2)).astype(int)
    )
    return grads.mean() + (2 if f == coulomb_friction else 0) * gaussian_pdf(
        -x, mu=0, sigma=sigma
    )


def continuous_bi_gradient(f, x, M=100):
    x = np.array([x], dtype=np.float64)
    grads = (
        0 * x
        if f == coulomb_friction
        else (2 / (t)) * ((-t / 2 < x) & (x < t / 2)).astype(int)
    )
    return grads.mean() + (2 if f == coulomb_friction else 0) * gaussian_pdf(
        -x, mu=0, sigma=sigma
    )


def visualize_derivatives(f, M=100):
    xs = np.linspace(-1, 1, 1000)

    y1 = np.array([zo_gradient(f, x, M) for x in xs])
    y2 = np.array([fo_gradient(f, x, M) for x in xs])
    y3 = np.array([bi_gradient(f, x, M) for x in xs])
    y4 = np.array([continuous_bi_gradient(f, x, M) for x in xs])

    plt.plot(xs, y1, label="ZoBG derivative estimate")
    plt.plot(xs, y2, label="FoBG derivative estimate")
    if f == coulomb_friction:
        plt.plot(xs, y3, label="biBG derivative estimate")
        plt.plot(xs, y4, label="continuous biBG derivative estimate")
    plt.legend()
    plt.grid(True)
    plt.show()


def gradient_variance(f, grad_fn, x, M, N_samples=100):
    grads = np.array([grad_fn(f, x, M) for _ in range(N_samples)])
    mean_grad = np.mean(grads)
    var = np.mean((grads - mean_grad) ** 2) * N_samples / (N_samples - 1)
    return var


x = 0.1
M_values = np.arange(10, 1001, 10)
var_zo = []
var_fo = []
for M in M_values:
    var_zo.append(gradient_variance(coulomb_friction, zo_gradient, x, M))
    var_fo.append(gradient_variance(coulomb_friction, fo_gradient, x, M))

plt.figure(figsize=(8, 5))
plt.plot(M_values, var_zo, label="ZO gradient variance")
plt.plot(M_values, var_fo, label="FO gradient variance")
plt.xlabel("M")
plt.ylabel("Estimated variance")
plt.yscale("log")
plt.title("Variance of ZO and FO gradients as a function of M")
plt.legend()
plt.grid(True)
plt.show()


M_values = np.arange(10, 1001, 10)
var_zo = []
var_fo = []
for M in M_values:
    var_zo.append(gradient_variance(smoothed_coulomb_friction, zo_gradient, x, M))
    var_fo.append(gradient_variance(smoothed_coulomb_friction, fo_gradient, x, M))

plt.figure(figsize=(8, 5))
plt.plot(M_values, var_zo, label="ZO gradient variance")
plt.plot(M_values, var_fo, label="FO gradient variance")
plt.xlabel("M")
plt.ylabel("Estimated variance")
plt.yscale("log")
plt.title("Variance of ZO and FO gradients as a function of M")
plt.legend()
plt.grid(True)
plt.show()

visualize(coulomb_friction)
visualize_derivatives(coulomb_friction, int(M / 10))
visualize(coulomb_friction)
visualize_derivatives(coulomb_friction, M)
visualize(coulomb_friction)
visualize_derivatives(smoothed_coulomb_friction, int(M / 10))
visualize(coulomb_friction)
visualize_derivatives(smoothed_coulomb_friction, M)
