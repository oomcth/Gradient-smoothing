import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sigma = 0.1
M = 500
Ns = 500
L = 3
rng = np.random.default_rng()


def f2(x, y):
    return np.where(x >= 0, x + np.abs(y), -1 + np.abs(y))


def grad_f2(Xs):
    x = Xs[:, 0]
    y = Xs[:, 1]
    dfdx = np.where(x > 0, 1.0, 0.0)
    dfdy = np.where(y > 0, 1.0, -1.0)
    return np.stack((dfdx, dfdy), axis=1)


def gaussian_density(omega, sigma):
    omega = np.asarray(omega)
    norm_sq = np.sum(omega**2, axis=-1)
    const = 1.0 / (2 * np.pi * sigma**2)
    return const * np.exp(-norm_sq / (2 * sigma**2))


def zo_gradient_2d(f, x, y):
    X = np.array([x, y])
    omegas = rng.normal(0, sigma, size=(M, 2))
    Xs = X + omegas
    f_vals = f(Xs[:, 0], Xs[:, 1])
    weights = -omegas / (sigma**2)
    grad_est = -(f_vals[:, None] * weights).mean(axis=0)
    return grad_est


def fo_gradient_2d(f, x, y):
    X = np.array([x, y])
    omegas = rng.normal(0, sigma, size=(M, 2))
    Xs = X + omegas
    grads_vals = grad_f2(Xs)
    y_s = rng.uniform(-L, L, size=Ns)
    z = np.stack((np.zeros_like(y_s), y_s), axis=1)
    p_surface = gaussian_density(z - X, sigma)
    surf_term = np.array([1.0, 0.0]) * (2 * L / Ns) * np.sum(p_surface)
    return grads_vals.mean(0) + surf_term


def bi_gradient_2d(f, x, y):
    X = np.array([x, y])[np.newaxis, :]
    y_s = rng.uniform(-L, L, size=Ns)
    z = np.stack((np.zeros_like(y_s), y_s), axis=1)
    p_surface = gaussian_density(z - X, sigma)
    surf_term = np.array([1.0, 0.0]) * (2 * L / Ns) * np.sum(p_surface)
    return grad_f2(X)[0] + surf_term


def continuous_bi_gradient_2d(f, x, y):
    X = np.array([x, y])[np.newaxis, :]
    y_s = rng.uniform(-L, L, size=Ns)
    z = np.stack((np.zeros_like(y_s), y_s), axis=1)
    p_surface = gaussian_density(z - X, sigma)
    surf_term = np.array([1.0, 0.0]) * (2 * L / Ns) * np.sum(p_surface)

    y2_s = rng.uniform(-L, L, size=Ns)
    z2 = np.stack((y2_s, np.zeros_like(y2_s)), axis=-1)
    return grad_f2(X)[0] + surf_term


def plot_f2_3d(f, x_range=(-1, 1), y_range=(-1, 1), n=200):
    xv = np.linspace(x_range[0], x_range[1], n)
    yv = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(xv, yv)
    Z = f(X, Y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_title("3D view of f2(x, y)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8)
    plt.show()


plot_f2_3d(f2, x_range=(-1, 1), y_range=(-1, 1))


Xmin = -0.4
Xmax = 0.4
xv = np.linspace(Xmin, Xmax, 40)
yv = np.linspace(-1, 1, 40)
X, Y = np.meshgrid(xv, yv)

U_zo = np.zeros_like(X)
V_zo = np.zeros_like(Y)
U_fo = np.zeros_like(X)
V_fo = np.zeros_like(Y)
U_bi = np.zeros_like(X)
V_bi = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        grad_zo = zo_gradient_2d(f2, X[i, j], Y[i, j])
        grad_fo = fo_gradient_2d(f2, X[i, j], Y[i, j])
        grad_bi = bi_gradient_2d(f2, X[i, j], Y[i, j])
        U_zo[i, j], V_zo[i, j] = grad_zo
        U_fo[i, j], V_fo[i, j] = grad_fo
        U_bi[i, j], V_bi[i, j] = grad_bi


def plot_vector_field(X, Y, U, V, title):
    norm = np.sqrt(U**2 + V**2)
    Udir = U / (norm + 1e-8)
    Vdir = V / (norm + 1e-8)
    fig, ax = plt.subplots(figsize=(7, 6))
    cont = ax.contourf(X, Y, norm, levels=30, cmap="magma", alpha=0.8)
    step = 2
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        Udir[::step, ::step],
        Vdir[::step, ::step],
        color="white",
        scale=25,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(cont, ax=ax, label="‖∇f(x,y)‖")
    plt.show()


plot_vector_field(X, Y, U_zo, V_zo, "Gradient field (Zero-Order)")
plot_vector_field(X, Y, U_fo, V_fo, "Gradient field (First-Order)")
plot_vector_field(X, Y, U_bi, V_bi, "Gradient field (Bias-Term)")

grad_true = grad_f2(np.stack([X.ravel(), Y.ravel()], axis=1)).reshape(X.shape + (2,))
U_true, V_true = grad_true[..., 0], grad_true[..., 1]


def mse(U1, V1, U2, V2):
    return np.mean((U1 - U2) ** 2 + (V1 - V2) ** 2)


print(f"Zero-Order vs fo : {mse(U_zo, V_zo, U_fo, V_fo):.4e}")
print(f"Bias-Term vs fo : {mse(U_bi, V_bi, U_true, V_fo):.4e}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ["Zero-Order", "First-Order", "Bias-Term"]
fields = [(U_zo, V_zo), (U_fo, V_fo), (U_bi, V_bi)]

for ax, (U_, V_), title in zip(axes, fields, titles):
    norm = np.sqrt(U_**2 + V_**2)
    Udir = U_ / (norm + 1e-8)
    Vdir = V_ / (norm + 1e-8)
    cont = ax.contourf(X, Y, norm, levels=30, cmap="magma", alpha=0.8)
    step = 2
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        Udir[::step, ::step],
        Vdir[::step, ::step],
        color="white",
        scale=25,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.colorbar(cont, ax=axes.ravel().tolist(), label="‖∇f(x,y)‖")
plt.suptitle("Comparison of the three gradient estimations")
plt.show()
