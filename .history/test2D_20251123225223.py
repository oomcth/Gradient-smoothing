import numpy as np
import matplotlib.pyplot as plt

sigma = 0.1
M = 500_000
rng = np.random.default_rng()


def f2(x, y):
    return np.where(x >= 0, x + np.abs(y), -1 + np.abs(y))


def grad_f2(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    dfdx = np.where(x > 0, 1.0, 0.0)
    dfdx = np.where(x == 0, np.nan, dfdx)

    # df/dy : signe de y (+1 ou -1)
    dfdy = np.where(y > 0, 1.0, -1.0)
    dfdy = np.where(y == 0, np.nan, dfdy)

    return np.stack((dfdx, dfdy), axis=-1)


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
    grads = grad_f2(Xs)
    p_vals = gaussian_density(omegas, sigma)
    grads_vals = grads * p_vals[:, None]
    return grads_vals.mean(0)


xv = np.linspace(-1, 1, 40)
yv = np.linspace(-1, 1, 40)
X, Y = np.meshgrid(xv, yv)

U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        # grad = zo_gradient_2d(f2, X[i, j], Y[i, j])
        grad = fo_gradient_2d(f2, X[i, j], Y[i, j])
        U[i, j], V[i, j] = grad[0], grad[1]

Z = f2(X, Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=False)

ax.set_title("Fonction avec discontinuité et discontinuité du gradient")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
cont = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.8)
ax.quiver(X, Y, U, V, color="red", scale=15)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Champ de gradients (Zero-Order) de f(x,y)")
plt.colorbar(cont, ax=ax, label="f(x,y)")
plt.show()
