import numpy as np
import matplotlib.pyplot as plt

sigma = 0.1
M = 500
Ns = 500
rng = np.random.default_rng()
L = 3


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
    grads = grad_f2(Xs)
    p_vals = gaussian_density(omegas, sigma)
    grads_vals = grads * p_vals[:, None]
    y_s = rng.uniform(-L, L, size=Ns)
    z = np.stack((np.zeros_like(y_s), y_s), axis=1)
    p_surface = gaussian_density(z - X, sigma)
    surf_term = np.array([1.0, 0.0]) * (2 * L / Ns) * np.sum(p_surface)
    return grads_vals.mean(0) + surf_term


def bi_gradient_2d(f, x, y):
    X = np.array([x, y])
    y_s = rng.uniform(-L, L, size=Ns)
    z = np.stack((np.zeros_like(y_s), y_s), axis=1)
    p_surface = gaussian_density(z - X, sigma)
    surf_term = np.array([1.0, 0.0]) * (2 * L / Ns) * np.sum(p_surface)
    return grad_f2(X) + surf_term


Xmin = -0.4
Xmax = 0.4
xv = np.linspace(Xmin, Xmax, 40)
yv = np.linspace(-1, 1, 40)
X, Y = np.meshgrid(xv, yv)

U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        grad = bi_gradient_2d(f2, X[i, j], Y[i, j])
        U[i, j], V[i, j] = grad[0], grad[1]

Z = f2(X, Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=False)
ax.set_title("Fonction avec discontinuité et discontinuité du gradient")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
cont = ax.contourf(X, Y, np.sqrt(U**2 + V**2), levels=30, cmap="magma", alpha=0.8)
step = 2
norm = np.sqrt(U**2 + V**2)
Udir = U / (norm + 1e-8)
Vdir = V / (norm + 1e-8)
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
ax.set_title("Champ de gradients (Zero-Order) de f(x,y)")
plt.colorbar(cont, ax=ax, label="‖∇f(x,y)‖")
plt.show()

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.6)
plt.streamplot(X, Y, U, V, color="red", density=1.2, arrowsize=1.2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Champ de gradients (streamlines)")
plt.colorbar(label="f(x,y)")
plt.show()
