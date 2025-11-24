import numpy as np
import matplotlib.pyplot as plt

# --- paramètres ---
sigma = 0.1
M = 500  # nombre d'échantillons pour le gradient
rng = np.random.default_rng()


# --- définition de la fonction R2 -> R ---
def f2(x, y):
    """Fonction avec discontinuité en x=0 et discontinuité du gradient en y=0."""
    return np.where(x >= 0, x + np.abs(y), -1 + np.abs(y))


# --- estimateur de gradient zéro-ordre ---
def zo_gradient_2d(f, x, y, sigma=0.1, M=50_00):
    """Estimation du gradient en (x, y) par l'estimateur zero-order."""
    # position courante
    X = np.array([x, y])

    # échantillons bruités autour de (x,y)
    omegas = rng.normal(0, sigma, size=(M, 2))
    Xs = X + omegas  # (M,2)

    # évaluations de f pour chaque point
    f_vals = f(Xs[:, 0], Xs[:, 1])  # (M,)

    # facteur lié à ∇_ω log p(ω)
    weights = -omegas / (sigma**2)  # (M,2)

    # estimation de l'espérance
    grad_est = -(f_vals[:, None] * weights).mean(axis=0)
    return grad_est  # (2,)


# --- on calcule le gradient sur une grille ---
xv = np.linspace(-1, 1, 40)
yv = np.linspace(-1, 1, 40)
X, Y = np.meshgrid(xv, yv)

U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        grad = zo_gradient_2d(f2, X[i, j], Y[i, j], sigma=sigma, M=M)
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


# --- visualisation du champ de gradients ---

fig, ax = plt.subplots(figsize=(8, 6))
cont = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.8)
ax.quiver(X, Y, U, V, color="red", scale=15)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Champ de gradients (Zero-Order) de f(x,y)")
plt.colorbar(cont, ax=ax, label="f(x,y)")
plt.show()
