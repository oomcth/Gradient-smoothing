import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return np.where(x >= 0, x + np.abs(y), -1 + np.abs(y))


x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=False)

ax.set_title("Fonction avec discontinuité et discontinuité du gradient")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")
plt.show()
