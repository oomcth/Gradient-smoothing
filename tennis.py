import numpy as np
import matplotlib.pyplot as plt


y_wall = 10.0
T = 2.0
P_target = np.array([8.0, 5.0])


def simulate(vx0, vy0):
    if vy0 <= 0:
        raise ValueError("vy0 doit être > 0 pour toucher le mur")
    t_contact = y_wall / vy0

    if T <= t_contact:
        xT = vx0 * T
        yT = vy0 * T
        ts = np.linspace(0, T, 100)
        xs = vx0 * ts
        ys = vy0 * ts
    else:
        x_contact = vx0 * t_contact
        t_after = T - t_contact
        xT = x_contact + vx0 * t_after
        yT = y_wall - vy0 * t_after

        ts1 = np.linspace(0, t_contact, 100)
        xs1 = vx0 * ts1
        ys1 = vy0 * ts1

        ts2 = np.linspace(t_contact, T, 100)
        xs2 = x_contact + vx0 * (ts2 - t_contact)
        ys2 = y_wall - vy0 * (ts2 - t_contact)

        ts = np.concatenate([ts1, ts2])
        xs = np.concatenate([xs1, xs2])
        ys = np.concatenate([ys1, ys2])
    return ts, xs, ys, np.array([xT, yT])


def loss_and_grad(v):
    vx0, vy0 = v
    t_contact = y_wall / vy0
    if T <= t_contact:
        xT = vx0 * T
        yT = vy0 * T
        L = (xT - P_target[0]) ** 2 + (yT - P_target[1]) ** 2
        dL_dvx = 2 * T * (xT - P_target[0])
        dL_dvy = 2 * T * (yT - P_target[1])
    else:
        xT = vx0 * T
        yT = 2 * y_wall - vy0 * T
        L = (xT - P_target[0]) ** 2 + (yT - P_target[1]) ** 2
        dL_dvx = 2 * T * (xT - P_target[0])
        dL_dvy = -2 * T * (yT - P_target[1])
    return L, np.array([dL_dvx, dL_dvy])


v0 = np.array([2.0, 12.0])
# v0 = np.array([2.0, 0.0])
# v0 = np.array([2.0, 5.1])
lr = 0.05
n_iter = 80

for it in range(n_iter):
    L, grad = loss_and_grad(v0)
    v0 -= lr * grad
    if it % 10 == 0:
        print(f"iter {it:03d} | v0=({v0[0]:.4f}, {v0[1]:.4f}) | loss={L:.6f}")


ts, xs, ys, Pf = simulate(v0[0], v0[1])
print(f"\nPosition finale: {Pf}, cible: {P_target}")


plt.figure(figsize=(6, 5))
plt.plot(xs, ys, label="trajectoire optimisée")
plt.axhline(y_wall, color="r", linestyle="--", label="mur y=10")
plt.scatter(*P_target, c="g", label="cible")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.title(f"Trajectoire (v0≈{v0})")
plt.show()
