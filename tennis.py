import numpy as np
import torch
import matplotlib.pyplot as plt

y_wall = 10.0
T = 2.0
P_target = torch.tensor([8.0, 5.0])
H = 2000
dt = 1e-2
v0 = torch.tensor([1.0, 1.0], requires_grad=True)
x0 = torch.tensor([0.0, 0.0])
rng = np.random.default_rng()
sigma = 0.1
M = 100
Ns = 100
L = 3


def gaussian_density(omega, sigma):
    omega = np.asarray(omega)
    norm_sq = np.sum(omega**2, axis=-1)
    const = 1.0 / (2 * np.pi * sigma**2)
    return const * np.exp(-norm_sq / (2 * sigma**2))


def f(x, v):
    assert not np.isnan(v).any()
    assert not np.isnan(x).any()
    x = np.atleast_2d(x)
    v = np.atleast_2d(v)
    x_new = x + dt * v
    mask = x_new[:, 1] >= y_wall
    out_x = np.empty_like(x)
    out_v = np.empty_like(v)
    out_x[~mask] = x_new[~mask]
    out_v[~mask] = v[~mask]
    if np.any(mask):
        safe_vy = np.where(np.abs(v[mask, 1]) < 1e-8, 1e-8, v[mask, 1])
        t_wall = (y_wall - x[mask, 1]) / safe_vy
        out_x[mask, 0] = x[mask, 0] + v[mask, 0] * dt
        out_x[mask, 1] = -(dt - t_wall) * v[mask, 1]
        out_v[mask, 0] = v[mask, 0]
        out_v[mask, 1] = -v[mask, 1]
    if x.shape[0] == 1:
        out_x = out_x[0]
        out_v = out_v[0]
    return out_x, out_v


def grad_f(x, v):
    J = np.eye(4)
    J[0, 2] = dt
    J[1, 3] = dt
    return np.repeat(J[np.newaxis, :, :], x.shape[0], axis=0)


def zo_gb(x, v):
    X = np.atleast_2d(x)
    V = np.atleast_2d(v)
    omegas = rng.normal(0, sigma, size=(M, 4))
    Xs = X + omegas[:, :2]
    Vs = V + omegas[:, 2:]
    X_next, V_next = f(Xs, Vs)
    out = np.zeros((4, 4))
    for i in range(4):
        f_vals = np.concatenate([X_next, V_next], axis=1)[:, i][:, None]
        weights = -omegas / (sigma**2)
        grad_est = -(f_vals * weights).mean(0)
        out[i] = grad_est
    return out


def fo_gb(x, v):
    X = np.atleast_2d(x)
    V = np.atleast_2d(v)
    omegas = rng.normal(0, sigma, size=(M, 4))
    Xs = X + omegas[:, :2]
    Vs = V + omegas[:, 2:]
    grads = grad_f(Xs, Vs).mean(0)

    vx_s = rng.uniform(-L, L, size=Ns)
    vy_s = np.ones(Ns) * y_wall
    pos = np.hstack((Xs, vx_s[:, None], vy_s[:, None]))

    p_surface = gaussian_density(pos - np.hstack((Xs, Vs)), sigma)
    surf_term = (
        np.hstack((Xs, Vs)) * np.array([0, 0, -2.0, 0.0])[None, :] * p_surface[:, None]
    )

    return grads + surf_term


x = np.array([0.5, 9.5])
v = np.array([0.3, -0.4])

J = zo_gb(x, v)
J = fo_gb(x, v)
print("Jacobien estimé :\n", J)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(J, cmap="coolwarm", origin="lower")
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(["x", "y", "vx", "vy"])
ax.set_yticklabels(["x'", "y'", "vx'", "vy'"])
plt.colorbar(im, ax=ax, label="∂output / ∂input")
plt.title("Jacobien estimé (ZO gradient)")
plt.show()


class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v):
        x_np = x.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        x_next, v_next = f(x_np, v_np)
        ctx.save_for_backward(x, v)
        ctx.x_np = x_np
        ctx.v_np = v_np
        return torch.from_numpy(np.hstack((x_next, v_next))).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        g = fo_gb(ctx.x_np, ctx.v_np)
        gx = g[:, :2]
        gv = g[:, 2:]
        grad_x = grad_output @ torch.tensor(gx, dtype=grad_output.dtype)
        grad_v = grad_output @ torch.tensor(gv, dtype=grad_output.dtype)
        return grad_x, grad_v


def simulate(v0, x0):
    x = x0.clone()
    v = v0
    for _ in range(H):
        out = Step.apply(x, v)
        x = out[:2]
        v = out[2:]
    return x


def loss_fn(x):
    return (x - P_target).pow(2).sum()


opt = torch.optim.Adam([v0], lr=0.05)
for i in range(100):
    opt.zero_grad()
    x_final = simulate(v0, x0)
    L = loss_fn(x_final)
    L.backward()
    opt.step()
    if i % 10 == 0:
        print(
            f"{i:03d} | v0=({v0[0].item():.4f},{v0[1].item():.4f}) | loss={L.item():.6f}"
        )

x_final = simulate(v0, x0).detach()
print(f"\nFinal pos: {x_final.numpy()}, target: {P_target.numpy()}")
