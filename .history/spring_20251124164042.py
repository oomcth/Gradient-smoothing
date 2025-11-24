import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use("TkAgg")
sigma = 0.1
M = 1000
rng = np.random.default_rng()


def F_contact_np(x1, x2, k):
    penetration = x1 + 1 - x2
    penetration = np.maximum(0.0, penetration)
    return np.array(-k * penetration)


def grad_F_contact(x1, x2, k):
    penetration = x1 + 1 - x2
    mask = (penetration > 0).astype(float)
    grad_x1 = -k * mask
    grad_x2 = k * mask
    grad = np.stack((grad_x1, grad_x2), axis=-1)
    return grad


def zo_gb(x1, x2, k):
    X = np.array([x1, x2])
    omegas = rng.normal(0, sigma, size=(M, 2))
    Xs = X + omegas
    f_vals = F_contact_np(Xs[:, 0], Xs[:, 1], k)
    weights = -omegas / (sigma**2)
    grad_est = -(f_vals[:, None] * weights).mean(axis=0)
    return np.array(grad_est)


def fo_gb(x1, x2, k):
    X = np.array([x1, x2])
    omegas = rng.normal(0, sigma, size=(M, 2))
    Xs = X + omegas
    grads_vals = grad_F_contact(Xs[:, 0], Xs[:, 1], k)
    return grads_vals.mean(0)


def bi_gb(x1, x2, k):
    X = np.array([x1, x2])
    Xs = X[np.newaxis, :]
    f_vals = grad_F_contact(Xs[:, 0], Xs[:, 1], k)
    return f_vals.mean(0)


class F_contact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, k, gtype=2):
        ctx.gtype = gtype
        ctx.k = k
        ctx.x1_np = x1.detach().cpu().numpy()
        ctx.x2_np = x2.detach().cpu().numpy()
        f = F_contact_np(ctx.x1_np, ctx.x2_np, k)
        out = torch.from_numpy(f).requires_grad_()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.gtype == 0:
            grad_x1, grad_x2 = zo_gb(ctx.x1_np, ctx.x2_np, ctx.k)
        if ctx.gtype == 1:
            grad_x1, grad_x2 = fo_gb(ctx.x1_np, ctx.x2_np, ctx.k)
        if ctx.gtype == 2:
            grad_x1, grad_x2 = bi_gb(ctx.x1_np, ctx.x2_np, ctx.k)
        return (
            grad_output * torch.from_numpy(np.array(grad_x1)),
            grad_output * torch.from_numpy(np.array(grad_x2)),
            None,
            None,
        )


class PushingEnv(nn.Module):
    def __init__(self, H=200, dt=0.01, k=100.0, c=1.0):
        super().__init__()
        self.H = H
        self.dt = dt
        self.k = k
        self.c = c
        self.x1s = []
        self.v1s = []
        self.x2s = []
        self.v2s = []

    def step(self, x1, v1, x2, v2, u, gtype):
        F_contact_ = F_contact.apply(x1, x2, self.k, gtype)
        F1 = F_contact_ - self.c * v1
        F2 = -F_contact_ - self.c * v2
        v1 = v1 + F1 * self.dt
        v2 = v2 + F2 * self.dt
        x1 = x1 + v1 * self.dt
        x2 = x2 + v2 * self.dt
        return x1, v1, x2, v2

    def simulate(self, u_seq, x1, v1, x2, v2, gtype):
        self.x1s = [x1.item()]
        self.x2s = [x2.item()]
        for t in range(self.H):
            x1, v1, x2, v2 = self.step(x1, v1, x2, v2, u_seq[t], gtype)
            self.x1s.append(x1.item())
            self.x2s.append(x2.item())
        return np.array(self.x1s), np.array(self.x2s)

    def forward(self, u_seq, x1_0, v1_0, x2_0, v2_0, goal, gtype):
        x1, v1, x2, v2 = x1_0, v1_0, x2_0, v2_0
        x1.requires_grad_()
        v1.requires_grad_()
        x2.requires_grad_()
        v2.requires_grad_()
        for t in range(self.H):
            x1, v1, x2, v2 = self.step(x1, v1, x2, v2, u_seq[t], gtype)
        loss = (x2 - goal).pow(2)
        return loss, x1, x2


Ns = 200
env = PushingEnv(H=200, k=100, c=1)
H = env.H
u_seq = torch.full((H,), 1.0, requires_grad=True)
goal = torch.tensor(5.0)

gtypes = [0, 1, 2]
names = ["zo_gb", "fo_gb", "bi_gb"]
means = []
vars_ = []

for gtype, name in zip(gtypes, names):
    grads = []
    for _ in range(Ns):
        env = PushingEnv(H=H, k=100, c=1)
        u_seq = torch.full((H,), 1.0, requires_grad=True)
        x1_0 = torch.tensor(0.0, requires_grad=True)
        v1_0 = torch.tensor(10.0, requires_grad=True)
        x2_0 = torch.tensor(2.0, requires_grad=True)
        v2_0 = torch.tensor(0.0, requires_grad=True)
        loss, _, _ = env(u_seq, x1_0, v1_0, x2_0, v2_0, goal, gtype)
        loss.backward()
        grads.append(v1_0.grad.item())
    grads = np.array(grads)
    print(name, "mean:", grads.mean(), "variance:", grads.var())
    means.append(grads.mean())
    vars_.append(grads.var())
    plt.figure()
    plt.hist(grads, bins=20, alpha=0.7)
    plt.title(name)

plt.figure()
x = np.arange(len(gtypes))
plt.bar(x - 0.2, means, width=0.4, label="mean")
plt.bar(x + 0.2, vars_, width=0.4, label="variance")
plt.xticks(x, names)
plt.legend()
plt.title("Gradient statistics v0")
plt.show()

env = PushingEnv(H=200, k=100, c=1)
u_seq = torch.full((env.H,), 1.0)
x1_0 = torch.tensor(0.0)
v1_0 = torch.tensor(10.0)
x2_0 = torch.tensor(2.0)
v2_0 = torch.tensor(0.0)
traj_x1, traj_x2 = env.simulate(u_seq, x1_0, v1_0, x2_0, v2_0, 2)

fig, ax = plt.subplots()
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 1)
ax.set_xlabel("Position")
ax.set_yticks([])
(block1,) = ax.plot([], [], "s", markersize=20, color="tab:blue", label="Block 1")
(block2,) = ax.plot([], [], "s", markersize=20, color="tab:red", label="Block 2")
(spring,) = ax.plot([], [], "k-", lw=2)
goal_line = ax.axvline(goal.item(), color="green", linestyle="--", label="Goal")
ax.legend()


def init():
    block1.set_data([], [])
    block2.set_data([], [])
    spring.set_data([], [])
    return block1, block2, spring


def update(frame):
    x1, x2 = traj_x1[frame], traj_x2[frame]
    block1.set_data([x1], [0])
    block2.set_data([x2], [0])
    spring.set_data([x1, x2], [0, 0])
    return block1, block2, spring


ani = FuncAnimation(
    fig, update, frames=len(traj_x1), init_func=init, blit=True, interval=30
)
plt.show()

epochs = 20
loss_hist = {}
v0_trained = {}

for gtype, name in zip(gtypes, names):
    losses = []
    v1_0 = torch.tensor(10.0, requires_grad=True)
    opt = torch.optim.SGD([v1_0], lr=0.1)
    for epoch in range(epochs):
        env = PushingEnv(H=H, k=100, c=1)
        u_seq = torch.full((H,), 1.0, requires_grad=True)
        x1_0 = torch.tensor(0.0)
        x2_0 = torch.tensor(2.0)
        v2_0 = torch.tensor(0.0)
        opt.zero_grad()
        loss, _, _ = env(u_seq, x1_0, v1_0, x2_0, v2_0, goal, gtype)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    loss_hist[name] = losses
    v0_trained[name] = v1_0.item()

plt.figure()
for name in names:
    plt.plot(loss_hist[name], label=name)
plt.legend()
plt.title("Loss per epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

for gtype, name in zip(gtypes, names):
    env = PushingEnv(H=200, k=100, c=1)
    u_seq = torch.full((env.H,), 1.0)
    x1_0 = torch.tensor(0.0)
    v1_0 = torch.tensor(v0_trained[name])
    x2_0 = torch.tensor(2.0)
    v2_0 = torch.tensor(0.0)
    traj_x1, traj_x2 = env.simulate(u_seq, x1_0, v1_0, x2_0, v2_0, gtype)
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Position")
    ax.set_yticks([])
    (block1,) = ax.plot([], [], "s", markersize=20, color="tab:blue", label="Block 1")
    (block2,) = ax.plot([], [], "s", markersize=20, color="tab:red", label="Block 2")
    (spring,) = ax.plot([], [], "k-", lw=2)
    goal_line = ax.axvline(goal.item(), color="green", linestyle="--", label="Goal")
    ax.legend()

    def init():
        block1.set_data([], [])
        block2.set_data([], [])
        spring.set_data([], [])
        return block1, block2, spring

    def update(frame):
        x1, x2 = traj_x1[frame], traj_x2[frame]
        block1.set_data([x1], [0])
        block2.set_data([x2], [0])
        spring.set_data([x1, x2], [0, 0])
        return block1, block2, spring

    ani = FuncAnimation(
        fig, update, frames=len(traj_x1), init_func=init, blit=True, interval=30
    )
    plt.title(name)
    plt.show()
