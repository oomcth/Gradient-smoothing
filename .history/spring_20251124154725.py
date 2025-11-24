import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use("TkAgg")


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

    def step(self, x1, v1, x2, v2, u):
        F1 = u.clone()
        penetration = x1 + 1 - x2
        contact = torch.clamp(penetration, min=0.0)
        F_contact = -self.k * contact
        F1 = F1 + F_contact - self.c * v1
        F2 = -F_contact - self.c * v2

        v1 = v1 + F1 * self.dt
        v2 = v2 + F2 * self.dt
        x1 = x1 + v1 * self.dt
        x2 = x2 + v2 * self.dt

        return x1, v1, x2, v2

    def simulate(self, u_seq, x1_0, v1_0, x2_0, v2_0):
        x1, v1 = x1_0, v1_0
        x2, v2 = x2_0, v2_0
        self.x1s.append(x1.item())
        self.v1s.append(v1.item())
        self.x2s.append(x2.item())
        self.v2s.append(v2.item())

        traj_x1, traj_x2 = [x1.item()], [x2.item()]

        for t in range(self.H):
            x1, v1, x2, v2 = self.step(x1, v1, x2, v2, u_seq[t])
            self.x1s.append(x1.item())
            self.v1s.append(v1.item())
            self.x2s.append(x2.item())
            self.v2s.append(v2.item())

        return np.array(traj_x1), np.array(traj_x2)

    def forward(self, u_seq, x1_0, v1_0, x2_0, v2_0, goal):
        x1, v1 = x1_0, v1_0
        x2, v2 = x2_0, v2_0
        for t in range(self.H):
            x1, v1, x2, v2 = self.step(x1, v1, x2, v2, u_seq[t])
        loss = (x2 - goal).pow(2)
        return loss, x1, x2

    def baxkward(self):
        pass


env = PushingEnv(H=200, k=100, c=1)
H = env.H
u_seq = torch.full((H,), 1.0, requires_grad=True)

u_seq.requires_grad_()
x1_0 = torch.tensor(0.0)
v1_0 = torch.tensor(1.0)
x2_0 = torch.tensor(2.0)
v2_0 = torch.tensor(0.0)
goal = torch.tensor(5.0)

loss, _, _ = env(u_seq, x1_0, v1_0, x2_0, v2_0, goal)
loss.backward()
print("Loss :", loss.item())
print("Gradient sur u_seq :", u_seq.grad[:5])

traj_x1, traj_x2 = env.simulate(u_seq.detach(), x1_0, v1_0, x2_0, v2_0)

fig, ax = plt.subplots()
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 1)
ax.set_xlabel("Position")
ax.set_yticks([])

(block1,) = ax.plot([], [], "s", markersize=20, color="tab:blue", label="Bloc 1")
(block2,) = ax.plot([], [], "s", markersize=20, color="tab:red", label="Bloc 2")
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
