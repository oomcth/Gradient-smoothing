import numpy as np
import torch
import torch.nn as nn


class PushingEnv(nn.Module):
    def __init__(self, H=200, dt=0.01, k=100.0, c=1.0):
        """
        H : horizon temporel
        dt : pas de temps
        k : raideur (spring stiffness)
        c : amortissement visqueux
        """
        super().__init__()
        self.H = H
        self.dt = dt
        self.k = k
        self.c = c

    def step(self, x1, v1, x2, v2, u):
        """Intégration simple d'une étape"""
        # appliquer la force de contrôle sur le bloc 1
        F1 = u.clone()

        penetration = x1 - x2 - 1.0
        contact = torch.clamp(penetration, min=0.0)

        # force de contact (spring + damping)
        F_contact = -self.k * contact - self.c * (v1 - v2)

        # appliquer les forces de contact
        F1 = F1 + F_contact
        F2 = -F_contact

        # mise à jour des vitesses
        v1 = v1 + F1 * self.dt
        v2 = v2 + F2 * self.dt

        # mise à jour des positions
        x1 = x1 + v1 * self.dt
        x2 = x2 + v2 * self.dt

        return x1, v1, x2, v2

    def forward(self, u_seq, x1_0, v1_0, x2_0, v2_0, goal):
        """
        u_seq : séquence de forces appliquées sur le bloc 1 [H]
        x1_0, v1_0, x2_0, v2_0 : états initiaux
        goal : position cible du bloc 2
        """
        x1, v1 = x1_0, v1_0
        x2, v2 = x2_0, v2_0

        for t in range(self.H):
            x1, v1, x2, v2 = self.step(x1, v1, x2, v2, u_seq[t])

        # coût final : distance du bloc 2 à la cible
        loss = (x2 - goal).pow(2)
        return loss, x1, x2


# initialisation
env = PushingEnv(H=200, k=100.0)
H = env.H

# séquence de forces aléatoires (paramètres à optimiser)
u_seq = torch.zeros(H, requires_grad=True)

# états initiaux
x1_0 = torch.tensor(0.0)
v1_0 = torch.tensor(0.0)
x2_0 = torch.tensor(2.0)
v2_0 = torch.tensor(0.0)
goal = torch.tensor(5.0)

# forward + backward
loss, _, _ = env(u_seq, x1_0, v1_0, x2_0, v2_0, goal)
loss.backward()

print("Loss :", loss.item())
print("Gradient sur u_seq :", u_seq.grad[:5])
