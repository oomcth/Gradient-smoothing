import numpy as np
import matplotlib.pyplot as plt

m = 1.0
mu = 0.5
dt = 0.01
T = 1.0
N = int(T / dt)
x_target = 1.0

def simulate(theta, noise=0.0):
    x, v = 0.0, theta + noise
    for _ in range(N):
        force = -mu * np.sign(v) if abs(v) > 1e-6 else 0.0
        a = force / m
        v += a * dt
        x += v * dt
    return x

def simulate_smooth(theta, noise=0.0, beta=50.0):
    x, v = 0.0, theta + noise
    for _ in range(N):
        force = -mu * np.tanh(beta*v)
        a = force / m
        v += a * dt
        x += v * dt
    return x

# Cost
def cost(x):
    return (x - x_target)**2

# Zeroth order gradient estimator (finite difference w/ noise sampling)
def zeroth_order_grad(theta, eps=0.1, samples=30):
    grads = []
    for _ in range(samples):
        noise = np.random.randn() * 0.05
        f_plus = cost(simulate(theta + eps, noise))
        f_minus = cost(simulate(theta - eps, noise))
        grads.append((f_plus - f_minus) / (2*eps))
    return np.mean(grads)

# First order gradient via smoothed dynamics (analytic via FD on smooth system)
def first_order_grad(theta, eps=1e-4, samples=30, beta=50.0):
    grads = []
    for _ in range(samples):
        noise = np.random.randn() * 0.05
        f_plus = cost(simulate_smooth(theta + eps, noise, beta))
        f_minus = cost(simulate_smooth(theta - eps, noise, beta))
        grads.append((f_plus - f_minus) / (2*eps))
    return np.mean(grads)

# Optimization loop
theta = 2.0
theta_zo = theta
theta_fo = theta

lr = 0.5
steps = 60
history_zo = []
history_fo = []

for _ in range(steps):
    history_zo.append(cost(simulate(theta_zo)))
    theta_zo -= lr * zeroth_order_grad(theta_zo)

    history_fo.append(cost(simulate(theta_fo)))
    theta_fo -= lr * first_order_grad(theta_fo)

# Plot results
plt.plot(history_zo, label="Zero-order (unbiased, high variance)")
plt.plot(history_fo, label="First-order (biased due to sign)")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.legend()
plt.title("Policy Gradient with Coulomb Friction")
plt.show()

print("Final theta (Zero-Order):", theta_zo)
print("Final theta (First-Order):", theta_fo)
