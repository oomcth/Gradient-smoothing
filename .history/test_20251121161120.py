import numpy as np



def coulomb_friction(x):
    return -1 if x < 0 else 1


def smoothed_coulomb_friction(x, t):
    assert t>0
    return -1 if x < -t/2 else 1 if t > t/2 else 2*x/t


def visualize_coulom(t):
    x = np.linspace(-10, 10, 1000)
    y1 = np.array([coulomb_friction(l) for l in x])
    y2 = np.array([smoothed_coulomb_friction(l, t) for l in x])