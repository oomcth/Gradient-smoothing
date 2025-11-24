import numpy as np
import matplotlib.pyplot as plt


def coulomb_friction(x):
    return -1 if x < 0 else 1


def smoothed_coulomb_friction(x, t):
    assert t>0
    return -1 if x < -t/2 else 1 if x > t/2 else 2*x/t


def visualize_coulomb(t):
    x = np.linspace(-1, 1, 1000)
    y1 = np.array([coulomb_friction(l) for l in x])
    y2 = np.array([smoothed_coulomb_friction(l, t) for l in x])
    
    plt.plot(x, y1, label='Coulomb friction')
    plt.plot(x, y2, label='Smoothed Coulomb friction')
    
    plt.legend()
    plt.xlabel("Input")
    plt.ylabel("Friction")
    plt.title("Coulomb vs Smoothed Coulomb friction")
    plt.grid(True)
    plt.show()




def FoGB():
    pass

visualize_coulomb(0.1)