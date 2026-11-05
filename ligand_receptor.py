# ligand_receptor.py

import numpy as np
import matplotlib.pyplot as plt

def Fact(L, Kd):
    """Fractional receptor activation: L / (Kd + L)"""
    return L / (Kd + L)

def Fact_costim(G, Kd1, S, Kd2):
    """Co-activation of one receptor by GLP1 (G, Kd1) and semaglutide (S, Kd2)"""
    return (Kd2 * G + Kd1 * S) / (Kd1 * Kd2 + Kd2 * G + Kd1 * S)

if __name__ == "__main__":
    Kd = 10.0                       # dissociation constant [nM]
    L = np.logspace(-2, 4, 500)     # ligand concentration, 0.01 to 10,000 [nM]

    plt.figure()
    plt.semilogx(L, Fact(L, Kd), lw=2)
    plt.axhline(0.5, ls="--", c="gray", lw=1)
    plt.axvline(Kd, ls="--", c="gray", lw=1)
    plt.ylim(0, 1)
    plt.xlabel("[L] (nM, log scale)")
    plt.ylabel("Fraction active (Fact)")
    plt.show()
