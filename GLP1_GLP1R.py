import numpy as np
import matplotlib.pyplot as plt
from ligand_receptor import Fact_costim

Kd_GLP1 = 0.17    
Kd_sema = 0.38    
GLP1 = {"Fasting": 0.015, "After meal": 0.040}    # [nM]

sema = np.logspace(-3, 3, 500)    # semaglutide, [nM]

for label, G in GLP1.items():
    plt.semilogx(sema, Fact_costim(G, Kd_GLP1, sema, Kd_sema), lw=2,
                 label=f"GLP1 {label.lower()} ({G} nM)")
    print(f"{label}: Fact = {Fact_costim(G, Kd_GLP1, 0, Kd_sema):.3f} with no semaglutide, "
          f"{Fact_costim(G, Kd_GLP1, 1, Kd_sema):.3f} with 1 nM semaglutide")

plt.axvline(1, ls="--", c="gray", lw=1)    # semaglutide ~ 1 nM
plt.ylim(0, 1)
plt.xlabel("[Semaglutide] (nM, log scale)")
plt.ylabel("Fraction active (Fact)")
plt.legend()
plt.show()
