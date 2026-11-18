"""Task 3: Cavalitide PK-PD simulation for the Module 3 final assignment."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Task 1 pharmacodynamics (nM)
KD_GLP1 = 0.17
KD_CAVALITIDE = 0.077
GLP1_FASTING = 0.015

# Task 2 pharmacokinetics (Cavalitide assumes Wegovy-like values except ka)
F_AVAILABLE = 0.01
KA = 0.006                 # 1/h
KE = 0.0039                # 1/h
V = 10.8                   # L
MW = 4113.6                # g/mol
FREE_FRACTION = 0.01
INTERVAL = 24               # h


def fraction_active(cavalitide_nM):
    """GLP1R active fraction for fasting endogenous GLP-1 plus Cavalitide."""
    return ((KD_CAVALITIDE * GLP1_FASTING) + (KD_GLP1 * cavalitide_nM)) / (
        (KD_GLP1 * KD_CAVALITIDE)
        + (KD_CAVALITIDE * GLP1_FASTING)
        + (KD_GLP1 * cavalitide_nM)
    )


def simulate(dose_mg, n_doses=100, points_per_day=240):
    """Return time (h), free Cavalitide (nM), and active GLP1R fraction."""
    def rhs(_t, y):
        depot_mg, blood_mg_per_L = y
        return [-KA * depot_mg, KA * depot_mg / V - KE * blood_mg_per_L]

    y = np.array([0.0, 0.0])
    times, concentrations = [], []
    for dose in range(n_doses):
        y[0] += F_AVAILABLE * dose_mg
        start = dose * INTERVAL
        t_eval = start + np.linspace(0, INTERVAL, points_per_day)
        solution = solve_ivp(rhs, (start, start + INTERVAL), y, t_eval=t_eval,
                             method="LSODA", rtol=1e-8, atol=1e-10)
        times.append(solution.t)
        concentrations.append(solution.y[1])
        y = solution.y[:, -1]

    time_h = np.concatenate(times)
    total_nM = np.concatenate(concentrations) * 1e-3 / MW * 1e9
    free_nM = FREE_FRACTION * total_nM
    return time_h, free_nM, fraction_active(free_nM)


if __name__ == "__main__":
    # 21 mg/day is the Task 2 Phase 1 exposure target; 29 mg/day targets ~90% GLP1R.
    t_21, c_21, fact_21 = simulate(21)
    t_29, c_29, fact_29 = simulate(29)
    final_day = t_29 >= t_29.max() - 24

    print(f"21 mg/day final-day mean Fact: {fact_21[t_21 >= t_21.max() - 24].mean():.3f}")
    print(f"29 mg/day final-day mean Fact: {fact_29[final_day].mean():.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t_21 / 24, c_21, "--", color="gray", label="21 mg/day")
    axes[0].plot(t_29 / 24, c_29, lw=2, label="29 mg/day")
    axes[0].set(ylabel="Free Cavalitide (nM)")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(t_21 / 24, fact_21, "--", color="gray", label="21 mg/day")
    axes[1].plot(t_29 / 24, fact_29, lw=2, label="29 mg/day")
    axes[1].axhline(0.90, color="black", ls=":", label="90% target")
    axes[1].set(xlabel="Time (days)", ylabel="Active GLP1R fraction", ylim=(0, 1.02))
    axes[1].grid(alpha=0.3); axes[1].legend()
    plt.tight_layout()
    plt.savefig("task3_cavalitide_pkpd.png", dpi=150)
    plt.show()
