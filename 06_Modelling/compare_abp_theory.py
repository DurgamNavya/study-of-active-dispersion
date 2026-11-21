import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Common ABP parameters
V0 = 1.0
DR = 0.1

simulations = {
    "Low (DT=0.005)": {
        "folder": "abp_results_dt_low",
        "DT": 0.005
    },
    "Baseline (DT=0.01)": {
        "folder": "abp_results_dt_baseline",
        "DT": 0.01
    },
    "High (DT=0.02)": {
        "folder": "abp_results_dt_high",
        "DT": 0.02
    }
}

plt.figure(figsize=(10, 6))

for label, params in simulations.items():

    data = pd.read_csv(
        f"{params['folder']}/msd.csv"
    )

    time = data["time"].to_numpy()
    msd = data["MSD"].to_numpy()

    plt.plot(
        time,
        msd,
        label=f"Simulated {label}"
    )

    DT = params["DT"]

    theoretical_msd = (
        4 * DT * time
        + (2 * V0**2 / DR**2)
        * (DR*time - 1 + np.exp(-DR*time))
    )

    plt.plot(
        time,
        theoretical_msd,
        linestyle="--",
        label=f"Theory {label}"
    )

plt.xlabel("Time")
plt.ylabel("Mean Squared Displacement")
plt.title("Simulated ABP MSD vs Theoretical ABP MSD")

plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "ABP_theory_vs_simulation.png",
    dpi=300
)

print("Graph saved: ABP_theory_vs_simulation.png")

plt.show()
