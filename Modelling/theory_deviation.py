import pandas as pd
import numpy as np

V0 = 1.0
DR = 0.1

simulations = {
    "Low (DT=0.005)": ("abp_results_dt_low", 0.005),
    "Baseline (DT=0.01)": ("abp_results_dt_baseline", 0.01),
    "High (DT=0.02)": ("abp_results_dt_high", 0.02)
}

selected_times = [5, 10, 20, 30, 40, 50]

print("\nSIMULATION VS THEORY")
print("=" * 85)

for label, (folder, DT) in simulations.items():

    data = pd.read_csv(f"{folder}/msd.csv")

    print(f"\n{label}")
    print("-" * 85)

    for t in selected_times:

        row = data.iloc[(data["time"] - t).abs().argsort()[:1]]

        simulated = row["MSD"].iloc[0]

        theoretical = (
            4 * DT * t
            + (2 * V0**2 / DR**2)
            * (DR*t - 1 + np.exp(-DR*t))
        )

        deviation = (
            (simulated - theoretical)
            / theoretical
        ) * 100

        print(
            f"Time = {t:2d} s | "
            f"Simulated = {simulated:9.3f} | "
            f"Theory = {theoretical:9.3f} | "
            f"Deviation = {deviation:7.2f}%"
        )
