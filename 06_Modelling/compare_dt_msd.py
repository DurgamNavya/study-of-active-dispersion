
import pandas as pd
import numpy as np

folders = {
    "Low (DT = 0.005)": "abp_results_dt_low",
    "Baseline (DT = 0.01)": "abp_results_dt_baseline",
    "High (DT = 0.02)": "abp_results_dt_high"
}

print("\nMSD SLOPE ANALYSIS")
print("-" * 55)

for label, folder in folders.items():

    data = pd.read_csv(f"{folder}/msd.csv")

    time = data["time"].to_numpy()
    msd = data["MSD"].to_numpy()

    slope, intercept = np.polyfit(time, msd, 1)

    print(f"\n{label}")
    print(f"Initial MSD: {msd[0]:.4f}")
    print(f"Final MSD: {msd[-1]:.4f}")
    print(f"Overall MSD slope: {slope:.4f}")
    print(f"Linear fit intercept: {intercept:.4f}")
