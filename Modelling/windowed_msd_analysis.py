import pandas as pd
import numpy as np

folders = {
    "Low (DT_TRANS=0.005)": "abp_results_dt_low",
    "Baseline (DT_TRANS=0.01)": "abp_results_dt_baseline",
    "High (DT_TRANS=0.02)": "abp_results_dt_high"
}

windows = [
    (0, 5),
    (5, 15),
    (15, 30),
    (30, 50)
]

print("\nWINDOWED MSD SLOPE ANALYSIS")
print("=" * 75)

for label, folder in folders.items():

    data = pd.read_csv(f"{folder}/msd.csv")

    time = data["time"].to_numpy()
    msd = data["MSD"].to_numpy()

    print(f"\n{label}")
    print("-" * 75)

    for start, end in windows:

        mask = (time >= start) & (time <= end)

        t = time[mask]
        y = msd[mask]

        slope, intercept = np.polyfit(t, y, 1)

        predicted = slope * t + intercept

        ss_res = np.sum((y - predicted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 1 - ss_res / ss_tot

        print(
            f"{start:2d}-{end:2d} s | "
            f"Slope = {slope:8.4f} | "
            f"R2 = {r_squared:.4f}"
        )
