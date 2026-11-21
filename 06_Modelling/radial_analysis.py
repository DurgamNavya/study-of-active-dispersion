import pandas as pd
import numpy as np

folders = {
    "Low (DT_TRANS=0.005)": "abp_results_dt_low",
    "Baseline (DT_TRANS=0.01)": "abp_results_dt_baseline",
    "High (DT_TRANS=0.02)": "abp_results_dt_high"
}

selected_times = [0, 10, 20, 30, 40, 50]

print("\nRADIAL COMPARISON")
print("=" * 90)

for label, folder in folders.items():

    data = pd.read_csv(f"{folder}/trajectories.csv")

    data["radius"] = np.sqrt(
        data["x"]**2 + data["y"]**2
    )

    print(f"\n{label}")
    print("-" * 90)

    for t in selected_times:

        frame = data[data["time"] == t]

        mean_r = frame["radius"].mean()
        max_r = frame["radius"].max()

        near_80 = 100 * np.mean(frame["radius"] >= 32)
        near_90 = 100 * np.mean(frame["radius"] >= 36)

        print(
            f"Time {t:2d} s | "
            f"Mean radius = {mean_r:7.3f} | "
            f"Max radius = {max_r:7.3f} | "
            f">80% = {near_80:5.1f}% | "
            f">90% = {near_90:5.1f}%"
        )
