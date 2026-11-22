
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path(__file__).resolve().parent

data_dir = (
    base_dir
    / "abp_analysis_results"
    / "seed_42"
    / "time_averaged_msd"
)

files = {
    "Inertial, N=100, T=200": "inertial_N100_T200_time_averaged_msd.csv",
    "Overdamped, N=100, T=200": "overdamped_N100_T200_time_averaged_msd.csv",
    "Inertial, N=100, T=50": "inertial_N100_T50_time_averaged_msd.csv",
    "Overdamped, N=100, T=50": "overdamped_N100_T50_time_averaged_msd.csv",
    "Inertial, N=400, T=50": "inertial_N400_T50_time_averaged_msd.csv",
    "Overdamped, N=400, T=50": "overdamped_N400_T50_time_averaged_msd.csv",
}

fig, ax = plt.subplots(figsize=(9, 6))

for label, filename in files.items():
    path = data_dir / filename

    df = pd.read_csv(path)

    # Remove zero values because logarithmic axes cannot display zero
    df = df[
        (df["lag_s"] > 0) &
        (df["TAMSD"] > 0)
    ]

    ax.loglog(
        df["lag_s"],
        df["TAMSD"],
        linewidth=1.5,
        label=label
    )

ax.set_xlabel("Lag time (model time units)")
ax.set_ylabel("Time-averaged MSD (model length units squared)")
ax.set_title("Time-Averaged Mean-Squared Displacement")

ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8)

plt.tight_layout()

output_path = data_dir / "TAMSD_combined_loglog.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"TAMSD graph saved at: {output_path}")
