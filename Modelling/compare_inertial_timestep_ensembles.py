from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RUNS = {
    "dt001": [
        "inertial_abp_cluster10_nonoverlap_results",
        "inertial_abp_cluster10_seed43_dt001_results",
        "inertial_abp_cluster10_seed44_dt001_results",
        "inertial_abp_cluster10_seed45_dt001_results",
        "inertial_abp_cluster10_seed46_dt001_results",
    ],

    "dt005": [
        "inertial_abp_cluster10_dt005_results",
        "inertial_abp_cluster10_seed43_dt005_results",
        "inertial_abp_cluster10_seed44_dt005_results",
        "inertial_abp_cluster10_seed45_dt005_results",
        "inertial_abp_cluster10_seed46_dt005_results",
    ],
}

OUTPUT_DIR = Path("inertial_timestep_ensemble_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)


def read_msd(folder):

    file = Path(folder) / "msd.csv"

    if not file.exists():
        raise FileNotFoundError(file)

    df = pd.read_csv(file)

    required = {"time", "MSD"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Unexpected columns in {file}: {df.columns.tolist()}"
        )

    df = df[["time", "MSD"]].sort_values("time")

    if df.isna().any().any():
        raise ValueError(f"Missing values in {file}")

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"Non-finite values in {file}")

    return df


data = {"dt001": [], "dt005": []}
summary = []

print("Reading simulation results...")

for timestep, folders in RUNS.items():

    for index, folder in enumerate(folders):

        df = read_msd(folder)

        data[timestep].append(df)

        final_msd = float(df["MSD"].iloc[-1])

        summary.append({
            "timestep": timestep,
            "run": index + 1,
            "folder": folder,
            "frames": len(df),
            "final_time": float(df["time"].iloc[-1]),
            "final_MSD": final_msd,
        })

        print(
            f"{timestep} | {folder} | "
            f"frames={len(df)} | final MSD={final_msd:.6f}"
        )


summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    OUTPUT_DIR / "final_msd_summary.csv",
    index=False
)


# --------------------------------------------------
# Final MSD ensemble statistics
# --------------------------------------------------

print("\nFINAL MSD ENSEMBLE STATISTICS")

for timestep in ["dt001", "dt005"]:

    values = np.array([
        df["MSD"].iloc[-1]
        for df in data[timestep]
    ])

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    sem = std / np.sqrt(len(values))

    print(f"\n{timestep}")
    print("Mean:", mean)
    print("Standard deviation:", std)
    print("Standard error:", sem)


# --------------------------------------------------
# Interpolate dt005 curves onto dt001 time grid
# --------------------------------------------------

time_grid = data["dt001"][0]["time"].to_numpy()

curves = {}

for timestep in ["dt001", "dt005"]:

    interpolated = []

    for df in data[timestep]:

        time = df["time"].to_numpy()
        msd = df["MSD"].to_numpy()

        if time[0] > time_grid[0] or time[-1] < time_grid[-1]:
            raise ValueError(
                f"{timestep} does not cover the common time range."
            )

        interpolated.append(
            np.interp(time_grid, time, msd)
        )

    curves[timestep] = np.array(interpolated)


# --------------------------------------------------
# Ensemble mean, SD and SEM
# --------------------------------------------------

ensemble = pd.DataFrame({"time": time_grid})

for timestep in ["dt001", "dt005"]:

    values = curves[timestep]

    ensemble[f"{timestep}_mean"] = np.mean(values, axis=0)
    ensemble[f"{timestep}_std"] = np.std(
        values, axis=0, ddof=1
    )
    ensemble[f"{timestep}_sem"] = (
        ensemble[f"{timestep}_std"] / np.sqrt(values.shape[0])
    )

ensemble.to_csv(
    OUTPUT_DIR / "ensemble_msd_comparison.csv",
    index=False
)


# --------------------------------------------------
# Linear-scale plot
# --------------------------------------------------

plt.figure(figsize=(9, 6))

for timestep, label in [
    ("dt001", "dt = 0.01"),
    ("dt005", "dt = 0.005"),
]:

    mean = ensemble[f"{timestep}_mean"].to_numpy()
    sem = ensemble[f"{timestep}_sem"].to_numpy()

    plt.plot(time_grid, mean, label=label)
    plt.fill_between(
        time_grid,
        mean - sem,
        mean + sem,
        alpha=0.2
    )

plt.xlabel("Time")
plt.ylabel("Ensemble mean MSD")
plt.title("Inertial ABP: Timestep Ensemble Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ensemble_msd_linear.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# Log-log plot
# --------------------------------------------------

plt.figure(figsize=(9, 6))

positive = time_grid > 0

for timestep, label in [
    ("dt001", "dt = 0.01"),
    ("dt005", "dt = 0.005"),
]:

    mean = ensemble[f"{timestep}_mean"].to_numpy()
    sem = ensemble[f"{timestep}_sem"].to_numpy()

    valid = positive & (mean > 0)

    plt.loglog(
        time_grid[valid],
        mean[valid],
        label=label
    )

    plt.fill_between(
        time_grid[valid],
        np.maximum(mean[valid] - sem[valid], 1e-12),
        mean[valid] + sem[valid],
        alpha=0.2
    )

plt.xlabel("Time")
plt.ylabel("Ensemble mean MSD")
plt.title("Inertial ABP: Timestep Comparison (Log-Log)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ensemble_msd_loglog.png",
    dpi=300
)

plt.close()


print("\nComparison completed.")
print("Output folder:", OUTPUT_DIR.resolve())
print("Files generated:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print(file.name)
