
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. LOAD SIMULATION DATA
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "abp_results"

data = pd.read_csv(RESULTS_DIR / "trajectories.csv")

print("Loaded trajectory data")
print("Number of particles:", data["track_id"].nunique())
print("Number of saved frames:", data["frame"].nunique())

# Arrange data consistently
data = data.sort_values(["track_id", "frame"])

# Check time interval
times = np.sort(data["time"].unique())
dt = np.mean(np.diff(times))

print("Time interval between saved frames:", dt)

# --------------------------------------------------
# 2. CALCULATE ORIENTATION AUTOCORRELATION (OACF)
# --------------------------------------------------

# OACF measures how long particles retain
# their original swimming direction.

orientation = data.pivot(
    index="frame",
    columns="track_id",
    values="theta"
).to_numpy()

# Unit orientation vectors
px = np.cos(orientation)
py = np.sin(orientation)

max_lag = len(times) - 1

oacf = np.zeros(max_lag + 1)

for lag in range(max_lag + 1):

    correlation = (
        px[:len(times)-lag] * px[lag:]
        + py[:len(times)-lag] * py[lag:]
    )

    oacf[lag] = np.nanmean(correlation)

oacf_time = np.arange(max_lag + 1) * dt

# --------------------------------------------------
# 3. CALCULATE VELOCITY AUTOCORRELATION (VACF)
# --------------------------------------------------

# Calculate velocities from successive positions.

positions = data.pivot(
    index="frame",
    columns="track_id",
    values=["x", "y"]
)

x = positions["x"].to_numpy()
y = positions["y"].to_numpy()

vx = np.diff(x, axis=0) / dt
vy = np.diff(y, axis=0) / dt

max_lag_v = len(vx) - 1

vacf = np.zeros(max_lag_v + 1)

for lag in range(max_lag_v + 1):

    correlation = (
        vx[:len(vx)-lag] * vx[lag:]
        + vy[:len(vy)-lag] * vy[lag:]
    )

    vacf[lag] = np.nanmean(correlation)

vacf_time = np.arange(max_lag_v + 1) * dt

# --------------------------------------------------
# 4. NORMALIZE CORRELATION FUNCTIONS
# --------------------------------------------------

if abs(oacf[0]) > 1e-12:
    oacf_normalized = oacf / oacf[0]
else:
    oacf_normalized = oacf

if abs(vacf[0]) > 1e-12:
    vacf_normalized = vacf / vacf[0]
else:
    vacf_normalized = vacf

# --------------------------------------------------
# 5. SAVE CSV RESULTS
# --------------------------------------------------

oacf_df = pd.DataFrame({
    "time_lag": oacf_time,
    "OACF": oacf,
    "OACF_normalized": oacf_normalized
})

vacf_df = pd.DataFrame({
    "time_lag": vacf_time,
    "VACF": vacf,
    "VACF_normalized": vacf_normalized
})

oacf_df.to_csv(RESULTS_DIR / "oacf_results.csv", index=False)
vacf_df.to_csv(RESULTS_DIR / "vacf_results.csv", index=False)

# --------------------------------------------------
# 6. PLOT OACF
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(oacf_time, oacf_normalized)

plt.xlabel("Time lag")
plt.ylabel("Normalized OACF")
plt.title("Orientation Autocorrelation Function")

plt.grid(True)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "oacf_plot.png", dpi=300)
plt.show()

# --------------------------------------------------
# 7. PLOT VACF
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(vacf_time, vacf_normalized)

plt.xlabel("Time lag")
plt.ylabel("Normalized VACF")
plt.title("Velocity Autocorrelation Function")

plt.grid(True)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "vacf_plot.png", dpi=300)
plt.show()

# --------------------------------------------------
# 8. FINAL SUMMARY
# --------------------------------------------------

print("\nCorrelation analysis completed.")

print("OACF at zero lag:", oacf_normalized[0])
print("VACF at zero lag:", vacf_normalized[0])

print("\nNew files saved:")
print("oacf_results.csv")
print("vacf_results.csv")
print("oacf_plot.png")
print("vacf_plot.png")
