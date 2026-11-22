import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = "inertial_abp_cluster10_nonoverlap_results"
OUTPUT_FOLDER = "inertial_abp_cluster10_correlation_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MAX_LAG_TIME = 20.0

df = pd.read_csv(
    os.path.join(FOLDER, "trajectories.csv")
)

# Arrange data consistently by time and particle
velocity_x = df.pivot(
    index="time", columns="particle", values="vx"
).sort_index()

velocity_y = df.pivot(
    index="time", columns="particle", values="vy"
).sort_index()

theta = df.pivot(
    index="time", columns="particle", values="theta"
).sort_index()

times = velocity_x.index.to_numpy()

vx = velocity_x.to_numpy()
vy = velocity_y.to_numpy()

velocities = np.stack((vx, vy), axis=2)

orientations = theta.to_numpy()

directions = np.stack((
    np.cos(orientations),
    np.sin(orientations)
), axis=2)

frames, particles, dimensions = velocities.shape

# Use a maximum lag of 10 seconds
time_step = times[1] - times[0]

MAX_LAG = min(
    int(round(MAX_LAG_TIME / time_step)),
    frames - 1
)

vacf = np.zeros(MAX_LAG + 1)
oacf = np.zeros(MAX_LAG + 1)

# Time-origin-averaged autocorrelation
for lag in range(MAX_LAG + 1):

    v1 = velocities[:frames-lag]
    v2 = velocities[lag:]

    p1 = directions[:frames-lag]
    p2 = directions[lag:]

    vacf[lag] = np.mean(
        np.sum(v1 * v2, axis=2)
    )

    oacf[lag] = np.mean(
        np.sum(p1 * p2, axis=2)
    )

# Normalize using zero-lag values
if vacf[0] == 0 or oacf[0] == 0:
    raise ValueError("Zero-lag correlation is zero. Check input data.")

vacf_normalized = vacf / vacf[0]
oacf_normalized = oacf / oacf[0]

time_lags = np.arange(MAX_LAG + 1) * time_step
# Save VACF
pd.DataFrame({
    "time_lag": time_lags,
    "VACF": vacf,
    "VACF_normalized": vacf_normalized
}).to_csv(
    os.path.join(OUTPUT_FOLDER, "vacf_time_averaged.csv"),
    index=False
)

# Save OACF
pd.DataFrame({
    "time_lag": time_lags,
    "OACF": oacf,
    "OACF_normalized": oacf_normalized
}).to_csv(
    os.path.join(OUTPUT_FOLDER, "oacf_time_averaged.csv"),
    index=False
)

# Plot VACF
plt.figure(figsize=(8, 6))

plt.plot(time_lags, vacf_normalized)

plt.xlabel("Time lag")
plt.ylabel("Normalized VACF")
plt.title("Inertial ABP: Time-Origin-Averaged VACF")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "VACF_time_averaged.png"),
    dpi=300
)

plt.close()

# Plot OACF
plt.figure(figsize=(8, 6))

plt.plot(time_lags, oacf_normalized)

plt.xlabel("Time lag")
plt.ylabel("Normalized OACF")
plt.title("Inertial ABP: Time-Origin-Averaged OACF")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "OACF_time_averaged.png"),
    dpi=300
)

plt.close()

# Inspect maximum speed and its occurrence
speed = df["speed"].to_numpy()
max_index = np.argmax(speed)
max_row = df.iloc[max_index]

print("TIME-ORIGIN-AVERAGED CORRELATION ANALYSIS")
print("Frames:", frames)
print("Particles:", particles)
print("Time interval:", time_lags[1] - time_lags[0])

print("\nVACF")
print("Initial:", vacf[0])
print("Final:", vacf[-1])
print("Normalized initial:", vacf_normalized[0])

print("\nOACF")
print("Initial:", oacf[0])
print("Final:", oacf[-1])
print("Normalized initial:", oacf_normalized[0])

print("Maximum lag time:", time_lags[-1])
print("Number of lag values:", len(time_lags))
print("Saved results in:", OUTPUT_FOLDER)
print("\nSaved time-averaged correlation files and plots.")
