
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "abp_results"

data = pd.read_csv(RESULTS_DIR / "trajectories.csv")

# Arrange particle orientations
orientation = data.pivot(
    index="frame",
    columns="track_id",
    values="theta"
).to_numpy()

px = np.cos(orientation)
py = np.sin(orientation)

times = np.sort(data["time"].unique())
dt = np.mean(np.diff(times))

# Restrict analysis to first 15 simulation time units
max_time = 15.0
max_lag = min(int(max_time / dt), len(times) - 1)

oacf = []

for lag in range(max_lag + 1):
    correlation = (
        px[:len(times)-lag] * px[lag:]
        + py[:len(times)-lag] * py[lag:]
    )
    oacf.append(np.nanmean(correlation))

oacf = np.array(oacf)
time_lag = np.arange(max_lag + 1) * dt

# Normalize
oacf = oacf / oacf[0]

# Theoretical OACF
DR = 0.1
theoretical = np.exp(-DR * time_lag)

# Save results
results = pd.DataFrame({
    "time_lag": time_lag,
    "simulated_OACF": oacf,
    "theoretical_OACF": theoretical
})

results.to_csv(
    RESULTS_DIR / "oacf_validation.csv",
    index=False
)

# Plot comparison
plt.figure(figsize=(8, 5))

plt.plot(time_lag, oacf, label="Simulated OACF")
plt.plot(
    time_lag,
    theoretical,
    "--",
    label="Theoretical OACF"
)

plt.xlabel("Time lag")
plt.ylabel("Normalized OACF")
plt.title("OACF: Simulation vs Theory")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "oacf_validation.png",
    dpi=300
)

plt.show()

print("OACF validation completed.")
print("Results saved in:", RESULTS_DIR)
