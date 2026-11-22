
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load MSD data from all three simulations
base = Path(__file__).parent

low = pd.read_csv(base / "abp_results_speed_low" / "msd.csv")
baseline = pd.read_csv(base / "abp_results_speed_baseline" / "msd.csv")
high = pd.read_csv(base / "abp_results_speed_high" / "msd.csv")

# Plot all three curves
plt.figure(figsize=(9, 6))

plt.plot(low["time"], low["MSD"], label="Low speed (V0 = 0.5)")
plt.plot(baseline["time"], baseline["MSD"], label="Baseline (V0 = 1.0)")
plt.plot(high["time"], high["MSD"], label="High speed (V0 = 1.5)")

plt.xlabel("Time")
plt.ylabel("Mean Squared Displacement (MSD)")
plt.title("Effect of Swimming Speed on Active Particle Dispersion")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save comparison graph
output = base / "speed_sensitivity_msd.png"
plt.savefig(output, dpi=300)

print("Comparison graph saved at:", output)

plt.show()
