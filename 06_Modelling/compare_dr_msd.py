
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).parent

low = pd.read_csv(base / "abp_results_dr_low" / "msd.csv")
baseline = pd.read_csv(base / "abp_results_dr_baseline" / "msd.csv")
high = pd.read_csv(base / "abp_results_dr_high" / "msd.csv")

plt.figure(figsize=(9, 6))

plt.plot(low["time"], low["MSD"], label="Low Dr = 0.05")
plt.plot(baseline["time"], baseline["MSD"], label="Baseline Dr = 0.1")
plt.plot(high["time"], high["MSD"], label="High Dr = 0.2")

plt.xlabel("Time")
plt.ylabel("Mean Squared Displacement (MSD)")
plt.title("Effect of Rotational Diffusion on Active Particle Dispersion")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output = base / "rotational_diffusion_sensitivity_msd.png"
plt.savefig(output, dpi=300)

print("Comparison graph saved at:", output)

plt.show()
