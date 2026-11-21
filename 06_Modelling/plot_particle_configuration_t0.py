
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load simulation trajectory
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "abp_results" / "trajectories.csv"

data = pd.read_csv(csv_path)

# Select the initial frame at t = 0
snapshot = data[data["time"] == 0].copy()

if snapshot.empty:
    raise ValueError("No particle data found at t = 0.")

# Verify particle count
print(f"Initial time: {snapshot['time'].iloc[0]}")
print(f"Number of particles: {len(snapshot)}")

# Create particle-configuration plot
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    snapshot["x"],
    snapshot["y"],
    s=28,
    alpha=0.8,
    label="Active Brownian particles"
)

# Draw the same circular simulation boundary
boundary = plt.Circle(
    (0, 0),
    40,
    fill=False,
    linestyle="--",
    linewidth=1.5,
    label="Circular boundary"
)

ax.add_patch(boundary)

ax.set_xlabel("X position (model length units)")
ax.set_ylabel("Y position (model length units)")
ax.set_title("Initial Particle Configuration at t = 0.00")

ax.set_aspect("equal", adjustable="box")
ax.legend(
    loc="upper right",
    frameon=True,
    framealpha=0.9
)
ax.grid(True, alpha=0.25)

# Save separately; do not overwrite the t = 50 image
output_path = csv_path.parent / "particle_configuration_t0.png"

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Image saved at: {output_path}")
