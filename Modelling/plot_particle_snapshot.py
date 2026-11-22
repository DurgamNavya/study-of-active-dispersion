
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load simulation trajectory
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "abp_results" / "trajectories.csv"

data = pd.read_csv(csv_path)

# Select the final saved frame
selected_frame = data["frame"].max()
snapshot = data[data["frame"] == selected_frame]

# Create particle-distribution plot
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    snapshot["x"],
    snapshot["y"],
    s=28,
    alpha=0.8,
    label="Active Brownian particles"
)

# Draw circular simulation boundary
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
ax.set_title(f"Particle Configuration at t = {snapshot['time'].iloc[0]:.2f}")

ax.set_aspect("equal", adjustable="box")
ax.legend()
ax.grid(True, alpha=0.25)

# Save image
output_path = csv_path.parent / "particle_configuration.png"

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Image saved at: {output_path}")
