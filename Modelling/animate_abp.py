
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# Input and output paths
input_file = Path("abp_results/trajectories.csv")
output_file = Path("abp_results/ABP_baseline_animation.gif")

# Load trajectory data
data = pd.read_csv(input_file)

frames = sorted(data["frame"].unique())
particle_count = data["track_id"].nunique()

# Circular dish dimensions
dish_radius = 40

fig, ax = plt.subplots(figsize=(7, 7))

ax.set_xlim(-42, 42)
ax.set_ylim(-42, 42)
ax.set_aspect("equal")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_title("Active Brownian Particle Simulation")

# Draw circular boundary
dish = plt.Circle(
    (0, 0), dish_radius,
    fill=False, linewidth=2
)
ax.add_patch(dish)

# Particle positions
scatter = ax.scatter([], [], s=18)

time_text = ax.text(
    0.03, 0.96, "",
    transform=ax.transAxes
)

def update(frame):
    current = data[data["frame"] == frame]

    scatter.set_offsets(
        current[["x", "y"]].to_numpy()
    )

    time_value = current["time"].iloc[0]
    time_text.set_text(
        f"Time = {time_value:.1f} | Particles = {particle_count}"
    )

    return scatter, time_text

animation = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=40,
    blit=True
)

animation.save(
    output_file,
    writer=PillowWriter(fps=25)
)

plt.close(fig)

print("Animation saved successfully:", output_file)
print("Particles:", particle_count)
print("Saved frames:", len(frames))
