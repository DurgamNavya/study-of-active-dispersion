
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

RUNS = {
    "Overlapping initialization": {
        "folder": "inertial_abp_cluster10_results",
        "label": "Overlapping initialization"
    },
    "Non-overlapping initialization": {
        "folder": "inertial_abp_cluster10_nonoverlap_results",
        "label": "Non-overlapping initialization"
    }
}

OUTPUT_DIR = "inertial_initialization_comparison"
PARTICLE_DIAMETER = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. CALCULATE OVERLAP STATISTICS AT EVERY FRAME
# ============================================================

def calculate_overlap_statistics(file_path):

    df = pd.read_csv(file_path)

    times = np.sort(df["time"].unique())

    overlap_counts = []
    maximum_overlaps = []
    maximum_speeds = []

    for t in times:

        frame = df[df["time"] == t].sort_values("particle")

        positions = frame[["x", "y"]].to_numpy()
        speeds = frame["speed"].to_numpy()

        displacement = (
            positions[:, np.newaxis, :]
            - positions[np.newaxis, :, :]
        )

        distances = np.linalg.norm(displacement, axis=2)

        i, j = np.triu_indices(len(positions), k=1)

        pair_distances = distances[i, j]

        overlaps = np.maximum(
            PARTICLE_DIAMETER - pair_distances,
            0.0
        )

        overlap_counts.append(
            np.sum(overlaps > 1e-10)
        )

        maximum_overlaps.append(
            overlaps.max()
        )

        maximum_speeds.append(
            speeds.max()
        )

    return pd.DataFrame({
        "time": times,
        "overlapping_pairs": overlap_counts,
        "maximum_overlap": maximum_overlaps,
        "maximum_speed": maximum_speeds
    })


# ============================================================
# 3. LOAD BOTH SIMULATIONS
# ============================================================

results = {}

for name, config in RUNS.items():

    trajectory_file = os.path.join(
        config["folder"],
        "trajectories.csv"
    )

    print(f"Analysing: {name}")

    results[name] = calculate_overlap_statistics(
        trajectory_file
    )


# ============================================================
# 4. SAVE COMBINED ANALYSIS DATA
# ============================================================

combined = []

for name, data in results.items():

    temp = data.copy()
    temp["initialization"] = name

    combined.append(temp)

combined_df = pd.concat(
    combined,
    ignore_index=True
)

combined_df.to_csv(
    os.path.join(OUTPUT_DIR, "initialization_comparison.csv"),
    index=False
)


# ============================================================
# 5. LOAD MSD DATA
# ============================================================

msd_data = {}

for name, config in RUNS.items():

    msd_file = os.path.join(
        config["folder"],
        "msd.csv"
    )

    msd_data[name] = pd.read_csv(msd_file)


# ============================================================
# 6. PLOT MAXIMUM OVERLAP VS TIME
# ============================================================

plt.figure(figsize=(9, 6))

for name, data in results.items():

    plt.plot(
        data["time"],
        data["maximum_overlap"],
        label=RUNS[name]["label"]
    )

plt.xlabel("Time")
plt.ylabel("Maximum overlap depth")
plt.title("Maximum Particle Overlap vs Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "maximum_overlap_vs_time.png"),
    dpi=300
)

plt.close()


# ============================================================
# 7. PLOT NUMBER OF OVERLAPPING PAIRS VS TIME
# ============================================================

plt.figure(figsize=(9, 6))

for name, data in results.items():

    plt.plot(
        data["time"],
        data["overlapping_pairs"],
        label=RUNS[name]["label"]
    )

plt.xlabel("Time")
plt.ylabel("Number of overlapping pairs")
plt.title("Overlapping Particle Pairs vs Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "overlapping_pairs_vs_time.png"),
    dpi=300
)

plt.close()


# ============================================================
# 8. PLOT MAXIMUM SPEED VS TIME
# ============================================================

plt.figure(figsize=(9, 6))

for name, data in results.items():

    plt.plot(
        data["time"],
        data["maximum_speed"],
        label=RUNS[name]["label"]
    )

plt.xlabel("Time")
plt.ylabel("Maximum particle speed")
plt.title("Maximum Particle Speed vs Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "maximum_speed_vs_time.png"),
    dpi=300
)

plt.close()


# ============================================================
# 9. PLOT MSD COMPARISON
# ============================================================

plt.figure(figsize=(9, 6))

for name, data in msd_data.items():

    plt.plot(
        data["time"],
        data["MSD"],
        label=RUNS[name]["label"]
    )

plt.xlabel("Time")
plt.ylabel("Mean squared displacement")
plt.title("MSD: Overlapping vs Non-overlapping Initialization")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "msd_initialization_comparison.png"),
    dpi=300
)

plt.close()


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print("\nINITIALIZATION COMPARISON SUMMARY")
print("=" * 60)

for name, data in results.items():

    msd_final = msd_data[name]["MSD"].iloc[-1]

    print(f"\n{name}")
    print("-" * 45)
    print("Maximum overlap:", data["maximum_overlap"].max())
    print("Maximum overlapping pairs:", data["overlapping_pairs"].max())
    print("Maximum particle speed:", data["maximum_speed"].max())
    print("Final MSD:", msd_final)

print("\nAnalysis completed.")
print("Results saved in:", OUTPUT_DIR)
