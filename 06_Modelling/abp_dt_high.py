
"""
CP302: Analysis of Active Dispersion
Model: 2D Active Brownian Particles (ABP)

Purpose:
Simulate dispersion of an initially concentrated cluster of
self-propelled particles in a circular domain.

Model includes:
1. Self-propulsion
2. Translational Brownian motion
3. Rotational Brownian motion
4. Short-range harmonic repulsion
5. Reflecting circular boundary

IMPORTANT:
The default parameters are dimensionless test values.
They are NOT fitted experimental parameters.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#
# 1. SIMULATION PARAMETERS
#

N = 100                 # Number of particles
STEPS = 5000            # Number of simulation steps
DT = 0.01               # Simulation time step
SAVE_EVERY = 10         # Save positions every 10 steps

V0 = 1.0                # Self-propulsion speed
DT_TRANS = 0.02         # Translational diffusion coefficient
DR = 0.1                # Rotational diffusion coefficient

MOBILITY = 1.0          # Particle mobility
REPULSION_K = 10.0      # Harmonic repulsion strength
PARTICLE_DIAMETER = 1.0 # Effective particle diameter

DISH_RADIUS = 40.0      # Circular domain radius
INITIAL_CLUSTER_RADIUS = 5.0

RANDOM_SEED = 42

OUTPUT_DIR = "abp_results_dt_high"

#
# 2. INITIALISE PARTICLE POSITIONS AND ORIENTATIONS
#

def initialise_particles(number, cluster_radius, seed):

    rng = np.random.default_rng(seed)

    # Uniform random positions inside the initial cluster
    angles = rng.uniform(0, 2 * np.pi, number)

    radii = cluster_radius * np.sqrt(
        rng.uniform(0, 1, number)
    )

    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    positions = np.column_stack((x, y))

    # Random initial orientations
    theta = rng.uniform(0, 2 * np.pi, number)

    return positions, theta, rng


#
# 3. CALCULATE PARTICLE-PARTICLE REPULSION
#

def calculate_repulsion(positions):

    # Difference between every pair of particles
    displacement = (
        positions[:, np.newaxis, :]
        - positions[np.newaxis, :, :]
    )

    distances = np.linalg.norm(
        displacement, axis=2
    )

    # Avoid division by zero
    safe_distances = np.maximum(distances, 1e-12)

    overlap = np.maximum(
        PARTICLE_DIAMETER - distances, 0.0
    )

    # Harmonic repulsive force
    force = (
        REPULSION_K
        * overlap[:, :, np.newaxis]
        * displacement
        / safe_distances[:, :, np.newaxis]
    )

    # A particle does not exert a force on itself
    np.fill_diagonal(overlap, 0.0)

    total_force = np.sum(force, axis=1)

    return total_force


#
# 4. APPLY CIRCULAR REFLECTING BOUNDARY
#

def apply_boundary(positions, dish_radius):

    distances = np.linalg.norm(positions, axis=1)

    outside = distances > dish_radius

    if np.any(outside):

        radial_unit = (
            positions[outside]
            / distances[outside, np.newaxis]
        )

        # Reflect overshooting positions back inside
        reflected_radius = (
            2.0 * dish_radius - distances[outside]
        )

        # Handle unusually large overshoots safely
        reflected_radius = np.maximum(
            reflected_radius, 0.0
        )

        positions[outside] = (
            reflected_radius[:, np.newaxis]
            * radial_unit
        )

    return positions


#
# 5. CALCULATE VELOCITY FROM SAVED POSITIONS
#

def calculate_velocity(position_history, time_history):

    # Shape: time x particles x coordinates
    positions = np.asarray(position_history)

    times = np.asarray(time_history)

    # Finite difference between consecutive saved frames
    velocity = np.diff(positions, axis=0) / np.diff(
        times
    )[:, np.newaxis, np.newaxis]

    speed = np.linalg.norm(velocity, axis=2)

    return velocity, speed


# ============================================================
# 6. MAIN ABP SIMULATION
# ============================================================

def run_simulation():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    positions, theta, rng = initialise_particles(
        N,
        INITIAL_CLUSTER_RADIUS,
        RANDOM_SEED
    )

    position_history = []
    orientation_history = []
    time_history = []

    # Save initial configuration
    position_history.append(positions.copy())
    orientation_history.append(theta.copy())
    time_history.append(0.0)

    print("Starting CP302 ABP simulation...")

    for step in range(1, STEPS + 1):

        # --------------------------------------------
        # A. Self-propulsion direction
        # --------------------------------------------

        direction = np.column_stack((
            np.cos(theta),
            np.sin(theta)
        ))

        active_velocity = V0 * direction

        # --------------------------------------------
        # B. Particle interaction force
        # --------------------------------------------

        force = calculate_repulsion(positions)

        interaction_velocity = MOBILITY * force

        # --------------------------------------------
        # C. Translational Brownian noise
        # --------------------------------------------

        translational_noise = (
            np.sqrt(2.0 * DT_TRANS * DT)
            * rng.normal(size=(N, 2))
        )

        # --------------------------------------------
        # D. Update particle positions
        # --------------------------------------------

        positions = (
            positions
            + (active_velocity + interaction_velocity) * DT
            + translational_noise
        )

        # --------------------------------------------
        # E. Apply reflecting boundary
        # --------------------------------------------

        positions = apply_boundary(
            positions,
            DISH_RADIUS
        )

        # --------------------------------------------
        # F. Update particle orientations
        # --------------------------------------------

        rotational_noise = (
            np.sqrt(2.0 * DR * DT)
            * rng.normal(size=N)
        )

        theta = theta + rotational_noise

        # Keep angles between 0 and 2*pi
        theta = np.mod(theta, 2.0 * np.pi)

        # --------------------------------------------
        # G. Save simulation data
        # --------------------------------------------

        if step % SAVE_EVERY == 0:

            position_history.append(
                positions.copy()
            )

            orientation_history.append(
                theta.copy()
            )

            time_history.append(step * DT)

        if step % 500 == 0:
            print(
                f"Completed {step}/{STEPS} steps"
            )

    print("Simulation completed.")

    return (
        np.asarray(position_history),
        np.asarray(orientation_history),
        np.asarray(time_history)
    )


# ============================================================
# 7. SAVE PARTICLE TRAJECTORIES
# ============================================================

def save_trajectories(
    position_history,
    orientation_history,
    time_history
):

    rows = []

    number_of_frames = len(time_history)

    for frame in range(number_of_frames):

        for particle in range(N):

            x, y = position_history[frame, particle]

            rows.append({
                "track_id": particle + 1,
                "frame": frame,
                "time": time_history[frame],
                "x": x,
                "y": y,
                "theta": orientation_history[frame, particle]
            })

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        os.path.join(OUTPUT_DIR, "trajectories.csv"),
        index=False
    )

    return dataframe


# ============================================================
# 8. SAVE VELOCITY DATA
# ============================================================

def save_velocity(
    position_history,
    time_history
):

    velocity, speed = calculate_velocity(
        position_history,
        time_history
    )

    rows = []

    for frame in range(len(time_history) - 1):

        for particle in range(N):

            rows.append({
                "track_id": particle + 1,
                "frame": frame + 1,
                "time": time_history[frame + 1],
                "vx": velocity[frame, particle, 0],
                "vy": velocity[frame, particle, 1],
                "speed": speed[frame, particle]
            })

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        os.path.join(OUTPUT_DIR, "velocity.csv"),
        index=False
    )

    return dataframe


# ============================================================
# 9. CALCULATE MEAN SQUARED DISPLACEMENT
# ============================================================

def calculate_msd(position_history):

    initial_positions = position_history[0]

    displacement = (
        position_history - initial_positions[np.newaxis, :, :]
    )

    squared_displacement = np.sum(
        displacement ** 2, axis=2
    )

    msd = np.mean(
        squared_displacement, axis=1
    )

    return msd


# ============================================================
# 10. PLOT INITIAL AND FINAL CONFIGURATIONS
# ============================================================

def plot_particle_configurations(position_history):

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5)
    )

    initial = position_history[0]
    final = position_history[-1]

    for ax, data, title in zip(
        axes,
        [initial, final],
        ["Initial cluster", "Final particle distribution"]
    ):

        ax.scatter(
            data[:, 0],
            data[:, 1],
            s=12
        )

        dish = plt.Circle(
            (0, 0),
            DISH_RADIUS,
            fill=False,
            linewidth=1.5
        )

        ax.add_patch(dish)

        ax.set_xlim(
            -DISH_RADIUS - 2,
            DISH_RADIUS + 2
        )

        ax.set_ylim(
            -DISH_RADIUS - 2,
            DISH_RADIUS + 2
        )

        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "initial_final_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 11. PLOT PARTICLE TRAJECTORIES
# ============================================================

def plot_trajectories(position_history):

    plt.figure(figsize=(8, 8))

    for particle in range(N):

        plt.plot(
            position_history[:, particle, 0],
            position_history[:, particle, 1],
            linewidth=0.6,
            alpha=0.7
        )

    dish = plt.Circle(
        (0, 0),
        DISH_RADIUS,
        fill=False,
        linewidth=1.5
    )

    plt.gca().add_patch(dish)

    plt.xlim(-DISH_RADIUS - 2, DISH_RADIUS + 2)
    plt.ylim(-DISH_RADIUS - 2, DISH_RADIUS + 2)

    plt.gca().set_aspect("equal")

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("ABP particle trajectories")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "particle_trajectories.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 12. PLOT MEAN SQUARED DISPLACEMENT
# ============================================================

def plot_msd(time_history, msd):

    plt.figure(figsize=(7, 5))

    plt.plot(
        time_history,
        msd,
        linewidth=2
    )

    plt.xlabel("Time")
    plt.ylabel("Mean squared displacement")
    plt.title("MSD of active Brownian particles")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "mean_squared_displacement.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    positions, orientations, times = run_simulation()

    trajectories = save_trajectories(
        positions,
        orientations,
        times
    )

    velocities = save_velocity(
        positions,
        times
    )

    msd = calculate_msd(positions)

    pd.DataFrame({
        "time": times,
        "MSD": msd
    }).to_csv(
        os.path.join(OUTPUT_DIR, "msd.csv"),
        index=False
    )

    plot_particle_configurations(positions)

    plot_trajectories(positions)

    plot_msd(times, msd)

    print("\nAll outputs saved in:", OUTPUT_DIR)
    print("Number of particles:", N)
    print("Number of saved frames:", len(times))
    print("Total simulation time:", times[-1])
    print("Mean simulated speed:", velocities["speed"].mean())
