
"""
CP302: Study of Active Dispersion
Model: Inertial Active Brownian Particles

Based on:
H. Lowen, J. Chem. Phys. 152, 040901 (2020).
DOI: 10.1063/1.5134455

Model:
- Translational inertia included
- Orientation remains overdamped
- Translational and rotational noise
- Harmonic pairwise repulsion
- Circular reflecting boundary

Parameters are dimensionless exploratory values.
They are not fitted to experimental data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SIMULATION PARAMETERS
# ============================================================

N = 100
STEPS = 5000
DT = 0.01
SAVE_EVERY = 1

V0 = 1.0
DT_TRANS = 0.01
DR = 0.1

MOBILITY = 1.0
GAMMA = 1.0 / MOBILITY

# Translational inertia
MASS = 0.1
TAU_M = MASS / GAMMA

REPULSION_K = 10.0
PARTICLE_DIAMETER = 1.0

DISH_RADIUS = 40.0
INITIAL_CLUSTER_RADIUS = 10.0

RANDOM_SEED = 45
OUTPUT_DIR = "inertial_abp_cluster10_seed45_dt001_results"

# ============================================================
# 2. INITIALISE PARTICLES
# ============================================================

def initialise_particles(number, cluster_radius, seed):

    # Keep the original random sequence for orientations
    rng = np.random.default_rng(seed)

    rng.uniform(0, 2 * np.pi, number)
    rng.uniform(0, 1, number)

    theta = rng.uniform(0, 2 * np.pi, number)

    # Separate random generator for non-overlapping positions
    position_rng = np.random.default_rng(seed + 1)

    positions = np.empty((number, 2), dtype=float)

    min_distance = PARTICLE_DIAMETER
    max_attempts_per_particle = 100000

    total_attempts = 0

    for i in range(number):

        placed = False

        for attempt in range(max_attempts_per_particle):

            total_attempts += 1

            angle = position_rng.uniform(0, 2 * np.pi)

            radius = cluster_radius * np.sqrt(
                position_rng.uniform(0, 1)
            )

            candidate = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle)
            ])

            if i == 0:
                positions[i] = candidate
                placed = True
                break

            distances = np.linalg.norm(
                positions[:i] - candidate,
                axis=1
            )

            if np.all(distances >= min_distance):
                positions[i] = candidate
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place particle {i + 1}. "
                "Increase INITIAL_CLUSTER_RADIUS."
            )

    print("Initial configuration generated.")
    print(f"Particles placed: {number}")
    print(f"Minimum separation required: {min_distance}")
    print(f"Total placement attempts: {total_attempts}")

    return positions, theta, rng

# ============================================================
# 3. PARTICLE-PARTICLE REPULSION
# ============================================================

def calculate_repulsion(positions):

    displacement = (
        positions[:, np.newaxis, :]
        - positions[np.newaxis, :, :]
    )

    distances = np.linalg.norm(displacement, axis=2)

    safe_distances = np.maximum(distances, 1e-12)

    overlap = np.maximum(
        PARTICLE_DIAMETER - distances, 0.0
    )

    np.fill_diagonal(overlap, 0.0)

    force = (
        REPULSION_K
        * overlap[:, :, np.newaxis]
        * displacement
        / safe_distances[:, :, np.newaxis]
    )

    return np.sum(force, axis=1)


# ============================================================
# 4. VELOCITY-AWARE CIRCULAR BOUNDARY
# ============================================================

def apply_boundary(positions, velocities):

    distances = np.linalg.norm(positions, axis=1)

    outside = distances > DISH_RADIUS

    if np.any(outside):

        radial_unit = (
            positions[outside]
            / distances[outside, np.newaxis]
        )

        # Place particles back on the circular boundary
        positions[outside] = (
            DISH_RADIUS * radial_unit
        )

        # Reverse the outward radial velocity component
        radial_velocity = np.sum(
            velocities[outside] * radial_unit,
            axis=1
        )

        outward = radial_velocity > 0

        velocities_outside = velocities[outside].copy()

        velocities_outside[outward] -= (
            2.0
            * radial_velocity[outward, np.newaxis]
            * radial_unit[outward]
        )

        velocities[outside] = velocities_outside

    return positions, velocities


# ============================================================
# 5. CALCULATE MSD
# ============================================================

def calculate_msd(position_history):

    positions = np.asarray(position_history)

    initial_positions = positions[0]

    displacement = positions - initial_positions

    squared_displacement = np.sum(
        displacement ** 2, axis=2
    )

    return np.mean(squared_displacement, axis=1)


# ============================================================
# 6. MAIN INERTIAL SIMULATION
# ============================================================

def run_simulation():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    positions, theta, rng = initialise_particles(
        N,
        INITIAL_CLUSTER_RADIUS,
        RANDOM_SEED
    )

    # Initial orientation vectors
    direction = np.column_stack((
        np.cos(theta),
        np.sin(theta)
    ))

    # Initialise velocity using self-propulsion only
    velocities = V0 * direction

    position_history = [positions.copy()]
    velocity_history = [velocities.copy()]
    orientation_history = [theta.copy()]
    time_history = [0.0]

    # Thermal energy corresponding to the existing
    # translational diffusion coefficient
    KBT = GAMMA * DT_TRANS

    # Exact relaxation factor for the linear
    # underdamped Langevin velocity equation
    relaxation = np.exp(-DT / TAU_M)

    noise_amplitude = np.sqrt(
        (KBT / MASS)
        * (1.0 - relaxation ** 2)
    )

    print("Starting CP302 inertial ABP simulation...")
    print(f"Mass: {MASS}")
    print(f"Friction coefficient: {GAMMA}")
    print(f"Translational relaxation time: {TAU_M}")
    print(f"Time step: {DT}")
    print(f"Total simulation time: {STEPS * DT}")

    for step in range(1, STEPS + 1):

        # --------------------------------------------
        # A. Self-propulsion direction
        # --------------------------------------------

        direction = np.column_stack((
            np.cos(theta),
            np.sin(theta)
        ))

        # --------------------------------------------
        # B. Interaction force
        # --------------------------------------------

        force = calculate_repulsion(positions)

        # --------------------------------------------
        # C. Underdamped Langevin velocity update
        # --------------------------------------------

        terminal_velocity = (
            V0 * direction
            + force / GAMMA
        )

        random_force = rng.normal(
            0.0, 1.0, size=(N, 2)
        )

        velocities = (
            relaxation * velocities
            + (1.0 - relaxation) * terminal_velocity
            + noise_amplitude * random_force
        )

        # --------------------------------------------
        # D. Update particle positions
        # --------------------------------------------

        positions = positions + velocities * DT

        # --------------------------------------------
        # E. Apply boundary condition
        # --------------------------------------------

        positions, velocities = apply_boundary(
            positions, velocities
        )

        # --------------------------------------------
        # F. Rotational Brownian dynamics
        # --------------------------------------------

        theta += np.sqrt(2.0 * DR * DT) * rng.normal(
            0.0, 1.0, N
        )

        theta = np.mod(theta, 2.0 * np.pi)

        # --------------------------------------------
        # G. Save simulation data
        # --------------------------------------------

        if step % SAVE_EVERY == 0:

            position_history.append(positions.copy())
            velocity_history.append(velocities.copy())
            orientation_history.append(theta.copy())

            time_history.append(step * DT)

        if step % 500 == 0:
            print(f"Completed step {step}/{STEPS}")

    # ========================================================
    # 7. CONVERT DATA TO ARRAYS
    # ========================================================

    position_history = np.asarray(position_history)
    velocity_history = np.asarray(velocity_history)
    orientation_history = np.asarray(orientation_history)
    time_history = np.asarray(time_history)

    msd = calculate_msd(position_history)

    speed_history = np.linalg.norm(
        velocity_history, axis=2
    )

    # ========================================================
    # 8. SAVE TRAJECTORIES
    # ========================================================

    rows = []

    for frame, time in enumerate(time_history):

        for particle in range(N):

            rows.append({
                "time": time,
                "particle": particle,
                "x": position_history[frame, particle, 0],
                "y": position_history[frame, particle, 1],
                "vx": velocity_history[frame, particle, 0],
                "vy": velocity_history[frame, particle, 1],
                "theta": orientation_history[frame, particle],
                "speed": speed_history[frame, particle]
            })

    trajectories_df = pd.DataFrame(rows)

    trajectories_df.to_csv(
        os.path.join(OUTPUT_DIR, "trajectories.csv"),
        index=False
    )

    # ========================================================
    # 9. SAVE MSD
    # ========================================================

    msd_df = pd.DataFrame({
        "time": time_history,
        "MSD": msd
    })

    msd_df.to_csv(
        os.path.join(OUTPUT_DIR, "msd.csv"),
        index=False
    )

    # ========================================================
    # 10. SAVE VELOCITY AUTOCORRELATION
    # ========================================================

    initial_velocity = velocity_history[0]

    vacf = np.mean(
        np.sum(
            velocity_history * initial_velocity[np.newaxis, :, :],
            axis=2
        ),
        axis=1
    )

    vacf_normalized = vacf / vacf[0] if vacf[0] != 0 else vacf

    pd.DataFrame({
        "time_lag": time_history,
        "VACF": vacf,
        "VACF_normalized": vacf_normalized
    }).to_csv(
        os.path.join(OUTPUT_DIR, "vacf_results.csv"),
        index=False
    )

    # ========================================================
    # 11. SAVE ORIENTATION AUTOCORRELATION
    # ========================================================

    initial_direction = np.column_stack((
        np.cos(orientation_history[0]),
        np.sin(orientation_history[0])
    ))

    orientation_vectors = np.stack((
        np.cos(orientation_history),
        np.sin(orientation_history)
    ), axis=2)

    oacf = np.mean(
        np.sum(
            orientation_vectors
            * initial_direction[np.newaxis, :, :],
            axis=2
        ),
        axis=1
    )

    oacf_normalized = (
        oacf / oacf[0] if oacf[0] != 0 else oacf
    )

    pd.DataFrame({
        "time_lag": time_history,
        "OACF": oacf,
        "OACF_normalized": oacf_normalized
    }).to_csv(
        os.path.join(OUTPUT_DIR, "oacf_results.csv"),
        index=False
    )

    # ========================================================
    # 12. PLOT MSD
    # ========================================================

    plt.figure(figsize=(8, 6))

    plt.plot(time_history, msd, linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Mean Squared Displacement")
    plt.title("Inertial ABP: MSD")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "MSD.png"),
        dpi=300
    )

    plt.close()

    # ========================================================
    # 13. PLOT VELOCITY AUTOCORRELATION
    # ========================================================

    plt.figure(figsize=(8, 6))

    plt.plot(
        time_history,
        vacf_normalized,
        linewidth=2
    )

    plt.xlabel("Time lag")
    plt.ylabel("Normalized VACF")
    plt.title("Inertial ABP: Velocity Autocorrelation")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "VACF.png"),
        dpi=300
    )

    plt.close()

    # ========================================================
    # 14. FINAL SUMMARY
    # ========================================================

    print("\nSimulation completed successfully.")

    print(f"Particles: {N}")
    print(f"Saved frames: {len(time_history)}")
    print(f"Final time: {time_history[-1]}")
    print(f"Final MSD: {msd[-1]:.6f}")
    print(f"Mean final speed: {np.mean(speed_history[-1]):.6f}")

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_simulation()
