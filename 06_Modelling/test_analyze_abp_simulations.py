import csv

import numpy as np
import pytest

from analyze_abp_simulations import (
    AnalysisInputError,
    RunSpec,
    displacement_velocities,
    ensemble_msd,
    load_trajectory_csv,
    orientation_autocorrelation,
    time_averaged_msd,
    time_origin_dot_correlation,
)


def _spec(folder, *, model="overdamped", n=1, frames=3, interval=1.0):
    return RunSpec(
        run_id=f"{model}_synthetic",
        group="synthetic",
        model=model,
        n_particles=n,
        duration_s=(frames - 1) * interval,
        folder=folder,
        metadata={"output_interval_s": interval, "expected_frames": frames},
        diagnostics={},
    )


def _write_csv(path, rows, header=("time_s", "particle", "x", "y", "theta", "vx", "vy")):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def test_ensemble_msd_zero_at_time_zero_and_known_motion():
    positions = np.array(
        [
            [[0.0, 0.0], [3.0, 4.0]],
            [[1.0, 0.0], [3.0, 6.0]],
        ]
    )
    msd = ensemble_msd(positions)
    assert msd[0] == 0.0
    assert msd[1] == pytest.approx((1.0 + 4.0) / 2.0)


def test_time_averaged_msd_for_constant_speed_motion():
    # x(t)=t at unit frame spacing: every k-frame displacement squared is k^2.
    positions = np.array([[[t, 0.0]] for t in range(5)], dtype=float)
    assert time_averaged_msd(positions) == pytest.approx(np.arange(5, dtype=float) ** 2)


def _direct_tamsd_and_origin_counts(positions):
    """Brute-force reference, deliberately independent of the production FFT."""
    frames, particles, dimensions = positions.shape
    assert dimensions == 2
    direct = np.zeros(frames, dtype=np.float64)
    origin_counts = np.zeros(frames, dtype=np.int64)
    for lag in range(frames):
        origins = frames - lag
        origin_counts[lag] = origins
        squared_displacement_sum = 0.0
        for particle in range(particles):
            for origin in range(origins):
                squared_distance = 0.0
                for coordinate in range(2):
                    difference = (
                        positions[origin + lag, particle, coordinate]
                        - positions[origin, particle, coordinate]
                    )
                    squared_distance += difference * difference
                squared_displacement_sum += squared_distance
        direct[lag] = squared_displacement_sum / (particles * origins)
    return direct, origin_counts


def test_fft_tamsd_matches_independent_direct_loops_and_origin_counts():
    # Particle 0 is stationary. Particles 1 and 2 have distinct, non-constant
    # trajectories in both coordinates.
    positions = np.array(
        [
            [[2.0, -1.0], [0.0, 0.0], [-1.0, 4.0]],
            [[2.0, -1.0], [1.0, 2.0], [-2.0, 2.0]],
            [[2.0, -1.0], [3.0, 1.0], [0.0, 5.0]],
            [[2.0, -1.0], [2.0, -1.0], [4.0, 5.0]],
            [[2.0, -1.0], [5.0, 3.0], [3.0, 1.0]],
        ],
        dtype=np.float64,
    )

    fft_result = time_averaged_msd(positions)
    direct_result, origin_counts = _direct_tamsd_and_origin_counts(positions)

    assert fft_result == pytest.approx(direct_result, rel=1e-11, abs=1e-12)
    # M=5: lag zero has five origins; the maximum lag M-1 has one.
    assert origin_counts.tolist() == [5, 4, 3, 2, 1]
    assert origin_counts[0] == positions.shape[0]
    assert origin_counts[-1] == 1


def test_stationary_particle_has_zero_tamsd_at_every_lag():
    stationary_positions = np.repeat(
        np.array([[[2.0, -3.0]]], dtype=np.float64), repeats=5, axis=0
    )
    assert time_averaged_msd(stationary_positions) == pytest.approx(np.zeros(5), abs=1e-12)


def test_oacf_is_one_at_zero_lag():
    angles = np.array([[0.1, 1.5], [0.7, 1.1], [1.2, 0.3]])
    oacf = orientation_autocorrelation(angles)
    assert oacf[0] == pytest.approx(1.0)
    assert np.isfinite(oacf).all()


def test_displacement_velocity_and_vacf_zero_lag():
    positions = np.array([[[0.0, 0.0]], [[1.0, 0.0]], [[3.0, 0.0]]])
    velocity = displacement_velocities(positions, 1.0)
    assert velocity[:, 0, 0] == pytest.approx([1.0, 2.0])
    assert np.all(velocity[:, 0, 1] == 0.0)
    vacf = time_origin_dot_correlation(velocity)
    assert vacf[0] == pytest.approx((1.0**2 + 2.0**2) / 2.0)


def test_overdamped_blank_velocities_are_accepted_and_not_zero_state(tmp_path):
    rows = [
        (0.0, 0, 0.0, 0.0, 0.25, "", ""),
        (1.0, 0, 1.0, 0.0, 0.50, "", ""),
    ]
    _write_csv(tmp_path / "trajectories.csv", rows)
    data = load_trajectory_csv(_spec(tmp_path, frames=2, interval=1.0))
    assert data.positions.shape == (2, 1, 2)
    assert data.orientations[1, 0] == pytest.approx(0.50)
    assert not hasattr(data, "velocities")


def test_nonfinite_coordinates_are_rejected(tmp_path):
    _write_csv(
        tmp_path / "trajectories.csv",
        [(0.0, 0, 0.0, 0.0, 0.0, "", ""), (1.0, 0, "nan", 0.0, 0.0, "", "")],
    )
    with pytest.raises(AnalysisInputError, match="Non-finite position/orientation"):
        load_trajectory_csv(_spec(tmp_path, frames=2))


def test_inertial_nonfinite_velocity_is_rejected(tmp_path):
    _write_csv(
        tmp_path / "trajectories.csv",
        [
            (0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 0, 1.0, 0.0, 0.1, "inf", 0.0),
        ],
    )
    with pytest.raises(AnalysisInputError, match="Non-finite inertial velocity"):
        load_trajectory_csv(_spec(tmp_path, model="inertial", frames=2))


def test_invalid_particle_order_and_overdamped_velocity_are_rejected(tmp_path):
    _write_csv(
        tmp_path / "trajectories.csv",
        [(0.0, 1, 0.0, 0.0, 0.0, "", ""), (1.0, 0, 1.0, 0.0, 0.0, "", "")],
    )
    with pytest.raises(AnalysisInputError, match="Particle order or timestamp mismatch"):
        load_trajectory_csv(_spec(tmp_path, frames=2))

    _write_csv(
        tmp_path / "trajectories.csv",
        [(0.0, 0, 0.0, 0.0, 0.0, 0.0, ""), (1.0, 0, 1.0, 0.0, 0.0, "", "")],
    )
    with pytest.raises(AnalysisInputError, match="velocity fields must be blank"):
        load_trajectory_csv(_spec(tmp_path, frames=2))
