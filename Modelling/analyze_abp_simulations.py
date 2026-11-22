"""Analysis pipeline for the six seed-42 CP302  simulations.

This module reads the pilot CSVs without importing or executing simulation code.
The default command writes analysis products to a new, non-overwriting folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_VERSION = "1.0"
CSV_COLUMNS = ("time_s", "particle", "x", "y", "theta", "vx", "vy")
MODELLING_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = MODELLING_DIR / "abp_simulation_results" / "seed_42"
DEFAULT_OUTPUT_ROOT = MODELLING_DIR / "abp_analysis_results" / "seed_42"

# Explicitly enumerate the approved six-run pilot. Metadata is checked against
# this matrix rather than inferred from directory names alone.
CONFIGURATIONS = (
    ("N100_T50", "overdamped", 100, 50.0),
    ("N100_T50", "inertial", 100, 50.0),
    ("N400_T50", "overdamped", 400, 50.0),
    ("N400_T50", "inertial", 400, 50.0),
    ("N100_T200", "overdamped", 100, 200.0),
    ("N100_T200", "inertial", 100, 200.0),
)


class AnalysisInputError(ValueError):
    """Raised when a pilot input is missing or fails integrity checks."""


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    group: str
    model: str
    n_particles: int
    duration_s: float
    folder: Path
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]

    @property
    def output_interval_s(self) -> float:
        return float(self.metadata["output_interval_s"])

    @property
    def expected_frames(self) -> int:
        return int(self.metadata["expected_frames"])


@dataclass
class TrajectoryData:
    spec: RunSpec
    positions: np.ndarray  # frames x particles x 2
    orientations: np.ndarray  # frames x particles


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisInputError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisInputError(f"Expected a JSON object in {path}")
    return value


def _is_empty_errors(value: Any) -> bool:
    return value in (None, [], {}, "")


def _close(a: float, b: float, *, atol: float = 1e-10) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def discover_runs(input_root: Path | str = DEFAULT_INPUT_ROOT) -> list[RunSpec]:
    """Validate run records and the exact six-configuration directory matrix."""
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise AnalysisInputError(f"Pilot input root does not exist: {root}")

    expected_groups = {item[0] for item in CONFIGURATIONS}
    actual_groups = {entry.name for entry in root.iterdir() if entry.is_dir()}
    if actual_groups != expected_groups:
        raise AnalysisInputError(
            f"Expected configuration folders {sorted(expected_groups)}, found {sorted(actual_groups)}"
        )

    specs: list[RunSpec] = []
    for group, model, n_particles, duration_s in CONFIGURATIONS:
        folder = root / group / model
        if not folder.is_dir():
            raise AnalysisInputError(f"Missing run directory: {folder}")
        model_dirs = {entry.name for entry in (root / group).iterdir() if entry.is_dir()}
        if model_dirs != {"overdamped", "inertial"}:
            raise AnalysisInputError(
                f"{root / group} must contain exactly overdamped and inertial directories"
            )

        metadata = _read_json(folder / "run_metadata.json")
        diagnostics = _read_json(folder / "diagnostics.json")
        run_id = f"{model}_{group}"
        if metadata.get("run_id") != run_id or diagnostics.get("run_id") != run_id:
            raise AnalysisInputError(f"Run ID mismatch in {folder}")
        if metadata.get("execution_status") != "completed" or diagnostics.get("execution_status") != "completed":
            raise AnalysisInputError(f"Run is not completed according to both records: {folder}")
        if not _is_empty_errors(metadata.get("error")):
            raise AnalysisInputError(f"Metadata contains an error for {run_id}: {metadata['error']}")
        if not _is_empty_errors(metadata.get("finalization_errors")):
            raise AnalysisInputError(f"Metadata has finalization errors for {run_id}")
        if not _is_empty_errors(diagnostics.get("errors")) or not _is_empty_errors(
            diagnostics.get("finalization_errors")
        ):
            raise AnalysisInputError(f"Diagnostics contain errors for {run_id}")

        if metadata.get("model") != model or int(metadata.get("n_particles", -1)) != n_particles:
            raise AnalysisInputError(f"Model or particle count mismatch in {folder}")
        if not _close(metadata.get("duration_s", math.nan), duration_s):
            raise AnalysisInputError(f"Duration mismatch in {folder}")
        if int(metadata.get("seed", -1)) != 42:
            raise AnalysisInputError(f"Expected seed 42 in {folder}")

        dt = float(metadata.get("dt_s", math.nan))
        output_dt = float(metadata.get("output_interval_s", math.nan))
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(output_dt) or output_dt <= 0:
            raise AnalysisInputError(f"Invalid timestep metadata in {folder}")
        step_ratio = duration_s / dt
        save_ratio = output_dt / dt
        if not _close(step_ratio, round(step_ratio)) or not _close(save_ratio, round(save_ratio)):
            raise AnalysisInputError(f"Duration/output interval is incompatible with DT in {folder}")
        expected_steps = int(round(step_ratio))
        expected_frames = expected_steps // int(round(save_ratio)) + 1
        expected_rows = expected_frames * n_particles

        expected_metadata = {
            "expected_steps": expected_steps,
            "completed_steps": expected_steps,
            "expected_frames": expected_frames,
            "expected_particle_rows": expected_rows,
        }
        for key, expected in expected_metadata.items():
            if int(metadata.get(key, -1)) != expected:
                raise AnalysisInputError(f"Metadata {key} mismatch in {folder}")
        expected_diagnostics = {
            "completed_steps": expected_steps,
            "expected_frames": expected_frames,
            "actual_frames": expected_frames,
            "expected_particle_rows": expected_rows,
            "actual_particle_rows": expected_rows,
        }
        for key, expected in expected_diagnostics.items():
            if int(diagnostics.get(key, -1)) != expected:
                raise AnalysisInputError(f"Diagnostics {key} mismatch in {folder}")
        for key in (
            "non_finite_positions_detected",
            "non_finite_orientations_detected",
            "unresolved_boundary_crossing",
        ):
            if diagnostics.get(key) is not False:
                raise AnalysisInputError(f"Diagnostics flag {key} is not false in {folder}")
        velocity_flag = diagnostics.get("non_finite_velocities_detected")
        expected_velocity_flag = False if model == "inertial" else None
        if velocity_flag is not expected_velocity_flag:
            raise AnalysisInputError(f"Unexpected velocity diagnostic for {run_id}")
        if model == "inertial" and diagnostics.get("inertial_step_diagnostics_complete") is not True:
            raise AnalysisInputError(f"Inertial step diagnostics are incomplete for {run_id}")

        specs.append(
            RunSpec(run_id, group, model, n_particles, duration_s, folder, metadata, diagnostics)
        )

    # Each model pair must share its initial conditions and all non-model setup.
    by_group: dict[str, list[RunSpec]] = {}
    for spec in specs:
        by_group.setdefault(spec.group, []).append(spec)
    for group, pair in by_group.items():
        if len(pair) != 2:
            raise AnalysisInputError(f"Expected a matched pair for {group}")
        od, inertial = sorted(pair, key=lambda item: item.model != "overdamped")
        if od.metadata.get("initial_conditions_sha256") != inertial.metadata.get(
            "initial_conditions_sha256"
        ):
            raise AnalysisInputError(f"Initial-condition hashes differ for {group}")
        for key in ("seed", "dt_s", "output_interval_s", "dish_radius", "initial_cluster_radius"):
            if od.metadata.get(key) != inertial.metadata.get(key):
                raise AnalysisInputError(f"Paired metadata differs for {group}: {key}")
        if od.metadata.get("physical_parameters") != inertial.metadata.get("physical_parameters"):
            raise AnalysisInputError(f"Paired physical parameters differ for {group}")

    intervals = {round(spec.output_interval_s, 12) for spec in specs}
    if len(intervals) != 1:
        raise AnalysisInputError("All six runs must use the same output interval")
    return specs


def load_trajectory_csv(spec: RunSpec) -> TrajectoryData:
    """Read one CSV into compact position/orientation arrays and validate every row."""
    path = spec.folder / "trajectories.csv"
    positions = np.empty((spec.expected_frames, spec.n_particles, 2), dtype=np.float64)
    orientations = np.empty((spec.expected_frames, spec.n_particles), dtype=np.float64)
    row_number = 0
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if tuple(header or ()) != CSV_COLUMNS:
                raise AnalysisInputError(
                    f"Unexpected CSV header in {path}: {header!r}; expected {CSV_COLUMNS!r}"
                )
            for row in reader:
                if len(row) != len(CSV_COLUMNS):
                    raise AnalysisInputError(
                        f"Expected seven columns at {path}, CSV row {row_number + 2}"
                    )
                if row_number >= spec.expected_frames * spec.n_particles:
                    raise AnalysisInputError(f"Too many particle rows in {path}")
                frame, particle = divmod(row_number, spec.n_particles)
                try:
                    time_s = float(row[0])
                    particle_id = int(row[1])
                    x, y, theta = (float(row[index]) for index in (2, 3, 4))
                except (ValueError, TypeError) as exc:
                    raise AnalysisInputError(
                        f"Invalid numeric value in {path}, CSV row {row_number + 2}"
                    ) from exc
                expected_time = frame * spec.output_interval_s
                if particle_id != particle or not _close(time_s, expected_time):
                    raise AnalysisInputError(
                        f"Particle order or timestamp mismatch in {path}, CSV row {row_number + 2}"
                    )
                if not np.isfinite((x, y, theta)).all():
                    raise AnalysisInputError(
                        f"Non-finite position/orientation in {path}, CSV row {row_number + 2}"
                    )
                vx_text, vy_text = row[5], row[6]
                if spec.model == "overdamped":
                    if vx_text != "" or vy_text != "":
                        raise AnalysisInputError(
                            f"Overdamped velocity fields must be blank in {path}, row {row_number + 2}"
                        )
                else:
                    try:
                        vx, vy = float(vx_text), float(vy_text)
                    except (ValueError, TypeError) as exc:
                        raise AnalysisInputError(
                            f"Missing/non-numeric inertial velocity in {path}, row {row_number + 2}"
                        ) from exc
                    if not np.isfinite((vx, vy)).all():
                        raise AnalysisInputError(
                            f"Non-finite inertial velocity in {path}, row {row_number + 2}"
                        )
                positions[frame, particle] = (x, y)
                orientations[frame, particle] = theta
                row_number += 1
    except OSError as exc:
        raise AnalysisInputError(f"Cannot read trajectory CSV {path}: {exc}") from exc

    expected_rows = spec.expected_frames * spec.n_particles
    if row_number != expected_rows:
        raise AnalysisInputError(f"{path} has {row_number} rows; expected {expected_rows}")
    return TrajectoryData(spec, positions, orientations)


def ensemble_msd(positions: np.ndarray) -> np.ndarray:
    """Return time-zero ensemble MSD, averaged over particles at each frame."""
    if positions.ndim != 3 or positions.shape[2] != 2 or positions.shape[0] < 1:
        raise ValueError("positions must have shape (frames, particles, 2)")
    displacement = positions - positions[0:1]
    values = np.mean(np.sum(displacement * displacement, axis=2), axis=1)
    values[0] = 0.0
    return values


def _fft_autocorrelation(values: np.ndarray, *, center: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    length = values.size
    if length == 0:
        raise ValueError("autocorrelation input must not be empty")
    if center:
        values = values - np.mean(values)
    fft_length = 1 << (2 * length - 1).bit_length()
    transformed = np.fft.rfft(values, n=fft_length)
    return np.fft.irfft(transformed * transformed.conjugate(), n=fft_length)[:length]


def time_averaged_msd(positions: np.ndarray) -> np.ndarray:
    """Compute TAMSD over all available origins using FFT autocorrelations.

    For lag k this returns the mean of |r_i(t+k)-r_i(t)|^2 over particles and
    all M-k valid time origins. Per-particle centering is used only to improve
    numerical conditioning; it does not change any displacement.
    """
    if positions.ndim != 3 or positions.shape[2] != 2 or positions.shape[0] < 1:
        raise ValueError("positions must have shape (frames, particles, 2)")
    frames, particles, _ = positions.shape
    if particles < 1:
        raise ValueError("positions must contain at least one particle")
    accumulated = np.zeros(frames, dtype=np.float64)
    for particle in range(particles):
        for dimension in range(2):
            signal = positions[:, particle, dimension].astype(np.float64, copy=True)
            signal -= np.mean(signal)
            prefix_square = np.concatenate(([0.0], np.cumsum(signal * signal)))
            autocorrelation = _fft_autocorrelation(signal)
            for lag in range(frames):
                count = frames - lag
                left = prefix_square[count]
                right = prefix_square[frames] - prefix_square[lag]
                squared_displacements = left + right - 2.0 * autocorrelation[lag]
                accumulated[lag] += max(float(squared_displacements), 0.0)
    result = accumulated / (particles * np.arange(frames, 0, -1, dtype=np.float64))
    result[0] = 0.0
    return result


def time_origin_dot_correlation(vectors: np.ndarray) -> np.ndarray:
    """Return mean dot(a(t),a(t+lag)) over all origins and particles."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 3 or vectors.shape[2] != 2 or vectors.shape[0] < 1:
        raise ValueError("vectors must have shape (frames, particles, 2)")
    frames, particles, _ = vectors.shape
    if particles < 1 or not np.isfinite(vectors).all():
        raise ValueError("vectors must be non-empty and finite")
    accumulated = np.zeros(frames, dtype=np.float64)
    for particle in range(particles):
        for dimension in range(2):
            accumulated += _fft_autocorrelation(vectors[:, particle, dimension])
    counts = particles * np.arange(frames, 0, -1, dtype=np.float64)
    return accumulated / counts


def orientation_autocorrelation(orientations: np.ndarray) -> np.ndarray:
    """Compute time-origin-averaged cos(theta(t+lag)-theta(t))."""
    orientations = np.asarray(orientations, dtype=np.float64)
    if orientations.ndim != 2 or orientations.shape[0] < 1 or orientations.shape[1] < 1:
        raise ValueError("orientations must have shape (frames, particles)")
    if not np.isfinite(orientations).all():
        raise ValueError("orientations must be finite")
    directions = np.stack((np.cos(orientations), np.sin(orientations)), axis=2)
    result = time_origin_dot_correlation(directions)
    result[0] = 1.0
    return result


def displacement_velocities(positions: np.ndarray, output_interval_s: float) -> np.ndarray:
    """Estimate interval-averaged velocities from saved position increments."""
    if not math.isfinite(output_interval_s) or output_interval_s <= 0:
        raise ValueError("output_interval_s must be finite and positive")
    if positions.ndim != 3 or positions.shape[2] != 2 or positions.shape[0] < 2:
        raise ValueError("positions must contain at least two frames")
    return np.diff(positions, axis=0) / output_interval_s


def local_msd_slopes(time_s: np.ndarray, msd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite-difference d(MSD)/dt and log-log exponent beta.

    This is diagnostic output only. It neither chooses a fit window nor reports
    a diffusion coefficient.
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    msd = np.asarray(msd, dtype=np.float64)
    valid = (time_s > 0) & (msd > 0) & np.isfinite(time_s) & np.isfinite(msd)
    t = time_s[valid]
    y = msd[valid]
    if t.size < 3 or np.any(np.diff(t) <= 0):
        raise ValueError("At least three increasing positive time/MSD points are needed")
    return t, np.gradient(y, t), np.gradient(np.log(y), np.log(t))


def _compute_run(data: TrajectoryData) -> dict[str, Any]:
    spec = data.spec
    msd = ensemble_msd(data.positions)
    tamsd = time_averaged_msd(data.positions)
    oacf = orientation_autocorrelation(data.orientations)
    velocities = displacement_velocities(data.positions, spec.output_interval_s)
    vacf = time_origin_dot_correlation(velocities)
    vacf_zero = float(vacf[0])
    normalized_vacf = vacf / vacf_zero if vacf_zero != 0.0 else np.zeros_like(vacf)
    time = np.arange(spec.expected_frames, dtype=np.float64) * spec.output_interval_s
    lag_o = np.arange(spec.expected_frames, dtype=np.float64) * spec.output_interval_s
    lag_v = np.arange(velocities.shape[0], dtype=np.float64) * spec.output_interval_s
    slope_t, slope_linear, slope_beta = local_msd_slopes(time, msd)
    return {
        "spec": spec,
        "time_s": time,
        "ensemble_msd": msd,
        "tamsd_lag_s": lag_o,
        "tamsd": tamsd,
        "oacf_lag_s": lag_o,
        "oacf": oacf,
        "vacf_lag_s": lag_v,
        "vacf": vacf,
        "vacf_normalized": normalized_vacf,
        "local_slope_time_s": slope_t,
        "local_dmsd_dt": slope_linear,
        "local_log_slope_beta": slope_beta,
    }


def _write_csv(path: Path, header: tuple[str, ...], rows: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _write_run_tables(result: dict[str, Any], output_root: Path) -> None:
    spec: RunSpec = result["spec"]
    n = spec.n_particles
    _write_csv(
        output_root / "ensemble_msd" / f"{spec.run_id}_ensemble_msd.csv",
        ("time_s", "MSD", "particle_count"),
        zip(result["time_s"], result["ensemble_msd"], [n] * len(result["time_s"])),
    )
    _write_csv(
        output_root / "time_averaged_msd" / f"{spec.run_id}_time_averaged_msd.csv",
        ("lag_s", "TAMSD", "particle_count", "time_origin_count"),
        zip(
            result["tamsd_lag_s"], result["tamsd"], [n] * len(result["tamsd"]),
            np.arange(spec.expected_frames, 0, -1),
        ),
    )
    _write_csv(
        output_root / "local_msd_slopes" / f"{spec.run_id}_local_msd_slopes.csv",
        ("time_s", "local_dMSD_dt", "local_log_slope_beta"),
        zip(result["local_slope_time_s"], result["local_dmsd_dt"], result["local_log_slope_beta"]),
    )
    _write_csv(
        output_root / "correlations" / f"{spec.run_id}_oacf.csv",
        ("lag_s", "OACF", "OACF_normalized", "particle_count", "time_origin_count"),
        zip(
            result["oacf_lag_s"], result["oacf"], result["oacf"], [n] * len(result["oacf"]),
            np.arange(spec.expected_frames, 0, -1),
        ),
    )
    velocity_origins = len(result["vacf"])
    _write_csv(
        output_root / "correlations" / f"{spec.run_id}_displacement_vacf.csv",
        (
            "lag_s", "VACF_displacement", "VACF_displacement_normalized",
            "particle_count", "time_origin_count",
        ),
        zip(
            result["vacf_lag_s"], result["vacf"], result["vacf_normalized"],
            [n] * velocity_origins, np.arange(velocity_origins, 0, -1),
        ),
    )


def _label(spec: RunSpec) -> str:
    return f"{spec.model}, N={spec.n_particles}, T={spec.duration_s:g} s"


def _save_msd_plot(results: dict[str, dict[str, Any]], path: Path, *, loglog: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for result in results.values():
        spec: RunSpec = result["spec"]
        t, y = result["time_s"], result["ensemble_msd"]
        if loglog:
            mask = (t > 0) & (y > 0) & np.isfinite(t) & np.isfinite(y)
            ax.loglog(t[mask], y[mask], label=_label(spec))
        else:
            ax.plot(t, y, label=_label(spec))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ensemble MSD (model length units²)")
    ax.set_title("Ensemble Mean-Squared Displacement (MSD)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_msd_comparison(
    results: dict[str, dict[str, Any]], ids: tuple[str, str], path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for run_id in ids:
        result = results[run_id]
        spec: RunSpec = result["spec"]
        ax.plot(result["time_s"], result["ensemble_msd"], label=_label(spec))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ensemble MSD (model length units²)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_correlation_comparison(
    results: dict[str, dict[str, Any]], ids: tuple[str, str], path: Path, *, kind: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for run_id in ids:
        result = results[run_id]
        spec: RunSpec = result["spec"]
        if kind == "oacf":
            time, values = result["oacf_lag_s"], result["oacf"]
            ylabel = "OACF (normalized; dimensionless)"
            title = "Orientation autocorrelation: matched models"
        else:
            time, values = result["vacf_lag_s"], result["vacf_normalized"]
            ylabel = "Normalized displacement-based VACF (dimensionless)"
            title = "Displacement-based VACF: matched models"
        ax.plot(time, values, label=_label(spec))
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_all(results: dict[str, dict[str, Any]], output_root: Path) -> None:
    plots = output_root / "plots"
    _save_msd_plot(results, plots / "ensemble_msd_linear.png", loglog=False)
    _save_msd_plot(results, plots / "ensemble_msd_loglog.png", loglog=True)

    pairs = (
        ("N100_T50", "overdamped_N100_T50", "inertial_N100_T50"),
        ("N400_T50", "overdamped_N400_T50", "inertial_N400_T50"),
        ("N100_T200", "overdamped_N100_T200", "inertial_N100_T200"),
    )
    for group, od, inertial in pairs:
        _save_msd_comparison(
            results, (od, inertial), plots / f"msd_matched_{group}.png",
            f"Ensemble MSD: overdamped vs inertial ({group.replace('_', ', ')})",
        )
        _save_correlation_comparison(
            results, (od, inertial), plots / f"oacf_matched_{group}.png", kind="oacf"
        )
        _save_correlation_comparison(
            results, (od, inertial), plots / f"displacement_vacf_matched_{group}.png", kind="vacf"
        )

    for model in ("overdamped", "inertial"):
        _save_msd_comparison(
            results,
            (f"{model}_N100_T50", f"{model}_N400_T50"),
            plots / f"msd_N100_vs_N400_T50_{model}.png",
            f"Ensemble MSD: N=100 vs N=400 at 50 s ({model})",
        )
        _save_msd_comparison(
            results,
            (f"{model}_N100_T50", f"{model}_N100_T200"),
            plots / f"msd_T50_vs_T200_N100_{model}.png",
            f"Ensemble MSD: 50 s vs 200 s, N=100 ({model})",
        )


def run_analysis(
    input_root: Path | str = DEFAULT_INPUT_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Validate, analyze, and write outputs; refuse an existing output folder."""
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {output_root}")

    specs = discover_runs(input_root)
    results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        data = load_trajectory_csv(spec)
        results[spec.run_id] = _compute_run(data)
        del data

    # Recheck immediately before making the first output directory.
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    for name in ("ensemble_msd", "time_averaged_msd", "local_msd_slopes", "correlations", "plots"):
        (output_root / name).mkdir()

    for result in results.values():
        _write_run_tables(result, output_root)
    _plot_all(results, output_root)

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(Path(input_root).resolve()),
        "output_root": str(output_root),
        "runs": [
            {
                "run_id": result["spec"].run_id,
                "model": result["spec"].model,
                "n_particles": result["spec"].n_particles,
                "duration_s": result["spec"].duration_s,
                "output_interval_s": result["spec"].output_interval_s,
                "seed": result["spec"].metadata["seed"],
                "initial_conditions_sha256": result["spec"].metadata[
                    "initial_conditions_sha256"
                ],
            }
            for result in results.values()
        ],
        "estimators": {
            "ensemble_msd": "mean_i |r_i(t)-r_i(0)|^2",
            "time_averaged_msd": "mean_i,t |r_i(t+lag)-r_i(t)|^2 over all valid origins",
            "oacf": "mean_i,t cos(theta_i(t+lag)-theta_i(t)); normalized by lag zero",
            "displacement_vacf": "mean_i,t v_i(t) dot v_i(t+lag), v=delta_r/output_interval; normalized by lag zero",
            "diffusion": "No coefficient is automatically fitted or reported.",
        },
    }
    (output_root / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        output = run_analysis(args.input_root, args.output_root)
    except (AnalysisInputError, FileExistsError, OSError, ValueError, FloatingPointError) as exc:
        parser.exit(2, f"Analysis stopped: {exc}\n")
    print(f"Analysis completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
