"""Safe, explicit driver for the six approved CP302 pilot configurations.

Importing this module has no filesystem or simulation side effects. Simulations
start only through the explicit ``--run`` or ``--all`` command-line options.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

try:  # Supports both direct script execution and package-style imports.
    from . import cp302_numerical_core as core
except ImportError:
    import cp302_numerical_core as core


DT = 0.005
OUTPUT_INTERVAL = 0.01
SEED = 42
MODELLING_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = MODELLING_DIR / "abp_simulation_results"


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    model: str
    n_particles: int
    duration: float
    dish_radius: float
    initial_cluster_radius: float
    seed: int = SEED
    dt: float = DT
    output_interval: float = OUTPUT_INTERVAL

    def relative_output_path(self) -> Path:
        model_folder = "overdamped" if self.model == "overdamped" else "inertial"
        return Path(f"seed_{self.seed}") / f"N{self.n_particles}_T{int(self.duration)}" / model_folder


CONFIGURATIONS: tuple[RunConfig, ...] = (
    RunConfig("overdamped_N100_T50", "overdamped", 100, 50.0, 40.0, 5.0),
    RunConfig("inertial_N100_T50", "inertial", 100, 50.0, 40.0, 5.0),
    RunConfig("overdamped_N400_T50", "overdamped", 400, 50.0, 80.0, 10.0),
    RunConfig("inertial_N400_T50", "inertial", 400, 50.0, 80.0, 10.0),
    RunConfig("overdamped_N100_T200", "overdamped", 100, 200.0, 40.0, 5.0),
    RunConfig("inertial_N100_T200", "inertial", 100, 200.0, 40.0, 5.0),
)

CONFIG_BY_ID = {config.run_id: config for config in CONFIGURATIONS}


class PreflightError(RuntimeError):
    """Raised before simulation if configuration or destination is unsafe."""


class TrajectoryIntegrityError(IOError):
    """Raised when the saved trajectory does not match its declared schema."""

    def __init__(self, message: str, actual_frames: int, actual_rows: int) -> None:
        super().__init__(message)
        self.actual_frames = actual_frames
        self.actual_rows = actual_rows


class RunRecordWriteError(IOError):
    """Raised when either initial run record cannot be persisted."""


@dataclass(frozen=True)
class RunPlan:
    config: RunConfig
    output_dir: Path
    steps: int
    save_stride: int
    expected_frames: int
    expected_particle_rows: int


def _integer_ratio(numerator: float, denominator: float, label: str) -> int:
    ratio = numerator / denominator
    nearest = round(ratio)
    if not math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-10):
        raise PreflightError(f"{label} must be an integer multiple of its interval")
    return int(nearest)


def build_plan(
    config: RunConfig,
    output_root: Path | str = RESULTS_ROOT,
    *,
    check_destination: bool = True,
) -> RunPlan:
    """Validate one run without creating directories or opening output files."""
    if config.model not in {"overdamped", "inertial"}:
        raise PreflightError(f"Unsupported model: {config.model}")
    if config.n_particles < 1 or config.seed < 0:
        raise PreflightError("Particle count must be positive and seed nonnegative")
    if config.dt <= 0 or config.output_interval <= 0 or config.duration <= 0:
        raise PreflightError("DT, output interval, and duration must be positive")
    if config.dish_radius <= 0 or config.initial_cluster_radius <= 0:
        raise PreflightError("Geometry radii must be positive")
    steps = _integer_ratio(config.duration, config.dt, "Duration")
    save_stride = _integer_ratio(config.output_interval, config.dt, "Output interval")
    if steps < 1 or save_stride < 1 or steps % save_stride:
        raise PreflightError("Final time must fall exactly on an output time")

    requested_root = Path(output_root).expanduser().absolute()
    if requested_root == RESULTS_ROOT.absolute() and RESULTS_ROOT.is_symlink():
        raise PreflightError("The pilot results root must not be a symbolic link")
    root = requested_root.resolve()
    if requested_root == RESULTS_ROOT.absolute() and root != requested_root:
        raise PreflightError("The pilot results root resolves through a redirected path")
    output_dir = (root / config.relative_output_path()).resolve()
    if not output_dir.is_relative_to(root) or output_dir == root:
        raise PreflightError("Output path escapes the configured results root")
    if check_destination and output_dir.exists():
        raise PreflightError(f"Refusing to reuse existing output directory: {output_dir}")
    return RunPlan(
        config=config,
        output_dir=output_dir,
        steps=steps,
        save_stride=save_stride,
        expected_frames=steps // save_stride + 1,
        expected_particle_rows=config.n_particles * (steps // save_stride + 1),
    )


def validate_all(output_root: Path | str = RESULTS_ROOT) -> tuple[RunPlan, ...]:
    """Preflight all six destinations and reject duplicate paths."""
    if len(CONFIGURATIONS) != 6:
        raise PreflightError("The pilot must contain exactly six configurations")
    plans = tuple(build_plan(config, output_root) for config in CONFIGURATIONS)
    paths = [plan.output_dir for plan in plans]
    if len(set(paths)) != len(paths):
        raise PreflightError("Pilot output paths must be unique")
    return plans


def _core_parameters(config: RunConfig) -> core.Parameters:
    """Construct the approved physical parameters using the current core API."""
    return core.Parameters(
        n_particles=config.n_particles,
        dt=config.dt,
        dt_trans=0.01,
        dr=0.1,
        mobility=1.0,
        mass=0.1,
        v0=1.0,
        repulsion_k=10.0,
        particle_diameter=1.0,
        dish_radius=config.dish_radius,
        initial_cluster_radius=config.initial_cluster_radius,
    )


def initial_conditions_hash(initial: core.InitialConditions) -> str:
    """Hash canonical float64 position/orientation arrays for paired runs."""
    digest = hashlib.sha256()
    for name, values in (("positions", initial.positions), ("orientations", initial.orientations)):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_initial_records(
    metadata_path: Path,
    diagnostics_path: Path,
    metadata: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    """Attempt both initial records and fail closed if either write fails."""
    failures: list[str] = []
    for label, path, record in (
        ("run_metadata.json", metadata_path, metadata),
        ("diagnostics.json", diagnostics_path, diagnostics),
    ):
        try:
            _write_json(path, record)
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    if failures:
        message = "Initial run-record write failed before simulation: " + "; ".join(failures)
        diagnostics["execution_status"] = "incomplete"
        diagnostics["errors"].extend(failures)
        diagnostics["incomplete_run_information"] = (
            "Initial metadata/diagnostics persistence failed before the simulation loop; "
            "zero simulation steps were completed."
        )
        metadata["execution_status"] = "incomplete"
        metadata["error"] = message
        metadata["completed_steps"] = 0
        raise RunRecordWriteError(message)


def _finalize_run_records(
    metadata_path: Path,
    diagnostics_path: Path,
    metadata: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    status: str,
    ended_at: str,
    primary_error: str | None,
) -> tuple[str, list[str]]:
    """Best-effort persist final status without masking an earlier failure."""
    finalization_errors: list[str] = []

    def stamp_records() -> None:
        diagnostics["execution_status"] = status
        diagnostics["ended_at_utc"] = ended_at
        diagnostics["finalization_errors"] = list(finalization_errors)
        metadata["execution_status"] = status
        metadata["ended_at_utc"] = ended_at
        metadata["error"] = primary_error
        metadata["completed_steps"] = diagnostics["completed_steps"]
        metadata["finalization_errors"] = list(finalization_errors)

    # Retry both records independently. A diagnostics write failure downgrades
    # the manifest before it is written; a manifest write failure triggers a
    # later rewrite of diagnostics with incomplete status.
    for _ in range(3):
        errors_before_pass = len(finalization_errors)
        stamp_records()
        try:
            _write_json(diagnostics_path, diagnostics)
        except Exception as exc:
            message = f"diagnostics.json: {type(exc).__name__}: {exc}"
            finalization_errors.append(message)
            diagnostics["errors"].append(message)
            diagnostics["incomplete_run_information"] = (
                "Finalization write failed; final status forced to incomplete."
            )
            status = "incomplete"

        stamp_records()
        try:
            _write_json(metadata_path, metadata)
        except Exception as exc:
            message = f"run_metadata.json: {type(exc).__name__}: {exc}"
            finalization_errors.append(message)
            diagnostics["errors"].append(message)
            diagnostics["incomplete_run_information"] = (
                "Finalization write failed; final status forced to incomplete."
            )
            status = "incomplete"
        if len(finalization_errors) == errors_before_pass:
            return status, finalization_errors
    stamp_records()
    return status, finalization_errors


def _manifest(
    plan: RunPlan,
    params: core.Parameters,
    initial_hash: str,
    core_hash: str,
    started_at: str,
    status: str,
    ended_at: str | None = None,
    error: str | None = None,
    completed_steps: int = 0,
) -> dict[str, Any]:
    config = plan.config
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "model": config.model,
        "n_particles": config.n_particles,
        "duration_s": config.duration,
        "dt_s": config.dt,
        "output_interval_s": config.output_interval,
        "seed": config.seed,
        "dish_radius": config.dish_radius,
        "initial_cluster_radius": config.initial_cluster_radius,
        "physical_parameters": {
            "dt_trans": params.dt_trans,
            "dr": params.dr,
            "mobility": params.mobility,
            "gamma": params.gamma,
            "mass": params.mass,
            "tau_m": params.tau_m,
            "kbt": params.kbt,
            "v0": params.v0,
            "repulsion_k": params.repulsion_k,
            "particle_diameter": params.particle_diameter,
        },
        "equations_note": (
            "Uses cp302_numerical_core; inertial OU substep is exact only for frozen force "
            "and orientation, and the interacting predictor-corrector is approximate."
        ),
        "numerical_core_file": core.__file__,
        "numerical_core_sha256": core_hash,
        "initial_conditions_sha256": initial_hash,
        "rng_design": (
            "SeedSequence(seed).spawn(6) streams: initialization positions, initialization "
            "orientations, overdamped translation, inertial translation, rotation, bridge. "
            "Initial conditions are shared by a pair; translational noise paths are model-specific."
        ),
        "expected_steps": plan.steps,
        "completed_steps": completed_steps,
        "save_stride_steps": plan.save_stride,
        "expected_frames": plan.expected_frames,
        "expected_particle_rows": plan.expected_particle_rows,
        "output_directory": str(plan.output_dir),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "execution_status": status,
        "error": error,
    }


def _new_diagnostics(plan: RunPlan, started_at: str) -> dict[str, Any]:
    return {
        "run_id": plan.config.run_id,
        "execution_status": "running",
        "started_at_utc": started_at,
        "ended_at_utc": None,
        "expected_frames": plan.expected_frames,
        "actual_frames": 0,
        "expected_particle_rows": plan.expected_particle_rows,
        "actual_particle_rows": 0,
        "boundary_corrections": 0,
        "inertial_refinements": None if plan.config.model == "overdamped" else 0,
        "maximum_refinement_depth": None if plan.config.model == "overdamped" else 0,
        "inertial_step_diagnostics_complete": None if plan.config.model == "overdamped" else True,
        "unresolved_boundary_crossing": False,
        "non_finite_positions_detected": False,
        "non_finite_orientations_detected": False,
        "non_finite_velocities_detected": None if plan.config.model == "overdamped" else False,
        "completed_steps": 0,
        "errors": [],
        "incomplete_run_information": None,
    }


def _check_finite(state: Any, model: str, diagnostics: dict[str, Any]) -> None:
    diagnostics["non_finite_positions_detected"] = not np.isfinite(state.positions).all()
    diagnostics["non_finite_orientations_detected"] = not np.isfinite(state.orientations).all()
    if model == "inertial":
        diagnostics["non_finite_velocities_detected"] = not np.isfinite(state.velocities).all()
    if any(
        diagnostics[key]
        for key in (
            "non_finite_positions_detected",
            "non_finite_orientations_detected",
            "non_finite_velocities_detected",
        )
    ):
        raise FloatingPointError("Non-finite simulation state detected")


def _validate_initial_state(state: Any, model: str, diagnostics: dict[str, Any]) -> None:
    """Validate the model's actual initial-state representation directly."""
    if model == "inertial":
        if not isinstance(state, core.InertialState):
            raise TypeError("Inertial initialization must return an InertialState")
        finite_state = state
    elif model == "overdamped":
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("Overdamped initialization must be a (positions, orientations) pair")
        finite_state = core.InitialConditions(*state)
    else:
        raise PreflightError(f"Unsupported model: {model}")
    _check_finite(finite_state, model, diagnostics)


def _trajectory_rows(writer: csv.writer, time_s: float, state: Any, model: str) -> int:
    if model == "overdamped":
        positions, orientations = state
        if positions.ndim != 2 or positions.shape[1] != 2 or orientations.shape != (len(positions),):
            raise ValueError("Overdamped trajectory state has inconsistent shapes")
        for particle, (position, theta) in enumerate(zip(positions, orientations)):
            writer.writerow((time_s, particle, position[0], position[1], theta, "", ""))
        return len(positions)
    else:
        if not isinstance(state, core.InertialState):
            raise TypeError("Inertial trajectory output requires an InertialState")
        if (
            state.positions.ndim != 2 or state.positions.shape[1] != 2
            or state.velocities.shape != state.positions.shape
            or state.orientations.shape != (len(state.positions),)
        ):
            raise ValueError("Inertial trajectory state has inconsistent shapes")
        for particle, (position, velocity, theta) in enumerate(
            zip(state.positions, state.velocities, state.orientations)
        ):
            writer.writerow((time_s, particle, position[0], position[1], theta, velocity[0], velocity[1]))
        return len(state.positions)


TRAJECTORY_COLUMNS = ("time_s", "particle", "x", "y", "theta", "vx", "vy")


def _validate_trajectory_file(path: Path, plan: RunPlan) -> tuple[int, int]:
    """Read back and validate the complete CSV before a run can complete."""
    rows_read = 0
    frames_seen = 0

    def fail(message: str) -> None:
        raise TrajectoryIntegrityError(message, frames_seen, rows_read)

    try:
        with path.open("r", newline="", encoding="utf-8") as trajectory_file:
            reader = csv.DictReader(trajectory_file)
            if tuple(reader.fieldnames or ()) != TRAJECTORY_COLUMNS:
                fail("Trajectory CSV header does not match the required seven-column schema")
            for row in reader:
                if rows_read >= plan.expected_particle_rows:
                    fail("Trajectory CSV contains more particle rows than expected")
                if None in row or any(value is None for value in row.values()):
                    fail(f"Trajectory row has an inconsistent number of columns at row {rows_read + 2}")
                frame_index, particle_index = divmod(rows_read, plan.config.n_particles)
                if particle_index == 0:
                    frames_seen += 1
                expected_time = frame_index * plan.config.output_interval
                try:
                    saved_time = float(row["time_s"])
                    saved_particle = int(row["particle"])
                    numeric = [float(row[name]) for name in ("x", "y", "theta")]
                    if plan.config.model == "inertial":
                        numeric.extend(float(row[name]) for name in ("vx", "vy"))
                except (TypeError, ValueError, KeyError):
                    fail(f"Malformed trajectory values at CSV row {rows_read + 2}")
                if not all(math.isfinite(value) for value in numeric):
                    fail(f"Non-finite trajectory value at CSV row {rows_read + 2}")
                if not math.isclose(saved_time, expected_time, rel_tol=0.0, abs_tol=1e-10):
                    fail(f"Unexpected saved time at CSV row {rows_read + 2}")
                if saved_particle != particle_index:
                    fail(f"Unexpected particle order at CSV row {rows_read + 2}")
                if plan.config.model == "overdamped" and (row["vx"] != "" or row["vy"] != ""):
                    fail("Overdamped rows must leave both velocity fields blank")
                if plan.config.model == "inertial" and (row["vx"] == "" or row["vy"] == ""):
                    fail("Inertial rows must contain both velocity components")
                rows_read += 1
    except TrajectoryIntegrityError:
        raise
    except OSError as exc:
        fail(f"Could not read back trajectory CSV: {exc}")

    if rows_read % plan.config.n_particles != 0:
        fail("Trajectory CSV ends with an incomplete particle frame")
    if frames_seen != plan.expected_frames or rows_read != plan.expected_particle_rows:
        fail(
            f"Trajectory CSV integrity count mismatch: {frames_seen} frames and {rows_read} rows "
            f"(expected {plan.expected_frames} and {plan.expected_particle_rows})"
        )
    return frames_seen, rows_read


def _record_step_diagnostics(diagnostics: dict[str, Any], step_diagnostics: core.StepDiagnostics) -> None:
    diagnostics["boundary_corrections"] += step_diagnostics.endpoint_corrections
    diagnostics["inertial_refinements"] += step_diagnostics.refinements
    diagnostics["maximum_refinement_depth"] = max(
        diagnostics["maximum_refinement_depth"], step_diagnostics.maximum_depth
    )


def _record_failure(
    diagnostics: dict[str, Any],
    exc: BaseException,
    *,
    inertial_step_in_progress: bool = False,
) -> str:
    error_text = f"{type(exc).__name__}: {exc}"
    diagnostics["errors"].append(error_text)
    diagnostics["execution_status"] = "incomplete"
    if diagnostics.get("incomplete_run_information") is None:
        diagnostics["incomplete_run_information"] = "Run failed; partial trajectory preserved."
    if inertial_step_in_progress:
        diagnostics["inertial_step_diagnostics_complete"] = False
        diagnostics["incomplete_run_information"] += (
            " The failing inertial macro-step raised before returning its internal correction/refinement counts."
        )
    if isinstance(exc, core.UnresolvedBoundaryCrossing):
        diagnostics["unresolved_boundary_crossing"] = True
        diagnostics["incomplete_run_information"] = (
            "Stopped at unresolved inertial boundary crossing; partial trajectory preserved."
        )
    if isinstance(exc, TrajectoryIntegrityError):
        diagnostics["actual_frames"] = exc.actual_frames
        diagnostics["actual_particle_rows"] = exc.actual_rows
    if isinstance(exc, Exception):
        diagnostics["traceback"] = traceback.format_exc()
    return error_text


def run_configuration(
    config: RunConfig,
    *,
    initial: core.InitialConditions | None = None,
    output_root: Path | str = RESULTS_ROOT,
) -> dict[str, Any]:
    """Execute one explicitly selected run and preserve partial files on failure."""
    plan = build_plan(config, output_root)
    params = _core_parameters(config)
    if initial is None:
        initial = core.initialize_particles(config.n_particles, config.initial_cluster_radius, config.seed)
    if initial.positions.shape != (config.n_particles, 2) or initial.orientations.shape != (config.n_particles,):
        raise PreflightError("Initial-condition shapes do not match the run configuration")
    initial_hash = initial_conditions_hash(initial)
    streams = core.make_rng_streams(config.seed)
    core_hash = _sha256_file(Path(core.__file__).resolve())

    # The destination is created only after every non-filesystem preflight check.
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    status = "running"
    error_text = None
    diagnostics = _new_diagnostics(plan, started_at)
    metadata = _manifest(plan, params, initial_hash, core_hash, started_at, status)
    metadata_path = plan.output_dir / "run_metadata.json"
    diagnostics_path = plan.output_dir / "diagnostics.json"

    trajectory_path = plan.output_dir / "trajectories.csv"
    inertial_step_in_progress = False
    try:
        _write_initial_records(metadata_path, diagnostics_path, metadata, diagnostics)
        if config.model == "overdamped":
            state: Any = (initial.positions.copy(), initial.orientations.copy())
        else:
            state = core.initialize_inertial_state(initial, params.v0)

        with trajectory_path.open("w", newline="", encoding="utf-8") as trajectory_file:
            writer = csv.writer(trajectory_file)
            writer.writerow(TRAJECTORY_COLUMNS)
            _validate_initial_state(state, config.model, diagnostics)
            diagnostics["actual_particle_rows"] += _trajectory_rows(writer, 0.0, state, config.model)
            diagnostics["actual_frames"] += 1

            for step in range(1, plan.steps + 1):
                if config.model == "overdamped":
                    positions, orientations, corrections = core.overdamped_step(
                        state[0], state[1], params,
                        streams.overdamped_translation, streams.rotation,
                    )
                    state = (positions, orientations)
                    diagnostics["boundary_corrections"] += corrections
                    finite_state = core.InitialConditions(*state)
                else:
                    inertial_step_in_progress = True
                    state, step_diagnostics = core.inertial_step(
                        state, params, streams.inertial_translation, streams.rotation, streams.bridge
                    )
                    _record_step_diagnostics(diagnostics, step_diagnostics)
                    inertial_step_in_progress = False
                    finite_state = state
                _check_finite(finite_state, config.model, diagnostics)
                diagnostics["completed_steps"] += 1

                if step % plan.save_stride == 0:
                    time_s = step * config.dt
                    diagnostics["actual_particle_rows"] += _trajectory_rows(
                        writer, time_s, state, config.model
                    )
                    diagnostics["actual_frames"] += 1
            trajectory_file.flush()

        diagnostics["actual_frames"], diagnostics["actual_particle_rows"] = (
            _validate_trajectory_file(trajectory_path, plan)
        )
        status = "completed"
    except core.UnresolvedBoundaryCrossing as exc:
        error_text = _record_failure(
            diagnostics, exc, inertial_step_in_progress=inertial_step_in_progress
        )
        status = "incomplete"
    except KeyboardInterrupt as exc:
        error_text = _record_failure(
            diagnostics, exc, inertial_step_in_progress=inertial_step_in_progress
        )
        status = "incomplete"
        raise
    except Exception as exc:  # Persist an auditable failure record and the partial CSV.
        error_text = _record_failure(
            diagnostics, exc, inertial_step_in_progress=inertial_step_in_progress
        )
        status = "incomplete"
    finally:
        ended_at = _utc_now()
        status, finalization_errors = _finalize_run_records(
            metadata_path,
            diagnostics_path,
            metadata,
            diagnostics,
            status=status,
            ended_at=ended_at,
            primary_error=error_text,
        )
        if finalization_errors:
            print(
                "Run finalization errors: " + "; ".join(finalization_errors),
                file=sys.stderr,
            )
    return diagnostics


def _run_selected(configs: tuple[RunConfig, ...]) -> list[dict[str, Any]]:
    """Preflight every requested destination before running any configuration."""
    plans = tuple(build_plan(config) for config in configs)
    if len({plan.output_dir for plan in plans}) != len(plans):
        raise PreflightError("Selected run output paths are not unique")
    shared_initials: dict[tuple[int, int], core.InitialConditions] = {}
    results = []
    for config in configs:
        key = (config.n_particles, config.seed)
        if key not in shared_initials:
            shared_initials[key] = core.initialize_particles(
                config.n_particles, config.initial_cluster_radius, config.seed
            )
        results.append(run_configuration(config, initial=shared_initials[key]))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run", choices=tuple(CONFIG_BY_ID), help="run exactly one configuration")
    selection.add_argument("--all", action="store_true", help="run all six configurations sequentially")
    selection.add_argument("--list", action="store_true", help="list configurations without running simulations")
    args = parser.parse_args(argv)
    if args.list:
        for config in CONFIGURATIONS:
            plan = build_plan(config, check_destination=False)
            print(f"{config.run_id}: {config.model}, N={config.n_particles}, T={config.duration:g} s, "
                  f"steps={plan.steps}, frames={plan.expected_frames}, output={plan.output_dir}")
        return 0
    selected = CONFIGURATIONS if args.all else (CONFIG_BY_ID[args.run],)
    try:
        results = _run_selected(selected)
    except PreflightError as exc:
        parser.error(str(exc))
    return 0 if all(item["execution_status"] == "completed" for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
