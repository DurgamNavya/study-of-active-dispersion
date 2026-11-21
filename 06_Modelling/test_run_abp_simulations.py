"""Non-simulation tests for the CP302 pilot driver."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import run_abp_simulations as pilot
import cp302_numerical_core as core


def test_exactly_six_configurations_and_approved_values(tmp_path: Path) -> None:
    assert len(pilot.CONFIGURATIONS) == 6
    assert len(pilot.CONFIG_BY_ID) == 6
    plans = pilot.validate_all(tmp_path / "fresh_results")
    assert [plan.steps for plan in plans] == [10000, 10000, 10000, 10000, 40000, 40000]
    assert [plan.expected_frames for plan in plans] == [5001, 5001, 5001, 5001, 20001, 20001]

    expected = {
        ("overdamped", 100, 50.0, 40.0, 5.0),
        ("inertial", 100, 50.0, 40.0, 5.0),
        ("overdamped", 400, 50.0, 80.0, 10.0),
        ("inertial", 400, 50.0, 80.0, 10.0),
        ("overdamped", 100, 200.0, 40.0, 5.0),
        ("inertial", 100, 200.0, 40.0, 5.0),
    }
    actual = {
        (c.model, c.n_particles, c.duration, c.dish_radius, c.initial_cluster_radius)
        for c in pilot.CONFIGURATIONS
    }
    assert actual == expected
    assert all(c.seed == 42 and c.dt == 0.005 and c.output_interval == 0.01
               for c in pilot.CONFIGURATIONS)

    for config in pilot.CONFIGURATIONS:
        params = pilot._core_parameters(config)
        assert params.dt_trans == 0.01
        assert params.dr == 0.1
        assert params.mobility == 1.0
        assert params.mass == 0.1
        assert params.v0 == 1.0
        assert params.repulsion_k == 10.0
        assert params.particle_diameter == 1.0


def test_unique_output_paths(tmp_path: Path) -> None:
    plans = pilot.validate_all(tmp_path / "new")
    paths = [plan.output_dir for plan in plans]
    assert len(paths) == 6
    assert len(set(paths)) == 6
    assert all(not path.exists() for path in paths)


def test_existing_destination_is_refused(tmp_path: Path) -> None:
    config = pilot.CONFIGURATIONS[0]
    destination = tmp_path / "results" / config.relative_output_path()
    destination.mkdir(parents=True)
    with pytest.raises(pilot.PreflightError, match="Refusing to reuse"):
        pilot.build_plan(config, tmp_path / "results")


def test_manifest_diagnostics_fields_and_status_handling(tmp_path: Path) -> None:
    config = pilot.CONFIGURATIONS[0]
    plan = pilot.build_plan(config, tmp_path / "fresh")
    params = pilot._core_parameters(config)
    initial = core.initialize_particles(config.n_particles, config.initial_cluster_radius, config.seed)
    initial_hash = pilot.initial_conditions_hash(initial)
    started = "2026-01-01T00:00:00+00:00"
    completed = pilot._manifest(plan, params, initial_hash, "abc123", started, "completed", "2026-01-01T00:01:00+00:00")
    assert completed["execution_status"] == "completed"
    assert completed["initial_conditions_sha256"] == initial_hash
    assert completed["numerical_core_sha256"] == "abc123"
    assert completed["expected_frames"] == 5001
    assert completed["expected_particle_rows"] == 500100
    assert completed["physical_parameters"]["kbt"] == pytest.approx(0.01)
    assert "model-specific" in completed["rng_design"]

    diagnostics = pilot._new_diagnostics(plan, started)
    assert diagnostics["execution_status"] == "running"
    assert diagnostics["actual_frames"] == 0
    assert diagnostics["actual_particle_rows"] == 0
    assert diagnostics["non_finite_positions_detected"] is False
    assert diagnostics["non_finite_orientations_detected"] is False
    assert diagnostics["non_finite_velocities_detected"] is None
    assert diagnostics["inertial_refinements"] is None
    assert diagnostics["inertial_step_diagnostics_complete"] is None

    failed = pilot._manifest(plan, params, initial_hash, "abc123", started, "incomplete", "later", "test error")
    assert failed["execution_status"] == "incomplete"
    assert failed["error"] == "test error"
    assert failed["completed_steps"] == 0


def test_initial_conditions_and_hash_are_reproducible() -> None:
    first = core.initialize_particles(100, 5.0, 42)
    second = core.initialize_particles(100, 5.0, 42)
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.orientations, second.orientations)
    assert pilot.initial_conditions_hash(first) == pilot.initial_conditions_hash(second)


def test_import_does_not_execute_or_create_results_directory(tmp_path: Path) -> None:
    import shutil

    module_dir = Path(pilot.__file__).resolve().parent
    isolated_dir = tmp_path / "isolated_modelling"
    isolated_dir.mkdir()

    shutil.copy2(module_dir / "run_abp_simulations.py", isolated_dir)
    shutil.copy2(module_dir / "cp302_numerical_core.py", isolated_dir)

    code = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(isolated_dir)!r}); "
        "import run_abp_simulations; "
        "assert not run_abp_simulations.RESULTS_ROOT.exists(); "
        "assert len(run_abp_simulations.CONFIGURATIONS) == 6"
    )

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (isolated_dir / "abp_simulation_results").exists()
def test_invalid_timing_is_rejected(tmp_path: Path) -> None:
    base = pilot.CONFIGURATIONS[0]
    bad = pilot.RunConfig(
        "bad_timing", base.model, base.n_particles, base.duration,
        base.dish_radius, base.initial_cluster_radius, base.seed,
        dt=0.006, output_interval=0.01,
    )
    with pytest.raises(pilot.PreflightError):
        pilot.build_plan(bad, tmp_path / "fresh")


@pytest.mark.parametrize("model", ["overdamped", "inertial"])
def test_trajectory_rows_match_seven_column_header(model: str) -> None:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    if model == "overdamped":
        state = (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.2, 0.4]))
    else:
        state = core.InertialState(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.5, 0.6], [0.7, 0.8]]),
            np.array([0.2, 0.4]),
        )
    written = pilot._trajectory_rows(writer, 0.0, state, model)
    rows = list(csv.reader(io.StringIO(output.getvalue(), newline="")))
    assert written == 2
    assert len(rows) == 2
    assert all(len(row) == len(pilot.TRAJECTORY_COLUMNS) == 7 for row in rows)
    if model == "overdamped":
        assert all(row[5:] == ["", ""] for row in rows)
    else:
        assert all(row[5:] != ["", ""] for row in rows)


def test_initial_inertial_validation_uses_returned_state_instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = pilot.CONFIG_BY_ID["inertial_N100_T50"]
    plan = pilot.build_plan(config, tmp_path / "fresh")
    initial = core.initialize_particles(config.n_particles, config.initial_cluster_radius, config.seed)
    state = core.initialize_inertial_state(initial, 1.0)
    assert isinstance(state, core.InertialState)
    seen: list[object] = []
    monkeypatch.setattr(pilot, "_check_finite", lambda checked, model, diagnostics: seen.append(checked))

    pilot._validate_initial_state(state, "inertial", pilot._new_diagnostics(plan, "start"))
    assert len(seen) == 1
    assert seen[0] is state

    with pytest.raises(TypeError, match="must return an InertialState"):
        pilot._validate_initial_state(tuple(state), "inertial", pilot._new_diagnostics(plan, "start"))


def test_trajectory_readback_integrity_for_both_models(tmp_path: Path) -> None:
    for model in ("overdamped", "inertial"):
        config = pilot.RunConfig(f"tiny_{model}", model, 2, 0.01, 10.0, 1.0)
        plan = pilot.build_plan(config, tmp_path / model)
        path = tmp_path / f"{model}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(pilot.TRAJECTORY_COLUMNS)
            if model == "overdamped":
                state = (np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.1, 0.2]))
            else:
                state = core.InertialState(
                    np.array([[0.0, 0.0], [1.0, 1.0]]),
                    np.array([[1.0, 0.0], [0.0, 1.0]]),
                    np.array([0.1, 0.2]),
                )
            pilot._trajectory_rows(writer, 0.0, state, model)
            pilot._trajectory_rows(writer, 0.01, state, model)
        assert pilot._validate_trajectory_file(path, plan) == (2, 4)

        # A truncated artifact must fail integrity validation and report actual counts.
        with path.open("r", newline="", encoding="utf-8") as stream:
            saved_lines = stream.readlines()
        path.write_text("".join(saved_lines[:-1]), encoding="utf-8")
        with pytest.raises(pilot.TrajectoryIntegrityError) as caught:
            pilot._validate_trajectory_file(path, plan)
        assert caught.value.actual_rows == 3
        assert caught.value.actual_frames == 2


def test_boundary_and_refinement_diagnostics_accumulate() -> None:
    diagnostics = {
        "boundary_corrections": 2,
        "inertial_refinements": 3,
        "maximum_refinement_depth": 1,
    }
    pilot._record_step_diagnostics(diagnostics, core.StepDiagnostics(4, 5, 3))
    assert diagnostics == {
        "boundary_corrections": 6,
        "inertial_refinements": 8,
        "maximum_refinement_depth": 3,
    }


def test_failures_are_marked_incomplete_and_unresolved_is_flagged() -> None:
    diagnostics = pilot._new_diagnostics(pilot.build_plan(pilot.CONFIGURATIONS[1], Path("unused")), "start")
    error = pilot._record_failure(
        diagnostics, core.UnresolvedBoundaryCrossing("crossing"), inertial_step_in_progress=True
    )
    assert error == "UnresolvedBoundaryCrossing: crossing"
    assert diagnostics["execution_status"] == "incomplete"
    assert diagnostics["unresolved_boundary_crossing"] is True
    assert diagnostics["inertial_step_diagnostics_complete"] is False

    interrupted = pilot._new_diagnostics(pilot.build_plan(pilot.CONFIGURATIONS[0], Path("unused")), "start")
    pilot._record_failure(interrupted, KeyboardInterrupt())
    assert interrupted["execution_status"] == "incomplete"
    assert interrupted["errors"] == ["KeyboardInterrupt: "]


@pytest.mark.parametrize("failed_record", ["run_metadata.json", "diagnostics.json"])
def test_initial_record_write_failure_is_finalized_incomplete(
    failed_record: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = pilot.CONFIGURATIONS[0]
    plan = pilot.build_plan(config, tmp_path / "root")
    params = pilot._core_parameters(config)
    initial = core.initialize_particles(config.n_particles, config.initial_cluster_radius, config.seed)
    started = "2026-01-01T00:00:00+00:00"
    metadata = pilot._manifest(
        plan, params, pilot.initial_conditions_hash(initial), "corehash", started, "running"
    )
    diagnostics = pilot._new_diagnostics(plan, started)
    diagnostics_path = tmp_path / "diagnostics.json"
    metadata_path = tmp_path / "run_metadata.json"
    real_write = pilot._write_json
    failed_once = False

    def fail_selected_write(path: Path, record: dict[str, object]) -> None:
        nonlocal failed_once
        if path.name == failed_record and not failed_once:
            failed_once = True
            raise OSError("injected initial write failure")
        real_write(path, record)

    monkeypatch.setattr(pilot, "_write_json", fail_selected_write)
    with pytest.raises(pilot.RunRecordWriteError) as caught:
        pilot._write_initial_records(metadata_path, diagnostics_path, metadata, diagnostics)
    pilot._record_failure(diagnostics, caught.value)
    assert diagnostics["completed_steps"] == 0

    status, errors = pilot._finalize_run_records(
        metadata_path,
        diagnostics_path,
        metadata,
        diagnostics,
        status="incomplete",
        ended_at="2026-01-01T00:00:01+00:00",
        primary_error=str(caught.value),
    )
    assert status == "incomplete"
    assert errors == []
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert saved_metadata["execution_status"] == "incomplete"
    assert saved_diagnostics["execution_status"] == "incomplete"
    assert saved_metadata["completed_steps"] == 0
    assert saved_diagnostics["completed_steps"] == 0
    assert "zero simulation steps" in saved_diagnostics["incomplete_run_information"]


def test_finalization_failure_does_not_replace_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = pilot.CONFIGURATIONS[0]
    plan = pilot.build_plan(config, tmp_path / "root")
    params = pilot._core_parameters(config)
    initial = core.initialize_particles(config.n_particles, config.initial_cluster_radius, config.seed)
    started = "2026-01-01T00:00:00+00:00"
    metadata = pilot._manifest(
        plan, params, pilot.initial_conditions_hash(initial), "corehash", started, "incomplete",
        error="RuntimeError: original simulation failure",
    )
    diagnostics = pilot._new_diagnostics(plan, started)
    pilot._record_failure(diagnostics, RuntimeError("original simulation failure"))
    metadata_path = tmp_path / "run_metadata.json"
    diagnostics_path = tmp_path / "diagnostics.json"
    real_write = pilot._write_json
    failed_once = False

    def fail_metadata_once(path: Path, record: dict[str, object]) -> None:
        nonlocal failed_once
        if path == metadata_path and not failed_once:
            failed_once = True
            raise OSError("injected finalization failure")
        real_write(path, record)

    monkeypatch.setattr(pilot, "_write_json", fail_metadata_once)
    status, errors = pilot._finalize_run_records(
        metadata_path,
        diagnostics_path,
        metadata,
        diagnostics,
        status="incomplete",
        ended_at="2026-01-01T00:00:01+00:00",
        primary_error="RuntimeError: original simulation failure",
    )
    assert status == "incomplete"
    assert len(errors) == 1
    assert "injected finalization failure" in errors[0]
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert saved_metadata["error"] == "RuntimeError: original simulation failure"
    assert saved_metadata["finalization_errors"] == errors
    assert saved_diagnostics["execution_status"] == "incomplete"
