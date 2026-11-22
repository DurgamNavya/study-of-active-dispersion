# CP302 Active Brownian Particle (ABP) Simulations

This simulation workflow is a new implementation built on `cp302_numerical_core.py`. It is
separate from the original `abp_model.py` and `inertial_abp_model.py`; it does
not alter or overwrite either baseline or their result folders.

> **Warning:** Selecting a run with `--run` or `--all` launches a simulation.
> Importing `run_abp_simulations.py` and using `--list` do not launch simulations.

## Configurations

Every run uses seed 42, `DT=0.005 s`, and saves at `0.01 s` intervals.

| ID | Model | N | Duration | Dish radius | Initial cluster radius |
|---|---|---:|---:|---:|---:|
| `overdamped_N100_T50` | Overdamped ABP | 100 | 50 s | 40 | 5 |
| `inertial_N100_T50` | Translational-inertial ABP | 100 | 50 s | 40 | 5 |
| `overdamped_N400_T50` | Overdamped ABP | 400 | 50 s | 80 | 10 |
| `inertial_N400_T50` | Translational-inertial ABP | 400 | 50 s | 80 | 10 |
| `overdamped_N100_T200` | Overdamped ABP | 100 | 200 s | 40 | 5 |
| `inertial_N100_T200` | Translational-inertial ABP | 100 | 200 s | 40 | 5 |

Other physical parameters are `DT_TRANS=0.01`, `DR=0.1`, `MOBILITY=1`,
`MASS=0.1`, `V0=1`, `REPULSION_K=10`, and `PARTICLE_DIAMETER=1`. The core
derives `GAMMA=1/MOBILITY`, `TAU_M=MASS/GAMMA`, and
`KBT=GAMMA*DT_TRANS`.

## Equations and numerical core

The overdamped equations are

\[
 d\mathbf{x}_i = [v_0\mathbf{u}(\theta_i)+\mu\mathbf{F}_i]dt
                 + \sqrt{2D_T}\,d\mathbf{W}_i,\qquad
 d\theta_i = \sqrt{2D_R}\,dB_i.
\]

The pair force is the core's harmonic repulsion for overlapping disks:

\[
 \mathbf{F}_i = k\sum_{j\ne i}
  (\sigma-r_{ij})_+\frac{\mathbf{x}_i-\mathbf{x}_j}{r_{ij}}.
\]

The inertial model has **translational inertia only**; its orientation remains
overdamped:

\[
 d\mathbf{x}_i=\mathbf{v}_i dt,\qquad
 m\,d\mathbf{v}_i=[-\gamma\mathbf{v}_i+\gamma v_0\mathbf{u}(\theta_i)
                    +\mathbf{F}_i]dt+\sqrt{2\gamma k_BT}\,d\mathbf{W}_i,
 \qquad d\theta_i=\sqrt{2D_R}\,dB_i.
\]

The driver uses `Parameters`, `initialize_particles`,
`initialize_inertial_state`, `make_rng_streams`, `overdamped_step`, and
`inertial_step` from the existing numerical core. The overdamped core step is
Eulerâ€“Maruyama with projected circular-boundary positions. The inertial step
uses the exact joint frozen-force, fixed-orientation OU positionâ€“velocity
transition within a midpoint force predictorâ€“corrector. That predictorâ€“
corrector is an approximation to the complete interacting system; exactness of
the OU subproblem does not make the full interacting step exact.

Boundary handling is finite-resolution. Overdamped endpoints outside the disk
are projected to the disk. Inertial endpoints are projected and outward normal
velocity is reflected. Adaptive bridge refinement detects some cross-and-return
events and can raise an unresolved-boundary error at its hard limit. The method
does not simulate exact continuous first-passage times or guarantee localization
of every wall collision.

The core's conditional OU bridge uses a pseudoinverse with `rcond=1e-14`.
Near-singular covariance directions below that relative cutoff may be discarded;
this is a numerical limitation of the bridge conditioning calculation.

## Paired initialization and RNG design

For `--all`, one initial position/orientation realization is generated per
`(N, seed)` and shared by both models and durations with that same pair. A
separately selected run regenerates the same arrays deterministically. The
canonical little-endian float64 position and orientation arrays are SHA-256
hashed; the hash is recorded in each run manifest.

The core expands the seed through `SeedSequence(seed).spawn(6)` into distinct
streams for initialization positions, initialization orientations, overdamped
translation, inertial translation, rotation, and inertial bridge operations.
Both paired models therefore have matching initial conditions and reproducible
streams. Their translational noise paths are model-specific and are not
identical merely because their user-facing seed is the same. Rotation is drawn
from its own deterministic stream. No RNG behavior in the numerical core is
changed by the driver.

## Selecting runs

From the project root, list IDs and destinations without creating them:

```powershell
python Modelling/run_abp_simulations.py --list
```

Run one configuration explicitly:

```powershell
python Modelling/run_abp_simulations.py --run overdamped_N100_T50
```

Running all six requires the explicit `--all` selection:

```powershell
python Modelling/run_abp_simulations.py --all
```

The driver preflights every selected destination before running. Existing
destinations cause refusal. Runs are serial, and an incomplete run preserves
its partial trajectory and failure records. The driver does not resume, delete,
reuse, or overwrite results.

## Output paths and schema

Each run is written beneath
`Modelling/abp_simulation_results/seed_42/`, with separate `N100_T50`,
`N400_T50`, or `N100_T200` and `overdamped`/`inertial` directories. The exact
six paths are derived from the configuration IDs in the driver.

Each run folder contains:

- `run_metadata.json`: configuration, all physical parameters, numerical-core
  filename and SHA-256, initial-condition SHA-256, RNG description, expected
  step/frame/row counts, timestamps, and final execution status.
- `trajectories.csv`: one row per particle at each saved time, with columns
  `time_s, particle, x, y, theta, vx, vy`. Overdamped `vx` and `vy` are blank:
  the model has no instantaneous velocity state. Any later finite-difference
  velocity is a distinct analysis estimate and must be labeled as such.
- `diagnostics.json`: expected and actual frames/particle rows, boundary
  corrections, inertial refinement and maximum-depth counts (where relevant),
  unresolved crossings, non-finite state flags, errors, timestamps, and status.
  If an inertial macro-step raises before returning its `StepDiagnostics`, the
  file marks those diagnostics incomplete because the core does not expose that
  failed step's internal correction/refinement totals.

At `DT=0.005 s`, the output stride is two integration steps. The expected frame
counts include `t=0`: 5,001 frames for 50 s and 20,001 frames for 200 s. Rows
are streamed to CSV rather than keeping full histories in memory.

## Status and failure handling

Metadata and diagnostics are initially written with status `running`. If either
initial write fails, the driver stops before entering the simulation loop,
records zero completed steps, and attempts to persist an `incomplete` status.
A run is marked `completed` only after the CSV has been reread and its header,
particle order, times, numeric values, model-specific velocity fields, and total
frame/row counts have all passed validation. A detected non-finite state,
unresolved boundary crossing, other exception, or interruption marks the run
`incomplete` and preserves partial output. Finalization write errors are
recorded separately from the original run error and force incomplete status.
For overdamped runs, `non_finite_velocities_detected` is `null`, because velocity
is not a state variable in that model; its CSV velocity columns are blank.

The driver refuses an existing run destination and does not write to the
baseline result folders `abp_results` or `inertial_abp_results_v2`.


## Repository Data Availability

Simulation source code, analysis scripts, tests, selected numerical outputs, and visualizations are included in this repository.

The generated `trajectories.csv` files are excluded because of their large storage requirements. The original trajectory data is retained separately in the project working directory.

Two incomplete simulation attempts are also excluded from the repository:
- The unfinished `N1000_T5` benchmark.
- The interrupted `N100_T200` run.

The completed simulation outputs and their analysis products are documented in the corresponding result directories. Re-running simulations requires the appropriate Python dependencies and configuration described in this document.

