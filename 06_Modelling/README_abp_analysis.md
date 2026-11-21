# CP302 Active Brownian Particle (ABP) Analysis

This analysis pipeline reads the six completed seed-42 simulations. It is
separate from the simulation drivers and numerical core; it does not run or
alter a simulation. It validates the full input set before producing analysis
files and refuses to write if its output directory already exists.

## Run

From the project root:

```powershell
python 06_Modelling/analyze_abp_simulations.py
```

Optional `--input-root` and `--output-root` arguments select different paths.
The default input is `06_Modelling/abp_simulation_results/seed_42/`; the default
output is the new directory `06_Modelling/abp_analysis_results/seed_42/`.
If that output path already exists, the script exits without overwriting it.

## Inputs and validation

The script expects exactly these run directories:

```text
N100_T50/overdamped/
N100_T50/inertial/
N400_T50/overdamped/
N400_T50/inertial/
N100_T200/overdamped/
N100_T200/inertial/
```

Each directory must contain `run_metadata.json`, `diagnostics.json`, and
`trajectories.csv`. The trajectory schema is exactly:

```text
time_s,particle,x,y,theta,vx,vy
```

The loader checks completed status and matching metadata/diagnostic counts,
model/N/duration/seed, frame timing, row count, particle order, finite positions
and orientations, finite inertial velocities, blank overdamped velocity fields,
and matching initial-condition hashes and physical setup within each model pair.
It processes one trajectory at a time and retains only position/orientation
arrays plus the much smaller analysis curves.

`time_s` and `lag_s` are seconds. `vx,vy` in inertial source rows are model
state velocities, but this pipeline does not use them for the fair-comparison
VACF. Positions and MSD use the model's length units; no physical length
calibration is recorded in the historical simulation metadata. MSD is therefore reported in
model length units squared. Displacement velocity is model length units per
second.

## Estimators

### Ensemble MSD

For each saved time, the pipeline calculates

\[
\operatorname{MSD}_{ens}(t)=\frac{1}{N}\sum_i
\left|\mathbf r_i(t)-\mathbf r_i(0)\right|^2.
\]

This uses the initial frame as the single time origin. The CSV has columns
`time_s,MSD,particle_count`; the first MSD value is checked/set to zero.

### Time-averaged MSD

For lag \(k\) across \(M\) saved frames, it calculates

\[
\operatorname{TAMSD}(k)=\frac{1}{N(M-k)}\sum_i\sum_{n=0}^{M-k-1}
\left|\mathbf r_i(n+k)-\mathbf r_i(n)\right|^2.
\]

This is a separate estimator, not another name for ensemble MSD. FFT
autocorrelations reduce the cost from nested origin-by-lag loops. The output
CSV includes `lag_s,TAMSD,particle_count,time_origin_count`. Time-origin
averaging can reduce noise only when origins are suitable samples of the same
process. These runs begin from a cluster and are confined, so stationarity must
not be assumed without checking.

### OACF

For each lag, the script averages

\[
C_\theta(k)=\left\langle\cos(\theta_i(n+k)-\theta_i(n))\right\rangle_{i,n}
\]

over particles and all valid time origins. Both models use this same estimator;
it is normalized by its zero-lag value. The CSV columns are
`lag_s,OACF,OACF_normalized,particle_count,time_origin_count`.

### Displacement-based VACF

For **both** models, interval velocities are computed from saved positions:

\[
\mathbf v_i(n)=\frac{\mathbf r_i(n+1)-\mathbf r_i(n)}{\Delta t_{out}}.
\]

The VACF is the average of \(\mathbf v_i(n)\cdot\mathbf v_i(n+k)\) over
particles and valid time origins, normalized by its zero-lag value. Its CSV is
`<run_id>_displacement_vacf.csv`. It is distinct from an inertial VACF using
the stored state velocities. Overdamped `vx,vy` blanks are validated and never
interpreted as zero velocities. Since overdamped Brownian velocity is not an
ordinary instantaneous state variable, this interval-displacement estimator
is the common finite-resolution comparison.

## Outputs

The output directory contains:

```text
analysis_manifest.json
ensemble_msd/<run_id>_ensemble_msd.csv                 # six files
time_averaged_msd/<run_id>_time_averaged_msd.csv       # six files
local_msd_slopes/<run_id>_local_msd_slopes.csv         # six files
correlations/<run_id>_oacf.csv                         # six files
correlations/<run_id>_displacement_vacf.csv            # six files
plots/ensemble_msd_linear.png
plots/ensemble_msd_loglog.png
plots/msd_matched_<group>.png                          # three paired plots
plots/msd_N100_vs_N400_T50_<model>.png                 # two plots
plots/msd_T50_vs_T200_N100_<model>.png                 # two plots
plots/oacf_matched_<group>.png                         # three paired plots
plots/displacement_vacf_matched_<group>.png            # three paired plots
```

Log-log MSD plots omit zero time and nonpositive MSD values. OACF and
displacement-VACF plots use linear axes so negative correlation estimates,
when present, remain visible. The local-slope CSV uses finite differences to
report `dMSD/dt` and the log-log exponent
\(\beta=d\log(\mathrm{MSD})/d\log(t)\); it does not fit a window or calculate
a diffusion coefficient.

## Diffusion interpretation

Inspect the generated ensemble MSD curves and local slopes before choosing any
candidate interval. A roughly constant positive `dMSD/dt`, or a log-log
exponent \(\beta\) near 1 over a sustained interval, can indicate a candidate
linear-MSD regime. A candidate must also lie after relevant short-time
crossovers and before confinement or other finite-size effects flatten the
curve. In the inertial model, examine the crossover around the recorded
\(\tau_m=0.1\) s; active reorientation and interactions add longer timescales.
Reflecting circular walls ultimately bound displacement, so the full 50 s or
200 s interval must not be presumed diffusive. If a defensible 2D linear regime
is established, its fitted slope would correspond to
\(D_{eff}=\text{slope}/4\). No such fit or coefficient is produced here.

There is only one seed per configuration. Particles interact and are not
independent replicate runs; the six curves do not provide robust uncertainty
estimates. Multiple seeds are needed for defensible ensemble uncertainty.

## Tests

Run only the focused analysis tests with:

```powershell
python -B -m pytest -p no:cacheprovider -q 06_Modelling/test_analyze_abp_simulations.py
```

The tests use synthetic trajectories and do not access or change the simulation outputs
simulation outputs.
