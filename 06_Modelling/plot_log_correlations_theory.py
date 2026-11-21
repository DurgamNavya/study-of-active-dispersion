
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "log_scale_plots"
OUTPUT.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. OACF: log-log plot
# --------------------------------------------------

oacf = pd.read_csv(ROOT / "abp_results" / "oacf_results.csv")
oacf = oacf[
    (oacf["time_lag"] > 0) &
    (oacf["OACF"] > 0)
]

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(oacf["time_lag"], oacf["OACF"],
          linewidth=2, label="OACF")

ax.set_xlabel("Time lag (log scale)")
ax.set_ylabel("Orientation autocorrelation (log scale)")
ax.set_title("Orientation Autocorrelation - Log-Log")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "OACF_loglog.png", dpi=300)
plt.close(fig)


# --------------------------------------------------
# 2. OACF validation: simulated vs theoretical
# --------------------------------------------------

validation = pd.read_csv(
    ROOT / "abp_results" / "oacf_validation.csv"
)

validation = validation[
    (validation["time_lag"] > 0) &
    (validation["simulated_OACF"] > 0) &
    (validation["theoretical_OACF"] > 0)
]

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(
    validation["time_lag"],
    validation["simulated_OACF"],
    linewidth=2,
    label="Simulated OACF"
)
ax.loglog(
    validation["time_lag"],
    validation["theoretical_OACF"],
    linestyle="--",
    linewidth=2,
    label="Theoretical OACF"
)

ax.set_xlabel("Time lag (log scale)")
ax.set_ylabel("OACF (log scale)")
ax.set_title("OACF Validation - Log-Log")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "OACF_validation_loglog.png", dpi=300)
plt.close(fig)


# --------------------------------------------------
# 3. VACF: log time axis, symmetric-log y axis
# Negative VACF values cannot be shown on a normal
# logarithmic y-axis.
# --------------------------------------------------

vacf = pd.read_csv(
    ROOT / "abp_results" / "vacf_results.csv"
)

vacf = vacf[vacf["time_lag"] > 0]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    vacf["time_lag"],
    vacf["VACF"],
    linewidth=1.8,
    label="VACF"
)

ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=0.01)

ax.axhline(0, linewidth=1)
ax.set_xlabel("Time lag (log scale)")
ax.set_ylabel("Velocity autocorrelation (symmetric-log scale)")
ax.set_title("Velocity Autocorrelation - Log Time / Symlog")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "VACF_logx_symlogy.png", dpi=300)
plt.close(fig)


# --------------------------------------------------
# 4. Theoretical vs simulated MSD
# Same parameters as compare_abp_theory.py
# --------------------------------------------------

V0 = 1.0
DR = 0.1

simulations = {
    "Low (DT=0.005)": {
        "folder": "abp_results_dt_low",
        "DT": 0.005
    },
    "Baseline (DT=0.01)": {
        "folder": "abp_results_dt_baseline",
        "DT": 0.01
    },
    "High (DT=0.02)": {
        "folder": "abp_results_dt_high",
        "DT": 0.02
    }
}

fig, ax = plt.subplots(figsize=(9, 6))

for label, params in simulations.items():

    data = pd.read_csv(
        ROOT / params["folder"] / "msd.csv"
    )

    time = data["time"].to_numpy()
    msd = data["MSD"].to_numpy()

    theoretical_msd = (
        4 * params["DT"] * time
        + (2 * V0**2 / DR**2)
        * (DR * time - 1 + np.exp(-DR * time))
    )

    valid = (
        (time > 0) &
        (msd > 0) &
        (theoretical_msd > 0)
    )

    ax.loglog(
        time[valid],
        msd[valid],
        linewidth=2,
        label=f"Simulated {label}"
    )

    ax.loglog(
        time[valid],
        theoretical_msd[valid],
        linestyle="--",
        linewidth=1.8,
        label=f"Theory {label}"
    )

ax.set_xlabel("Time (log scale)")
ax.set_ylabel("Mean Squared Displacement (log scale)")
ax.set_title("Simulated vs Theoretical ABP MSD - Log-Log")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(
    OUTPUT / "ABP_theory_vs_simulation_loglog.png",
    dpi=300
)
plt.close(fig)


print("All remaining logarithmic plots generated.")
print("1. OACF log-log")
print("2. OACF validation log-log")
print("3. VACF log-time / symmetric-log correlation")
print("4. ABP theory vs simulation log-log")
print("Output folder:", OUTPUT)
