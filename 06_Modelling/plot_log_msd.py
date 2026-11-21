
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "log_scale_plots"
OUTPUT.mkdir(exist_ok=True)

# All available simulation cases
cases = {
    "Main baseline": "abp_results",
    "DR baseline": "abp_results_dr_baseline",
    "DR low": "abp_results_dr_low",
    "DR high": "abp_results_dr_high",
    "DT baseline": "abp_results_dt_baseline",
    "DT low": "abp_results_dt_low",
    "DT high": "abp_results_dt_high",
    "DT no repulsion": "abp_results_dt_no_repulsion",
    "Speed baseline": "abp_results_speed_baseline",
    "Speed low": "abp_results_speed_low",
    "Speed high": "abp_results_speed_high",
}

def load_msd(folder):
    file = ROOT / folder / "msd.csv"

    if not file.exists():
        raise FileNotFoundError(f"Missing file: {file}")

    df = pd.read_csv(file)

    # Keep only positive time and MSD for log-log plotting
    df = df[(df["time"] > 0) & (df["MSD"] > 0)]
    return df.sort_values("time")


def save_individual_plots():
    for label, folder in cases.items():
        df = load_msd(folder)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(df["time"], df["MSD"], linewidth=2)

        ax.set_xlabel("Time (log scale)")
        ax.set_ylabel("Mean Squared Displacement (log scale)")
        ax.set_title(f"MSD - {label}")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        safe_name = folder.replace("abp_results", "main", 1)
        fig.tight_layout()
        fig.savefig(
            OUTPUT / f"{safe_name}_MSD_loglog.png",
            dpi=300
        )
        plt.close(fig)


def save_comparison(title, selected_cases, filename):
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, folder in selected_cases:
        df = load_msd(folder)
        ax.loglog(
            df["time"],
            df["MSD"],
            linewidth=2,
            label=label
        )

    ax.set_xlabel("Time (log scale)")
    ax.set_ylabel("Mean Squared Displacement (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=300)
    plt.close(fig)


# 1. Individual MSD plots for all cases
save_individual_plots()

# 2. Rotational diffusion comparison
save_comparison(
    "Rotational Diffusion Sensitivity - Log-Log MSD",
    [
        ("Low DR", "abp_results_dr_low"),
        ("Baseline DR", "abp_results_dr_baseline"),
        ("High DR", "abp_results_dr_high"),
    ],
    "rotational_diffusion_MSD_loglog.png"
)

# 3. Translational diffusion comparison
save_comparison(
    "Translational Diffusion Sensitivity - Log-Log MSD",
    [
        ("Low DT", "abp_results_dt_low"),
        ("Baseline DT", "abp_results_dt_baseline"),
        ("High DT", "abp_results_dt_high"),
    ],
    "translational_diffusion_MSD_loglog.png"
)

# 4. Speed sensitivity comparison
save_comparison(
    "Particle Speed Sensitivity - Log-Log MSD",
    [
        ("Low speed", "abp_results_speed_low"),
        ("Baseline speed", "abp_results_speed_baseline"),
        ("High speed", "abp_results_speed_high"),
    ],
    "speed_sensitivity_MSD_loglog.png"
)

# 5. Repulsion versus no repulsion
save_comparison(
    "Effect of Repulsion - Log-Log MSD",
    [
        ("With repulsion", "abp_results"),
        ("Without repulsion", "abp_results_dt_no_repulsion"),
    ],
    "repulsion_comparison_MSD_loglog.png"
)

print("Log-log MSD plots generated successfully.")
print("Number of individual plots:", len(cases))
print("Number of comparison plots: 4")
print("Output folder:", OUTPUT)
