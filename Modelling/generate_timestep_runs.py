from pathlib import Path
import re

seeds = [43, 44, 45, 46]

templates = {
    "dt001": "inertial_abp_cluster10_nonoverlap.py",
    "dt005": "inertial_abp_cluster10_dt005.py"
}

for timestep, template in templates.items():

    source = Path(template).read_text(encoding="utf-8")

    for seed in seeds:

        script_name = (
            f"inertial_abp_cluster10_seed{seed}_{timestep}.py"
        )

        output_folder = (
            f"inertial_abp_cluster10_seed{seed}_{timestep}_results"
        )

        updated = re.sub(
            r"^RANDOM_SEED\s*=\s*42\s*$",
            f"RANDOM_SEED = {seed}",
            source,
            flags=re.MULTILINE
        )

        updated = re.sub(
            r'^OUTPUT_DIR\s*=\s*"[^"]+"$',
            f'OUTPUT_DIR = "{output_folder}"',
            updated,
            flags=re.MULTILINE
        )

        if updated == source:
            raise RuntimeError(
                f"Configuration was not changed for {script_name}"
            )

        Path(script_name).write_text(
            updated,
            encoding="utf-8"
        )

        print("Created:", script_name)
        print("Output:", output_folder)

print("\nAll eight scripts generated.")
