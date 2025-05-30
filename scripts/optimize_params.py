import subprocess
import shlex
import re
import csv
import ast
from pathlib import Path
from itertools import product

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / "optimize_params.txt"

# --- Parse config file ---
with open(CONFIG_FILE, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

base_command = None
param_grid = {}
experiment_name = None
mode = None

for line in lines:
    if line.startswith("##"):
        if "NAME" in line.upper():
            mode = "name"
        elif "RUN" in line.upper():
            mode = "run"
        elif "PARAMETERS" in line.upper():
            mode = "params"
        continue

    if mode == "name":
        experiment_name = line.strip()
    elif mode == "run":
        base_command = line.strip()
    elif mode == "params":
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            # Skip deprecated parameters
            if key in ("--power_noise_change", "--power_distribution_change"):
                print(f"⚠️ Skipping deprecated parameter: {key}")
                continue
            try:
                values = ast.literal_eval(val)
                if isinstance(values, list):
                    param_grid[key] = values
            except Exception as e:
                print(f"⚠️ Skipping invalid parameter list: {line}")

# --- Validate ---
if not experiment_name:
    raise ValueError("Missing experiment name under ## NAME ##")
if not base_command:
    raise ValueError("Missing base command under ## RUN COMMAND ##")
if not param_grid:
    raise ValueError("No parameters found under ## PARAMETERS ##")

# --- Prepare output folder ---
RUN_DIR = PROJECT_ROOT / "results" / "optimize_params" / experiment_name
RUN_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RUN_DIR / "results.csv"
CONFIG_COPY = RUN_DIR / "optimize_params.txt"

# --- Copy config for reproducibility ---
with open(CONFIG_FILE, "r") as src, open(CONFIG_COPY, "w") as dst:
    dst.write(src.read())

# --- Build parameter combinations ---
original_param_keys = list(param_grid.keys())
clean_param_keys = [k.lstrip("-") for k in original_param_keys]
combinations = list(product(*(param_grid[k] for k in original_param_keys)))

# --- Run experiments ---
results = []

for combo in combinations:
    param_args = [f"{k}={v}" for k, v in zip(original_param_keys, combo)]
    full_command = base_command + " " + " ".join(param_args)

    # Add required flags if not already present
    if "--dont_show_plot" not in full_command:
        full_command += " --dont_show_plot"
    if "--noise_type" not in full_command:
        full_command += " --noise_type=wgn"  # Default fallback

    print(f"\n▶ Running: {full_command}")
    try:
        result = subprocess.run(
            shlex.split(full_command), capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed:\n{e.stderr}")
        continue

    output = result.stdout
    print(output)

    mse_match = re.search(r"MSE:\s*([0-9.]+)", output)
    mse = float(mse_match.group(1)) if mse_match else None

    result_row = {k: v for k, v in zip(clean_param_keys, combo)}
    result_row["MSE"] = mse
    results.append(result_row)

# --- Save CSV ---
if results:
    with open(RESULTS_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=clean_param_keys + ["MSE"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Results saved to {RESULTS_CSV}")

    # --- Print best result ---
    best_result = min(
        results, key=lambda x: x["MSE"] if x["MSE"] is not None else float("inf")
    )
    print("\n🏆 Best parameter configuration:")
    for k in clean_param_keys:
        print(f"  {k} = {best_result[k]}")
    print(f"  MSE = {best_result['MSE']:.6f}")
else:
    print("⚠️ No results to write.")
