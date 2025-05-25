import subprocess
import shlex
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / "plot_filter_sizes.txt"

# --- Parse config file ---
with open(CONFIG_FILE, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

default_args = []
filter_sizes = []
commands = []

current_label = None
for line in lines:
    if line.startswith("##"):
        continue
    elif line.startswith("K="):
        try:
            filter_sizes = eval(line.split("=", 1)[1].strip())
        except Exception as e:
            raise ValueError(f"Invalid K line: {line}")
    elif line.startswith("--"):
        default_args.append(line)
    elif line.startswith("#"):
        current_label = line.lstrip("#").strip()
    elif line.startswith("python") and current_label:
        commands.append((current_label, line.strip()))
        current_label = None  # Reset

# --- Run each command for each K and collect MSEs ---
mse_results = {label: [] for label, _ in commands}

for label, raw_command in commands:
    print(f"\n▶ Running {label}...")

    for K in filter_sizes:
        full_command = raw_command + " " + " ".join(default_args)
        full_command += f" --filter_size={K} --dont_show_plot"

        print(f"  → K={K}: {full_command}")

        try:
            result = subprocess.run(
                shlex.split(full_command),
                capture_output=True,
                text=True,
                check=True,
                cwd=PROJECT_ROOT,
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed at K={K}:\n{e.stderr}")
            mse_results[label].append(None)
            continue

        output = result.stdout + result.stderr
        match = re.search(r"MSE:\s*([0-9.]+)", output)
        mse = float(match.group(1)) if match else None
        print(f"     MSE: {mse}")
        mse_results[label].append(mse)

# --- Plot ---
if mse_results:
    for label, mses in mse_results.items():
        if all(m is None for m in mses):
            continue
        plt.plot(filter_sizes, mses, marker="o", label=label)

    plt.xlabel("Filter size (K)")
    plt.ylabel("MSE")
    plt.xticks(filter_sizes)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No MSEs were successfully parsed.")
