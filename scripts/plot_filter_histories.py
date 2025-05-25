import subprocess
import shlex
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns

sns.set_theme()

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_FILE = SCRIPT_DIR / "plot_filter_histories.txt"

# --- Parse config file ---
with open(CONFIG_FILE, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

default_args = []
commands = []

current_label = None
for line in lines:
    if line.startswith("##"):
        continue
    elif line.startswith("--"):
        default_args.append(line)
    elif line.startswith("#"):
        current_label = line.lstrip("#").strip()
    elif line.startswith("python") and current_label:
        commands.append((current_label, line.strip()))
        current_label = None  # Reset for next

# --- Run each command and collect filter histories ---
histories = {}

for label, raw_command in commands:
    full_command = raw_command + " " + " ".join(default_args)

    if "--dont_show_plot" not in full_command:
        full_command += " --dont_show_plot"
    if "--print_filter_distances" not in full_command:
        full_command += " --print_filter_distances"

    print(f"\n▶ Running: {label}\n{full_command}")

    try:
        result = subprocess.run(
            shlex.split(full_command),
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,  # Important for relative paths in main.py
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed for {label}:\n{e.stderr}")
        continue

    stdout = result.stdout
    match = re.search(r"Filter distances:\s*(\[[^\]]+\])", stdout)
    if match:
        try:
            distances = np.array(json.loads(match.group(1)))
            histories[label] = distances  # ✅ store in the result dict
            print(f"✅ Collected filter history for {label} (length {len(distances)})")
        except Exception as e:
            print(f"⚠️ Failed to parse distances for {label}: {e}")
    else:
        print(f"⚠️ No filter distances found in output for {label}")

# --- Plot results ---
if histories:
    for label, history in histories.items():
        plt.plot(history, label=label)

    # plt.title("Filter Estimation Error (L2 Norm) over Time")
    plt.xlabel("Sample Index")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No histories were successfully parsed.")
