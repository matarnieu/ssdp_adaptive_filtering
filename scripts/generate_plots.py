import os
import subprocess
import re
from pathlib import Path
import shlex
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Setup base paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
COMMAND_FILE = SCRIPT_DIR / "generate_plots.txt"

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = PROJECT_ROOT / "results" / "generate_plots" / timestamp
RESULTS_DIR = RUN_DIR / "plots"
BAR_PLOTS_DIR = RUN_DIR / "bar_plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BAR_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MSE_LOG = RUN_DIR / "MSE.txt"
RUN_COMMAND_COPY = RUN_DIR / "run_command.txt"

# --- Copy original command file to run directory ---
with open(COMMAND_FILE, "r") as src, open(RUN_COMMAND_COPY, "w") as dst:
    dst.write(src.read())

# --- Parse commands.txt ---
default_args = {}
commands = []
method_order = []

mode = None
pending_filename = None

with open(COMMAND_FILE, "r") as f:
    for line in f:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("##"):
            if "DEFAULT" in stripped.upper():
                mode = "defaults"
            elif "RUN" in stripped.upper():
                mode = "commands"
            continue

        if mode == "defaults":
            parts = stripped.split(maxsplit=1)
            if len(parts) == 2:
                default_args[parts[0]] = parts[1]
            elif len(parts) == 1:
                default_args[parts[0]] = True
        elif mode == "commands":
            if stripped.startswith("#"):
                pending_filename = stripped[1:].strip()
            elif stripped.startswith("python"):
                parts = stripped.split("#", 1)
                command = parts[0].strip()
                inline_filename = parts[1].strip() if len(parts) > 1 else None
                filename = inline_filename or pending_filename
                commands.append((command, filename))
                pending_filename = None

                # --- Extract method name and track order ---
                if filename.startswith("gwf"):
                    method = "_".join(filename.split("_")[:2]).upper()
                else:
                    method = filename.split("_")[0].upper()
                if method not in method_order:
                    method_order.append(method)


# --- Inject defaults and plot filename ---
def inject_defaults(command: str, defaults: dict, plot_filename: str = None) -> str:
    tokens = shlex.split(command)
    existing_args = {tok.split("=")[0] for tok in tokens if tok.startswith("--")}

    for arg, val in defaults.items():
        if arg not in existing_args:
            tokens.append(arg)
            if val is not True:
                tokens.append(str(val))

    if plot_filename and "--plot_filename" not in existing_args:
        tokens.append("--plot_filename")
        tokens.append(str(RESULTS_DIR / plot_filename))

    if "--dont_show_plot" not in existing_args:
        tokens.append("--dont_show_plot")

    return " ".join(shlex.quote(tok) for tok in tokens)


# --- Run and collect MSEs ---
mse_results = []

for cmd, filename in commands:
    full_cmd = inject_defaults(cmd, default_args, plot_filename=filename)
    print(f"\n▶ Running: {full_cmd}")

    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed:\n{e.stderr}")
        continue

    output = result.stdout
    print(output)

    # Extract MSE
    mse_match = re.search(r"MSE:\s*([0-9.]+)", output)
    mse_value = float(mse_match.group(1)) if mse_match else -1

    expected_plot = RESULTS_DIR / filename if filename else None
    if expected_plot and expected_plot.exists():
        mse_results.append(f"{filename} {mse_value:.6f}")
    else:
        print(f"⚠️ Plot not found: {expected_plot}")

# --- Save MSE log ---
with open(MSE_LOG, "w") as f:
    f.write("\n".join(mse_results))

print(f"\n✅ All commands completed. Results saved to:\n{RUN_DIR}")

# ========== Create bar plots from MSE results ==========
sns.set_theme(style="darkgrid")

# Parse MSE results
entries = []
for line in mse_results:
    filename, mse = line.strip().split()

    # Full method name
    if filename.startswith("gwf"):
        method = "_".join(filename.split("_")[:2]).upper()
    else:
        method = filename.split("_")[0].upper()

    match = re.search(
        r"(stationary_fixed|stationary_smooth|stationary_abrupt|powerchange_fixed)",
        filename,
    )
    if match:
        scenario = match.group(1)
        entries.append((method, scenario, float(mse)))

df = pd.DataFrame(entries, columns=["Method", "Scenario", "MSE"])

pretty_names = {
    "stationary_fixed": "Stationary Fixed",
    "stationary_smooth": "Stationary Smooth",
    "stationary_abrupt": "Stationary Abrupt",
    "powerchange_fixed": "Power Change Fixed",
}

# --- Plot each scenario ---
for scenario in df["Scenario"].unique():
    subset = df[df["Scenario"] == scenario].copy()
    subset["Method"] = pd.Categorical(
        subset["Method"], categories=method_order, ordered=True
    )
    subset = subset.sort_values("Method")

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=subset, x="Method", y="MSE", palette="deep", ci=None, order=method_order
    )

    ax.set_xticklabels([label.replace("_", " ") for label in method_order], fontsize=12)

    """plt.title(
        f"MSE - {pretty_names.get(scenario, scenario)}", fontsize=16, weight="bold"
    )"""
    plt.xlabel("Method", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.4f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            xytext=(0, 4),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(BAR_PLOTS_DIR / f"{scenario}.png", dpi=300)
    plt.close()

print(f"\n📊 MSE bar plots saved in: {BAR_PLOTS_DIR}")
