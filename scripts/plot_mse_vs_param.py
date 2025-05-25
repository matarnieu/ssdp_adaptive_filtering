import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns

"""
python plotting_scripts/plot_mse_vs_param.py results/optimize_params/gwf_swc_noise_power_change_fixed_window_size/results.csv window_size "\alpha"
"""
sns.set_theme()


def plot_mse_vs_param(filepath, param, param_clean):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Ensure correct column names
    if param not in df.columns or "MSE" not in df.columns:
        raise ValueError(f"CSV must contain '{param}' and 'MSE' columns.")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df[param], df["MSE"], marker="o")
    plt.xlabel(f"${param_clean}$")
    plt.ylabel("MSE")
    # plt.title(f"MSE vs. {param_clean}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot MSE vs. a parameter from a CSV file."
    )
    parser.add_argument("filepath", type=str, help="Path to the CSV file")
    parser.add_argument(
        "param", type=str, help="Name of the parameter column in the CSV"
    )
    parser.add_argument(
        "param_clean",
        type=str,
        help="Clean name to use for labeling the parameter axis",
    )

    args = parser.parse_args()
    plot_mse_vs_param(args.filepath, args.param, args.param_clean)


if __name__ == "__main__":
    main()
