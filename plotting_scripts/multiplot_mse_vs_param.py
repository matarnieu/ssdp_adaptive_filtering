import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns
import os

"""
Example:
python plotting_scripts/multiplot_mse_vs_param.py \
 results/optimize_params/gwf_ema_stationary_fixed/results.csv \
 results/optimize_params/gwf_ema_stationary_smooth/results.csv \
 lambda "\lambda" \
  --labels "Fixed Filter" "Smoothly Changing Filter"
"""

sns.set_theme()


def plot_mse_vs_param(filepaths, labels, param, param_clean):
    if len(filepaths) != len(labels):
        raise ValueError("Number of filepaths and labels must match.")

    plt.figure(figsize=(8, 5))

    for filepath, label in zip(filepaths, labels):
        df = pd.read_csv(filepath)

        if param not in df.columns or "MSE" not in df.columns:
            raise ValueError(f"CSV must contain '{param}' and 'MSE' columns.")

        plt.plot(df[param], df["MSE"], marker="o", label=label)

    plt.xlabel(f"${param_clean}$")
    plt.ylabel("MSE")
    # plt.title(f"MSE vs. {param_clean}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot MSE vs. a parameter from one or more CSV files."
    )
    parser.add_argument("filepaths", nargs="+", help="Path(s) to the CSV file(s)")
    parser.add_argument(
        "--labels", nargs="+", required=True, help="Clean labels for each file"
    )
    parser.add_argument(
        "param", type=str, help="Name of the parameter column in the CSV"
    )
    parser.add_argument(
        "param_clean",
        type=str,
        help="Clean name to use for labeling the parameter axis",
    )

    args = parser.parse_args()
    plot_mse_vs_param(args.filepaths, args.labels, args.param, args.param_clean)


if __name__ == "__main__":
    main()
