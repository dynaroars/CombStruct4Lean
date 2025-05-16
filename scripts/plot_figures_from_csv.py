import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_figure(csv_path, x_col, hue_col, output_path):
    df = pd.read_csv(csv_path)
    # Set style and font sizes
    sns.set(style="whitegrid", font_scale=2.25)  # Increase overall font scale

    # Lines of Code figure
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x=x_col, y="Ratio", hue="Dataset", data=df)

    # Adjust layout with more padding to prevent text cutoff
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory


def main():
    os.makedirs("figures", exist_ok=True)
    plot_figure(
        csv_path="figures/loc.csv",
        x_col="Lines of code range",
        hue_col="Ratio",
        output_path="figures/loc_plot.pdf",
    )
    plot_figure(
        csv_path="figures/def.csv",
        x_col="Number of definitions",
        hue_col="Ratio",
        output_path="figures/def_plot.pdf",
    )


if __name__ == "__main__":
    main()
