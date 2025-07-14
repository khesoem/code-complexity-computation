import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def main():
    # 1 – Load the two data sets
    eeg_df = pd.read_csv("eeg_CzMetric.csv")                  # expects columns: Participant, SnippetID, …, czmetric
    metrics_df = pd.read_csv("snippet_metrics.csv")  # expects columns: Participant, SnippetID, SurveyCL, …

    # 2 – (Optional) drop Snippet 13 if you want to keep parity with the old analysis
    #eeg_df = eeg_df[eeg_df["SnippetID"] != 13]
    #metrics_df = metrics_df[metrics_df["SnippetID"] != 13]

    # 3 – Average per snippet
    eeg_snip = eeg_df.groupby("SnippetID", as_index=False)["czmetric"].mean()
    met_snip = metrics_df.groupby("SnippetID", as_index=False)["SurveyCL"].mean()

    # 4 – Merge the two summaries
    merged = pd.merge(eeg_snip, met_snip, on="SnippetID").dropna(subset=["czmetric", "SurveyCL"])

    # 5 – Correlation statistics
    r, p = pearsonr(merged["czmetric"], merged["SurveyCL"])
    print(f"Correlation (r) : {r:.4f}")
    print(f"P-value        : {p:.4e}")

    # 6 – Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(merged["czmetric"], merged["SurveyCL"], label="Snippet means")

    # Add a simple least-squares regression line
    m, b = np.polyfit(merged["czmetric"], merged["SurveyCL"], 1)
    x_vals = np.linspace(merged["czmetric"].min(), merged["czmetric"].max(), 100)
    plt.plot(x_vals, m * x_vals + b, linewidth=2, label=f"y = {m:.2f}x + {b:.2f}")

    plt.title("Average Manual CL vs. EEG")
    plt.xlabel("Average EEG czmetric")
    plt.ylabel("Average manual CL")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
