import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def main():
    # 1 – Load the two data sets  ──▶  NOTE: both now have “CL” as the value column
    eeg_df     = pd.read_csv("eeg.csv")             # columns: Participant, SnippetID, …, CL
    metrics_df = pd.read_csv("survey_cl.csv") # columns: Participant, SnippetID, …, CL

    # 2 – (Optional) drop Snippet 13 if you kept it in the old analysis
    # eeg_df     = eeg_df[eeg_df["SnippetID"] != 13]
    # metrics_df = metrics_df[metrics_df["SnippetID"] != 13]

    # 3 – Average CL per snippet, then give each version a unique name so they don’t collide later
    eeg_snip = (
        eeg_df
        .groupby("SnippetID", as_index=False)["CL"]
        .mean()
        .rename(columns={"CL": "CL_EEG"})          # ★ renamed
    )
    met_snip = (
        metrics_df
        .groupby("SnippetID", as_index=False)["CL"]
        .mean()
        .rename(columns={"CL": "CL_MAN"})          # ★ renamed
    )

    # 4 – Merge the two summaries (now no missing values to drop)
    merged = pd.merge(eeg_snip, met_snip, on="SnippetID")

    # 5 – Correlation statistics
    r, p = pearsonr(merged["CL_EEG"], merged["CL_MAN"])
    print(f"Correlation (r) : {r:.4f}")
    print(f"P-value        : {p:.4e}")

    # 6 – Plot
    #plt.figure(figsize=(8, 6))
    #plt.scatter(merged["CL_EEG"], merged["CL_MAN"], label="Snippet means")

    # Least‑squares regression line
    #m, b = np.polyfit(merged["CL_EEG"], merged["CL_MAN"], 1)
    #x_vals = np.linspace(merged["CL_EEG"].min(), merged["CL_EEG"].max(), 100)
    #plt.plot(x_vals, m * x_vals + b, linewidth=2, label=f"y = {m:.2f}x + {b:.2f}")

    #plt.title("Average Manual CL vs. EEG CL")
    #plt.xlabel("Average EEG CL")
    #plt.ylabel("Average Manual CL")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()


if __name__ == "__main__":
    main()
