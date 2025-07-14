#!/usr/bin/env python3
"""
aggregate_channel_quality.py
----------------------------

Summarises channel_quality.csv (one row per Participant × Channel)
into aggregate statistics across all available participants.

Usage
-----
    python aggregate_channel_quality.py --out summary.csv

If --out is omitted the script just prints to the console.

Outputs
-------
A tidy CSV (or console table) with columns:

    Channel,
    N_Participants,
    Mean_AvgCorr, SD_AvgCorr, Median_AvgCorr,
    Mean_ZCorr,  SD_ZCorr,  Median_ZCorr,
    Mean_RansacCorr,
    BadProp

where BadProp = fraction of participants whose channel is flagged
bad by   (ZCorr < -2) OR (RansacCorr < 0.75).

© 2025
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate EEG channel quality metrics")
    p.add_argument("--out", help="Write summary to this file (CSV). If omitted, prints.")
    return p.parse_args()


def load_data() -> pd.DataFrame:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "channel_quality.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        sys.exit(f"Error reading {csv_path}: {e}")
    required = {"Participant", "Channel", "AvgCorr", "ZCorr", "RansacCorr"}
    missing = required.difference(df.columns)
    if missing:
        sys.exit(f"CSV is missing required columns: {', '.join(missing)}")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # Define helper lambdas that ignore NaNs
    nanmean = lambda x: np.nanmean(x.values) if len(x) else np.nan
    nanstd  = lambda x: np.nanstd(x.values, ddof=1) if len(x) else np.nan
    nanmed  = lambda x: np.nanmedian(x.values) if len(x) else np.nan

    # Bad‑channel rule: ZCorr < –2 OR RansacCorr < 0.75
    bad_mask = (df["ZCorr"] < -2) | (df["RansacCorr"] < 0.75)
    df = df.assign(Bad=bad_mask)

    agg = (
        df.groupby("Channel")
          .agg(
              N_Participants = ("Participant", "nunique"),
              Mean_AvgCorr   = ("AvgCorr", nanmean),
              SD_AvgCorr     = ("AvgCorr", nanstd),
              Median_AvgCorr = ("AvgCorr", nanmed),
              Mean_ZCorr     = ("ZCorr", nanmean),
              SD_ZCorr       = ("ZCorr", nanstd),
              Median_ZCorr   = ("ZCorr", nanmed),
              Mean_RansacCorr= ("RansacCorr", nanmean),
              BadProp        = ("Bad", nanmean),
          )
          .reset_index()
          .sort_values("Channel")
    )
    return agg


def main():
    args = parse_args()
    df = load_data()
    summary = aggregate(df)

    if args.out:
        summary.to_csv(args.out, index=False)
        print(f"[OK] Summary written to {args.out}")
    else:
        # Pretty console print
        with pd.option_context("display.max_columns", None,
                               "display.float_format", "{:.3f}".format):
            print(summary)


if __name__ == "__main__":
    main()
