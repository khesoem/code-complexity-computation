import pandas as pd
import statsmodels.formula.api as smf

def compute_correlations(target_prediction, cccp_metric, old_suffix):
    print(f"Computing correlations for predictor: {target_prediction}, CCCP metric: {cccp_metric}, old suffix: {old_suffix}")

    # 1. Load the data
    df = pd.read_csv("../../results/final_results.csv")

    # 2. Drop missing data in any of the relevant columns
    needed_cols = ["subject","ID","manual_cl","DD","Halstead","LOC","Cyclomatic","CCCPPD","CCCPMPI","DDold","Halsteadold","LOCold","Cyclomaticold","CCCPPDold","CCCPMPIold","eeg_cl_fzpz34","eeg_cl_fz34pz34","eeg_cl_fzpz"]
    df = df.dropna(subset=needed_cols).copy()

    # 3. Convert subject and ID to categorical, reset the index
    df["subject"] = df["subject"].astype("category")
    df["ID"] = df["ID"].astype("category")
    df.reset_index(drop=True, inplace=True)

    # 4. (Optional) Scale all metrics (M1..M6)
    #    We'll create new columns scale_M1..scale_M6
    metric_names = [f"DD{old_suffix}",f"Halstead{old_suffix}",f"LOC{old_suffix}",f"Cyclomatic{old_suffix}",f"{cccp_metric}{old_suffix}"]
    for m in metric_names:
        df[f"scale_{m}"] = (df[m] - df[m].mean()) / df[m].std()

    # -----------------------------------------------------------
    # PART A: Compare six separate models, one metric at a time.
    # -----------------------------------------------------------

    results_list = []
    for m in metric_names:
        # Build a formula with only one scaled metric as a predictor
        formula = f"{target_prediction} ~ scale_{m}"

        # Fit the mixed model
        md = smf.mixedlm(
            formula=formula,
            data=df,
            groups=df["subject"],     # random intercept by subject
            re_formula="~1",          # random intercept only
            vc_formula={"ID": "0 + C(ID)"}  # random intercept by ID
        )
        mdf = md.fit(method='lbfgs')

        # Store model information
        results_list.append({
            "metric": m,
            "AIC": mdf.aic,
            "BIC": mdf.bic,
            "LogLik": mdf.llf,
            "coef": mdf.params.get(f"scale_{m}", float('nan')),
            "p-value": mdf.pvalues.get(f"scale_{m}", float('nan'))
        })

    # Convert results to a DataFrame for easy viewing
    results_df = pd.DataFrame(results_list)
    results_df.sort_values(by="AIC", inplace=True)
    print("=== Comparison of Single-Metric Models (sorted by AIC) ===")
    print(results_df)

    # -----------------------------------------------------------
    # PART B: Fit one combined model with all six metrics
    # -----------------------------------------------------------

    # Build a formula with all six scaled metrics
    all_metrics_formula = (
        f'{target_prediction} ~ ' + ' + '.join([f'scale_{m}' for m in metric_names])
    )

    md_all = smf.mixedlm(
        formula=all_metrics_formula,
        data=df,
        groups=df["subject"],
        re_formula="~1",
        vc_formula={"ID": "0 + C(ID)"}
    )
    mdf_all = md_all.fit(method='lbfgs')

    print("\n=== Combined Model with All 5 Metrics ===")
    print(mdf_all.summary())

compute_correlations("eeg_cl_fzpz34", "CCCPPD", "")
compute_correlations("eeg_cl_fz34pz34", "CCCPPD", "")
compute_correlations("eeg_cl_fzpz", "CCCPPD", "")
compute_correlations("eeg_cl_fzpz34", "CCCPMPI", "")
compute_correlations("eeg_cl_fz34pz34", "CCCPMPI", "")
compute_correlations("eeg_cl_fzpz", "CCCPMPI", "")
# import numpy as np
# df = pd.read_csv("../../results/final_results.csv")
# olddf = pd.read_csv("../../results/revised_complexities_and_eeg_cl.csv")
# df["eeg_cl_fzpz34"] = pd.to_numeric(df["eeg_cl_fzpz34"], errors="coerce")
# olddf["rating"] = pd.to_numeric(olddf["rating"], errors="coerce")
# df = df[np.isfinite(df["eeg_cl_fzpz34"])]
# olddf = olddf[np.isfinite(olddf["rating"])]
# df["eeg_cl_fzpz34"] = df["eeg_cl_fzpz34"].round(0).astype(int)
# olddf["rating"] = olddf["rating"].round(0).astype(int)
#
# eeg_vals = df["eeg_cl_fzpz34"].tolist()
# old_eeg_vals = olddf["rating"].tolist()
# if eeg_vals != old_eeg_vals:
#     print("EEG values do not match between the two datasets. Please check the data.")
#
# print(eeg_vals)