import pandas as pd
import statsmodels.formula.api as smf

def compute_correlations(target_prediction, cccp_metric, old_suffix):
    print(f"Computing correlations for predictor: {target_prediction}, CCCP metric: {cccp_metric}, old suffix: {old_suffix}")

    # 1. Load the data
    df = pd.read_csv("../../results/complexities_and_CLs.csv")

    # 2. Drop missing data in any of the relevant columns
    needed_cols = ["subject","ID","manual_cl","DD","Halstead","LOC","Cyclomatic","CCCPPD","CCCPMPI","DDold","Halsteadold","LOCold","Cyclomaticold","CCCPPDold","CCCPMPIold",f"{target_prediction}"]
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

    return results_df, mdf_all

pdes, pdec = compute_correlations("eeg_cl_fzpz34", "CCCPPD", "")
# compute_correlations("eeg_cl_fz34pz34", "CCCPPD", "")
# compute_correlations("eeg_cl_fzpz", "CCCPPD", "")
pdms, pdmc = compute_correlations("manual_cl", "CCCPPD", "")
mpes, mpec = compute_correlations("eeg_cl_fzpz34", "CCCPMPI", "")
# compute_correlations("eeg_cl_fz34pz34", "CCCPMPI", "")
# compute_correlations("eeg_cl_fzpz", "CCCPMPI", "")
mpms, mpmc = compute_correlations("manual_cl", "CCCPMPI", "")

print(f'Cyclomatic & {pdes.loc[pdes["metric"] == "Cyclomatic"]["LogLik"].iloc[0].round(2)} & {pdes.loc[pdes["metric"] == "Cyclomatic"]["coef"].iloc[0].round(2)} & {"<0.001" if pdes.loc[pdes["metric"] == "Cyclomatic"]["p-value"].iloc[0].round(3) < 0.001 else pdes.loc[pdes["metric"] == "Cyclomatic"]["p-value"].iloc[0].round(3)}', end=" & ")
print(f'{pdms.loc[pdms["metric"] == "Cyclomatic"]["LogLik"].iloc[0].round(2)} & {pdms.loc[pdms["metric"] == "Cyclomatic"]["coef"].iloc[0].round(2)} & {"<0.001" if pdms.loc[pdms["metric"] == "Cyclomatic"]["p-value"].iloc[0].round(3) < 0.001 else pdms.loc[pdms["metric"] == "Cyclomatic"]["p-value"].iloc[0].round(3)} \\\\')

print(f'DD & {pdes.loc[pdes["metric"] == "DD"]["LogLik"].iloc[0].round(2)} & {pdes.loc[pdes["metric"] == "DD"]["coef"].iloc[0].round(2)} & {"<0.001" if pdes.loc[pdes["metric"] == "DD"]["p-value"].iloc[0].round(3) < 0.001 else pdes.loc[pdes["metric"] == "DD"]["p-value"].iloc[0].round(3)}', end=" & ")
print(f'{pdms.loc[pdms["metric"] == "DD"]["LogLik"].iloc[0].round(2)} & {pdms.loc[pdms["metric"] == "DD"]["coef"].iloc[0].round(2)} & {"<0.001" if pdms.loc[pdms["metric"] == "DD"]["p-value"].iloc[0].round(3) < 0.001 else pdms.loc[pdms["metric"] == "DD"]["p-value"].iloc[0].round(3)} \\\\')

print(f'Halstead & {pdes.loc[pdes["metric"] == "Halstead"]["LogLik"].iloc[0].round(2)} & {pdes.loc[pdes["metric"] == "Halstead"]["coef"].iloc[0].round(2)} & {"<0.001" if pdes.loc[pdes["metric"] == "Halstead"]["p-value"].iloc[0].round(3) < 0.001 else pdes.loc[pdes["metric"] == "Halstead"]["p-value"].iloc[0].round(3)}', end=" & ")
print(f'{pdms.loc[pdms["metric"] == "Halstead"]["LogLik"].iloc[0].round(2)} & {pdms.loc[pdms["metric"] == "Halstead"]["coef"].iloc[0].round(2)} & {"<0.001" if pdms.loc[pdms["metric"] == "Halstead"]["p-value"].iloc[0].round(3) < 0.001 else pdms.loc[pdms["metric"] == "Halstead"]["p-value"].iloc[0].round(3)} \\\\')

print(f'LOC & {pdes.loc[pdes["metric"] == "LOC"]["LogLik"].iloc[0].round(2)} & {pdes.loc[pdes["metric"] == "LOC"]["coef"].iloc[0].round(2)} & {"<0.001" if pdes.loc[pdes["metric"] == "LOC"]["p-value"].iloc[0].round(3) < 0.001 else pdes.loc[pdes["metric"] == "LOC"]["p-value"].iloc[0].round(3)}', end=" & ")
print(f'{pdms.loc[pdms["metric"] == "LOC"]["LogLik"].iloc[0].round(2)} & {pdms.loc[pdms["metric"] == "LOC"]["coef"].iloc[0].round(2)} & {"<0.001" if pdms.loc[pdms["metric"] == "LOC"]["p-value"].iloc[0].round(3) < 0.001 else pdms.loc[pdms["metric"] == "LOC"]["p-value"].iloc[0].round(3)} \\\\')

print(f'CCCP-PD & {pdes.loc[pdes["metric"] == "CCCPPD"]["LogLik"].iloc[0].round(2)} & {pdes.loc[pdes["metric"] == "CCCPPD"]["coef"].iloc[0].round(2)} & {"<0.001" if pdes.loc[pdes["metric"] == "CCCPPD"]["p-value"].iloc[0].round(3) < 0.001 else pdes.loc[pdes["metric"] == "CCCPPD"]["p-value"].iloc[0].round(3)}', end=" & ")
print(f'{pdms.loc[pdms["metric"] == "CCCPPD"]["LogLik"].iloc[0].round(2)} & {pdms.loc[pdms["metric"] == "CCCPPD"]["coef"].iloc[0].round(2)} & {"<0.001" if pdms.loc[pdms["metric"] == "CCCPPD"]["p-value"].iloc[0].round(3) < 0.001 else pdms.loc[pdms["metric"] == "CCCPPD"]["p-value"].iloc[0].round(3)} \\\\')

print(f'CCCP-MPI & {mpes.loc[mpes["metric"] == "CCCPMPI"]["LogLik"].iloc[0].round(2)} & {mpes.loc[mpes["metric"] == "CCCPMPI"]["coef"].iloc[0].round(2)} & {"<0.001" if mpes.loc[mpes["metric"] == "CCCPMPI"]["p-value"].iloc[0].round(3) < 0.001 else mpes.loc[mpes["metric"] == "CCCPMPI"]["p-value"].iloc[0].round(3)}', end=" & ")
print(f'{mpms.loc[mpms["metric"] == "CCCPMPI"]["LogLik"].iloc[0].round(2)} & {mpms.loc[mpms["metric"] == "CCCPMPI"]["coef"].iloc[0].round(2)} & {"<0.001" if mpms.loc[mpms["metric"] == "CCCPMPI"]["p-value"].iloc[0].round(3) < 0.001 else mpms.loc[mpms["metric"] == "CCCPMPI"]["p-value"].iloc[0].round(3)} \\\\')
