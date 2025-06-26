import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# 1. Load the data
df = pd.read_csv("../../results/revised_complexities_and_manual_cl.csv")

# 2. Drop missing data in any of the relevant columns
needed_cols = ["subject","ID","rating","DD","Halstead","LOC","Cyclomatic","CCCPPD","CCCPMDI","DDold","Halsteadold","LOCold","Cyclomaticold","CCCPPDold","CCCPMPIold"]
df = df.dropna(subset=needed_cols).copy()

# 3. Convert subject and ID to categorical, reset the index
df["subject"] = df["subject"].astype("category")
df["ID"] = df["ID"].astype("category")
df.reset_index(drop=True, inplace=True)

# 4. (Optional) Scale all metrics (M1..M6)
#    We'll create new columns scale_M1..scale_M6
metric_names = ["DD","Halstead","LOC","Cyclomatic","CCCPPD","CCCPMDI","DDold","Halsteadold","LOCold","Cyclomaticold","CCCPPDold","CCCPMPIold"]
for m in metric_names:
    df[f"scale_{m}"] = (df[m] - df[m].mean()) / df[m].std()

# -----------------------------------------------------------
# PART A: Compare six separate models, one metric at a time.
# -----------------------------------------------------------

results_list = []
for m in metric_names:
    # Build a formula with only one scaled metric as a predictor
    formula = f"rating ~ scale_{m}"

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
print('rating ~ ' + ' + '.join([f'scale_{m}' for m in metric_names]))

all_metrics_formula = (
    'rating ~ ' + ' + '.join([f'scale_{m}' for m in metric_names])
)

md_all = smf.mixedlm(
    formula=all_metrics_formula,
    data=df,
    groups=df["subject"],
    re_formula="~1",
    vc_formula={"ID": "0 + C(ID)"}
)
mdf_all = md_all.fit(method='lbfgs')

print("\n=== Combined Model with All 12 Metrics ===")
print(mdf_all.summary())
