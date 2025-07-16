#!/usr/bin/env python3
"""
EEG ↔️ Code‑Complexity Correlation Utility (survey‑only edition)
--------------------------------------------------------------

This revision makes **no changes to the mixed‑effects logic** but restores the
behaviour of the former “all‑in‑one CSV” workflow when the metrics table now
contains *one row per snippet* (instead of one per participant‑snippet pair).

➤ **Automatic join mode**

* If the metrics table has a *unique* row for every `SnippetID`, the script
  joins on `SnippetID` only (many‑to‑one) and happily propagates the same metric
  values to every participant.
* Otherwise it falls back to a strict `Participant + SnippetID` join, exactly
  like the original.

No command‑line flags: tweak the constants below and run `python eeg_cc_complexity.py`.

"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import statsmodels.formula.api as smf

###############################################################################
# Configuration                                                                #
###############################################################################

TARGET_PREDICTION: str = "CL"                     # Column to predict
SURVEY_PATH: Path = Path("eeg.csv")          # CL ratings (Participant, SnippetID, CL)
METRICS_PATH: Path = Path("snippet_metrics.csv") # Complexity metrics table
MCC_METRICS: tuple[str, ...] = ("MCCPD", "MCCMPI")  # MCC metrics to iterate over
OLD_SUFFIX: str = ""                               # Legacy suffix for column names, if any

###############################################################################
# Helper functions                                                             #
###############################################################################

def _safe_get(mapping: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    """Return *mapping[key]* or *default* if missing/NaN/inf."""
    val = mapping.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

###############################################################################
# Core utility                                                                 #
###############################################################################

def compute_correlations(
    mcc_metric: str,
    *,
    target_prediction: str = TARGET_PREDICTION,
    survey_path: Path = SURVEY_PATH,
    metrics_path: Path = METRICS_PATH,
    old_suffix: str = OLD_SUFFIX,
) -> None:
    """Fit mixed‑effects models and print two succinct summary lines."""

    # ------------------------------------------------------------------
    # 1. Load & harmonise data                                          #
    # ------------------------------------------------------------------
    key_cols: List[str] = ["Participant", "SnippetID"]

    survey_df = pd.read_csv(survey_path)
    metrics_df = pd.read_csv(metrics_path)

    # Ensure consistent dtypes for merge keys
    for col in key_cols:
        if col in survey_df.columns:
            survey_df[col] = survey_df[col].astype(str)
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].astype(str)

    if target_prediction not in survey_df.columns:
        raise KeyError(f"Column '{target_prediction}' not found in {survey_path}.")

    # ------------------------------------------------------------------
    # 1a. Determine join strategy                                       #
    # ------------------------------------------------------------------
    #   • If metrics already repeat for every participant (old style),
    #     merge on Participant+SnippetID (1‑to‑1).
    #   • If each SnippetID appears exactly once, treat metrics as
    #     snippet‑level constants -> drop Participant and merge on SnippetID.

    snippet_counts = metrics_df.groupby("SnippetID").size()
    metrics_unique_per_snippet = (snippet_counts == 1).all()

    if metrics_unique_per_snippet:
        # Drop Participant (if present) – many participants share one metric row
        metrics_join = metrics_df.drop(columns=["Participant"], errors="ignore")
        df = pd.merge(
            survey_df,
            metrics_join,
            on="SnippetID",
            how="inner",
            validate="many_to_one",
        )
    else:
        # Old behaviour: need exact (Participant, SnippetID) match
        df = pd.merge(
            survey_df,
            metrics_df,
            on=key_cols,
            how="inner",
            validate="one_to_one",
        )

    # ------------------------------------------------------------------
    # 2. Drop rows with missing values                                  #
    # ------------------------------------------------------------------
    metric_names: List[str] = [
        f"DD{old_suffix}",
        f"Halstead{old_suffix}",
        f"LOC{old_suffix}",
        f"Cyclomatic{old_suffix}",
        f"{mcc_metric}{old_suffix}",
    ]
    needed_cols: List[str] = ["Participant", "SnippetID", *metric_names, target_prediction]
    df = df.dropna(subset=needed_cols)
    if df.empty:
        print(f"⚠️  No data after filtering – {mcc_metric} skipped.")
        return

    # ------------------------------------------------------------------
    # 3. Prepare categorical keys & scale predictors                    #
    # ------------------------------------------------------------------
    df["Participant"] = df["Participant"].astype("category")
    df["SnippetID"] = df["SnippetID"].astype("category")

    for m in metric_names:
        df[f"scale_{m}"] = (df[m] - df[m].mean()) / df[m].std()

    ####################################################################
    # PART A – Single‑predictor models                                 #
    ####################################################################
    single_results: List[Dict[str, Any]] = []
    for m in metric_names:
        formula = f"{target_prediction} ~ scale_{m}"
        try:
            md = smf.mixedlm(
                formula=formula,
                data=df,
                groups=df["Participant"],
                re_formula="~1",
                vc_formula={"SnippetID": "0 + C(SnippetID)"},
            )
            with warnings.catch_warnings():  # suppress convergence chatter
                warnings.simplefilter("ignore")
                mdf = md.fit(method="lbfgs")
            single_results.append(
                {
                    "metric": m,
                    "coef": _safe_get(mdf.params, f"scale_{m}"),
                    "p": _safe_get(mdf.pvalues, f"scale_{m}"),
                }
            )
        except ValueError:
            single_results.append({"metric": m, "coef": float("nan"), "p": float("nan")})

    focal_metric = f"{mcc_metric}{old_suffix}"
    single_by_metric = {r["metric"]: r for r in single_results}
    focal_single = single_by_metric.get(focal_metric, {"coef": float("nan"), "p": float("nan")})
    best_single = max(single_results, key=lambda r: float("-inf") if pd.isna(r["coef"]) else r["coef"])

    best_single_flag = best_single["metric"] == focal_metric

    ####################################################################
    # PART B – Combined model                                          #
    ####################################################################
    combined_coef = float("nan")
    combined_p = float("nan")
    best_comb_metric: str | None = None
    try:
        formula_all = f"{target_prediction} ~ " + " + ".join(f"scale_{m}" for m in metric_names)
        md_all = smf.mixedlm(
            formula=formula_all,
            data=df,
            groups=df["Participant"],
            re_formula="~1",
            vc_formula={"SnippetID": "0 + C(SnippetID)"},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdf_all = md_all.fit(method="lbfgs")
        params = mdf_all.params
        pvals = mdf_all.pvalues
        combined_coef = _safe_get(params, f"scale_{focal_metric}")
        combined_p = _safe_get(pvals, f"scale_{focal_metric}")

        comb_metrics_coefs = {m: _safe_get(params, f"scale_{m}") for m in metric_names}
        best_comb_metric = max(comb_metrics_coefs, key=comb_metrics_coefs.get)
    except ValueError:
        pass

    best_comb_flag = best_comb_metric == focal_metric if best_comb_metric else False

    ####################################################################
    # 4. Two‑line report                                               #
    ####################################################################
    def _fmt_line(label: str, coef: float, pval: float, is_best: bool, best_info: str | None = None) -> str:
        base = f"{label:<9}{focal_metric:<9}coef = {coef:.3g}  p = {pval:.3g}"
        if pd.isna(coef):
            return f"{base}  (model failed)"
        if is_best:
            return f"{base}  (best)"
        return f"{base}  (¬best, top = {best_info})"

    single_best_info = f"{best_single['metric']} {best_single['coef']:.3g}" if not best_single_flag else None
    combined_best_info = (
        f"{best_comb_metric} {_safe_get(params, f'scale_{best_comb_metric}'):.3g}" if not best_comb_flag and best_comb_metric else None
    )

    print(_fmt_line("SINGLE:", focal_single["coef"], focal_single["p"], best_single_flag, single_best_info))
    print(_fmt_line("COMBINED:", combined_coef, combined_p, best_comb_flag, combined_best_info))

###############################################################################
# Script entry point                                                          #
###############################################################################

if __name__ == "__main__":
    for metric in MCC_METRICS:
        compute_correlations(metric)
