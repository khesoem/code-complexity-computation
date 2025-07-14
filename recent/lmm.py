#!/usr/bin/env python3
"""
EEG ↔️ Code-Complexity Correlation Utility
-----------------------------------------

This script fits (linear) mixed-effects models to quantify how strongly various
code-complexity metrics predict either

* an **EEG-based comprehension difficulty proxy** (default: ``ThetaAlphaRatio``), or
* **self-reported comprehension difficulty**, i.e. the ``SurveyCL`` ratings in
  *snippet_metrics.csv* (activate via the ``--survey`` flag).

The script prints *exactly two* machine-readable lines per run:

    1. **SINGLE**   – coefficient & p-value of the focal metric in the
       single-predictor ("one metric at a time") model, plus whether it is the
       highest (most-positive) coefficient.
    2. **COMBINED** – the same, but for the full multi-predictor model.

Example output for ``cccp_metric = "CCCPMPI"`` and the default EEG target::

    SINGLE:  CCCPMPI  coef = 0.052  p = 0.172  (best)
    COMBINED:CCCPMPI  coef = 0.106  p = 0.205  (best)

When the focal metric is **not** the most-positive coefficient, the line
mentions the actual best metric, e.g.::

    SINGLE:  CCCPPD  coef = 0.034  p = 0.233  (¬best, top = LOC 0.078)

Command-line usage
~~~~~~~~~~~~~~~~~~

```
python eeg_cc_complexity_updated.py            # EEG-based (ThetaAlphaRatio)
python eeg_cc_complexity_updated.py --survey   # SurveyCL ratings
```
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import statsmodels.formula.api as smf

###############################################################################
# Helper functions                                                            #
###############################################################################

def _safe_get(mapping: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    """Return *mapping[key]* or *default* if the key is absent or the value is
    not finite (mixed-effects sometimes returns NaNs when the fit fails).
    """
    val = mapping.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

###############################################################################
# Core utility                                                                #
###############################################################################

def compute_correlations(
    target_prediction: str,
    cccp_metric: str,
    *,
    old_suffix: str = "",
    prediction_source: str = "eeg",  # "eeg" (default) or "metrics"
    eeg_path: str | Path = "eeg.csv",
    metrics_path: str | Path = "snippet_metrics.csv",
) -> None:
    """Fit mixed-effects models and print two succinct summary lines.

    Parameters
    ----------
    target_prediction : str
        Column to be predicted (either in *eeg.csv* or *snippet_metrics.csv*).
    cccp_metric : str
        Name of the focal CCCP metric, *without* the ``old_suffix``.
    old_suffix : str, optional
        Suffix appended to metric names in legacy tables (default: "").
    prediction_source : {"eeg", "metrics"}, optional
        Where to read *target_prediction* from. ``"eeg"`` expects the column
        in *eeg.csv* (default). ``"metrics"`` expects it in *snippet_metrics.csv*.
    eeg_path, metrics_path : str or pathlib.Path, optional
        File locations (defaults point to the repository root).
    """

    # ------------------------------------------------------------------
    # 1. Load & harmonise data                                          #
    # ------------------------------------------------------------------
    key_cols: List[str] = ["Participant", "SnippetID"]

    metrics_df = pd.read_csv(metrics_path)
    for col in key_cols:
        metrics_df[col] = metrics_df[col].astype(str)

    if prediction_source == "eeg":
        eeg_df = pd.read_csv(eeg_path)
        for col in key_cols:
            eeg_df[col] = eeg_df[col].astype(str)

        if target_prediction not in eeg_df.columns:
            raise KeyError(
                f"Column '{target_prediction}' not found in {eeg_path}. "
                "Did you mean to use '--survey'?"
            )

        eeg_cols = [*key_cols, target_prediction]
        df = pd.merge(
            metrics_df,
            eeg_df[eeg_cols],
            on=key_cols,
            how="inner",
            validate="one_to_one",
        )
    elif prediction_source == "metrics":
        if target_prediction not in metrics_df.columns:
            raise KeyError(
                f"Column '{target_prediction}' not found in {metrics_path}."
            )
        df = metrics_df.copy()
    else:
        raise ValueError("prediction_source must be 'eeg' or 'metrics'")

    # ------------------------------------------------------------------
    # 2. Drop rows with missing values                                  #
    # ------------------------------------------------------------------
    metric_names: List[str] = [
        f"DD{old_suffix}",
        f"Halstead{old_suffix}",
        f"LOC{old_suffix}",
        f"Cyclomatic{old_suffix}",
        f"{cccp_metric}{old_suffix}",
    ]
    needed_cols: List[str] = [*key_cols, *metric_names, target_prediction]
    df = df.dropna(subset=needed_cols)
    if df.empty:
        print(f"⚠️  No data after filtering – {cccp_metric} skipped.")
        return

    # ------------------------------------------------------------------
    # 3. Prepare categorical keys & scale predictors                    #
    # ------------------------------------------------------------------
    df["Participant"] = df["Participant"].astype("category")
    df["SnippetID"] = df["SnippetID"].astype("category")

    for m in metric_names:
        df[f"scale_{m}"] = (df[m] - df[m].mean()) / df[m].std()

    ####################################################################
    # PART A – Single-predictor models                                 #
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
            # Model failed – record NaNs so the focal metric can still be found
            single_results.append(
                {"metric": m, "coef": float("nan"), "p": float("nan")}
            )

    focal_metric = f"{cccp_metric}{old_suffix}"
    single_by_metric = {r["metric"]: r for r in single_results}
    focal_single = single_by_metric.get(
        focal_metric, {"coef": float("nan"), "p": float("nan")}
    )
    best_single = max(
        single_results,
        key=lambda r: float("-inf") if pd.isna(r["coef"]) else r["coef"],
    )  # most-positive coef

    best_single_flag = best_single["metric"] == focal_metric

    ####################################################################
    # PART B – Combined model                                          #
    ####################################################################
    combined_coef = float("nan")
    combined_p = float("nan")
    best_comb_metric: str | None = None
    try:
        formula_all = (
            f"{target_prediction} ~ " + " + ".join(f"scale_{m}" for m in metric_names)
        )
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

        # determine best combined coefficient among considered metrics
        comb_metrics_coefs = {m: _safe_get(params, f"scale_{m}") for m in metric_names}
        best_comb_metric = max(comb_metrics_coefs, key=comb_metrics_coefs.get)
    except ValueError:
        pass  # leave nan values; printing logic below will cope

    best_comb_flag = best_comb_metric == focal_metric if best_comb_metric else False

    ####################################################################
    # 4. Two-line report                                               #
    ####################################################################
    def _fmt_line(
        label: str,
        coef: float,
        pval: float,
        is_best: bool,
        best_info: str | None = None,
    ) -> str:
        base = f"{label:<9}{focal_metric:<9}coef = {coef:.3g}  p = {pval:.3g}"
        if pd.isna(coef):
            return f"{base}  (model failed)"
        if is_best:
            return f"{base}  (best)"
        return f"{base}  (¬best, top = {best_info})"

    # build best-info strings only if needed
    single_best_info = (
        f"{best_single['metric']} {best_single['coef']:.3g}"
        if not best_single_flag
        else None
    )
    combined_best_info = None
    if not best_comb_flag and best_comb_metric:
        combined_best_info = (
            f"{best_comb_metric} {_safe_get(params, f'scale_{best_comb_metric}'):.3g}"
        )

    print(
        _fmt_line(
            "SINGLE:",
            focal_single["coef"],
            focal_single["p"],
            best_single_flag,
            single_best_info,
        )
    )
    print(
        _fmt_line(
            "COMBINED:",
            combined_coef,
            combined_p,
            best_comb_flag,
            combined_best_info,
        )
    )

###############################################################################
# CLI entry-point                                                             #
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Fit mixed-effects models for code-complexity metrics against EEG "
            "or SurveyCL comprehension difficulty targets."
        )
    )
    parser.add_argument(
        "--survey",
        action="store_true",
        help="Use SurveyCL ratings (snippet_metrics.csv) instead of EEG-based ThetaAlphaRatio predictions.",
    )

    args = parser.parse_args()

    target_prediction = "SurveyCL" if args.survey else "ThetaAlphaRatio"
    prediction_source = "metrics" if args.survey else "eeg"

    for metric in ("CCCPPD", "CCCPMPI"):
        compute_correlations(
            target_prediction=target_prediction,
            cccp_metric=metric,
            prediction_source=prediction_source,
        )
