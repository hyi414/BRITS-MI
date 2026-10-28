"""Summaries retain attempted/valid denominators, per-term errors, and paired contrasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def summarize(output):
    output = Path(output)
    records, coefficients = [], []
    for path in sorted((output / "jobs").glob("*/result.json")):
        data = json.loads(path.read_text())
        metadata = {k: data["job"][k] for k in ["kind", "scenario", "n", "m", "missing_percentage", "seed", "mechanism"]}
        records.extend([{**metadata, **r} for r in data["metrics"]])
        coefficients.extend([{**metadata, **r} for r in data["coefficients"]])
    if not records:
        return
    frame = pd.DataFrame(records)
    frame.to_csv(output / "metrics_by_run.csv", index=False)
    keys = ["kind", "scenario", "n", "m", "mechanism", "missing_percentage", "method"]
    validity = frame.groupby(keys, as_index=False).agg(
        n_attempted=("seed", "nunique"), n_valid=("status", lambda x: int(x.eq("valid").sum())))
    validity["n_invalid"] = validity.n_attempted - validity.n_valid
    validity.to_csv(output / "validity_summary.csv", index=False)
    summary = []
    metrics = [x for x in ["coverage", "reference_interval_inclusion", "coefficient_mse", "mean_model_se",
        "imputation_rmse", "imputation_nrmse", "trajectory_nrmse", "cell_interval_inclusion", "seconds",
        "auroc", "auprc", "log_loss", "apparent_auroc", "apparent_auprc", "apparent_log_loss"] if x in frame]
    for group, part in frame.groupby(keys):
        for metric in metrics:
            x = part.loc[part.status.eq("valid"), metric].dropna()
            summary.append(dict(zip(keys, group)) | dict(metric=metric, mean=x.mean(), sd=x.std(ddof=1),
                mcse=x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                n_valid=len(x), n_attempted=len(part)))
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    if coefficients:
        terms = pd.DataFrame(coefficients)
        terms.to_csv(output / "coefficient_by_run.csv", index=False)
        grouped = terms.groupby(keys + ["term"], as_index=False).agg(
            bias=("signed_error", "mean"), mse=("squared_error", "mean"), coverage=("covered", "mean"),
            empirical_se=("estimate", "std"), error_sd=("signed_error", "std"),
            average_model_se=("standard_error", "mean"), n_valid=("seed", "nunique"))
        grouped["bias_mcse"] = grouped.error_sd / np.sqrt(grouped.n_valid)
        grouped["coverage_mcse"] = np.sqrt(grouped.coverage * (1 - grouped.coverage) / grouped.n_valid)
        grouped = grouped.merge(validity[keys + ["n_attempted"]], on=keys, validate="many_to_one")
        grouped["interval_target"] = np.where(grouped.kind.eq("real"), "pre-mask reference estimate", "population coefficient")
        grouped.to_csv(output / "term_summary.csv", index=False)
        grouped["absolute_bias"] = grouped.bias.abs()
        overall = grouped.groupby(keys, as_index=False).agg(
            mean_absolute_bias=("absolute_bias", "mean"), mean_signed_bias=("bias", "mean"),
            mean_mse=("mse", "mean"), mean_coverage=("coverage", "mean"),
            mean_empirical_se=("empirical_se", "mean"), mean_model_se=("average_model_se", "mean"))
        overall = overall.merge(validity, on=keys, validate="one_to_one")
        overall["interval_target"] = np.where(overall.kind.eq("real"), "pre-mask reference estimate", "population coefficient")
        overall.to_csv(output / "overall_association.csv", index=False)
    tests = []
    grouping = ["kind", "scenario", "n", "m", "mechanism", "missing_percentage"]
    for group, part in frame.groupby(grouping):
        for metric in [x for x in ["auroc", "auprc", "coefficient_mse", "imputation_nrmse"] if x in frame]:
            wide = part.loc[part.status.eq("valid")].pivot(index="seed", columns="method", values=metric)
            if "BRITS-MI" not in wide:
                continue
            for comparator in [x for x in ["MICE", "missForest", "Mean imputation"] if x in wide]:
                paired = wide[["BRITS-MI", comparator]].dropna()
                if len(paired) < 2:
                    continue
                difference = paired["BRITS-MI"] - paired[comparator]
                se = difference.std(ddof=1) / np.sqrt(len(difference))
                p = stats.ttest_1samp(difference, 0).pvalue if se > 0 else np.nan
                tests.append(dict(zip(grouping, group)) | dict(metric=metric, comparator=comparator,
                    n_pairs=len(paired), mean_difference=difference.mean(), difference_se=se,
                    lower=difference.mean() - stats.t.ppf(.975, len(paired)-1)*se,
                    upper=difference.mean() + stats.t.ppf(.975, len(paired)-1)*se, p_value=p))
    if tests:
        table = pd.DataFrame(tests)
        table["p_holm"] = np.nan
        for _, part in table.groupby(grouping + ["metric"]):
            valid = part.p_value.notna()
            table.loc[part.index[valid], "p_holm"] = multipletests(part.loc[valid, "p_value"], method="holm")[1]
        table.to_csv(output / "paired_comparisons.csv", index=False)
    print(frame.groupby(["missing_percentage", "method", "status"]).size().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
