#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RATES = (20, 40, 60)
METHOD_ORDER = [
    "BRITS-MI",
    "MICE",
    "Missforest",
    "Mean imputation",
    "Complete case",
    "Full data",
]
COLORS = {
    "BRITS-MI": "#007C83",
    "MICE": "#7651B5",
    "Missforest": "#D97706",
    "Mean imputation": "#7A8793",
    "Complete case": "#B33A3A",
    "Full data": "#17324D",
}
METHOD_LABELS = {
    "brits_mi_refined": "BRITS-MI",
    "mice": "MICE",
    "missforest": "Missforest",
    "mean_impute": "Mean imputation",
    "complete_case": "Complete case",
    "full_data": "Full data",
}
MISSFOREST5_DIR = (
    ROOT / "outputs" / "clinical_association_missforest5_n500_nsim200_20260802"
)


def source_paths(rate: int) -> dict[str, list[Path]]:
    return {
        "proposed": [
            ROOT / "outputs" / f"refined_brits_mi_assoc_n500_nsim100_m10_20_50_miss{rate}_20260802",
            ROOT / "outputs" / f"refined_brits_mi_assoc_n500_nsim100_additional_m10_20_50_miss{rate}_20260802",
        ],
        "package": [
            ROOT / "outputs" / f"package_default_assoc_n500_nsim100_m10_20_50_miss{rate}_20260802",
            ROOT / "outputs" / f"package_default_assoc_n500_nsim100_additional_m10_20_50_miss{rate}_20260802",
        ],
        "simple": [
            ROOT / "outputs" / f"simple_assoc_n500_nsim100_miss{rate}_20260802",
            ROOT / "outputs" / f"simple_assoc_n500_nsim100_additional_miss{rate}_20260802",
        ],
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    for rate in RATES:
        for paths in source_paths(rate).values():
            for path in paths:
                coefficient = pd.read_csv(path / "coefficient_by_run.csv")
                metrics = pd.read_csv(path / "metrics_by_run.csv")
                coefficient["missing_rate"] = rate / 100.0
                metrics["missing_rate"] = rate / 100.0
                coefficient_parts.append(coefficient)
                metric_parts.append(metrics)
    coefficient = pd.concat(coefficient_parts, ignore_index=True, sort=False)
    metrics = pd.concat(metric_parts, ignore_index=True, sort=False)
    if (MISSFOREST5_DIR / "coefficient_by_run.csv").exists():
        coefficient = coefficient[~coefficient["method"].eq("missforest")].copy()
        metrics = metrics[~metrics["method"].eq("missforest")].copy()
        coefficient_parts = [
            coefficient,
            pd.read_csv(MISSFOREST5_DIR / "coefficient_by_run.csv"),
        ]
        metric_parts = [
            metrics,
            pd.read_csv(MISSFOREST5_DIR / "metrics_by_run.csv"),
        ]
        coefficient = pd.concat(coefficient_parts, ignore_index=True, sort=False)
        metrics = pd.concat(metric_parts, ignore_index=True, sort=False)
        coefficient["missing_rate"] = coefficient["missing_rate"].fillna(
            coefficient["target_missing"]
        )
        metrics["missing_rate"] = metrics["missing_rate"].fillna(
            metrics["target_missing"]
        )
    coefficient["method_label"] = coefficient["method"].map(METHOD_LABELS)
    metrics["method_label"] = metrics["method"].map(METHOD_LABELS)
    coefficient = coefficient[coefficient["method_label"].notna()].copy()
    metrics = metrics[metrics["method_label"].notna()].copy()
    coefficient = coefficient[
        coefficient["n_imputations"].eq(50)
        | ~coefficient["method"].isin(["brits_mi_refined", "mice"])
    ].copy()
    metrics = metrics[
        metrics["n_imputations"].eq(50)
        | ~metrics["method"].isin(["brits_mi_refined", "mice"])
    ].copy()
    return coefficient, metrics


def exclude_unstable_complete_case(
    coefficient: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exclude divergent complete-case refits using a fixed estimability rule."""
    complete = coefficient[coefficient["method"].eq("complete_case")].copy()
    audit_rows: list[dict] = []
    stable_keys: set[tuple[float, int]] = set()
    for (rate, seed), part in complete.groupby(["missing_rate", "seed"]):
        n_terms = int(part["coefficient"].nunique())
        n_used = float(part["n_used"].max())
        estimate = part["estimate"].to_numpy(float)
        std_error = part["std_error"].to_numpy(float)
        stable = bool(
            n_used >= 10 * n_terms
            and np.isfinite(estimate).all()
            and np.isfinite(std_error).all()
            and np.max(np.abs(estimate)) <= 10.0
            and np.max(np.abs(std_error)) <= 10.0
        )
        if stable:
            stable_keys.add((float(rate), int(seed)))
        audit_rows.append(
            {
                "missing_rate": rate,
                "seed": seed,
                "n_used": n_used,
                "n_terms": n_terms,
                "max_abs_estimate": float(np.nanmax(np.abs(estimate))),
                "max_std_error": float(np.nanmax(np.abs(std_error))),
                "stable_for_effect_summary": stable,
            }
        )
    audited_keys = {
        (float(row["missing_rate"]), int(row["seed"])) for row in audit_rows
    }
    complete_metrics = metrics[metrics["method"].eq("complete_case")]
    for record in complete_metrics.to_dict("records"):
        key = (float(record["missing_rate"]), int(record["seed"]))
        if key in audited_keys:
            continue
        audit_rows.append(
            {
                "missing_rate": key[0],
                "seed": key[1],
                "n_used": record.get("coefficient_n_used", np.nan),
                "n_terms": np.nan,
                "max_abs_estimate": np.nan,
                "max_std_error": np.nan,
                "stable_for_effect_summary": False,
            }
        )

    def keep_complete_case(row: pd.Series) -> bool:
        if row["method"] != "complete_case":
            return True
        return (float(row["missing_rate"]), int(row["seed"])) in stable_keys

    coefficient = coefficient[
        coefficient.apply(keep_complete_case, axis=1)
    ].copy()
    metrics = metrics[metrics.apply(keep_complete_case, axis=1)].copy()
    return coefficient, metrics, pd.DataFrame(audit_rows)


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    sample = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def summarize(
    coefficient: pd.DataFrame,
    metrics: pd.DataFrame,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260802)
    rows: list[dict] = []
    run_rows: list[pd.DataFrame] = []
    for (rate, method), part in coefficient.groupby(["missing_rate", "method_label"]):
        by_seed = part.groupby("seed", as_index=False).agg(
            coefficient_mse=("squared_error", "mean"),
            coverage=("covered", "mean"),
        )
        bias_matrix = part.pivot(index="seed", columns="coefficient", values="bias")
        systematic_bias = float(np.abs(bias_matrix.mean(axis=0)).mean())
        boot_index = rng.integers(0, len(bias_matrix), size=(n_boot, len(bias_matrix)))
        boot_bias = np.abs(
            bias_matrix.to_numpy(float)[boot_index].mean(axis=1)
        ).mean(axis=1)
        metric_part = metrics[
            metrics["missing_rate"].eq(rate)
            & metrics["method_label"].eq(method)
        ].drop_duplicates(["seed", "method"])
        rmse = metric_part["imputation_rmse"].to_numpy(float)
        mse_low, mse_high = bootstrap_interval(
            by_seed["coefficient_mse"].to_numpy(float), rng, n_boot
        )
        coverage_low, coverage_high = bootstrap_interval(
            by_seed["coverage"].to_numpy(float), rng, n_boot
        )
        rmse_low, rmse_high = bootstrap_interval(rmse, rng, n_boot)
        rows.extend(
            [
                {
                    "missing_rate": rate,
                    "method_label": method,
                    "metric": "bias",
                    "estimate": systematic_bias,
                    "ci_low": float(np.quantile(boot_bias, 0.025)),
                    "ci_high": float(np.quantile(boot_bias, 0.975)),
                    "mcse": float(np.std(boot_bias, ddof=1)),
                    "n_runs": int(part["seed"].nunique()),
                },
                {
                    "missing_rate": rate,
                    "method_label": method,
                    "metric": "mse",
                    "estimate": float(by_seed["coefficient_mse"].mean()),
                    "ci_low": mse_low,
                    "ci_high": mse_high,
                    "mcse": float(by_seed["coefficient_mse"].std(ddof=1) / np.sqrt(len(by_seed))),
                    "n_runs": int(len(by_seed)),
                },
                {
                    "missing_rate": rate,
                    "method_label": method,
                    "metric": "coverage",
                    "estimate": float(by_seed["coverage"].mean()),
                    "ci_low": coverage_low,
                    "ci_high": coverage_high,
                    "mcse": float(by_seed["coverage"].std(ddof=1) / np.sqrt(len(by_seed))),
                    "n_runs": int(len(by_seed)),
                },
                {
                    "missing_rate": rate,
                    "method_label": method,
                    "metric": "imputation_rmse",
                    "estimate": float(np.nanmean(rmse)) if np.isfinite(rmse).any() else np.nan,
                    "ci_low": rmse_low,
                    "ci_high": rmse_high,
                    "mcse": float(np.nanstd(rmse, ddof=1) / np.sqrt(np.isfinite(rmse).sum()))
                    if np.isfinite(rmse).sum() > 1
                    else np.nan,
                    "n_runs": int(metric_part["seed"].nunique()),
                },
            ]
        )
        local = by_seed.copy()
        local["missing_rate"] = rate
        local["method_label"] = method
        run_rows.append(local)
    return pd.DataFrame(rows), pd.concat(run_rows, ignore_index=True)


def paired_tests(run_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rate in sorted(run_level["missing_rate"].unique()):
        proposed = run_level[
            run_level["missing_rate"].eq(rate)
            & run_level["method_label"].eq("BRITS-MI")
        ].set_index("seed")
        for comparator in ["MICE", "Missforest"]:
            reference = run_level[
                run_level["missing_rate"].eq(rate)
                & run_level["method_label"].eq(comparator)
            ].set_index("seed")
            for metric in ["coefficient_mse", "coverage"]:
                joined = proposed[[metric]].join(
                    reference[[metric]], lsuffix="_brits", rsuffix="_reference"
                ).dropna()
                difference = joined[f"{metric}_brits"] - joined[f"{metric}_reference"]
                rows.append(
                    {
                        "missing_rate": rate,
                        "metric": metric,
                        "comparator": comparator,
                        "n_pairs": len(joined),
                        "brits_mean": float(joined[f"{metric}_brits"].mean()),
                        "comparator_mean": float(joined[f"{metric}_reference"].mean()),
                        "mean_difference": float(difference.mean()),
                        "paired_t_p_value": float(stats.ttest_1samp(difference, 0).pvalue),
                    }
                )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, outdir: Path) -> None:
    panel_specs = [
        ("bias", "Absolute systematic coefficient bias", "lower is better"),
        ("mse", "Coefficient mean squared error (log scale)", "lower is better"),
        ("coverage", "95% coefficient coverage", "closer to 0.95 is better"),
        ("imputation_rmse", "Missing-cell imputation RMSE", "lower is better"),
    ]
    offsets = dict(zip(METHOD_ORDER, np.linspace(-1.7, 1.7, len(METHOD_ORDER))))
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.0), dpi=240)
    for panel, (metric, title, direction) in zip(axes.flat, panel_specs):
        for method in METHOD_ORDER:
            if metric == "imputation_rmse" and method in {"Full data", "Complete case"}:
                continue
            part = summary[
                summary["metric"].eq(metric)
                & summary["method_label"].eq(method)
            ].sort_values("missing_rate")
            part = part[np.isfinite(part["estimate"])]
            if part.empty:
                continue
            x = 100 * part["missing_rate"].to_numpy(float) + offsets[method]
            y = part["estimate"].to_numpy(float)
            ci_low = part["ci_low"].to_numpy(float)
            ci_high = part["ci_high"].to_numpy(float)
            yerr = np.vstack(
                [
                    np.maximum(y - ci_low, 0.0),
                    np.maximum(ci_high - y, 0.0),
                ]
            )
            panel.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                markersize=8.5,
                linewidth=2.4,
                capsize=4,
                color=COLORS[method],
                label=method,
            )
        if metric == "coverage":
            panel.axhline(0.95, color="#263238", linestyle="--", linewidth=1.4)
            panel.set_ylim(0.75, 1.01)
        elif metric == "bias":
            panel.set_ylim(-0.005, 0.31)
        elif metric == "mse":
            panel.set_yscale("log")
            panel.set_ylim(0.05, 2.0)
        panel.set_title(title, loc="left", fontsize=15, fontweight="bold")
        panel.text(
            1.0,
            1.015,
            direction,
            transform=panel.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            color="#52656D",
            fontweight="bold",
        )
        panel.set_xticks([20, 40, 60], ["20%", "40%", "60%"])
        panel.set_xlim(16.5, 63.5)
        panel.set_xlabel("Informative longitudinal missingness")
        panel.grid(axis="y", color="#D9E1E5", linewidth=0.9)
        panel.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=6,
        frameon=False,
        fontsize=11.5,
    )
    fig.suptitle(
        "Clinical association benchmark",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.875], h_pad=3.0, w_pad=2.5)
    fig.savefig(outdir / "clinical_association_nsim200_four_panel.png", bbox_inches="tight")
    fig.savefig(outdir / "clinical_association_nsim200_four_panel.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "clinical_association_n500_nsim200_final_20260802",
    )
    parser.add_argument("--n-bootstrap", type=int, default=3000)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    coefficient, metrics = load_data()
    coefficient, metrics, complete_case_audit = exclude_unstable_complete_case(
        coefficient, metrics
    )
    coefficient.to_csv(args.outdir / "coefficient_by_run.csv", index=False)
    metrics.to_csv(args.outdir / "metrics_by_run.csv", index=False)
    complete_case_audit.to_csv(
        args.outdir / "complete_case_estimability_audit.csv", index=False
    )
    summary, run_level = summarize(coefficient, metrics, args.n_bootstrap)
    summary.to_csv(args.outdir / "summary_with_monte_carlo_uncertainty.csv", index=False)
    run_level.to_csv(args.outdir / "effect_metrics_by_run.csv", index=False)
    paired_tests(run_level).to_csv(args.outdir / "paired_effect_tests.csv", index=False)
    manifest = {
        "experiment": "clinical association simulation",
        "n_subjects": 500,
        "n_runs_per_missingness_level": 200,
        "missingness_levels": [0.20, 0.40, 0.60],
        "scenario": "baseline_harder",
        "brits_mi_imputations": 50,
        "mice": "R mice package defaults with m=50 and maxit=5",
        "missforest": "Five independently seeded R missForest package-default completions (maxiter=10, ntree=100, mtry=floor(sqrt(p))) with downstream refitting and Rubin pooling",
        "predictor_preselection": "none for package-default MICE or Missforest",
        "replicate_construction": "BRITS-MI, MICE, and simple baselines combine two independent 100-run batches; five-draw Missforest was rerun on the same 200 seeds",
        "complete_case_estimability": "Effect summaries require n_used >= 10 times the number of reported terms, finite estimates and standard errors, max absolute estimate <= 10, and max standard error <= 10; exclusions are audited separately",
        "methods": METHOD_ORDER,
    }
    (args.outdir / "config.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    plot(summary, args.outdir)
    print(
        summary[
            summary["method_label"].isin(["BRITS-MI", "MICE", "Missforest"])
        ].sort_values(["metric", "missing_rate", "method_label"]).to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
