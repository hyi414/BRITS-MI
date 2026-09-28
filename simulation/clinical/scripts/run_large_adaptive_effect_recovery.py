#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
warnings.filterwarnings("ignore", message="overflow encountered in exp", category=RuntimeWarning)


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "scripts" / "test_posterior_vc_smc_brits.py"

SPEC = importlib.util.spec_from_file_location("posterior_test", TEST_PATH)
posterior_test = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(posterior_test)

sim = posterior_test.sim
hist_sim = posterior_test.hist_sim
dgp1 = posterior_test.dgp1
BIOMARKERS = posterior_test.BIOMARKERS
TRUE_BETA = posterior_test.TRUE_BETA
COEFFICIENT_LABELS = posterior_test.COEFFICIENT_LABELS


LOCKED_SCENARIOS = {
    "mild_irregular": {
        "label": "Mild irregularity",
        "time": np.array([0.00, 0.15, 0.32, 0.54, 0.78, 1.00]),
        "late_accel_mult": 0.85,
        "pulse_mult": 0.85,
        "noise_mult": 0.90,
        "missing_abnormality_mult": 0.85,
        "missing_slope_mult": 0.85,
    },
    "baseline_irregular": sim.SCENARIOS["baseline_irregular"],
    "strong_irregular": sim.SCENARIOS["strong_irregular"],
}
sim.SCENARIOS.update(LOCKED_SCENARIOS)

METHOD_LABELS = {
    "full_data": "Oracle (full data)",
    "complete_case": "Complete case",
    "adaptive_posterior_vc_smc_brits": "BRITS",
    "mice": "MICE",
    "missforest": "Missforest",
    "mean_impute": "Mean imputation",
}
METHOD_ORDER = [
    "full_data",
    "adaptive_posterior_vc_smc_brits",
    "mice",
    "missforest",
    "mean_impute",
    "complete_case",
]
METHOD_COLORS = {
    "full_data": "#111827",
    "adaptive_posterior_vc_smc_brits": "#005f73",
    "mice": "#7b2cbf",
    "missforest": "#4EA8DE",
    "mean_impute": "#6b7280",
    "complete_case": "#9b2226",
}

DGP_TEXT = """DGP variables affecting longitudinal values:
severity = N(0,1) + 0.36 diabetes + 0.22 hypertension - 0.08 male sex;
chronic_slope = N(0,0.32^2) + 0.22 diabetes + 0.15 hypertension;
late_activity = N(0,1) + 0.18 diabetes + 0.12 hypertension;
late_acceleration = N(0,1) + 0.70 late_activity + 0.22 diabetes + 0.18 hypertension.
AST, ALT, and platelet trajectories are generated from severity, chronic slope,
late acceleration after time 0.45, a subject-specific pulse, biomarker-specific
clinical shifts, and scenario-specific late-acceleration, pulse, and noise
multipliers. Platelet effects have the opposite disease direction.

DGP variables affecting missingness:
missingness is non-monotone at biomarker-cell level and depends on fibrosis
outcome Y, diabetes, hypertension, visit time, biomarker type, abnormal observed
biomarker value (AST, ALT, and low platelet), slope pressure
0.30 slope_AST + 0.22 slope_ALT - 0.28 slope_platelet, late activity, and
scenario-specific abnormality/slope multipliers. The intercept is calibrated by
replicate to the target late-cell missingness.
"""


def _safe_nanmean(values: np.ndarray) -> float:
    if not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def _run_one(
    seed: int,
    n: int,
    n_imputations: int,
    target_missing: float,
    scenario: str,
    variance_factors: tuple[float, ...],
    calibration_fraction: float,
    calibration_draws: int,
    adaptive_target: float,
) -> tuple[list[dict], list[dict], dict]:
    rng = np.random.default_rng(seed)
    diabetes = rng.binomial(1, 0.36, size=n).astype(float)
    htn = rng.binomial(1, sim._sigmoid(-0.20 + 1.05 * diabetes), size=n).astype(float)
    male = rng.binomial(1, 0.46, size=n).astype(float)
    trajectory, time, latent = sim._simulate_irregular_trajectory(rng, diabetes, htn, male, scenario)
    true_features = dgp1._trajectory_features(trajectory, time)
    x_true = dgp1._design(true_features, diabetes, htn, male)
    eta = -0.84 + sum(TRUE_BETA[name] * x_true[name].to_numpy() for name in TRUE_BETA)
    outcome = rng.binomial(1, sim._sigmoid(eta)).astype(int)

    observed = sim._make_missingness_targeted(
        trajectory,
        outcome,
        diabetes,
        htn,
        latent,
        time,
        rng,
        target_missing,
        scenario,
    )
    trajectory_obs = np.where(observed, trajectory, np.nan)
    base_imp = dgp1._build_base_frame(trajectory_obs, observed, diabetes, htn, male, time)
    outcome_imp = dgp1._outcome_features(true_features, latent, outcome, base_imp, diabetes, htn, male, rng)
    late_true_cells = hist_sim._cell_frame(trajectory)
    late_obs_cells = hist_sim._cell_frame(trajectory_obs)
    history_by_cell = posterior_test._build_history_by_cell(trajectory_obs, observed, time)

    core, vc_smc_brits = posterior_test._make_vc_smc_brits_draws(
        late_obs_cells,
        base_imp,
        history_by_cell,
        outcome_imp,
        trajectory_obs,
        observed,
        time,
        outcome,
        diabetes,
        htn,
        male,
        rng,
        n_imputations,
    )
    selected_factor, selected_calibration = posterior_test._choose_adaptive_variance_factor(
        trajectory,
        observed,
        true_features,
        latent,
        outcome,
        diabetes,
        htn,
        male,
        time,
        rng,
        variance_factors,
        calibration_fraction,
        calibration_draws,
        adaptive_target,
    )
    adaptive_draws = posterior_test.posterior_variance_calibrated_brits_draws(
        vc_smc_brits,
        late_obs_cells,
        base_imp,
        history_by_cell,
        outcome_imp,
        trajectory_obs,
        time,
        outcome,
        diabetes,
        htn,
        male,
        rng,
        n_imputations,
        variance_factor=selected_factor,
    )
    methods = {
        "full_data": [late_true_cells],
        "complete_case": [late_obs_cells],
        "adaptive_posterior_vc_smc_brits": adaptive_draws,
        "mice": core["mice"],
        "missforest": core["missforest"],
        "mean_impute": core["mean_impute"],
    }

    coef_rows: list[dict] = []
    metric_rows: list[dict] = []
    missing_late = ~observed[:, -3:, :]
    true_late_cells = trajectory[:, -3:, :]
    true_summary = true_features.to_numpy()
    for method, cell_draws in methods.items():
        completed_trajectories = [hist_sim._frame_to_late_trajectory(draw, trajectory_obs) for draw in cell_draws]
        feature_draws = [dgp1._trajectory_features(comp, time) for comp in completed_trajectories]
        pooled = dgp1._pool_logit_fits(feature_draws, outcome, diabetes, htn, male)
        fit_success = pooled is not None
        fit_n_used = 0
        fit_n_draws = 0
        if pooled is not None:
            params, bse, within_se, between_sd, ci, n_used, n_fit_draws = pooled
            fit_n_used = int(n_used)
            fit_n_draws = int(n_fit_draws)
            for name, truth in TRUE_BETA.items():
                lo, hi = ci.loc[name]
                coef_rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "scenario_label": LOCKED_SCENARIOS[scenario]["label"],
                        "target_missing": target_missing,
                        "n_imputations": n_imputations,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "coefficient": name,
                        "coefficient_label": COEFFICIENT_LABELS[name],
                        "true_value": truth,
                        "estimate": float(params[name]),
                        "std_error": float(bse[name]),
                        "within_std_error": float(within_se[name]),
                        "between_imputation_sd": float(between_sd[name]),
                        "bias": float(params[name] - truth),
                        "abs_bias": float(abs(params[name] - truth)),
                        "squared_error": float((params[name] - truth) ** 2),
                        "covered": float(lo <= truth <= hi),
                        "n_used": int(n_used),
                        "n_fit_draws": int(n_fit_draws),
                        "selected_variance_factor": (
                            float(selected_factor) if method == "adaptive_posterior_vc_smc_brits" else np.nan
                        ),
                    }
                )

        if method in {"full_data", "complete_case"}:
            imputation_rmse = 0.0 if method == "full_data" else np.nan
            trajectory_summary_rmse = 0.0 if method == "full_data" else np.nan
        else:
            completed_mean = np.mean(np.stack([comp[:, -3:, :] for comp in completed_trajectories], axis=0), axis=0)
            imputation_rmse = float(np.sqrt(np.nanmean((completed_mean[missing_late] - true_late_cells[missing_late]) ** 2)))
            summary_mean = np.nanmean(np.stack([fd.to_numpy() for fd in feature_draws], axis=0), axis=0)
            trajectory_summary_rmse = float(np.sqrt(_safe_nanmean((summary_mean - true_summary) ** 2)))
        metric_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "scenario_label": LOCKED_SCENARIOS[scenario]["label"],
                "target_missing": target_missing,
                "n_imputations": n_imputations,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "imputation_rmse": imputation_rmse,
                "trajectory_summary_rmse": trajectory_summary_rmse,
                "coefficient_fit_success": float(fit_success),
                "coefficient_n_used": fit_n_used,
                "coefficient_n_fit_draws": fit_n_draws,
                "selected_variance_factor": (
                    float(selected_factor) if method == "adaptive_posterior_vc_smc_brits" else np.nan
                ),
            }
        )

    selected = selected_calibration[selected_calibration["selected"]].iloc[0]
    audit = {
        "seed": seed,
        "scenario": scenario,
        "scenario_label": LOCKED_SCENARIOS[scenario]["label"],
        "target_missing": target_missing,
        "n_imputations": n_imputations,
        "n": int(n),
        "outcome_rate": float(outcome.mean()),
        "late_cell_missing_rate": float(missing_late.mean()),
        "cell_missing_rate": float((~observed).mean()),
        "complete_late_rate": float(observed[:, -3:, :].all(axis=(1, 2)).mean()),
        "selected_variance_factor": float(selected_factor),
        "validation_coverage": float(selected["validation_coverage"]),
        "validation_rmse": float(selected["validation_rmse"]),
        "validation_width": float(selected["validation_width"]),
        "n_validation_cells": int(selected["n_validation_cells"]),
    }
    return coef_rows, metric_rows, audit


def _run_task(task):
    return _run_one(*task)


def summarize_effects(coef_df: pd.DataFrame, metric_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_term = (
        coef_df.groupby(
            [
                "scenario",
                "scenario_label",
                "target_missing",
                "n_imputations",
                "method",
                "method_label",
                "coefficient",
                "coefficient_label",
            ],
            as_index=False,
        )
        .agg(
            true_value=("true_value", "first"),
            estimate_mean=("estimate", "mean"),
            signed_bias=("bias", "mean"),
            mean_abs_run_bias=("abs_bias", "mean"),
            coefficient_mse=("squared_error", "mean"),
            coverage=("covered", "mean"),
            empirical_se=("estimate", lambda x: float(np.std(x, ddof=1))),
            average_se=("std_error", "mean"),
            within_se=("within_std_error", "mean"),
            between_imputation_sd=("between_imputation_sd", "mean"),
            n_runs=("seed", "nunique"),
            n_used_mean=("n_used", "mean"),
        )
        .reset_index(drop=True)
    )
    by_term["abs_signed_bias"] = by_term["signed_bias"].abs()
    by_term["se_ratio"] = by_term["average_se"] / by_term["empirical_se"].replace(0, np.nan)
    overall_coef = (
        by_term.groupby(["scenario", "scenario_label", "target_missing", "n_imputations", "method", "method_label"], as_index=False)
        .agg(
            mean_abs_signed_bias=("abs_signed_bias", "mean"),
            mean_abs_run_bias=("mean_abs_run_bias", "mean"),
            coefficient_mse=("coefficient_mse", "mean"),
            coverage=("coverage", "mean"),
            empirical_se=("empirical_se", "mean"),
            average_se=("average_se", "mean"),
            se_ratio=("se_ratio", "mean"),
            between_imputation_sd=("between_imputation_sd", "mean"),
            n_runs=("n_runs", "min"),
            n_used_mean=("n_used_mean", "mean"),
        )
    )
    metric_summary = (
        metric_df.groupby(["scenario", "scenario_label", "target_missing", "n_imputations", "method", "method_label"], as_index=False)
        .agg(
            imputation_rmse_mean=("imputation_rmse", "mean"),
            imputation_rmse_sd=("imputation_rmse", "std"),
            trajectory_summary_rmse_mean=("trajectory_summary_rmse", "mean"),
            trajectory_summary_rmse_sd=("trajectory_summary_rmse", "std"),
            coefficient_fit_rate=("coefficient_fit_success", "mean"),
            coefficient_n_used_mean=("coefficient_n_used", "mean"),
        )
    )
    overall = metric_summary.merge(
        overall_coef,
        on=["scenario", "scenario_label", "target_missing", "n_imputations", "method", "method_label"],
        how="left",
    )
    return by_term, overall


def _order_methods(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method_order"] = out["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return out.sort_values(["scenario", "method_order"])


def plot_overall_tradeoff(overall: pd.DataFrame, outdir: Path) -> None:
    data = _order_methods(overall)
    scenarios = list(LOCKED_SCENARIOS)
    metrics = [
        ("mean_abs_signed_bias", "coverage", "Overall |signed bias|", "95% CI coverage"),
        ("imputation_rmse_mean", "coverage", "Imputation RMSE", "95% CI coverage"),
        ("trajectory_summary_rmse_mean", "coverage", "Trajectory-summary RMSE", "95% CI coverage"),
        ("empirical_se", "average_se", "Empirical SE", "Average model SE"),
    ]
    fig, axes = plt.subplots(len(scenarios), len(metrics), figsize=(21, 12.2), dpi=240, constrained_layout=True)
    for r, scenario in enumerate(scenarios):
        subset = data[data["scenario"] == scenario]
        for c, (xcol, ycol, xlabel, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            for _, row in subset.iterrows():
                if xcol in {"imputation_rmse_mean", "trajectory_summary_rmse_mean"} and not np.isfinite(row[xcol]):
                    continue
                if not (np.isfinite(row[xcol]) and np.isfinite(row[ycol])):
                    continue
                method = row["method"]
                size = 155 if method == "adaptive_posterior_vc_smc_brits" else 80
                marker = "*" if method == "adaptive_posterior_vc_smc_brits" else "s" if method in {"mice", "missforest"} else "o"
                ax.scatter(
                    row[xcol],
                    row[ycol],
                    s=size,
                    marker=marker,
                    color=METHOD_COLORS.get(method, "#777777"),
                    edgecolor="white",
                    linewidth=0.9,
                    zorder=3,
                )
                label = METHOD_LABELS[method]
                label = label.replace("Complete case", "CC")
                label = label.replace("Oracle (full data)", "Oracle")
                ax.annotate(label, (row[xcol], row[ycol]), xytext=(5, 3), textcoords="offset points", fontsize=7.1)
            if ycol == "coverage":
                ax.axhline(0.95, color="#333333", linestyle="--", linewidth=1.0)
                ax.set_ylim(max(0.70, subset[ycol].min() - 0.04), 1.0)
            if (xcol, ycol) == ("empirical_se", "average_se"):
                finite = subset[np.isfinite(subset[xcol]) & np.isfinite(subset[ycol])]
                if not finite.empty:
                    lo = min(finite[xcol].min(), finite[ycol].min()) * 0.90
                    hi = max(finite[xcol].max(), finite[ycol].max()) * 1.06
                    ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=1.0)
                    ax.set_xlim(lo, hi)
                    ax.set_ylim(lo, hi)
            else:
                finite = subset[np.isfinite(subset[xcol])]
                if not finite.empty:
                    xmin, xmax = finite[xcol].min(), finite[xcol].max()
                    pad = max((xmax - xmin) * 0.14, 0.01)
                    ax.set_xlim(max(0, xmin - pad), xmax + pad)
            if r == 0:
                ax.set_title(f"{xlabel} vs {ylabel}", fontsize=11.2, weight="bold")
            if c == 0:
                ax.set_ylabel(f"{LOCKED_SCENARIOS[scenario]['label']}\n{ylabel}", fontsize=9.2)
            else:
                ax.set_ylabel(ylabel, fontsize=9.2)
            ax.set_xlabel(xlabel, fontsize=9.2)
            ax.grid(color="#dddddd", linewidth=0.7)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.suptitle("Large deterministic simulation: effect recovery and uncertainty trade-offs", fontsize=16, weight="bold")
    fig.savefig(outdir / "large_overall_tradeoff_grid.png", bbox_inches="tight")
    plt.close(fig)


def plot_predictor_bias_coverage(by_term: pd.DataFrame, outdir: Path) -> None:
    coeff_order = list(TRUE_BETA)
    scenarios = list(LOCKED_SCENARIOS)
    methods = ["full_data", "adaptive_posterior_vc_smc_brits", "mice", "missforest", "mean_impute", "complete_case"]
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(15.5, 14.5), dpi=240, constrained_layout=True)
    if len(scenarios) == 1:
        axes = np.asarray(axes).reshape(1, 2)
    for r, scenario in enumerate(scenarios):
        subset = by_term[(by_term["scenario"] == scenario) & (by_term["method"].isin(methods))].copy()
        subset["coef_order"] = subset["coefficient"].map({c: i for i, c in enumerate(coeff_order)})
        for c, (metric, title) in enumerate([("signed_bias", "Signed coefficient bias"), ("coverage", "95% CI coverage")]):
            ax = axes[r, c]
            y = np.arange(len(coeff_order))
            offsets = np.linspace(-0.27, 0.27, len(methods))
            for idx, method in enumerate(methods):
                data = subset[subset["method"] == method].set_index("coefficient").reindex(coeff_order)
                ax.scatter(
                    data[metric].to_numpy(dtype=float),
                    y + offsets[idx],
                    s=44 if method != "adaptive_posterior_vc_smc_brits" else 86,
                    marker="*" if method == "adaptive_posterior_vc_smc_brits" else "o",
                    color=METHOD_COLORS.get(method, "#777777"),
                    edgecolor="white",
                    linewidth=0.7,
                    label=METHOD_LABELS[method],
                )
            if metric == "signed_bias":
                ax.axvline(0, color="#333333", linewidth=1.0)
            else:
                ax.axvline(0.95, color="#333333", linestyle="--", linewidth=1.0)
                ax.set_xlim(0.55, 1.0)
            ax.set_yticks(y)
            ax.set_yticklabels([COEFFICIENT_LABELS[c] for c in coeff_order], fontsize=7.8)
            ax.invert_yaxis()
            ax.set_title(f"{LOCKED_SCENARIOS[scenario]['label']}: {title}", fontsize=11.2, weight="bold")
            ax.set_xlabel(title, fontsize=9.2)
            ax.grid(axis="x", color="#dddddd", linewidth=0.7)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Individual predictor effect recovery across irregular longitudinal scenarios", fontsize=16, weight="bold")
    fig.savefig(outdir / "large_predictor_bias_coverage.png", bbox_inches="tight")
    plt.close(fig)


def plot_selected_factor(audit: pd.DataFrame, outdir: Path) -> None:
    counts = pd.crosstab(audit["scenario_label"], audit["selected_variance_factor"])
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=240, constrained_layout=True)
    counts.plot(kind="bar", ax=ax, edgecolor="white", colormap="viridis")
    ax.set_title("Selected posterior variance factor across repeated runs", fontsize=13, weight="bold")
    ax.set_xlabel("Irregular longitudinal scenario")
    ax.set_ylabel("Number of repeated runs")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Selected factor", frameon=False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(outdir / "large_selected_factor_distribution.png", bbox_inches="tight")
    plt.close(fig)
    counts.to_csv(outdir / "large_selected_factor_counts.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("outputs/large_adaptive_effect_recovery_n500_nsim1000"))
    parser.add_argument("--nsim", type=int, default=1000)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--m", type=int, default=50)
    parser.add_argument("--target-missing", type=float, default=0.60)
    parser.add_argument("--seed-start", type=int, default=500000)
    parser.add_argument("--scenarios", nargs="+", choices=list(LOCKED_SCENARIOS), default=list(LOCKED_SCENARIOS))
    parser.add_argument("--variance-factors", type=float, nargs="+", default=[1.3, 1.7, 2.1, 2.5])
    parser.add_argument("--calibration-fraction", type=float, default=0.18)
    parser.add_argument("--calibration-draws", type=int, default=20)
    parser.add_argument("--adaptive-target", type=float, default=0.86)
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    variance_factors = tuple(float(x) for x in args.variance_factors)
    tasks = []
    job = 0
    for scenario in args.scenarios:
        for rep in range(args.nsim):
            tasks.append(
                (
                    args.seed_start + job * 100000 + rep,
                    args.n,
                    args.m,
                    args.target_missing,
                    scenario,
                    variance_factors,
                    args.calibration_fraction,
                    args.calibration_draws,
                    args.adaptive_target,
                )
            )
        job += 1

    coef_rows: list[dict] = []
    metric_rows: list[dict] = []
    audit_rows: list[dict] = []
    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            for c_rows, m_rows, audit in executor.map(_run_task, tasks, chunksize=1):
                coef_rows.extend(c_rows)
                metric_rows.extend(m_rows)
                audit_rows.append(audit)
    else:
        for task in tasks:
            c_rows, m_rows, audit = _run_task(task)
            coef_rows.extend(c_rows)
            metric_rows.extend(m_rows)
            audit_rows.append(audit)

    coef_df = pd.DataFrame(coef_rows)
    metric_df = pd.DataFrame(metric_rows)
    audit_df = pd.DataFrame(audit_rows)
    by_term, overall = summarize_effects(coef_df, metric_df)
    coef_df.to_csv(args.outdir / "coefficient_by_run.csv", index=False)
    metric_df.to_csv(args.outdir / "metrics_by_run.csv", index=False)
    audit_df.to_csv(args.outdir / "missingness_audit.csv", index=False)
    by_term.to_csv(args.outdir / "effect_recovery_by_predictor.csv", index=False)
    overall.to_csv(args.outdir / "effect_recovery_overall.csv", index=False)
    _order_methods(overall).to_csv(args.outdir / "effect_recovery_main_table.csv", index=False)
    plot_overall_tradeoff(overall, args.outdir)
    plot_predictor_bias_coverage(by_term, args.outdir)
    plot_selected_factor(audit_df, args.outdir)

    config = {
        "nsim": args.nsim,
        "n": args.n,
        "m": args.m,
        "target_missing": args.target_missing,
        "seed_start": args.seed_start,
        "scenarios": {
            name: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in LOCKED_SCENARIOS[name].items()}
            for name in args.scenarios
        },
        "variance_factors": list(variance_factors),
        "calibration_fraction": args.calibration_fraction,
        "calibration_draws": args.calibration_draws,
        "adaptive_target": args.adaptive_target,
        "methods": METHOD_LABELS,
        "dgp_text": DGP_TEXT,
        "n_workers": args.n_workers,
    }
    with open(args.outdir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    (args.outdir / "dgp_design_note.txt").write_text(DGP_TEXT, encoding="utf-8")

    display = _order_methods(overall)[
        [
            "scenario_label",
            "method_label",
            "mean_abs_signed_bias",
            "coverage",
            "empirical_se",
            "average_se",
            "se_ratio",
            "imputation_rmse_mean",
            "trajectory_summary_rmse_mean",
            "coefficient_fit_rate",
            "n_used_mean",
            "n_runs",
        ]
    ]
    print(display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
