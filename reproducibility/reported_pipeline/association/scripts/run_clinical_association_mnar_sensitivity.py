#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_neural_brits_mi_downstream_gradient as neural  # noqa: E402
import run_package_default_association_comparators as comparators  # noqa: E402
import run_refined_brits_mi_association as refined  # noqa: E402
from package_default_imputation import run_package_imputer  # noqa: E402
from test_anchored_brits_mi_blend import blend_draws, missforest_anchor  # noqa: E402


SCENARIO = "baseline_harder"
HIST_SIM = neural.run_large.hist_sim
METHOD_ORDER = [
    "brits_mi_refined",
    "mice",
    "missforest",
    "mean_impute",
    "complete_case",
    "full_data",
]
METHOD_LABELS = {
    "brits_mi_refined": "BRITS-MI",
    "mice": "MICE",
    "missforest": "Missforest",
    "mean_impute": "Mean imputation",
    "complete_case": "Complete case",
    "full_data": "Full data",
}
COLORS = {
    "brits_mi_refined": "#007C83",
    "mice": "#7651B5",
    "missforest": "#D97706",
    "mean_impute": "#7A8793",
    "complete_case": "#B33A3A",
    "full_data": "#17324D",
}
STRENGTH_LABELS = {
    0.0: "Locked reference\n(extra OR 1.00)",
    0.5: "Moderate MNAR\n(extra OR 1.65)",
    1.0: "Strong MNAR\n(extra OR 2.72)",
}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _calibrate_missingness(
    component: np.ndarray,
    rng: np.random.Generator,
    target_late_missing: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    late = np.zeros_like(component, dtype=bool)
    late[:, -3:, :] = True
    lo, hi = -8.0, 8.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        expected = float(neural.sim._sigmoid(mid + component)[late].mean())
        if expected < target_late_missing:
            lo = mid
        else:
            hi = mid
    intercept = 0.5 * (lo + hi)
    probability_missing = neural.sim._sigmoid(intercept + component)
    observed = rng.binomial(1, 1.0 - probability_missing).astype(bool)
    return observed, probability_missing, intercept


def generate_mnar_dataset(
    seed: int,
    n: int,
    target_missing: float,
    mnar_strength: float,
) -> tuple[dict, dict]:
    """Add a direct current-cell MNAR departure to the locked generator.

    At strength zero this reproduces the locked clinical-association dataset
    exactly. Positive strengths multiply a standardized current-cell
    abnormality that is unavailable when that cell is missing. The intercept is
    recalibrated so marginal late-cell missingness remains fixed.
    """
    rng = np.random.default_rng(seed)
    diabetes = rng.binomial(1, 0.36, size=n).astype(float)
    hypertension = rng.binomial(
        1,
        neural.sim._sigmoid(-0.20 + 1.05 * diabetes),
        size=n,
    ).astype(float)
    male = rng.binomial(1, 0.46, size=n).astype(float)
    trajectory, time_grid, latent = neural.sim._simulate_irregular_trajectory(
        rng,
        diabetes,
        hypertension,
        male,
        SCENARIO,
    )
    true_features = neural.dgp1._trajectory_features(trajectory, time_grid)
    true_design = neural.dgp1._design(
        true_features,
        diabetes,
        hypertension,
        male,
    )
    eta = -0.84 + sum(
        neural.TRUE_BETA[name] * true_design[name].to_numpy()
        for name in neural.TRUE_BETA
    )
    outcome = rng.binomial(1, neural.sim._sigmoid(eta)).astype(int)

    cfg = neural.sim.SCENARIOS[SCENARIO]
    slope_pressure = (
        0.30 * true_features["slope_ast"].to_numpy()
        + 0.22 * true_features["slope_alt"].to_numpy()
        - 0.28 * true_features["slope_platelet"].to_numpy()
    )
    visit_shift = np.array([-0.30, -0.06, -0.16, -0.50, 0.12, 0.62])
    marker_shift = np.array([0.05, -0.02, 0.14])
    abnormality = np.stack(
        [trajectory[:, :, 0], trajectory[:, :, 1], -trajectory[:, :, 2]],
        axis=2,
    )
    base_component = (
        1.05 * outcome[:, None, None]
        + 0.28 * diabetes[:, None, None]
        + 0.16 * hypertension[:, None, None]
        + visit_shift[None, :, None]
        + marker_shift[None, None, :]
        + float(cfg["missing_abnormality_mult"]) * 0.18 * abnormality
        + float(cfg["missing_slope_mult"])
        * 0.26
        * slope_pressure[:, None, None]
        * (time_grid[None, :, None] > 0.50)
        + 0.08 * latent["late_activity"][:, None, None]
        + rng.normal(scale=0.24, size=trajectory.shape)
    )

    center = trajectory.mean(axis=0, keepdims=True)
    scale = trajectory.std(axis=0, ddof=1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    current_value_z = (trajectory - center) / scale
    direction = np.array([1.0, 1.0, -1.0]).reshape(1, 1, 3)
    unobserved_abnormality = np.clip(current_value_z * direction, -3.0, 3.0)
    component = base_component + mnar_strength * unobserved_abnormality
    observed, probability_missing, intercept = _calibrate_missingness(
        component,
        rng,
        target_missing,
    )
    trajectory_obs = np.where(observed, trajectory, np.nan)
    base = neural.dgp1._build_base_frame(
        trajectory_obs,
        observed,
        diabetes,
        hypertension,
        male,
        time_grid,
    )
    late_observed = HIST_SIM._cell_frame(trajectory_obs)
    data = {
        "trajectory": trajectory.astype(np.float32),
        "trajectory_obs": trajectory_obs.astype(np.float32),
        "observed": observed,
        "time": time_grid.astype(np.float32),
        "times": np.broadcast_to(
            time_grid.astype(np.float32),
            (n, len(time_grid)),
        ).copy(),
        "static": np.column_stack([diabetes, hypertension, male]).astype(np.float32),
        "diabetes": diabetes,
        "hypertension": hypertension,
        "male": male,
        "outcome": outcome,
        "true_features": true_features,
        "target_missing": float(target_missing),
        "base": base,
        "late_observed": late_observed,
    }

    late = np.zeros_like(observed, dtype=bool)
    late[:, -3:, :] = True
    late_values = unobserved_abnormality[late]
    late_missing = (~observed)[late]
    lower = late_values <= np.quantile(late_values, 0.20)
    upper = late_values >= np.quantile(late_values, 0.80)
    audit = {
        "seed": seed,
        "mnar_strength": mnar_strength,
        "mnar_odds_ratio_per_sd": math.exp(mnar_strength),
        "target_late_missing": target_missing,
        "realized_late_missing": float(late_missing.mean()),
        "all_cell_missing": float((~observed).mean()),
        "low_abnormality_missing": float(late_missing[lower].mean()),
        "high_abnormality_missing": float(late_missing[upper].mean()),
        "high_minus_low_missing": float(
            late_missing[upper].mean() - late_missing[lower].mean()
        ),
        "calibrated_intercept": intercept,
        "mean_model_missing_probability": float(probability_missing[late].mean()),
    }
    return data, audit


def _score_cell_draws(
    data: dict,
    draws: list[pd.DataFrame],
    seed: int,
    method: str,
    target_missing: float,
) -> tuple[list[dict], dict]:
    comparators.METHOD_LABELS.update(METHOD_LABELS)
    return comparators.score_method(
        data,
        draws,
        seed,
        SCENARIO,
        method,
        target_missing,
    )


def _run_one(task: tuple) -> tuple[list[dict], list[dict], list[dict]]:
    (
        seed,
        n,
        m,
        missforest_draws,
        target_missing,
        mnar_strength,
        mean_weight,
        noise_weight,
        max_epochs,
    ) = task
    torch.set_num_threads(1)
    data, audit = generate_mnar_dataset(
        seed,
        n,
        target_missing,
        mnar_strength,
    )
    started = time.time()
    coefficient_rows: list[dict] = []
    metric_rows: list[dict] = []

    train_config = argparse.Namespace(
        m_values=[m],
        crossfit_folds=3,
        hidden_size=48,
        max_epochs=max_epochs,
    )
    train_args = refined.training_args(m, train_config)
    raw_draws, _, _, fold_audit, predictive = neural.cross_fitted_completions(
        data,
        seed,
        train_args,
        train_args.lambda_down,
        "brits_mi_gradient",
        torch.device("cpu"),
    )
    anchor = missforest_anchor(data, seed)
    completed = blend_draws(
        raw_draws,
        anchor,
        data["observed"],
        mean_weight,
        noise_weight,
    )
    neural.METHOD_LABELS["brits_mi_refined"] = "BRITS-MI"
    rows, metric = neural.evaluate_completions(
        data,
        completed,
        seed,
        SCENARIO,
        "brits_mi_refined",
        predictive,
    )
    coefficient_rows.extend(rows)
    metric_rows.append(metric)

    frame = pd.concat(
        [
            data["late_observed"].reset_index(drop=True),
            data["base"].reset_index(drop=True),
        ],
        axis=1,
    )
    mice_draws, mice_metadata = run_package_imputer(
        frame,
        comparators.TARGET_COLUMNS,
        "mice",
        m,
        seed + 720260,
    )
    rows, metric = _score_cell_draws(
        data,
        mice_draws,
        seed,
        "mice",
        target_missing,
    )
    coefficient_rows.extend(rows)
    metric_rows.append(metric)

    missforest_trajectories = [anchor]
    for draw_index in range(1, missforest_draws):
        draws, _ = run_package_imputer(
            frame,
            comparators.TARGET_COLUMNS,
            "missforest",
            1,
            seed + 8_142_119 + 1009 * draw_index,
        )
        missforest_trajectories.append(
            HIST_SIM._frame_to_late_trajectory(
                draws[0],
                data["trajectory_obs"],
            )
        )
    missforest_cells = [
        HIST_SIM._cell_frame(draw) for draw in missforest_trajectories
    ]
    rows, metric = _score_cell_draws(
        data,
        missforest_cells,
        seed,
        "missforest",
        target_missing,
    )
    coefficient_rows.extend(rows)
    metric_rows.append(metric)

    late_full = HIST_SIM._cell_frame(data["trajectory"])
    late_observed = data["late_observed"].copy()
    mean_completed = late_observed.copy()
    for column in comparators.TARGET_COLUMNS:
        mean_completed[column] = mean_completed[column].fillna(
            mean_completed[column].mean()
        )
    references = {
        "full_data": [late_full],
        "complete_case": [late_observed],
        "mean_impute": [mean_completed],
    }
    for method, draws in references.items():
        try:
            rows, metric = _score_cell_draws(
                data,
                draws,
                seed,
                method,
                target_missing,
            )
            coefficient_rows.extend(rows)
            metric_rows.append(metric)
        except Exception as exc:
            metric_rows.append(
                {
                    "seed": seed,
                    "scenario": SCENARIO,
                    "scenario_label": neural.SCENARIOS[SCENARIO]["label"],
                    "target_missing": target_missing,
                    "n_imputations": len(draws),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "imputation_rmse": np.nan,
                    "trajectory_summary_rmse": np.nan,
                    "coefficient_fit_success": 0.0,
                    "coefficient_n_used": 0,
                    "coefficient_n_fit_draws": 0,
                    "failure": repr(exc),
                }
            )

    for row in coefficient_rows:
        row["mnar_strength"] = mnar_strength
        row["mnar_odds_ratio_per_sd"] = math.exp(mnar_strength)
    for row in metric_rows:
        row["mnar_strength"] = mnar_strength
        row["mnar_odds_ratio_per_sd"] = math.exp(mnar_strength)
        row["runtime_total_seconds"] = float(time.time() - started)
    audit.update(
        {
            "mice_m": m,
            "mice_version": mice_metadata.get("package_version", ""),
            "missforest_draws": missforest_draws,
            "folds_completed": len(fold_audit),
            "runtime_total_seconds": float(time.time() - started),
        }
    )
    return coefficient_rows, metric_rows, [audit]


def _bootstrap_coefficient_metric(
    frame: pd.DataFrame,
    metric: str,
    seed: int,
    n_bootstrap: int = 1000,
) -> tuple[float, float, float, float]:
    seeds = np.sort(frame["seed"].unique())
    if metric == "absolute_bias":
        matrix = frame.pivot(index="seed", columns="coefficient", values="bias").loc[
            seeds
        ].to_numpy(float)
        statistic = lambda values: float(np.abs(values.mean(axis=0)).mean())
    elif metric == "coefficient_mse":
        matrix = (
            frame.groupby("seed")["squared_error"].mean().reindex(seeds).to_numpy(float)[:, None]
        )
        statistic = lambda values: float(values.mean())
    elif metric == "coverage":
        matrix = frame.groupby("seed")["covered"].mean().reindex(seeds).to_numpy(float)[:, None]
        statistic = lambda values: float(values.mean())
    else:
        raise ValueError(f"Unsupported coefficient metric: {metric}")
    estimate = statistic(matrix)
    if len(seeds) < 2:
        return estimate, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    values_array = np.asarray(
        [
            statistic(matrix[rng.integers(0, len(seeds), size=len(seeds))])
            for _ in range(n_bootstrap)
        ]
    )
    return (
        estimate,
        float(values_array.std(ddof=1)),
        float(np.quantile(values_array, 0.025)),
        float(np.quantile(values_array, 0.975)),
    )


def summarize(
    coefficients: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_term = (
        coefficients.groupby(
            [
                "mnar_strength",
                "mnar_odds_ratio_per_sd",
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
            coefficient_mse=("squared_error", "mean"),
            coverage=("covered", "mean"),
            empirical_se=("estimate", lambda values: float(np.std(values, ddof=1))),
            average_se=("std_error", "mean"),
            n_runs=("seed", "nunique"),
        )
        .reset_index(drop=True)
    )
    by_term["abs_signed_bias"] = by_term["signed_bias"].abs()
    by_term["se_ratio"] = by_term["average_se"] / by_term["empirical_se"].replace(
        0,
        np.nan,
    )

    run_effects = (
        coefficients.groupby(
            [
                "mnar_strength",
                "mnar_odds_ratio_per_sd",
                "method",
                "method_label",
                "seed",
            ],
            as_index=False,
        )
        .agg(
            mean_abs_run_bias=("abs_bias", "mean"),
            coefficient_mse=("squared_error", "mean"),
            coverage=("covered", "mean"),
            average_se=("std_error", "mean"),
        )
    )
    run_effects = run_effects.merge(
        metrics[
            [
                "mnar_strength",
                "method",
                "seed",
                "imputation_rmse",
                "trajectory_summary_rmse",
            ]
        ],
        on=["mnar_strength", "method", "seed"],
        how="left",
    )

    rows: list[dict] = []
    for (strength, method), part in coefficients.groupby(
        ["mnar_strength", "method"]
    ):
        method_label = METHOD_LABELS[method]
        term_part = by_term[
            by_term["mnar_strength"].eq(strength)
            & by_term["method"].eq(method)
        ]
        run_part = run_effects[
            run_effects["mnar_strength"].eq(strength)
            & run_effects["method"].eq(method)
        ]
        metrics_part = metrics[
            metrics["mnar_strength"].eq(strength)
            & metrics["method"].eq(method)
        ]

        for metric_name in ["absolute_bias", "coefficient_mse", "coverage"]:
            estimate, mcse, low, high = _bootstrap_coefficient_metric(
                part,
                metric_name,
                seed=73_001 + int(round(100 * strength)) + 101 * METHOD_ORDER.index(method),
            )
            rows.append(
                {
                    "mnar_strength": strength,
                    "mnar_odds_ratio_per_sd": math.exp(strength),
                    "method": method,
                    "method_label": method_label,
                    "metric": metric_name,
                    "estimate": estimate,
                    "mcse": mcse,
                    "ci_low": low,
                    "ci_high": high,
                    "n_runs": int(part["seed"].nunique()),
                }
            )

        for metric_name, column in [
            ("imputation_rmse", "imputation_rmse"),
            ("trajectory_summary_rmse", "trajectory_summary_rmse"),
        ]:
            values = metrics_part[column].dropna().to_numpy(float)
            estimate = float(np.mean(values)) if len(values) else np.nan
            mcse = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
            rows.append(
                {
                    "mnar_strength": strength,
                    "mnar_odds_ratio_per_sd": math.exp(strength),
                    "method": method,
                    "method_label": method_label,
                    "metric": metric_name,
                    "estimate": estimate,
                    "mcse": mcse,
                    "ci_low": estimate - 1.96 * mcse if np.isfinite(mcse) else np.nan,
                    "ci_high": estimate + 1.96 * mcse if np.isfinite(mcse) else np.nan,
                    "n_runs": int(metrics_part["seed"].nunique()),
                }
            )

        rows.append(
            {
                "mnar_strength": strength,
                "mnar_odds_ratio_per_sd": math.exp(strength),
                "method": method,
                "method_label": method_label,
                "metric": "se_ratio",
                "estimate": float(term_part["se_ratio"].mean()),
                "mcse": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_runs": int(part["seed"].nunique()),
            }
        )
    return by_term, run_effects, pd.DataFrame(rows)


def exclude_unstable_complete_case(
    coefficients: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict] = []
    stable_keys: set[tuple[int, float]] = set()
    complete = coefficients[coefficients["method"].eq("complete_case")]
    for (seed, strength), part in complete.groupby(["seed", "mnar_strength"]):
        n_terms = int(part["coefficient"].nunique())
        n_used = float(part["n_used"].max())
        estimates = part["estimate"].to_numpy(float)
        standard_errors = part["std_error"].to_numpy(float)
        stable = bool(
            n_terms == len(neural.TRUE_BETA)
            and n_used >= 10 * n_terms
            and np.isfinite(estimates).all()
            and np.isfinite(standard_errors).all()
            and np.max(np.abs(estimates)) <= 10.0
            and np.max(np.abs(standard_errors)) <= 10.0
        )
        key = (int(seed), float(strength))
        if stable:
            stable_keys.add(key)
        audit_rows.append(
            {
                "seed": key[0],
                "mnar_strength": key[1],
                "n_terms": n_terms,
                "n_used": n_used,
                "max_abs_estimate": float(np.nanmax(np.abs(estimates))),
                "max_standard_error": float(np.nanmax(np.abs(standard_errors))),
                "stable_for_effect_summary": stable,
            }
        )

    coefficient_keep = ~coefficients["method"].eq("complete_case") | coefficients.apply(
        lambda row: (int(row["seed"]), float(row["mnar_strength"])) in stable_keys,
        axis=1,
    )
    metric_keep = ~metrics["method"].eq("complete_case") | metrics.apply(
        lambda row: (int(row["seed"]), float(row["mnar_strength"])) in stable_keys,
        axis=1,
    )
    return (
        coefficients.loc[coefficient_keep].copy(),
        metrics.loc[metric_keep].copy(),
        pd.DataFrame(audit_rows),
    )


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5), dpi=220)
    panels = [
        ("absolute_bias", "A  Absolute coefficient bias", "lower is better"),
        ("coefficient_mse", "B  Coefficient mean squared error", "lower is better"),
        ("coverage", "C  95% interval coverage", "target = 0.95"),
        ("se_ratio", "D  Average model SE / empirical SE", "target = 1.00"),
        ("imputation_rmse", "E  Missing-cell RMSE", "lower is better"),
        ("trajectory_summary_rmse", "F  Trajectory-summary RMSE", "lower is better"),
    ]
    strengths = sorted(summary["mnar_strength"].unique())
    x_base = np.arange(len(strengths), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(METHOD_ORDER))
    for axis, (metric, title, direction) in zip(axes.flat, panels):
        local = summary[summary["metric"].eq(metric)]
        for method, offset in zip(METHOD_ORDER, offsets):
            if metric in {"imputation_rmse", "trajectory_summary_rmse"} and method in {
                "full_data",
                "complete_case",
            }:
                continue
            part = local[local["method"].eq(method)].set_index("mnar_strength")
            if part.empty:
                continue
            values = np.asarray(
                [part.at[strength, "estimate"] if strength in part.index else np.nan for strength in strengths]
            )
            lows = np.asarray(
                [part.at[strength, "ci_low"] if strength in part.index else np.nan for strength in strengths]
            )
            highs = np.asarray(
                [part.at[strength, "ci_high"] if strength in part.index else np.nan for strength in strengths]
            )
            error = np.maximum(0.0, np.vstack([values - lows, highs - values]))
            finite_error = np.isfinite(error).all(axis=0)
            axis.plot(
                x_base + offset,
                values,
                color=COLORS[method],
                linewidth=1.8,
                marker="o",
                markersize=7.5,
                label=METHOD_LABELS[method],
                zorder=3,
            )
            if finite_error.any():
                axis.errorbar(
                    (x_base + offset)[finite_error],
                    values[finite_error],
                    yerr=error[:, finite_error],
                    fmt="none",
                    ecolor=COLORS[method],
                    elinewidth=1.5,
                    capsize=3,
                    zorder=2,
                )
        if metric == "coverage":
            axis.axhline(0.95, color="#31444B", linestyle="--", linewidth=1.3)
        if metric == "se_ratio":
            axis.axhline(1.0, color="#31444B", linestyle="--", linewidth=1.3)
        axis.set_xticks(x_base)
        axis.set_xticklabels([STRENGTH_LABELS.get(value, f"gamma={value:g}") for value in strengths])
        axis.set_title(title, loc="left", fontsize=13, weight="bold")
        axis.text(
            0.98,
            0.95,
            direction,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            color="#52656D",
            weight="bold",
        )
        axis.grid(axis="y", color="#D7DEE2", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=6,
        frameon=False,
        fontsize=10.5,
    )
    fig.suptitle(
        "Clinical-association sensitivity to additional current-value MNAR dependence",
        fontsize=17,
        weight="bold",
        y=0.998,
    )
    fig.text(
        0.5,
        0.008,
        "Late-cell missingness is fixed at 40%; gamma is the added log-odds coefficient per SD of the unobserved current-cell abnormality.",
        ha="center",
        fontsize=10.5,
        color="#42555D",
    )
    fig.tight_layout(rect=[0.025, 0.04, 0.995, 0.925], h_pad=2.7, w_pad=2.2)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_by_term(by_term: pd.DataFrame, output: Path) -> None:
    methods = ["brits_mi_refined", "mice", "missforest"]
    strengths = sorted(by_term["mnar_strength"].unique())
    term_order = list(neural.TRUE_BETA)
    term_labels = {
        name: neural.COEFFICIENT_LABELS[name] for name in term_order
    }
    fig, axes = plt.subplots(
        len(strengths),
        2,
        figsize=(15.5, 4.2 * len(strengths)),
        dpi=220,
        sharey=True,
    )
    axes = np.atleast_2d(axes)
    y = np.arange(len(term_order))
    offsets = {"brits_mi_refined": -0.20, "mice": 0.0, "missforest": 0.20}
    for row, strength in enumerate(strengths):
        for method in methods:
            local = by_term[
                by_term["mnar_strength"].eq(strength)
                & by_term["method"].eq(method)
            ].set_index("coefficient")
            bias = np.asarray([local.at[name, "abs_signed_bias"] for name in term_order])
            coverage = np.asarray([local.at[name, "coverage"] for name in term_order])
            axes[row, 0].scatter(
                bias,
                y + offsets[method],
                s=42,
                color=COLORS[method],
                label=METHOD_LABELS[method],
                zorder=3,
            )
            axes[row, 1].scatter(
                coverage,
                y + offsets[method],
                s=42,
                color=COLORS[method],
                label=METHOD_LABELS[method],
                zorder=3,
            )
        axes[row, 0].set_title(
            f"{STRENGTH_LABELS.get(strength, strength).replace(chr(10), ' ')}: absolute bias",
            loc="left",
            weight="bold",
        )
        axes[row, 1].set_title(
            f"{STRENGTH_LABELS.get(strength, strength).replace(chr(10), ' ')}: coverage",
            loc="left",
            weight="bold",
        )
        axes[row, 1].axvline(0.95, color="#31444B", linestyle="--", linewidth=1.2)
        for axis in axes[row]:
            axis.grid(axis="x", color="#D7DEE2", linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_yticks(y)
            axis.set_yticklabels([term_labels[name] for name in term_order], fontsize=9.5)
            axis.invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Predictor-specific association recovery under MNAR departures",
        fontsize=16,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.925], h_pad=2.0, w_pad=2.5)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def paired_method_tests(run_effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    proposed = "brits_mi_refined"
    comparators_to_test = ["mice", "missforest", "mean_impute"]
    local = run_effects.copy()
    local["coverage_target_error"] = (local["coverage"] - 0.95).abs()
    metric_columns = [
        "mean_abs_run_bias",
        "coefficient_mse",
        "coverage_target_error",
        "imputation_rmse",
        "trajectory_summary_rmse",
    ]
    for strength in sorted(local["mnar_strength"].unique()):
        strength_data = local[local["mnar_strength"].eq(strength)]
        for comparator in comparators_to_test:
            comparison = strength_data[
                strength_data["method"].isin([proposed, comparator])
            ]
            for metric in metric_columns:
                wide = comparison.pivot(
                    index="seed",
                    columns="method",
                    values=metric,
                ).dropna()
                if proposed not in wide or comparator not in wide or len(wide) < 2:
                    continue
                difference = wide[proposed] - wide[comparator]
                test = stats.ttest_rel(wide[proposed], wide[comparator])
                rows.append(
                    {
                        "mnar_strength": strength,
                        "mnar_odds_ratio_per_sd": math.exp(strength),
                        "comparator": comparator,
                        "comparator_label": METHOD_LABELS[comparator],
                        "metric": metric,
                        "n_pairs": len(wide),
                        "brits_mi_mean": float(wide[proposed].mean()),
                        "comparator_mean": float(wide[comparator].mean()),
                        "mean_paired_difference": float(difference.mean()),
                        "paired_difference_se": float(
                            difference.std(ddof=1) / np.sqrt(len(difference))
                        ),
                        "paired_t_p_value": float(test.pvalue),
                    }
                )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["holm_p_value_within_strength"] = np.nan
    for strength, indices in output.groupby("mnar_strength").groups.items():
        p_values = output.loc[indices, "paired_t_p_value"].to_numpy(float)
        order = np.argsort(p_values)
        adjusted = np.empty_like(p_values)
        running = 0.0
        for rank, position in enumerate(order):
            value = min(1.0, (len(p_values) - rank) * p_values[position])
            running = max(running, value)
            adjusted[position] = running
        output.loc[indices, "holm_p_value_within_strength"] = adjusted
    return output


def plot_missingness_audit(audit: pd.DataFrame, output: Path) -> None:
    summary = (
        audit.groupby("mnar_strength", as_index=False)
        .agg(
            overall=("realized_late_missing", "mean"),
            overall_se=("realized_late_missing", lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))),
            low=("low_abnormality_missing", "mean"),
            low_se=("low_abnormality_missing", lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))),
            high=("high_abnormality_missing", "mean"),
            high_se=("high_abnormality_missing", lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))),
            gap=("high_minus_low_missing", "mean"),
            gap_se=("high_minus_low_missing", lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))),
        )
    )
    strengths = summary["mnar_strength"].to_numpy(float)
    x = np.arange(len(summary), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=220)
    series = [
        ("low", "low_se", "Least abnormal quintile", "#4EA8DE", -0.12),
        ("overall", "overall_se", "All late cells", "#7A8793", 0.0),
        ("high", "high_se", "Most abnormal quintile", "#C2410C", 0.12),
    ]
    for value, error, label, color, offset in series:
        axes[0].errorbar(
            x + offset,
            summary[value],
            yerr=1.96 * summary[error],
            color=color,
            marker="o",
            linewidth=2.0,
            markersize=7,
            capsize=3,
            label=label,
        )
    axes[0].axhline(0.40, color="#31444B", linestyle="--", linewidth=1.2)
    axes[0].set_title("A  Realized late-cell missingness", loc="left", weight="bold")
    axes[0].set_ylabel("Missing proportion")
    axes[0].legend(frameon=False, fontsize=9.5)
    axes[1].errorbar(
        x,
        summary["gap"],
        yerr=1.96 * summary["gap_se"],
        color="#007C83",
        marker="o",
        linewidth=2.3,
        markersize=8,
        capsize=3,
    )
    axes[1].set_title(
        "B  Most-minus-least abnormal missingness",
        loc="left",
        weight="bold",
    )
    axes[1].set_ylabel("Missing-proportion difference")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(
            [STRENGTH_LABELS.get(value, f"gamma={value:g}") for value in strengths]
        )
        axis.grid(axis="y", color="#D7DEE2", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Empirical separation of the MNAR sensitivity mechanisms",
        fontsize=15.5,
        weight="bold",
    )
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.93], w_pad=2.5)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--nsim", type=int, default=100)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--missforest-draws", type=int, default=5)
    parser.add_argument("--target-missing", type=float, default=0.40)
    parser.add_argument("--mnar-strengths", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--seed-start", type=int, default=1_500_000)
    parser.add_argument("--mean-weight", type=float, default=0.10)
    parser.add_argument("--noise-weight", type=float, default=0.70)
    parser.add_argument("--max-epochs", type=int, default=45)
    parser.add_argument("--n-workers", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    coefficient_path = args.outdir / "coefficient_by_run.csv"
    metric_path = args.outdir / "metrics_by_run.csv"
    audit_path = args.outdir / "missingness_audit.csv"
    coefficient_rows = (
        pd.read_csv(coefficient_path).to_dict("records")
        if args.resume and coefficient_path.exists()
        else []
    )
    metric_rows = (
        pd.read_csv(metric_path).to_dict("records")
        if args.resume and metric_path.exists()
        else []
    )
    audit_rows = (
        pd.read_csv(audit_path).to_dict("records")
        if args.resume and audit_path.exists()
        else []
    )
    completed = {
        (int(row["seed"]), float(row["mnar_strength"]))
        for row in metric_rows
        if row.get("method") == "brits_mi_refined"
    }
    tasks = [
        (
            args.seed_start + offset,
            args.n,
            args.m,
            args.missforest_draws,
            args.target_missing,
            strength,
            args.mean_weight,
            args.noise_weight,
            args.max_epochs,
        )
        for offset in range(args.nsim)
        for strength in args.mnar_strengths
        if (args.seed_start + offset, strength) not in completed
    ]

    def collect_result(index: int, task: tuple, result: tuple) -> None:
        coefficients, metrics, audits = result
        coefficient_rows.extend(coefficients)
        metric_rows.extend(metrics)
        audit_rows.extend(audits)
        if index == 1 or index % 5 == 0 or index == len(tasks):
            atomic_csv(pd.DataFrame(coefficient_rows), coefficient_path)
            atomic_csv(pd.DataFrame(metric_rows), metric_path)
            atomic_csv(pd.DataFrame(audit_rows), audit_path)
            print(f"MNAR sensitivity {index}/{len(tasks)}", flush=True)

    if tasks and args.n_workers <= 1:
        for index, task in enumerate(tasks, 1):
            try:
                result = _run_one(task)
            except Exception as exc:
                print(
                    f"failed seed={task[0]} strength={task[5]}: {exc!r}",
                    flush=True,
                )
                continue
            collect_result(index, task, result)
    elif tasks:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {executor.submit(_run_one, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(
                        f"failed seed={task[0]} strength={task[5]}: {exc!r}",
                        flush=True,
                    )
                    continue
                collect_result(index, task, result)

    coefficients = pd.DataFrame(coefficient_rows).drop_duplicates(
        ["seed", "mnar_strength", "method", "coefficient"],
        keep="last",
    )
    metrics = pd.DataFrame(metric_rows).drop_duplicates(
        ["seed", "mnar_strength", "method"],
        keep="last",
    )
    audits = pd.DataFrame(audit_rows).drop_duplicates(
        ["seed", "mnar_strength"],
        keep="last",
    )
    atomic_csv(coefficients, coefficient_path)
    atomic_csv(metrics, metric_path)
    atomic_csv(audits, audit_path)

    analysis_coefficients, analysis_metrics, complete_case_audit = (
        exclude_unstable_complete_case(coefficients, metrics)
    )
    atomic_csv(
        complete_case_audit,
        args.outdir / "complete_case_estimability_audit.csv",
    )
    by_term, run_effects, summary = summarize(
        analysis_coefficients,
        analysis_metrics,
    )
    atomic_csv(by_term, args.outdir / "association_recovery_by_predictor.csv")
    atomic_csv(run_effects, args.outdir / "effect_metrics_by_run.csv")
    atomic_csv(summary, args.outdir / "summary_with_monte_carlo_uncertainty.csv")
    atomic_csv(
        paired_method_tests(run_effects),
        args.outdir / "paired_method_tests.csv",
    )
    plot_summary(summary, args.outdir / "clinical_association_mnar_sensitivity")
    plot_by_term(by_term, args.outdir / "clinical_association_mnar_by_predictor")
    plot_missingness_audit(
        audits,
        args.outdir / "clinical_association_mnar_mechanism_audit",
    )

    config = vars(args).copy()
    config["outdir"] = str(config["outdir"])
    config.update(
        {
            "scenario": SCENARIO,
            "mnar_definition": (
                "The locked informative mechanism is augmented by gamma times the "
                "standardized current-cell abnormality (high AST, high ALT, or low "
                "platelet), a quantity unavailable when that cell is missing."
            ),
            "mnar_odds_ratios_per_sd": {
                str(value): math.exp(value) for value in args.mnar_strengths
            },
            "marginal_control": (
                "The missingness intercept is recalibrated within each replicate and "
                "MNAR strength to preserve the target late-cell missingness."
            ),
            "matched_design": (
                "The same seed generates subjects, trajectories, outcomes, random "
                "missingness perturbations, and Bernoulli uniforms across strengths."
            ),
            "methods": [METHOD_LABELS[name] for name in METHOD_ORDER],
            "brits_mi": (
                "Final refined BRITS-MI: cross-fitted bidirectional RITS-GRU proposal, "
                "downstream classification loss, Missforest conditional-mean anchor, "
                "mean weight 0.10, residual spread weight 0.70."
            ),
            "mice": "R mice package default predictive mean matching; m fixed by design.",
            "missforest": (
                "R missForest package defaults, independently seeded completions, "
                "Rubin pooling."
            ),
            "identification_warning": (
                "This is a stress analysis, not proof of identification under MNAR. "
                "Observed data alone do not identify an unrestricted MNAR mechanism."
            ),
        }
    )
    (args.outdir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    display = summary[
        summary["metric"].isin(
            [
                "absolute_bias",
                "coverage",
                "coefficient_mse",
                "imputation_rmse",
                "trajectory_summary_rmse",
            ]
        )
    ].pivot_table(
        index=["mnar_strength", "method_label"],
        columns="metric",
        values="estimate",
    )
    print(display.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
