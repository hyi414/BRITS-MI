#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SIM_PATH = ROOT / "scripts" / "run_gm_mi_dgp1_grid.py"

SPEC = importlib.util.spec_from_file_location("gm_grid", SIM_PATH)
sim = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(sim)

hist_sim = sim.hist_sim
dgp1 = sim.dgp1
BIOMARKERS = sim.BIOMARKERS
TRUE_BETA = sim.TRUE_BETA
COEFFICIENT_LABELS = sim.COEFFICIENT_LABELS

BASE_METHOD_LABELS = {
    "full_data": "Full data",
    "vc_smc_brits_mi": "VC-SMC-BRITS MI",
    "adaptive_posterior_vc_smc_brits": "Adaptive posterior VC-SMC-BRITS",
    "gm_brits_stochastic_mi": "GM-BRITS stochastic MI",
    "mice": "MICE",
    "missforest": "Missforest",
    "brits_impute_only": "BRITS imputation-only",
}


def _method_label(method: str) -> str:
    if method.startswith("posterior_vc_smc_brits_"):
        suffix = method.removeprefix("posterior_vc_smc_brits_")
        return f"Posterior VC-SMC-BRITS ({suffix.replace('_', '.')})"
    return BASE_METHOD_LABELS.get(method, method)


def _safe_method_suffix(value: float) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", f"{value:.2f}").strip("_")


def _fit_bootstrap_substantive_params(
    draw: pd.DataFrame,
    trajectory_obs: np.ndarray,
    time: np.ndarray,
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    rng: np.random.Generator,
) -> pd.Series | None:
    completed = hist_sim._frame_to_late_trajectory(draw, trajectory_obs)
    features = dgp1._trajectory_features(completed, time)
    design = dgp1._design(features, diabetes, htn, male)
    keep = design.notna().all(axis=1).to_numpy()
    idx = np.flatnonzero(keep)
    if len(idx) < 50:
        return None
    boot = rng.choice(idx, size=len(idx), replace=True)
    try:
        fit = dgp1.sm.GLM(
            outcome[boot],
            dgp1.sm.add_constant(design.iloc[boot].reset_index(drop=True), has_constant="add"),
            family=dgp1.sm.families.Binomial(),
        ).fit(maxiter=80, disp=False)
    except Exception:
        return None
    return fit.params.drop("const")


def _missingness_residual_scale(
    late_obs: pd.DataFrame,
    base_imp: pd.DataFrame,
    history_by_cell: dict[str, pd.DataFrame],
    outcome_imp: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, float]]:
    out: dict[str, tuple[np.ndarray, float]] = {}
    for col in late_obs.columns:
        predictors = [base_imp, history_by_cell[col], outcome_imp]
        predictors.append(hist_sim._history_outcome_interactions(history_by_cell[col], outcome_imp))
        x_base = pd.concat(predictors, axis=1)
        x_base.columns = [f"p{j}" for j in range(x_base.shape[1])]
        missing = late_obs[col].isna().to_numpy()
        miss_features = hist_sim._fit_missingness_propensity(missing, x_base)
        x = pd.concat([x_base, miss_features], axis=1)
        _, resid_sd = hist_sim._linear_prediction_and_resid_sd(late_obs[col].to_numpy(dtype=float), x)
        prob = miss_features["missingness_prob"].to_numpy(dtype=float)
        variance_multiplier = np.sqrt(np.clip(prob / (1.0 - prob), 0.20, 5.00))
        out[col] = variance_multiplier, float(resid_sd)
    return out


def posterior_variance_calibrated_brits_draws(
    seed_draws: list[pd.DataFrame],
    late_obs: pd.DataFrame,
    base_imp: pd.DataFrame,
    history_by_cell: dict[str, pd.DataFrame],
    outcome_imp: pd.DataFrame,
    trajectory_obs: np.ndarray,
    time: np.ndarray,
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    rng: np.random.Generator,
    n_imputations: int,
    variance_factor: float,
) -> list[pd.DataFrame]:
    """Bootstrap/posterior BRITS MI with extra variance for association-relevant cells.

    The seed draws provide the VC-SMC-BRITS conditional mean. This step adds
    model uncertainty through bootstrap outcome-model parameters and adds
    missingness-calibrated residual draws at originally missing cells.
    """
    if not seed_draws:
        return []

    late_missing = late_obs.isna()
    scale_info = _missingness_residual_scale(late_obs, base_imp, history_by_cell, outcome_imp)
    out_draws: list[pd.DataFrame] = []
    for m in range(n_imputations):
        base = seed_draws[m % len(seed_draws)].copy()
        params = _fit_bootstrap_substantive_params(base, trajectory_obs, time, outcome, diabetes, htn, male, rng)
        out = base.copy()
        subject_shift = rng.normal(scale=0.10, size=len(late_obs))

        for local_j, visit in enumerate(range(4, 7)):
            for marker in BIOMARKERS:
                col = f"{marker}_v{visit}"
                missing = late_missing[col].to_numpy()
                if not missing.any():
                    continue
                values = out[col].to_numpy(dtype=float).copy()
                variance_multiplier, resid_sd = scale_info[col]
                role = np.ones(len(values), dtype=float)
                if params is not None:
                    sensitivity = np.abs(hist_sim._cell_score_sensitivity(params, marker, visit, time, diabetes, htn))
                    denom = float(np.nanmedian(sensitivity[missing])) if np.isfinite(sensitivity[missing]).any() else 0.0
                    if denom > 1e-8:
                        role = np.clip(0.70 + 0.55 * sensitivity / denom, 0.70, 2.40)
                scale = variance_factor * 0.18 * resid_sd * variance_multiplier * role
                marker_sd = float(np.nanstd(values))
                if not np.isfinite(marker_sd) or marker_sd <= 0:
                    marker_sd = 1.0
                noise = rng.normal(scale=scale[missing], size=int(missing.sum()))
                noise = noise + subject_shift[missing] * 0.025 * marker_sd
                values[missing] = values[missing] + noise
                out[col] = values
        out_draws.append(out)
    return out_draws


def _build_history_by_cell(
    trajectory_obs: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
) -> dict[str, pd.DataFrame]:
    history_by_cell = {}
    for local_j, j in enumerate(range(3, 6), start=4):
        for k, name in enumerate(BIOMARKERS):
            history_by_cell[f"{name}_v{local_j}"] = hist_sim._history_features(trajectory_obs, observed, time, j, k)
    return history_by_cell


def _make_vc_smc_brits_draws(
    late_obs_cells: pd.DataFrame,
    base_imp: pd.DataFrame,
    history_by_cell: dict[str, pd.DataFrame],
    outcome_imp: pd.DataFrame,
    trajectory_obs: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    rng: np.random.Generator,
    n_imputations: int,
) -> tuple[dict[str, list[pd.DataFrame]], list[pd.DataFrame]]:
    core = sim._make_core_methods(
        late_obs_cells,
        base_imp,
        history_by_cell,
        outcome_imp,
        trajectory_obs,
        observed,
        time,
        rng,
        n_imputations,
    )
    smc_brits = hist_sim._targeted_association_fluctuation(
        core["gm_brits_stochastic_mi"],
        trajectory_obs,
        observed,
        time,
        outcome,
        diabetes,
        htn,
        male,
        dgp1._trajectory_features,
        dgp1._design,
        rng,
        n_steps=1,
        step_size=0.028,
    )
    vc_smc_brits = hist_sim._missingness_calibrated_stochastic_draws(
        smc_brits,
        late_obs_cells,
        base_imp,
        history_by_cell,
        outcome_imp,
        time,
        rng,
        use_attention=False,
    )
    return core, vc_smc_brits


def _artificial_late_mask(
    observed: np.ndarray,
    rng: np.random.Generator,
    fraction: float,
    min_per_cell: int = 20,
) -> np.ndarray:
    late_observed = observed[:, -3:, :]
    mask = np.zeros_like(late_observed, dtype=bool)
    for local_j in range(late_observed.shape[1]):
        for k in range(late_observed.shape[2]):
            idx = np.flatnonzero(late_observed[:, local_j, k])
            if len(idx) <= 2 * min_per_cell:
                continue
            n_hide = max(min_per_cell, int(round(fraction * len(idx))))
            n_hide = min(n_hide, len(idx) - min_per_cell)
            chosen = rng.choice(idx, size=n_hide, replace=False)
            mask[chosen, local_j, k] = True
    return mask


def _choose_adaptive_variance_factor(
    trajectory: np.ndarray,
    observed: np.ndarray,
    true_features: pd.DataFrame,
    latent: dict[str, np.ndarray],
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    candidate_factors: tuple[float, ...],
    calibration_fraction: float,
    calibration_draws: int,
    adaptive_target: float,
) -> tuple[float, pd.DataFrame]:
    """Select posterior spread using hidden-observed late biomarker cells."""
    validation_mask = _artificial_late_mask(observed, rng, calibration_fraction)
    if not validation_mask.any():
        default_factor = float(np.median(candidate_factors))
        return default_factor, pd.DataFrame(
            [{"factor": default_factor, "validation_coverage": np.nan, "validation_rmse": np.nan, "selected": True}]
        )

    observed_calib = observed.copy()
    observed_calib[:, -3:, :] = observed_calib[:, -3:, :] & ~validation_mask
    trajectory_calib_obs = np.where(observed_calib, trajectory, np.nan)
    base_imp_calib = dgp1._build_base_frame(trajectory_calib_obs, observed_calib, diabetes, htn, male, time)
    outcome_imp_calib = dgp1._outcome_features(true_features, latent, outcome, base_imp_calib, diabetes, htn, male, rng)
    late_calib_cells = hist_sim._cell_frame(trajectory_calib_obs)
    history_calib = _build_history_by_cell(trajectory_calib_obs, observed_calib, time)
    _, vc_seed_draws = _make_vc_smc_brits_draws(
        late_calib_cells,
        base_imp_calib,
        history_calib,
        outcome_imp_calib,
        trajectory_calib_obs,
        observed_calib,
        time,
        outcome,
        diabetes,
        htn,
        male,
        rng,
        calibration_draws,
    )

    rows: list[dict] = []
    truth = trajectory[:, -3:, :][validation_mask]
    for factor in candidate_factors:
        draws = posterior_variance_calibrated_brits_draws(
            vc_seed_draws,
            late_calib_cells,
            base_imp_calib,
            history_calib,
            outcome_imp_calib,
            trajectory_calib_obs,
            time,
            outcome,
            diabetes,
            htn,
            male,
            rng,
            calibration_draws,
            variance_factor=factor,
        )
        stack = np.stack(
            [hist_sim._frame_to_late_trajectory(draw, trajectory_calib_obs)[:, -3:, :] for draw in draws],
            axis=0,
        )
        pred = stack[:, validation_mask]
        mean_pred = pred.mean(axis=0)
        lo, hi = np.quantile(pred, [0.025, 0.975], axis=0)
        coverage = float(np.mean((lo <= truth) & (truth <= hi)))
        rmse = float(np.sqrt(np.mean((mean_pred - truth) ** 2)))
        width = float(np.mean(hi - lo))
        rows.append(
            {
                "factor": float(factor),
                "validation_coverage": coverage,
                "validation_rmse": rmse,
                "validation_width": width,
                "n_validation_cells": int(validation_mask.sum()),
            }
        )
    calib = pd.DataFrame(rows)
    # Hidden-cell prediction intervals are intentionally stricter than
    # downstream coefficient intervals. Select an elbow: keep candidates within
    # a small tolerance of the best validation coverage, then penalize RMSE,
    # interval width, and avoidable variance inflation. This avoids the old
    # behavior of always selecting the largest variance factor while preventing
    # the tuned selector from choosing clearly under-covered low-variance draws.
    eps = 1e-8
    rmse_span = max(float(calib["validation_rmse"].max() - calib["validation_rmse"].min()), eps)
    width_span = max(float(calib["validation_width"].max() - calib["validation_width"].min()), eps)
    factor_span = max(float(calib["factor"].max() - calib["factor"].min()), eps)
    calib["rmse_scaled"] = (calib["validation_rmse"] - float(calib["validation_rmse"].min())) / rmse_span
    calib["width_scaled"] = (calib["validation_width"] - float(calib["validation_width"].min())) / width_span
    calib["factor_scaled"] = (calib["factor"] - float(calib["factor"].min())) / factor_span
    max_coverage = float(calib["validation_coverage"].max())
    coverage_floor = min(max_coverage, adaptive_target) - 0.035
    near_best = calib[calib["validation_coverage"] >= coverage_floor].copy()
    if near_best.empty:
        near_best = calib.copy()
    near_best["coverage_gap"] = max_coverage - near_best["validation_coverage"]
    near_best["score"] = (
        0.55 * near_best["coverage_gap"]
        + 0.28 * near_best["rmse_scaled"]
        + 0.12 * near_best["width_scaled"]
        + 0.08 * near_best["factor_scaled"]
    )
    calib["score"] = np.nan
    calib.loc[near_best.index, "score"] = near_best["score"]
    selected_idx = int(near_best["score"].idxmin())
    selected_factor = float(calib.loc[selected_idx, "factor"])
    calib["selected"] = calib.index == selected_idx
    return selected_factor, calib


def run_one_test(
    seed: int,
    n: int,
    n_imputations: int,
    target_missing: float,
    scenario: str,
    variance_factors: tuple[float, ...],
    include_adaptive: bool = False,
    calibration_fraction: float = 0.18,
    calibration_draws: int = 20,
    adaptive_target: float = 0.86,
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
    history_by_cell = _build_history_by_cell(trajectory_obs, observed, time)

    core, vc_smc_brits = _make_vc_smc_brits_draws(
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
    methods: dict[str, list[pd.DataFrame]] = {
        "full_data": [late_true_cells],
        "vc_smc_brits_mi": vc_smc_brits,
        "gm_brits_stochastic_mi": core["gm_brits_stochastic_mi"],
        "mice": core["mice"],
        "missforest": core["missforest"],
        "brits_impute_only": core["brits_impute_only"],
    }
    for factor in variance_factors:
        methods[f"posterior_vc_smc_brits_{_safe_method_suffix(factor)}"] = posterior_variance_calibrated_brits_draws(
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
            variance_factor=factor,
        )
    selected_factor = np.nan
    selected_calibration = pd.DataFrame()
    if include_adaptive:
        selected_factor, selected_calibration = _choose_adaptive_variance_factor(
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
        methods["adaptive_posterior_vc_smc_brits"] = posterior_variance_calibrated_brits_draws(
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

    coef_rows: list[dict] = []
    metric_rows: list[dict] = []
    missing_late = ~observed[:, -3:, :]
    true_late_cells = trajectory[:, -3:, :]
    for method, cell_draws in methods.items():
        completed_trajectories = [hist_sim._frame_to_late_trajectory(draw, trajectory_obs) for draw in cell_draws]
        feature_draws = [dgp1._trajectory_features(comp, time) for comp in completed_trajectories]
        pooled = dgp1._pool_logit_fits(feature_draws, outcome, diabetes, htn, male)
        if pooled is not None:
            params, bse, within_se, between_sd, ci, n_used, n_fit_draws = pooled
            for name, truth in TRUE_BETA.items():
                lo, hi = ci.loc[name]
                coef_rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "scenario_label": sim.SCENARIOS[scenario]["label"],
                        "target_missing": target_missing,
                        "n_imputations": n_imputations,
                        "method": method,
                        "method_label": _method_label(method),
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

        class_metrics = dgp1._classification_metrics(feature_draws, outcome, diabetes, htn, male, seed)
        completed_mean = np.mean(np.stack([comp[:, -3:, :] for comp in completed_trajectories], axis=0), axis=0)
        impute_rmse = 0.0 if method == "full_data" else float(
            np.sqrt(np.nanmean((completed_mean[missing_late] - true_late_cells[missing_late]) ** 2))
        )
        summary_mse = float(np.nanmean((feature_draws[0].to_numpy() - true_features.to_numpy()) ** 2))
        metric_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "scenario_label": sim.SCENARIOS[scenario]["label"],
                "target_missing": target_missing,
                "n_imputations": n_imputations,
                "method": method,
                "method_label": _method_label(method),
                "imputation_rmse": impute_rmse,
                "summary_mse": summary_mse,
                "selected_variance_factor": (
                    float(selected_factor) if method == "adaptive_posterior_vc_smc_brits" else np.nan
                ),
                **class_metrics,
            }
        )

    audit = {
        "seed": seed,
        "scenario": scenario,
        "target_missing": target_missing,
        "n_imputations": n_imputations,
        "n": int(n),
        "outcome_rate": float(outcome.mean()),
        "late_cell_missing_rate": float(missing_late.mean()),
        "cell_missing_rate": float((~observed).mean()),
        "complete_late_rate": float(observed[:, -3:, :].all(axis=(1, 2)).mean()),
        "selected_variance_factor": float(selected_factor) if include_adaptive else np.nan,
    }
    if include_adaptive and not selected_calibration.empty:
        selected = selected_calibration[selected_calibration["selected"]].iloc[0]
        audit |= {
            "validation_coverage": float(selected["validation_coverage"]),
            "validation_rmse": float(selected["validation_rmse"]),
            "validation_width": float(selected.get("validation_width", np.nan)),
            "n_validation_cells": int(selected.get("n_validation_cells", 0)),
        }
    return coef_rows, metric_rows, audit


def _run_task(task: tuple[int, int, int, float, str, tuple[float, ...], bool, float, int, float]) -> tuple[list[dict], list[dict], dict]:
    return run_one_test(*task)


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
            mean_abs_bias=("abs_bias", "mean"),
            coefficient_mse=("squared_error", "mean"),
            coverage=("covered", "mean"),
            empirical_se=("estimate", lambda x: float(np.std(x, ddof=1))),
            average_se=("std_error", "mean"),
            within_se=("within_std_error", "mean"),
            between_imputation_sd=("between_imputation_sd", "mean"),
            n_runs=("seed", "nunique"),
        )
        .reset_index(drop=True)
    )
    by_term["abs_signed_bias"] = by_term["signed_bias"].abs()
    by_term["se_ratio"] = by_term["average_se"] / by_term["empirical_se"].replace(0, np.nan)

    coef_overall = (
        by_term.groupby(["scenario", "scenario_label", "target_missing", "n_imputations", "method", "method_label"], as_index=False)
        .agg(
            mean_abs_signed_bias=("abs_signed_bias", "mean"),
            mean_abs_run_bias=("mean_abs_bias", "mean"),
            coefficient_mse=("coefficient_mse", "mean"),
            coverage=("coverage", "mean"),
            empirical_se=("empirical_se", "mean"),
            average_se=("average_se", "mean"),
            se_ratio=("se_ratio", "mean"),
            between_imputation_sd=("between_imputation_sd", "mean"),
            n_runs=("n_runs", "min"),
        )
    )
    metric_summary = (
        metric_df.groupby(["scenario", "scenario_label", "target_missing", "n_imputations", "method", "method_label"], as_index=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auprc_mean=("auprc", "mean"),
            log_loss_mean=("log_loss", "mean"),
            imputation_rmse_mean=("imputation_rmse", "mean"),
            summary_mse_mean=("summary_mse", "mean"),
        )
    )
    overall = coef_overall.merge(
        metric_summary,
        on=["scenario", "scenario_label", "target_missing", "n_imputations", "method", "method_label"],
        how="left",
    )
    return by_term, overall


def plot_test_summary(overall: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        ("mean_abs_signed_bias", "Mean absolute signed bias"),
        ("coverage", "95% CI coverage"),
        ("average_se", "Average model SE"),
        ("imputation_rmse_mean", "Imputation RMSE"),
    ]
    scenarios = list(overall[["scenario", "scenario_label"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(len(metrics), len(scenarios), figsize=(14.5, 12.5), dpi=220, constrained_layout=True)
    if len(scenarios) == 1:
        axes = np.asarray(axes).reshape(len(metrics), 1)
    color_map = {
        "vc_smc_brits_mi": "#005f73",
        "gm_brits_stochastic_mi": "#ca6702",
        "mice": "#7b2cbf",
        "missforest": "#2a9d8f",
        "brits_impute_only": "#9b2226",
    }
    posterior_colors = ["#0a9396", "#94d2bd", "#ee9b00", "#bb3e03", "#ae2012"]
    posterior_methods = [m for m in overall["method"].unique() if m.startswith("posterior_vc_smc_brits_")]
    color_map.update({m: posterior_colors[i % len(posterior_colors)] for i, m in enumerate(sorted(posterior_methods))})
    for r, (metric, ylabel) in enumerate(metrics):
        for c, (scenario, label) in enumerate(scenarios):
            ax = axes[r, c]
            data = overall[overall["scenario"] == scenario].copy()
            data = data[data["method"] != "full_data"].sort_values(metric)
            y = np.arange(len(data))
            ax.barh(
                y,
                data[metric],
                color=[color_map.get(m, "#6b7280") for m in data["method"]],
                edgecolor="white",
                linewidth=0.8,
            )
            ax.set_yticks(y)
            ax.set_yticklabels(data["method_label"], fontsize=8)
            ax.invert_yaxis()
            ax.set_title(label if r == 0 else "", fontsize=12, weight="bold")
            ax.set_xlabel(ylabel, fontsize=9)
            if metric == "coverage":
                ax.axvline(0.95, color="#222222", linestyle="--", linewidth=1.0)
                ax.set_xlim(0.70, 1.0)
            ax.grid(axis="x", color="#dddddd", linewidth=0.7)
            ax.set_axisbelow(True)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
    fig.suptitle("Posterior VC-SMC-BRITS variance-calibration test", fontsize=16, weight="bold")
    fig.savefig(outdir / "posterior_vc_smc_brits_test_summary.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("outputs/posterior_vc_smc_brits_test"))
    parser.add_argument("--nsim", type=int, default=20)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--m", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=260000)
    parser.add_argument("--missing", type=float, default=0.60)
    parser.add_argument("--scenarios", nargs="+", choices=list(sim.SCENARIOS), default=list(sim.SCENARIOS))
    parser.add_argument("--variance-factors", type=float, nargs="+", default=[0.50, 0.90, 1.30, 1.70])
    parser.add_argument("--include-adaptive", action="store_true")
    parser.add_argument("--calibration-fraction", type=float, default=0.18)
    parser.add_argument("--calibration-draws", type=int, default=20)
    parser.add_argument("--adaptive-target", type=float, default=0.86)
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    factors = tuple(float(x) for x in args.variance_factors)
    tasks: list[tuple[int, int, int, float, str, tuple[float, ...], bool, float, int, float]] = []
    job = 0
    for scenario in args.scenarios:
        for rep in range(args.nsim):
            tasks.append(
                (
                    args.seed_start + job * 1000 + rep,
                    args.n,
                    args.m,
                    args.missing,
                    scenario,
                    factors,
                    args.include_adaptive,
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
    plot_test_summary(overall, args.outdir)
    with open(args.outdir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(vars(args) | {"variance_factors": list(factors)}, fh, indent=2, default=str)

    display = overall.sort_values(["scenario", "mean_abs_signed_bias"])
    print(
        display[
            [
                "scenario_label",
                "method_label",
                "mean_abs_signed_bias",
                "coverage",
                "empirical_se",
                "average_se",
                "se_ratio",
                "between_imputation_sd",
                "imputation_rmse_mean",
                "summary_mse_mean",
                "auroc_mean",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()
