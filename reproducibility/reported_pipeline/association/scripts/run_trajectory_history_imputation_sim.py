#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
COEF_SIM_PATH = ROOT / "scripts" / "run_outcome_aware_coefficient_sim.py"
SPEC = importlib.util.spec_from_file_location("coefsim", COEF_SIM_PATH)
coefsim = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(coefsim)


BIOMARKERS = coefsim.BIOMARKERS
METHOD_LABELS = {
    "full_data": "Full data",
    "tap_brits_outcome_history": "TAP-BRITS outcome + history",
    "tap_saits_outcome_history": "TAP-SAITS outcome + history",
    "cf_tap_brits_outcome_history": "Cross-fit TAP-BRITS",
    "cf_tap_saits_outcome_history": "Cross-fit TAP-SAITS",
    "gm_brits_stochastic_mi": "GM-BRITS stochastic MI",
    "gm_saits_stochastic_mi": "GM-SAITS stochastic MI",
    "stacked_deep_mice": "Stacked deep + MICE",
    "stacked_brits_saits": "Stacked BRITS + SAITS",
    "brits_outcome_history": "BRITS outcome + history",
    "saits_outcome_history": "SAITS-style outcome + history",
    "brits_outcome": "BRITS outcome",
    "brits_history_impute_only": "BRITS history only",
    "saits_impute_only": "SAITS-style impute only",
    "brits_impute_only": "BRITS impute only",
    "mice": "MICE",
    "missforest": "Missforest",
    "mean_impute": "Mean imputation",
    "complete_case": "Complete case",
}
METHOD_ORDER = tuple(METHOD_LABELS)
NORMAL_CRITICAL_VALUE = coefsim.NORMAL_CRITICAL_VALUE
BRITS_RESIDUAL_SCALE = coefsim.BRITS_RESIDUAL_SCALE
MICE_RESIDUAL_SCALE = coefsim.MICE_RESIDUAL_SCALE
MISSFOREST_RESIDUAL_SCALE = coefsim.MISSFOREST_RESIDUAL_SCALE


def _late_summary(trajectory: np.ndarray) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return coefsim._late_frame(np.nanmean(trajectory[:, -3:, :], axis=1))


def _late_summary_feature_builder(trajectory: np.ndarray, time: np.ndarray) -> pd.DataFrame:
    del time
    return _late_summary(trajectory)


def _late_summary_design_builder(
    features: pd.DataFrame,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
) -> pd.DataFrame:
    return coefsim._design(features, diabetes, htn, male)


def _cell_frame(trajectory: np.ndarray) -> pd.DataFrame:
    data = {}
    for local_j, j in enumerate(range(3, 6), start=4):
        for k, name in enumerate(BIOMARKERS):
            data[f"{name}_v{local_j}"] = trajectory[:, j, k]
    return pd.DataFrame(data)


def _frame_to_late_trajectory(frame: pd.DataFrame, base_trajectory: np.ndarray) -> np.ndarray:
    out = base_trajectory.copy()
    for local_j, j in enumerate(range(3, 6), start=4):
        for k, name in enumerate(BIOMARKERS):
            out[:, j, k] = frame[f"{name}_v{local_j}"].to_numpy(dtype=float)
    return out


def _history_features(
    trajectory_obs: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    target_j: int,
    target_k: int,
) -> pd.DataFrame:
    n = trajectory_obs.shape[0]
    before = np.arange(target_j)
    data: dict[str, np.ndarray] = {}
    marker_name = BIOMARKERS[target_k]

    for k, name in enumerate(BIOMARKERS):
        vals = trajectory_obs[:, before, k]
        mask = observed[:, before, k]
        count = mask.sum(axis=1).astype(float)
        miss_frac = 1.0 - count / max(len(before), 1)
        filled = np.where(mask, vals, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(filled, axis=1)
            std = np.nanstd(filled, axis=1)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std), std, 0.0)

        last = np.zeros(n)
        first = np.zeros(n)
        last_gap = np.repeat(time[target_j] + 1.0, n)
        slope = np.zeros(n)
        for i in range(n):
            idx = np.flatnonzero(mask[i])
            if idx.size:
                first_idx = idx[0]
                last_idx = idx[-1]
                first[i] = vals[i, first_idx]
                last[i] = vals[i, last_idx]
                last_gap[i] = time[target_j] - time[last_idx]
                if idx.size >= 2:
                    denom = max(time[last_idx] - time[first_idx], 1e-6)
                    slope[i] = (vals[i, last_idx] - vals[i, first_idx]) / denom

        prefix = "target" if k == target_k else name
        data[f"{prefix}_hist_mean"] = mean
        data[f"{prefix}_hist_std"] = std
        data[f"{prefix}_hist_last"] = last
        data[f"{prefix}_hist_slope"] = slope
        data[f"{prefix}_hist_missing_fraction"] = miss_frac
        data[f"{prefix}_hist_count"] = count
        data[f"{prefix}_hist_gap_to_last"] = last_gap

    data["target_marker_ast"] = np.repeat(float(marker_name == "ast"), n)
    data["target_marker_alt"] = np.repeat(float(marker_name == "alt"), n)
    data["target_marker_platelet"] = np.repeat(float(marker_name == "platelet"), n)
    data["target_time"] = np.repeat(float(time[target_j]), n)
    data["target_visit_index"] = np.repeat(float(target_j), n)
    return pd.DataFrame(data)


def _outcome_image_features(
    late_true: pd.DataFrame,
    latent: dict[str, np.ndarray],
    outcome: np.ndarray,
    base_imp: pd.DataFrame,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    residual = coefsim._outcome_residual(outcome, base_imp)
    late_trend = latent.get("late_trend", np.zeros(len(outcome)))
    image_ast = (
        0.58 * late_true["late_ast"].to_numpy()
        + 0.30 * latent["late_activity"]
        + 0.28 * late_trend
        + rng.normal(scale=0.70, size=len(outcome))
    )
    image_alt = (
        0.55 * late_true["late_alt"].to_numpy()
        + 0.26 * latent["late_activity"]
        + 0.24 * late_trend
        + rng.normal(scale=0.72, size=len(outcome))
    )
    image_platelet = (
        0.58 * late_true["late_platelet"].to_numpy()
        - 0.26 * latent["late_activity"]
        - 0.28 * late_trend
        + rng.normal(scale=0.70, size=len(outcome))
    )
    image_activity = 0.70 * latent["late_activity"] + 0.46 * late_trend + rng.normal(scale=0.62, size=len(outcome))
    image_aux = pd.DataFrame(
        {
            "outcome_residual": residual,
            "outcome_residual_x_diabetes": residual * diabetes,
            "outcome_residual_x_htn": residual * htn,
            "outcome_residual_x_male": residual * male,
            "image_ast_texture_score": image_ast,
            "image_alt_texture_score": image_alt,
            "image_platelet_texture_score": image_platelet,
            "image_ast_x_diabetes_score": image_ast * diabetes,
            "image_alt_x_diabetes_score": image_alt * diabetes,
            "image_platelet_x_htn_score": image_platelet * htn,
            "image_fibrosis_score": (
                0.42 * image_ast
                + 0.34 * image_alt
                - 0.44 * image_platelet
                + 0.42 * image_activity
                + rng.normal(scale=0.25, size=len(outcome))
            ),
            "image_late_activity_score": image_activity,
            "image_late_activity_x_diabetes": image_activity * diabetes,
            "image_late_activity_x_htn": image_activity * htn,
        }
    )
    return image_aux


def _linear_cell_impute(
    late_obs: pd.DataFrame,
    base_imp: pd.DataFrame,
    history_by_cell: dict[str, pd.DataFrame],
    outcome_imp: pd.DataFrame | None,
    rng: np.random.Generator,
    residual_scale: float,
    use_history: bool,
) -> pd.DataFrame:
    out = late_obs.copy()
    for col in late_obs.columns:
        predictors = [base_imp]
        if use_history:
            predictors.append(history_by_cell[col])
        if outcome_imp is not None:
            predictors.append(outcome_imp)
        x = pd.concat(predictors, axis=1)
        out[col] = coefsim._linear_impute(late_obs[col].to_numpy(dtype=float), x, rng, residual_scale)
    return out


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def _prepare_numeric_predictors(predictors: pd.DataFrame) -> pd.DataFrame:
    x = predictors.copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    for col in x.columns:
        values = x[col].to_numpy(dtype=float)
        if np.isnan(values).any():
            fill = float(np.nanmean(values)) if np.isfinite(np.nanmean(values)) else 0.0
            values = np.where(np.isfinite(values), values, fill)
        sd = float(np.std(values))
        if np.isfinite(sd) and sd > 1e-8:
            values = (values - float(np.mean(values))) / sd
        else:
            values = values * 0.0
        x[col] = values
    x.columns = [f"x{j}" for j in range(x.shape[1])]
    return x


def _fit_missingness_propensity(missing: np.ndarray, predictors: pd.DataFrame) -> pd.DataFrame:
    missing = np.asarray(missing, dtype=float)
    x = _prepare_numeric_predictors(predictors)
    mean_missing = float(np.mean(missing))
    if mean_missing <= 1e-6 or mean_missing >= 1.0 - 1e-6:
        prob = np.repeat(np.clip(mean_missing, 1e-4, 1.0 - 1e-4), len(missing))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = sm.GLM(missing, sm.add_constant(x, has_constant="add"), family=sm.families.Binomial()).fit(
                    maxiter=100, disp=False
                )
                prob = np.asarray(fit.predict(sm.add_constant(x, has_constant="add")), dtype=float)
            except Exception:
                prob = np.repeat(np.clip(mean_missing, 1e-4, 1.0 - 1e-4), len(missing))
    prob = np.clip(prob, 1e-4, 1.0 - 1e-4)
    return pd.DataFrame(
        {
            "missingness_prob": prob,
            "missingness_logit": _safe_logit(prob),
            "missingness_residual": missing - prob,
            "missingness_entropy": -(prob * np.log(prob) + (1.0 - prob) * np.log(1.0 - prob)),
        },
        index=predictors.index,
    )


def _posterior_linear_impute(
    target: np.ndarray,
    predictors: pd.DataFrame,
    rng: np.random.Generator,
    coefficient_scale: float = 0.75,
    residual_scale: float = 1.00,
) -> np.ndarray:
    observed = np.isfinite(target)
    out = target.copy()
    if observed.sum() < 20:
        fill = float(np.nanmean(target)) if np.isfinite(np.nanmean(target)) else 0.0
        out[~observed] = fill + rng.normal(scale=0.20, size=int((~observed).sum()))
        return out

    x = _prepare_numeric_predictors(predictors)
    x_all = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    x_obs = x_all[observed]
    y_obs = target[observed]
    ridge = 1e-2 * np.eye(x_obs.shape[1])
    xtx_inv = np.linalg.pinv(x_obs.T @ x_obs + ridge)
    beta_hat = xtx_inv @ x_obs.T @ y_obs
    resid = y_obs - x_obs @ beta_hat
    df = max(len(y_obs) - x_obs.shape[1], 1)
    sigma2 = float(max(np.sum(resid**2) / df, 1e-8))
    sigma2_draw = sigma2 * df / max(rng.chisquare(df), 1e-6)
    cov = coefficient_scale**2 * sigma2_draw * xtx_inv
    cov = (cov + cov.T) / 2.0
    try:
        beta_draw = rng.multivariate_normal(beta_hat, cov, check_valid="ignore")
    except Exception:
        beta_draw = beta_hat + rng.normal(scale=np.sqrt(np.maximum(np.diag(cov), 1e-10)))
    pred = x_all @ beta_draw
    draw = pred[~observed] + rng.normal(
        scale=residual_scale * np.sqrt(sigma2_draw),
        size=int((~observed).sum()),
    )
    q_low, q_high = np.nanquantile(y_obs, [0.01, 0.99])
    support_sd = float(np.nanstd(y_obs))
    buffer = max(0.20, 1.50 * support_sd if np.isfinite(support_sd) else 0.20)
    out[~observed] = np.clip(draw, float(q_low - buffer), float(q_high + buffer))
    return out


def _linear_prediction_and_resid_sd(target: np.ndarray, predictors: pd.DataFrame) -> tuple[np.ndarray, float]:
    observed = np.isfinite(target)
    if observed.sum() < 20:
        fill = float(np.nanmean(target)) if np.isfinite(np.nanmean(target)) else 0.0
        return np.repeat(fill, len(target)), 0.20
    x = _prepare_numeric_predictors(predictors)
    x_all = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    x_obs = x_all[observed]
    y_obs = target[observed]
    ridge = 1e-2 * np.eye(x_obs.shape[1])
    beta = np.linalg.pinv(x_obs.T @ x_obs + ridge) @ x_obs.T @ y_obs
    pred = x_all @ beta
    resid = y_obs - x_obs @ beta
    resid_sd = float(np.sqrt(max(np.sum(resid**2) / max(len(y_obs) - x_obs.shape[1], 1), 1e-8)))
    return pred, resid_sd


def _history_outcome_interactions(history: pd.DataFrame, outcome_imp: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=history.index)
    residual = outcome_imp["outcome_residual"].to_numpy()
    image_score = outcome_imp["image_fibrosis_score"].to_numpy()
    image_activity = outcome_imp["image_late_activity_score"].to_numpy()
    for col in (
        "target_hist_last",
        "target_hist_slope",
        "target_hist_mean",
        "target_hist_missing_fraction",
        "target_hist_gap_to_last",
    ):
        value = history[col].to_numpy()
        out[f"{col}_x_outcome_residual"] = value * residual
        out[f"{col}_x_image_fibrosis"] = value * image_score
    out["target_hist_slope_x_image_activity"] = history["target_hist_slope"].to_numpy() * image_activity
    out["target_hist_missing_fraction_x_image_activity"] = (
        history["target_hist_missing_fraction"].to_numpy() * image_activity
    )
    return out


def _parse_late_cell(col: str) -> tuple[str, int]:
    marker, visit = col.rsplit("_v", 1)
    return marker, int(visit)


def _attention_context_features(
    late_obs: pd.DataFrame,
    target_col: str,
    time: np.ndarray,
) -> pd.DataFrame:
    """SAITS-style context features from all late cells except the target cell.

    This is a compact tabular analogue of diagonally masked self-attention: a
    target cell cannot use its own value, but it can use observed same-marker,
    same-time, and cross-marker late context plus the corresponding masks.
    """
    filled = late_obs.copy()
    masks = late_obs.notna().astype(float)
    for col in filled.columns:
        filled[col] = filled[col].fillna(float(filled[col].mean()))

    target_marker, target_visit = _parse_late_cell(target_col)
    target_j = target_visit - 1
    directions = {"ast": 1.0, "alt": 1.0, "platelet": -1.0}
    data: dict[str, np.ndarray] = {}

    context_cols = [col for col in late_obs.columns if col != target_col]
    for col in context_cols:
        data[f"ctx_{col}"] = filled[col].to_numpy(dtype=float)
        data[f"ctx_{col}_observed"] = masks[col].to_numpy(dtype=float)

    same_marker_cols = [col for col in context_cols if _parse_late_cell(col)[0] == target_marker]
    same_time_cols = [col for col in context_cols if _parse_late_cell(col)[1] == target_visit]

    def _masked_mean(cols: list[str]) -> np.ndarray:
        if not cols:
            return np.zeros(len(late_obs))
        values = np.column_stack([filled[col].to_numpy(dtype=float) for col in cols])
        obs = np.column_stack([masks[col].to_numpy(dtype=float) for col in cols])
        denom = obs.sum(axis=1)
        return np.divide((values * obs).sum(axis=1), denom, out=np.zeros(len(late_obs)), where=denom > 0)

    data["attn_same_marker_mean"] = _masked_mean(same_marker_cols)
    data["attn_same_time_mean"] = _masked_mean(same_time_cols)
    data["attn_observed_fraction"] = masks[context_cols].mean(axis=1).to_numpy(dtype=float)
    data["attn_same_marker_observed_fraction"] = (
        masks[same_marker_cols].mean(axis=1).to_numpy(dtype=float) if same_marker_cols else np.zeros(len(late_obs))
    )
    data["attn_same_time_observed_fraction"] = (
        masks[same_time_cols].mean(axis=1).to_numpy(dtype=float) if same_time_cols else np.zeros(len(late_obs))
    )

    weights = []
    signed_values = []
    raw_values = []
    obs_values = []
    for col in context_cols:
        marker, visit = _parse_late_cell(col)
        source_j = visit - 1
        time_weight = np.exp(-abs(float(time[target_j] - time[source_j])) / 0.30)
        marker_weight = 1.00 if marker == target_marker else 0.74 if {marker, target_marker} <= {"ast", "alt"} else 0.62
        weight = float(time_weight * marker_weight)
        weights.append(weight)
        value = filled[col].to_numpy(dtype=float)
        raw_values.append(value)
        signed_values.append(value * directions[marker] * directions[target_marker])
        obs_values.append(masks[col].to_numpy(dtype=float))

    weight_arr = np.asarray(weights, dtype=float)[None, :]
    obs = np.column_stack(obs_values)
    raw = np.column_stack(raw_values)
    signed = np.column_stack(signed_values)
    denom = (obs * weight_arr).sum(axis=1)
    data["attn_raw_context"] = np.divide(
        (raw * obs * weight_arr).sum(axis=1), denom, out=np.zeros(len(late_obs)), where=denom > 0
    )
    data["attn_disease_direction_context"] = np.divide(
        (signed * obs * weight_arr).sum(axis=1), denom, out=np.zeros(len(late_obs)), where=denom > 0
    )
    data["attn_weighted_observed_mass"] = denom
    return pd.DataFrame(data, index=late_obs.index)


def _attention_outcome_interactions(context: pd.DataFrame, outcome_imp: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=context.index)
    residual = outcome_imp["outcome_residual"].to_numpy()
    image_score = outcome_imp["image_fibrosis_score"].to_numpy()
    image_activity = outcome_imp["image_late_activity_score"].to_numpy()
    for col in (
        "attn_same_marker_mean",
        "attn_same_time_mean",
        "attn_raw_context",
        "attn_disease_direction_context",
        "attn_observed_fraction",
        "attn_weighted_observed_mass",
    ):
        value = context[col].to_numpy(dtype=float)
        out[f"{col}_x_outcome_residual"] = value * residual
        out[f"{col}_x_image_fibrosis"] = value * image_score
    out["attn_disease_direction_context_x_image_activity"] = (
        context["attn_disease_direction_context"].to_numpy(dtype=float) * image_activity
    )
    return out


def _saits_style_cell_impute(
    late_obs: pd.DataFrame,
    base_imp: pd.DataFrame,
    history_by_cell: dict[str, pd.DataFrame],
    outcome_imp: pd.DataFrame | None,
    time: np.ndarray,
    rng: np.random.Generator,
    residual_scale: float,
    use_history: bool,
) -> pd.DataFrame:
    out = late_obs.copy()
    for col in late_obs.columns:
        context = _attention_context_features(late_obs, col, time)
        predictors = [base_imp, context]
        if use_history:
            predictors.append(history_by_cell[col])
        if outcome_imp is not None:
            predictors.append(outcome_imp)
            predictors.append(_attention_outcome_interactions(context, outcome_imp))
            if use_history:
                predictors.append(_history_outcome_interactions(history_by_cell[col], outcome_imp))
        x = pd.concat(predictors, axis=1)
        out[col] = coefsim._linear_impute(late_obs[col].to_numpy(dtype=float), x, rng, residual_scale)
    return out


def _generative_missingness_cell_impute(
    late_obs: pd.DataFrame,
    base_imp: pd.DataFrame,
    history_by_cell: dict[str, pd.DataFrame],
    outcome_imp: pd.DataFrame,
    time: np.ndarray,
    rng: np.random.Generator,
    use_attention: bool,
    use_history: bool = True,
) -> pd.DataFrame:
    """Explicit missingness-process stochastic multiple imputation analogue.

    For each late biomarker cell, first fit a selection/missingness model for
    the observed mask. The predicted missingness propensity is then included in
    a Bayesian linear imputation draw. This keeps the method stochastic and
    makes the missingness process explicit rather than relying only on a
    downstream classification loss.
    """
    out = late_obs.copy()
    for col in late_obs.columns:
        predictors = [base_imp]
        context = None
        if use_attention:
            context = _attention_context_features(late_obs, col, time)
            predictors.append(context)
        if use_history:
            predictors.append(history_by_cell[col])
        predictors.append(outcome_imp)
        if context is not None:
            predictors.append(_attention_outcome_interactions(context, outcome_imp))
        if use_history:
            predictors.append(_history_outcome_interactions(history_by_cell[col], outcome_imp))
        x_base = pd.concat(predictors, axis=1)
        x_base.columns = [f"p{j}" for j in range(x_base.shape[1])]
        miss_features = _fit_missingness_propensity(late_obs[col].isna().to_numpy(), x_base)
        x = pd.concat([x_base, miss_features], axis=1)
        stochastic = _posterior_linear_impute(
            late_obs[col].to_numpy(dtype=float),
            x,
            rng,
            coefficient_scale=0.18,
            residual_scale=0.55,
        )
        deterministic = coefsim._linear_impute(late_obs[col].to_numpy(dtype=float), x, rng, residual_scale=0.0)
        combined = deterministic.copy()
        missing = late_obs[col].isna().to_numpy()
        combined[missing] = 0.72 * deterministic[missing] + 0.28 * stochastic[missing]
        out[col] = combined
    return out


def _missingness_calibrated_stochastic_draws(
    seed_draws: list[pd.DataFrame],
    late_obs: pd.DataFrame,
    base_imp: pd.DataFrame,
    history_by_cell: dict[str, pd.DataFrame],
    outcome_imp: pd.DataFrame,
    time: np.ndarray,
    rng: np.random.Generator,
    use_attention: bool,
) -> list[pd.DataFrame]:
    """Use an explicit missingness model to calibrate MI uncertainty.

    The conditional mean comes from a strong deep-style imputer supplied in
    ``seed_draws``. The fitted missingness process changes the posterior spread
    of each missing cell, which is the part deterministic imputers usually
    understate.
    """
    calibrated: list[pd.DataFrame] = []
    missing_masks = {col: late_obs[col].isna().to_numpy() for col in late_obs.columns}
    cell_info: dict[str, tuple[np.ndarray, float, np.ndarray]] = {}
    for col in late_obs.columns:
        predictors = [base_imp]
        context = None
        if use_attention:
            context = _attention_context_features(late_obs, col, time)
            predictors.append(context)
        predictors.append(history_by_cell[col])
        predictors.append(outcome_imp)
        if context is not None:
            predictors.append(_attention_outcome_interactions(context, outcome_imp))
        predictors.append(_history_outcome_interactions(history_by_cell[col], outcome_imp))
        x_base = pd.concat(predictors, axis=1)
        x_base.columns = [f"p{j}" for j in range(x_base.shape[1])]
        miss_features = _fit_missingness_propensity(missing_masks[col], x_base)
        x = pd.concat([x_base, miss_features], axis=1)
        pred, resid_sd = _linear_prediction_and_resid_sd(late_obs[col].to_numpy(dtype=float), x)
        prob = miss_features["missingness_prob"].to_numpy(dtype=float)
        variance_multiplier = np.sqrt(np.clip(prob / (1.0 - prob), 0.20, 5.00))
        cell_info[col] = pred, resid_sd, variance_multiplier

    for draw in seed_draws:
        out = draw.copy()
        for col in late_obs.columns:
            missing = missing_masks[col]
            if not missing.any():
                continue
            pred, resid_sd, variance_multiplier = cell_info[col]
            values = out[col].to_numpy(dtype=float).copy()
            # Shrink extreme seed imputations toward the missingness-aware mean,
            # then add propensity-scaled posterior noise.
            values[missing] = (
                0.90 * values[missing]
                + 0.10 * pred[missing]
                + rng.normal(scale=0.22 * resid_sd * variance_multiplier[missing], size=int(missing.sum()))
            )
            out[col] = values
        calibrated.append(out)
    return calibrated


def _fit_downstream_residual(
    features: pd.DataFrame,
    design: pd.DataFrame,
    outcome: np.ndarray,
) -> tuple[pd.Series, np.ndarray] | None:
    keep = features.notna().all(axis=1).to_numpy()
    if keep.sum() < 50:
        return None
    try:
        result = sm.GLM(
            outcome[keep],
            sm.add_constant(design.loc[keep].reset_index(drop=True), has_constant="add"),
            family=sm.families.Binomial(),
        ).fit(maxiter=100, disp=False)
    except Exception:
        return None
    params = result.params.drop("const")
    x_all = sm.add_constant(design, has_constant="add")
    prob = np.asarray(result.predict(x_all), dtype=float)
    prob = np.clip(prob, 1e-4, 1.0 - 1e-4)
    return params, outcome - prob


def _fit_downstream_train_predict_residual(
    train_features: pd.DataFrame,
    train_design: pd.DataFrame,
    train_outcome: np.ndarray,
    predict_design: pd.DataFrame,
    predict_outcome: np.ndarray,
) -> tuple[pd.Series, np.ndarray] | None:
    keep = train_features.notna().all(axis=1).to_numpy()
    if keep.sum() < 50:
        return None
    try:
        result = sm.GLM(
            train_outcome[keep],
            sm.add_constant(train_design.loc[keep].reset_index(drop=True), has_constant="add"),
            family=sm.families.Binomial(),
        ).fit(maxiter=100, disp=False)
    except Exception:
        return None
    params = result.params.drop("const")
    prob = np.asarray(result.predict(sm.add_constant(predict_design, has_constant="add")), dtype=float)
    prob = np.clip(prob, 1e-4, 1.0 - 1e-4)
    return params, predict_outcome - prob


def _cell_score_sensitivity(
    params: pd.Series,
    marker: str,
    visit: int,
    time: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
) -> np.ndarray:
    """Approximate d eta / d biomarker cell for late-mean/slope outcome models."""
    sensitivity = np.zeros(len(diabetes), dtype=float)
    late_name = f"late_{marker}"
    if late_name in params:
        sensitivity += float(params[late_name]) / 3.0

    late_diabetes = f"late_{marker}_x_diabetes"
    if late_diabetes in params:
        sensitivity += float(params[late_diabetes]) * diabetes / 3.0

    late_htn = f"late_{marker}_x_htn"
    if late_htn in params:
        sensitivity += float(params[late_htn]) * htn / 3.0

    if marker == "platelet" and "late_platelet_x_htn" in params:
        sensitivity += float(params["late_platelet_x_htn"]) * htn / 3.0

    slope_name = f"slope_{marker}"
    if slope_name in params:
        dt = max(float(time[-1] - time[-3]), 1e-6)
        slope_derivative = -1.0 / dt if visit == 4 else 1.0 / dt if visit == 6 else 0.0
        sensitivity += float(params[slope_name]) * slope_derivative
        slope_diabetes = f"slope_{marker}_x_diabetes"
        if slope_diabetes in params:
            sensitivity += float(params[slope_diabetes]) * diabetes * slope_derivative
        slope_htn = f"slope_{marker}_x_htn"
        if slope_htn in params:
            sensitivity += float(params[slope_htn]) * htn * slope_derivative
        if marker == "platelet" and "slope_platelet_x_htn" in params:
            sensitivity += float(params["slope_platelet_x_htn"]) * htn * slope_derivative
    return sensitivity


def _targeted_association_fluctuation(
    cell_draws: list[pd.DataFrame],
    trajectory_obs: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    feature_builder,
    design_builder,
    rng: np.random.Generator,
    n_steps: int = 1,
    step_size: float = 0.025,
) -> list[pd.DataFrame]:
    """Target missing cells toward downstream estimating-equation fit.

    This is a simulation analogue of adding an association-preserving loss:
    after an initial imputation, fit the downstream model, compute outcome
    residuals, and apply a bounded fluctuation to originally missing cells in
    the direction that reduces the fitted score residual.
    """
    targeted: list[pd.DataFrame] = []
    late_missing = ~observed[:, -3:, :]
    for draw in cell_draws:
        out = draw.copy()
        for _ in range(n_steps):
            completed = _frame_to_late_trajectory(out, trajectory_obs)
            features = feature_builder(completed, time)
            design = design_builder(features, diabetes, htn, male)
            fit = _fit_downstream_residual(features, design, outcome)
            if fit is None:
                break
            params, residual = fit
            residual = np.clip(residual, -0.85, 0.85)
            for local_j, visit in enumerate(range(4, 7)):
                for k, marker in enumerate(BIOMARKERS):
                    missing = late_missing[:, local_j, k]
                    if not missing.any():
                        continue
                    col = f"{marker}_v{visit}"
                    sensitivity = _cell_score_sensitivity(params, marker, visit, time, diabetes, htn)
                    raw_delta = step_size * residual * sensitivity
                    finite = np.isfinite(raw_delta)
                    if not finite.any():
                        continue
                    scale = np.nanstd(out[col].to_numpy(dtype=float))
                    cap = max(0.18, 0.35 * float(scale) if np.isfinite(scale) else 0.18)
                    delta = np.clip(raw_delta, -cap, cap)
                    values = out[col].to_numpy(dtype=float).copy()
                    values[missing] = values[missing] + delta[missing]
                    out[col] = values
        for local_j, visit in enumerate(range(4, 7)):
            for k, marker in enumerate(BIOMARKERS):
                col = f"{marker}_v{visit}"
                missing = late_missing[:, local_j, k]
                if not missing.any():
                    continue
                sd = float(np.nanstd(out[col].to_numpy(dtype=float)))
                if np.isfinite(sd) and sd > 0:
                    values = out[col].to_numpy(dtype=float).copy()
                    values[missing] = values[missing] + rng.normal(scale=0.025 * sd, size=int(missing.sum()))
                    out[col] = values
        targeted.append(out)
    return targeted


def _crossfit_targeted_association_fluctuation(
    cell_draws: list[pd.DataFrame],
    trajectory_obs: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    feature_builder,
    design_builder,
    rng: np.random.Generator,
    n_folds: int = 5,
    n_steps: int = 1,
    step_size: float = 0.018,
) -> list[pd.DataFrame]:
    """Cross-fitted targeted fluctuation for association-preserving imputation.

    Each fold is corrected using downstream parameters estimated outside that
    fold. This is intended to separate outcome-informed imputation from
    in-sample classifier overfitting.
    """
    n = len(outcome)
    fold_id = np.arange(n) % max(n_folds, 2)
    rng.shuffle(fold_id)
    late_missing = ~observed[:, -3:, :]
    targeted: list[pd.DataFrame] = []

    for draw in cell_draws:
        out = draw.copy()
        for _ in range(n_steps):
            completed = _frame_to_late_trajectory(out, trajectory_obs)
            features = feature_builder(completed, time)
            design = design_builder(features, diabetes, htn, male)
            for fold in range(max(n_folds, 2)):
                test = fold_id == fold
                train = ~test
                fit = _fit_downstream_train_predict_residual(
                    features.loc[train].reset_index(drop=True),
                    design.loc[train].reset_index(drop=True),
                    outcome[train],
                    design.loc[test].reset_index(drop=True),
                    outcome[test],
                )
                if fit is None:
                    continue
                params, residual = fit
                residual = np.clip(residual, -0.75, 0.75)
                test_idx = np.flatnonzero(test)
                for local_j, visit in enumerate(range(4, 7)):
                    for k, marker in enumerate(BIOMARKERS):
                        missing = late_missing[test_idx, local_j, k]
                        if not missing.any():
                            continue
                        col = f"{marker}_v{visit}"
                        sensitivity = _cell_score_sensitivity(
                            params,
                            marker,
                            visit,
                            time,
                            diabetes[test],
                            htn[test],
                        )
                        raw_delta = step_size * residual * sensitivity
                        scale = np.nanstd(out[col].to_numpy(dtype=float))
                        cap = max(0.12, 0.20 * float(scale) if np.isfinite(scale) else 0.12)
                        delta = np.clip(raw_delta, -cap, cap)
                        values = out[col].to_numpy(dtype=float).copy()
                        rows = test_idx[missing]
                        values[rows] = values[rows] + delta[missing]
                        out[col] = values

        for local_j, visit in enumerate(range(4, 7)):
            for k, marker in enumerate(BIOMARKERS):
                col = f"{marker}_v{visit}"
                missing = late_missing[:, local_j, k]
                if not missing.any():
                    continue
                sd = float(np.nanstd(out[col].to_numpy(dtype=float)))
                if np.isfinite(sd) and sd > 0:
                    values = out[col].to_numpy(dtype=float).copy()
                    values[missing] = values[missing] + rng.normal(scale=0.035 * sd, size=int(missing.sum()))
                    out[col] = values
        targeted.append(out)
    return targeted


def _blend_cell_draws(
    left_draws: list[pd.DataFrame],
    right_draws: list[pd.DataFrame],
    late_obs: pd.DataFrame,
    left_weight: float = 0.55,
) -> list[pd.DataFrame]:
    """Convexly stack two completed-data engines only at originally missing cells."""
    n_draws = min(len(left_draws), len(right_draws))
    missing = late_obs.isna()
    blended: list[pd.DataFrame] = []
    for left, right in zip(left_draws[:n_draws], right_draws[:n_draws]):
        out = left.copy()
        values = left_weight * left.to_numpy(dtype=float) + (1.0 - left_weight) * right.to_numpy(dtype=float)
        out = pd.DataFrame(values, columns=left.columns, index=left.index)
        for col in out:
            observed = ~missing[col].to_numpy()
            out.loc[observed, col] = late_obs.loc[observed, col].to_numpy(dtype=float)
        blended.append(out)
    return blended


def _sequential_history_cell_impute(
    late_obs: pd.DataFrame,
    base_imp: pd.DataFrame,
    outcome_imp: pd.DataFrame | None,
    trajectory_obs: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    residual_scale: float,
) -> pd.DataFrame:
    """Impute late cells sequentially so later visits can use earlier filled states.

    This is a compact simulation analogue of a recurrent BRITS state: for each
    late cell, predictors are computed only from visits before the target time.
    After a cell is imputed, it becomes part of the completed state used by
    later target times. Original missingness summaries remain in the history
    features, so the model can still use observation-pattern information.
    """
    out = late_obs.copy()
    trajectory_work = trajectory_obs.copy()
    completed = observed.copy()

    for local_j, j in enumerate(range(3, 6), start=4):
        for k, name in enumerate(BIOMARKERS):
            col = f"{name}_v{local_j}"
            history = _history_features(trajectory_work, completed, time, j, k)
            predictors = [base_imp, history]
            if outcome_imp is not None:
                predictors.append(outcome_imp)
                predictors.append(_history_outcome_interactions(history, outcome_imp))
            x = pd.concat(predictors, axis=1)
            values = coefsim._linear_impute(late_obs[col].to_numpy(dtype=float), x, rng, residual_scale)
            out[col] = values
            trajectory_work[:, j, k] = values
            completed[:, j, k] = True
    return out


def _method_cell_draws(
    late_true_cells: pd.DataFrame,
    late_obs_cells: pd.DataFrame,
    base_imp: pd.DataFrame,
    history_by_cell: dict[str, pd.DataFrame],
    outcome_imp: pd.DataFrame,
    trajectory_obs: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    n_imputations: int,
) -> dict[str, list[pd.DataFrame]]:
    mean_fill = late_obs_cells.copy()
    for col in mean_fill:
        mean_fill[col] = mean_fill[col].fillna(float(mean_fill[col].mean()))

    methods: dict[str, list[pd.DataFrame]] = {
        "full_data": [late_true_cells],
        "complete_case": [late_obs_cells],
        "mean_impute": [mean_fill],
    }
    methods["brits_impute_only"] = [
        _linear_cell_impute(late_obs_cells, base_imp, history_by_cell, None, rng, 0.0, use_history=False)
    ] + [
        _linear_cell_impute(late_obs_cells, base_imp, history_by_cell, None, rng, BRITS_RESIDUAL_SCALE, False)
        for _ in range(max(n_imputations - 1, 0))
    ]
    methods["brits_history_impute_only"] = [
        _sequential_history_cell_impute(late_obs_cells, base_imp, None, trajectory_obs, observed, time, rng, 0.0)
    ] + [
        _sequential_history_cell_impute(
            late_obs_cells, base_imp, None, trajectory_obs, observed, time, rng, BRITS_RESIDUAL_SCALE
        )
        for _ in range(max(n_imputations - 1, 0))
    ]
    methods["brits_outcome"] = [
        _linear_cell_impute(late_obs_cells, base_imp, history_by_cell, outcome_imp, rng, 0.0, use_history=False)
    ] + [
        _linear_cell_impute(late_obs_cells, base_imp, history_by_cell, outcome_imp, rng, BRITS_RESIDUAL_SCALE, False)
        for _ in range(max(n_imputations - 1, 0))
    ]
    methods["brits_outcome_history"] = [
        _sequential_history_cell_impute(late_obs_cells, base_imp, outcome_imp, trajectory_obs, observed, time, rng, 0.0)
    ] + [
        _sequential_history_cell_impute(
            late_obs_cells, base_imp, outcome_imp, trajectory_obs, observed, time, rng, BRITS_RESIDUAL_SCALE
        )
        for _ in range(max(n_imputations - 1, 0))
    ]
    methods["saits_impute_only"] = [
        _saits_style_cell_impute(late_obs_cells, base_imp, history_by_cell, None, time, rng, 0.0, use_history=True)
    ] + [
        _saits_style_cell_impute(
            late_obs_cells, base_imp, history_by_cell, None, time, rng, BRITS_RESIDUAL_SCALE, use_history=True
        )
        for _ in range(max(n_imputations - 1, 0))
    ]
    methods["saits_outcome_history"] = [
        _saits_style_cell_impute(
            late_obs_cells, base_imp, history_by_cell, outcome_imp, time, rng, 0.0, use_history=True
        )
    ] + [
        _saits_style_cell_impute(
            late_obs_cells, base_imp, history_by_cell, outcome_imp, time, rng, BRITS_RESIDUAL_SCALE, use_history=True
        )
        for _ in range(max(n_imputations - 1, 0))
    ]
    mice_predictors = base_imp.copy()
    methods["mice"] = [
        coefsim._chained_linear_impute(late_obs_cells, mice_predictors, rng, n_iter=2, residual_scale=MICE_RESIDUAL_SCALE)
        for _ in range(n_imputations)
    ]
    methods["missforest"] = [
        coefsim._missforest_style_impute(
            late_obs_cells,
            mice_predictors,
            rng,
            n_iter=1,
            residual_scale=MISSFOREST_RESIDUAL_SCALE,
        )
        for _ in range(n_imputations)
    ]
    methods["gm_brits_stochastic_mi"] = _missingness_calibrated_stochastic_draws(
        methods["brits_outcome_history"],
        late_obs_cells,
        base_imp,
        history_by_cell,
        outcome_imp,
        time,
        rng,
        use_attention=False,
    )
    methods["gm_saits_stochastic_mi"] = _missingness_calibrated_stochastic_draws(
        methods["saits_outcome_history"],
        late_obs_cells,
        base_imp,
        history_by_cell,
        outcome_imp,
        time,
        rng,
        use_attention=True,
    )
    methods["stacked_deep_mice"] = _blend_cell_draws(
        methods["saits_outcome_history"],
        methods["mice"],
        late_obs_cells,
        left_weight=0.60,
    )
    methods["stacked_brits_saits"] = _blend_cell_draws(
        methods["brits_outcome_history"],
        methods["saits_outcome_history"],
        late_obs_cells,
        left_weight=0.50,
    )
    return methods


def _rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    if pos.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    return float((precision * (y_sorted == 1)).sum() / pos.sum())


def _classification_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-5, 1.0 - 1e-5)
    y = np.asarray(y).astype(int)
    pred = (prob >= 0.5).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "auroc": _rank_auc(y, prob),
        "auprc": _average_precision(y, prob),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(0.5 * (sens + spec)),
        "log_loss": float(-(y * np.log(prob) + (1 - y) * np.log(1 - prob)).mean()),
        "n_test": float(len(y)),
    }


def _fit_predict_probs(late_draws: list[pd.DataFrame], y: np.ndarray, diabetes: np.ndarray, htn: np.ndarray, male: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed + 771)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    test_size = max(100, int(0.30 * len(idx)))
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    probs = []
    y_ref = None
    for late in late_draws:
        keep_train = late.iloc[train_idx].notna().all(axis=1).to_numpy()
        keep_test = late.iloc[test_idx].notna().all(axis=1).to_numpy()
        if keep_train.sum() < 50 or keep_test.sum() < 50:
            continue
        train_rows = train_idx[keep_train]
        test_rows = test_idx[keep_test]
        x_train = coefsim._design(late.iloc[train_rows].reset_index(drop=True), diabetes[train_rows], htn[train_rows], male[train_rows])
        x_test = coefsim._design(late.iloc[test_rows].reset_index(drop=True), diabetes[test_rows], htn[test_rows], male[test_rows])
        try:
            result = sm.GLM(y[train_rows], sm.add_constant(x_train, has_constant="add"), family=sm.families.Binomial()).fit(
                maxiter=100, disp=False
            )
        except Exception:
            continue
        prob = np.asarray(result.predict(sm.add_constant(x_test, has_constant="add")), dtype=float)
        if y_ref is None:
            y_ref = y[test_rows]
            probs.append(prob)
        elif len(test_rows) == len(y_ref) and np.array_equal(y[test_rows], y_ref):
            probs.append(prob)
    if y_ref is None or not probs:
        return {"auroc": np.nan, "auprc": np.nan, "accuracy": np.nan, "balanced_accuracy": np.nan, "log_loss": np.nan, "n_test": 0.0}
    return _classification_metrics(y_ref, np.mean(np.vstack(probs), axis=0))


def _make_missingness(trajectory: np.ndarray, outcome: np.ndarray, diabetes: np.ndarray, htn: np.ndarray, latent: dict[str, np.ndarray], time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    late_time = (time >= time[-3]).astype(float)
    visit_shift = np.array([-0.30, -0.28, -0.20, -0.55, 0.18, 0.78])
    abnormality = np.stack([trajectory[:, :, 0], trajectory[:, :, 1], -trajectory[:, :, 2]], axis=2)
    marker_shift = np.array([0.06, 0.00, 0.12])
    late_trend = latent.get("late_trend", np.zeros(len(outcome)))
    logit_missing = (
        -1.78
        + 1.45 * outcome[:, None, None]
        + 0.22 * diabetes[:, None, None]
        + 0.14 * htn[:, None, None]
        + visit_shift[None, :, None]
        + marker_shift[None, None, :]
        + 0.12 * abnormality
        + 0.10 * latent["late_activity"][:, None, None] * late_time[None, :, None]
        + 0.14 * late_trend[:, None, None] * late_time[None, :, None]
        + rng.normal(scale=0.18, size=trajectory.shape)
    )
    return rng.binomial(1, 1.0 - coefsim._sigmoid(logit_missing)).astype(bool)


def _add_history_sensitive_late_trend(
    trajectory: np.ndarray,
    latent: dict[str, np.ndarray],
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    rng: np.random.Generator,
) -> None:
    late_trend = (
        rng.normal(size=len(diabetes))
        + 0.58 * latent["late_activity"]
        + 0.28 * diabetes
        + 0.18 * htn
        - 0.08 * male
    )
    trend_shape = np.array([-0.22, 0.52, 1.06])
    trajectory[:, 3:, 0] += 0.76 * late_trend[:, None] * trend_shape[None, :] + rng.normal(
        scale=0.09, size=(len(diabetes), 3)
    )
    trajectory[:, 3:, 1] += 0.64 * late_trend[:, None] * trend_shape[None, :] + rng.normal(
        scale=0.10, size=(len(diabetes), 3)
    )
    trajectory[:, 3:, 2] -= 0.72 * late_trend[:, None] * trend_shape[None, :] + rng.normal(
        scale=0.09, size=(len(diabetes), 3)
    )
    latent["late_trend"] = late_trend


def run_one(seed: int, n: int, n_imputations: int) -> tuple[list[dict], list[dict], dict]:
    rng = np.random.default_rng(seed)
    diabetes = rng.binomial(1, 0.36, size=n).astype(float)
    htn = rng.binomial(1, coefsim._sigmoid(-0.20 + 1.05 * diabetes), size=n).astype(float)
    male = rng.binomial(1, 0.46, size=n).astype(float)
    trajectory, time, latent = coefsim._simulate_longitudinal_biomarkers(rng, diabetes, htn, male)
    _add_history_sensitive_late_trend(trajectory, latent, diabetes, htn, male, rng)

    late_true = _late_summary(trajectory)
    x_true = coefsim._design(late_true, diabetes, htn, male)
    eta = -0.90 + sum(coefsim.TRUE_BETA[name] * x_true[name].to_numpy() for name in coefsim.TRUE_BETA)
    outcome = rng.binomial(1, coefsim._sigmoid(eta)).astype(int)

    observed = _make_missingness(trajectory, outcome, diabetes, htn, latent, time, rng)
    trajectory_obs = np.where(observed, trajectory, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        early_obs = np.nanmean(trajectory_obs[:, :3, :], axis=1)

    base_imp = coefsim._build_base_imputation_frame(early_obs, observed, diabetes, htn, male)
    outcome_imp = _outcome_image_features(late_true, latent, outcome, base_imp, diabetes, htn, male, rng)
    late_true_cells = _cell_frame(trajectory)
    late_obs_cells = _cell_frame(trajectory_obs)

    history_by_cell = {}
    for local_j, j in enumerate(range(3, 6), start=4):
        for k, name in enumerate(BIOMARKERS):
            history_by_cell[f"{name}_v{local_j}"] = _history_features(trajectory_obs, observed, time, j, k)

    methods = _method_cell_draws(
        late_true_cells,
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
    methods["tap_brits_outcome_history"] = _targeted_association_fluctuation(
        methods["brits_outcome_history"],
        trajectory_obs,
        observed,
        time,
        outcome,
        diabetes,
        htn,
        male,
        _late_summary_feature_builder,
        _late_summary_design_builder,
        rng,
    )
    methods["tap_saits_outcome_history"] = _targeted_association_fluctuation(
        methods["saits_outcome_history"],
        trajectory_obs,
        observed,
        time,
        outcome,
        diabetes,
        htn,
        male,
        _late_summary_feature_builder,
        _late_summary_design_builder,
        rng,
    )
    methods["cf_tap_brits_outcome_history"] = _crossfit_targeted_association_fluctuation(
        methods["brits_outcome_history"],
        trajectory_obs,
        observed,
        time,
        outcome,
        diabetes,
        htn,
        male,
        _late_summary_feature_builder,
        _late_summary_design_builder,
        rng,
    )
    methods["cf_tap_saits_outcome_history"] = _crossfit_targeted_association_fluctuation(
        methods["saits_outcome_history"],
        trajectory_obs,
        observed,
        time,
        outcome,
        diabetes,
        htn,
        male,
        _late_summary_feature_builder,
        _late_summary_design_builder,
        rng,
    )

    coef_rows: list[dict] = []
    metric_rows: list[dict] = []
    true_late_cells = trajectory[:, -3:, :]
    missing_late = ~observed[:, -3:, :]
    true_late_summary = _late_summary(trajectory)
    for method, cell_draws in methods.items():
        late_draws = [_late_summary(_frame_to_late_trajectory(draw, trajectory_obs)) for draw in cell_draws]
        pooled = coefsim._pool_logit_fits(late_draws, outcome, diabetes, htn, male)
        if pooled is not None:
            params, bse, within_se, between_sd, ci, n_used, n_fit_draws = pooled
            for name, truth in coefsim.TRUE_BETA.items():
                lo, hi = ci.loc[name]
                coef_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "coefficient": name,
                        "coefficient_label": coefsim.BIOMARKER_LABELS[name],
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
                    }
                )

        class_metrics = _fit_predict_probs(late_draws, outcome, diabetes, htn, male, seed)
        completed_mean = np.mean(
            np.stack([_frame_to_late_trajectory(draw, trajectory_obs)[:, -3:, :] for draw in cell_draws], axis=0),
            axis=0,
        )
        if method in {"full_data", "complete_case"}:
            impute_rmse = 0.0 if method == "full_data" else np.nan
        else:
            impute_rmse = float(np.sqrt(np.nanmean((completed_mean[missing_late] - true_late_cells[missing_late]) ** 2)))
        completed_summary = _late_summary(_frame_to_late_trajectory(cell_draws[0], trajectory_obs))
        summary_mse = float(np.nanmean((completed_summary.to_numpy() - true_late_summary.to_numpy()) ** 2))
        metric_rows.append(
            {
                "seed": seed,
                "method": method,
                "imputation_rmse": impute_rmse,
                "summary_mse": summary_mse,
                **class_metrics,
            }
        )

    audit = {
        "seed": seed,
        "n": int(n),
        "outcome_rate": float(outcome.mean()),
        "late_cell_missing_rate": float(missing_late.mean()),
        "cell_missing_rate": float((~observed).mean()),
        "late_ast_summary_missing_rate": float(_late_summary(trajectory_obs)["late_ast"].isna().mean()),
        "late_alt_summary_missing_rate": float(_late_summary(trajectory_obs)["late_alt"].isna().mean()),
        "late_platelet_summary_missing_rate": float(_late_summary(trajectory_obs)["late_platelet"].isna().mean()),
        "all_late_summaries_complete_rate": float(_late_summary(trajectory_obs).notna().all(axis=1).mean()),
    }
    return coef_rows, metric_rows, audit


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("method", as_index=False)
        .agg(
            imputation_rmse_mean=("imputation_rmse", "mean"),
            imputation_rmse_sd=("imputation_rmse", "std"),
            summary_mse_mean=("summary_mse", "mean"),
            summary_mse_sd=("summary_mse", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_sd=("log_loss", "std"),
            n_test_mean=("n_test", "mean"),
            n_runs=("seed", "nunique"),
        )
        .sort_values("auroc_mean", ascending=False)
    )


def draw_old_style_plots(metrics: pd.DataFrame, outdir: Path) -> None:
    labels = METHOD_LABELS
    order = [m for m in METHOD_ORDER if m in set(metrics["method"])]
    colors = {
        "Full data": "#74c7b0",
        "TAP-BRITS outcome + history": "#b2182b",
        "TAP-SAITS outcome + history": "#2166ac",
        "Cross-fit TAP-BRITS": "#ef8a62",
        "Cross-fit TAP-SAITS": "#67a9cf",
        "GM-BRITS stochastic MI": "#a6611a",
        "GM-SAITS stochastic MI": "#018571",
        "Stacked deep + MICE": "#6a3d9a",
        "Stacked BRITS + SAITS": "#cab2d6",
        "BRITS outcome + history": "#d95f02",
        "SAITS-style outcome + history": "#1b9e77",
        "BRITS outcome": "#f3a343",
        "BRITS history only": "#4c78a8",
        "SAITS-style impute only": "#66c2a5",
        "BRITS impute only": "#9b95d1",
        "MICE": "#f2d263",
        "Missforest": "#66a61e",
        "Mean imputation": "#c4a76b",
        "Complete case": "#b8b8b8",
    }

    specs = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("log_loss", "Log loss"),
        ("imputation_rmse", "Imputation RMSE"),
        ("summary_mse", "Trajectory summary MSE"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8), dpi=220)
    rng = np.random.default_rng(20260602)
    for ax, (metric, title) in zip(axes.ravel(), specs):
        vals = [metrics.loc[metrics["method"] == m, metric].dropna().to_numpy(dtype=float) for m in order]
        bp = ax.boxplot(vals, patch_artist=True, showfliers=False, widths=0.62)
        for patch, method in zip(bp["boxes"], order):
            label = labels[method]
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.88)
            patch.set_edgecolor("#333333")
        for idx, arr in enumerate(vals, start=1):
            if arr.size:
                ax.scatter(np.full(arr.size, idx) + rng.normal(0, 0.035, arr.size), arr, s=9, color="#222222", alpha=0.28)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([labels[m] for m in order], rotation=38, ha="right", fontsize=8)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Trajectory-history BRITS simulation: downstream and imputation metrics", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(outdir / "old_style_auc_imputation_boxplots.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("outputs/trajectory_history_outcome_sim"))
    parser.add_argument("--nsim", type=int, default=200)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument("--n-imputations", type=int, default=3)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    coef_rows: list[dict] = []
    metric_rows: list[dict] = []
    audits: list[dict] = []
    for seed in range(args.seed_start, args.seed_start + args.nsim):
        c_rows, m_rows, audit = run_one(seed, args.n, args.n_imputations)
        coef_rows.extend(c_rows)
        metric_rows.extend(m_rows)
        audits.append(audit)

    coef_df = coefsim.add_full_data_reference_errors(pd.DataFrame(coef_rows))
    coef_summary = coefsim.summarize(coef_df)
    coef_overall = coefsim.summarize_overall(coef_df)
    metrics = pd.DataFrame(metric_rows)
    metrics_summary = summarize_metrics(metrics)
    audit_df = pd.DataFrame(audits)

    coef_df.to_csv(args.outdir / "coefficient_by_run.csv", index=False)
    coef_summary.to_csv(args.outdir / "coefficient_summary_by_term.csv", index=False)
    coef_overall.to_csv(args.outdir / "coefficient_summary_scientific_biomarker_terms.csv", index=False)
    metrics.to_csv(args.outdir / "metrics_by_seed.csv", index=False)
    metrics_summary.to_csv(args.outdir / "metrics_aggregate.csv", index=False)
    audit_df.to_csv(args.outdir / "missingness_audit.csv", index=False)
    draw_old_style_plots(metrics, args.outdir)

    with open(args.outdir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "nsim": args.nsim,
                "n": args.n,
                "seed_start": args.seed_start,
                "n_imputations": args.n_imputations,
                "method_labels": METHOD_LABELS,
                "method_order": METHOD_ORDER,
                "history_features": (
                    "For each late biomarker cell, causal summaries before that time point are added: cumulative "
                    "mean, standard deviation, last value, slope, observed count, missing fraction, and gap to last "
                    "observation for the target marker and cross-marker histories."
                ),
                "purpose": (
                    "Tests whether adding pre-time trajectory summaries and missingness-history parameters improves "
                    "BRITS-style imputation beyond outcome/image supervision alone."
                ),
            },
            fh,
            indent=2,
        )
    print(coef_overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nClassification/imputation metrics:")
    print(metrics_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
