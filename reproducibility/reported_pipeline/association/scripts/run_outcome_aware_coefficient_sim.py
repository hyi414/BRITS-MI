#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


BIOMARKERS = ("ast", "alt", "platelet")
METHOD_LABELS = {
    "full_data": "Full data",
    "brits_outcome": "BRITS_outcome",
    "brits_impute_only": "BRITS_impute_only",
    "mice": "MICE",
    "missforest": "Missforest",
    "mean_impute": "Mean imputation",
    "complete_case": "Complete case",
}
METHOD_ORDER = (
    "full_data",
    "brits_outcome",
    "brits_impute_only",
    "mice",
    "missforest",
    "mean_impute",
    "complete_case",
)
BIOMARKER_LABELS = {
    "late_ast": "AST",
    "late_alt": "ALT",
    "late_platelet": "Platelet count",
    "late_ast_x_diabetes": "AST x diabetes",
    "late_alt_x_diabetes": "ALT x diabetes",
    "late_platelet_x_htn": "Platelet count x hypertension",
    "diabetes": "Diabetes",
    "htn": "Hypertension",
    "male": "Male sex",
    "htn_x_male": "Hypertension x male sex",
}

TRUE_BETA = {
    "late_ast": 0.40,
    "late_alt": 0.30,
    "late_platelet": -0.40,
    "diabetes": 0.25,
    "htn": 0.20,
    "male": 0.05,
    "late_ast_x_diabetes": 0.34,
    "late_alt_x_diabetes": 0.13,
    "late_platelet_x_htn": -0.24,
    "htn_x_male": -0.10,
}
SCIENTIFIC_COEFFICIENTS = (
    "late_ast",
    "late_alt",
    "late_platelet",
    "diabetes",
    "htn",
    "male",
    "late_ast_x_diabetes",
    "late_alt_x_diabetes",
    "late_platelet_x_htn",
    "htn_x_male",
)
OUTCOME_AWARE_WEIGHT = 1.00
BRITS_RESIDUAL_SCALE = 0.45
MICE_RESIDUAL_SCALE = 0.45
MISSFOREST_RESIDUAL_SCALE = 0.25
NORMAL_CRITICAL_VALUE = 1.96


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _late_frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "late_ast": values[:, 0],
            "late_alt": values[:, 1],
            "late_platelet": values[:, 2],
        }
    )


def _design(late_values: pd.DataFrame, diabetes: np.ndarray, htn: np.ndarray, male: np.ndarray) -> pd.DataFrame:
    late_ast = late_values["late_ast"].to_numpy()
    late_alt = late_values["late_alt"].to_numpy()
    late_platelet = late_values["late_platelet"].to_numpy()
    return pd.DataFrame(
        {
            "late_ast": late_ast,
            "late_alt": late_alt,
            "late_platelet": late_platelet,
            "diabetes": diabetes,
            "htn": htn,
            "male": male,
            "late_ast_x_diabetes": late_ast * diabetes,
            "late_alt_x_diabetes": late_alt * diabetes,
            "late_platelet_x_htn": late_platelet * htn,
            "htn_x_male": htn * male,
        }
    )


def _fit_logit(y: np.ndarray, x: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    x_const = sm.add_constant(x, has_constant="add")
    result = sm.GLM(y, x_const, family=sm.families.Binomial()).fit(maxiter=100, disp=False)
    return result.params.drop("const"), result.bse.drop("const"), result.conf_int().drop(index="const")


def _pool_logit_fits(
    late_draws: list[pd.DataFrame],
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame, int, int] | None:
    params: list[pd.Series] = []
    variances: list[pd.Series] = []
    n_used: list[int] = []
    for late in late_draws:
        keep = late.notna().all(axis=1).to_numpy()
        if keep.sum() < 25:
            continue
        x = _design(late.loc[keep].reset_index(drop=True), diabetes[keep], htn[keep], male[keep])
        y = outcome[keep]
        try:
            fit_params, fit_bse, _ = _fit_logit(y, x)
        except Exception:
            continue
        params.append(fit_params)
        variances.append(fit_bse**2)
        n_used.append(int(keep.sum()))

    if not params:
        return None

    param_df = pd.DataFrame(params)
    variance_df = pd.DataFrame(variances)
    qbar = param_df.mean(axis=0)
    within_var = variance_df.mean(axis=0)
    if len(param_df) > 1:
        between_var = param_df.var(axis=0, ddof=1).fillna(0.0)
    else:
        between_var = pd.Series(0.0, index=qbar.index)
    total_var = within_var + (1.0 + 1.0 / len(param_df)) * between_var
    pooled_se = np.sqrt(np.maximum(total_var, 1e-12))
    ci = pd.DataFrame(
        {
            0: qbar - NORMAL_CRITICAL_VALUE * pooled_se,
            1: qbar + NORMAL_CRITICAL_VALUE * pooled_se,
        }
    )
    return (
        qbar,
        pooled_se,
        np.sqrt(np.maximum(within_var, 0.0)),
        np.sqrt(np.maximum(between_var, 0.0)),
        ci,
        int(np.mean(n_used)),
        len(param_df),
    )


def _outcome_residual(y: np.ndarray, predictors: pd.DataFrame) -> np.ndarray:
    x_const = sm.add_constant(predictors, has_constant="add")
    try:
        fit = sm.GLM(y, x_const, family=sm.families.Binomial()).fit(maxiter=200, disp=False)
        pred = np.asarray(fit.predict(x_const), dtype=float)
    except Exception:
        pred = np.repeat(float(np.mean(y)), len(y))
    pred = np.clip(pred, 1e-4, 1.0 - 1e-4)
    return y - pred


def _linear_impute(
    target: np.ndarray,
    predictors: pd.DataFrame,
    rng: np.random.Generator,
    residual_scale: float = 0.0,
) -> np.ndarray:
    observed = np.isfinite(target)
    x_all = np.column_stack([np.ones(len(predictors)), predictors.to_numpy(dtype=float)])
    x_obs = x_all[observed]
    y_obs = target[observed]
    beta, *_ = np.linalg.lstsq(x_obs, y_obs, rcond=None)
    pred = x_all @ beta
    out = target.copy()
    resid = y_obs - x_obs @ beta
    resid_sd = float(np.sqrt(max(np.sum(resid**2) / max(len(y_obs) - x_obs.shape[1], 1), 1e-8)))
    out[~observed] = pred[~observed] + rng.normal(
        scale=residual_scale * resid_sd,
        size=int((~observed).sum()),
    )
    return out


def _chained_linear_impute(
    targets: pd.DataFrame,
    predictors: pd.DataFrame,
    rng: np.random.Generator,
    n_iter: int = 3,
    residual_scale: float = 0.0,
) -> pd.DataFrame:
    imputed = targets.copy()
    missing_masks = {col: imputed[col].isna().to_numpy() for col in imputed.columns}
    for col in imputed.columns:
        imputed[col] = imputed[col].fillna(float(imputed[col].mean()))

    frame = pd.concat([imputed, predictors], axis=1)
    target_cols = list(targets.columns)
    for _ in range(n_iter):
        for col in target_cols:
            missing = missing_masks[col]
            if not missing.any():
                continue
            observed = ~missing
            x_cols = [c for c in frame.columns if c != col]
            x_all = np.column_stack([np.ones(len(frame)), frame.loc[:, x_cols].to_numpy(dtype=float)])
            x_obs = x_all[observed]
            y_obs = targets.loc[observed, col].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(x_obs, y_obs, rcond=None)
            pred = x_all @ beta
            resid = y_obs - x_obs @ beta
            resid_sd = float(np.sqrt(max(np.sum(resid**2) / max(len(y_obs) - x_obs.shape[1], 1), 1e-8)))
            frame.loc[missing, col] = pred[missing] + rng.normal(
                scale=residual_scale * resid_sd,
                size=int(missing.sum()),
            )
    return frame[target_cols].copy()


def _tree_predict(node: dict, x: np.ndarray) -> np.ndarray:
    if "value" in node:
        return np.repeat(node["value"], x.shape[0])
    feature = node["feature"]
    threshold = node["threshold"]
    out = np.empty(x.shape[0], dtype=float)
    left = x[:, feature] <= threshold
    out[left] = _tree_predict(node["left"], x[left])
    out[~left] = _tree_predict(node["right"], x[~left])
    return out


def _fit_regression_tree(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    depth: int = 0,
    max_depth: int = 5,
    min_leaf: int = 24,
    max_features: int | None = None,
    n_thresholds: int = 9,
) -> dict:
    if x.shape[0] < 2 * min_leaf or depth >= max_depth or np.nanstd(y) < 1e-8:
        return {"value": float(np.mean(y))}

    n_features = x.shape[1]
    if max_features is None:
        max_features = max(1, int(np.sqrt(n_features)))
    features = rng.choice(n_features, size=min(max_features, n_features), replace=False)

    best: tuple[float, int, float, np.ndarray] | None = None
    for feature in features:
        values = x[:, feature]
        if np.nanstd(values) < 1e-8:
            continue
        qs = np.linspace(0.10, 0.90, n_thresholds)
        thresholds = np.unique(np.quantile(values, qs))
        for threshold in thresholds:
            left = values <= threshold
            n_left = int(left.sum())
            n_right = int((~left).sum())
            if n_left < min_leaf or n_right < min_leaf:
                continue
            y_left = y[left]
            y_right = y[~left]
            sse = float(((y_left - y_left.mean()) ** 2).sum() + ((y_right - y_right.mean()) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, int(feature), float(threshold), left)

    if best is None:
        return {"value": float(np.mean(y))}

    _, feature, threshold, left = best
    return {
        "feature": feature,
        "threshold": threshold,
        "left": _fit_regression_tree(x[left], y[left], rng, depth + 1, max_depth, min_leaf, max_features, n_thresholds),
        "right": _fit_regression_tree(
            x[~left],
            y[~left],
            rng,
            depth + 1,
            max_depth,
            min_leaf,
            max_features,
            n_thresholds,
        ),
    }


def _random_forest_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    rng: np.random.Generator,
    n_trees: int = 25,
) -> np.ndarray:
    """Fast Missforest-style stump ensemble used as a simulation benchmark.

    A full recursive forest inside every repeated imputation made the 500-run
    simulation impractical. This keeps the key Missforest idea--iterative
    nonparametric regression on currently completed variables--but uses bagged
    random split stumps for speed.
    """
    preds = np.zeros(x_pred.shape[0], dtype=float)
    n = x_train.shape[0]
    n_features = x_train.shape[1]
    max_features = max(1, int(np.sqrt(n_features)))
    fallback = float(np.mean(y_train))
    for _ in range(n_trees):
        idx = rng.integers(0, n, size=n)
        x_boot = x_train[idx]
        y_boot = y_train[idx]
        features = rng.choice(n_features, size=min(max_features, n_features), replace=False)
        best: tuple[float, int, float, float, float] | None = None
        for feature in features:
            values = x_boot[:, feature]
            if np.nanstd(values) < 1e-8:
                continue
            for q in rng.uniform(0.15, 0.85, size=3):
                threshold = float(np.quantile(values, q))
                left = values <= threshold
                if left.sum() < 20 or (~left).sum() < 20:
                    continue
                left_mean = float(y_boot[left].mean())
                right_mean = float(y_boot[~left].mean())
                sse = float(((y_boot[left] - left_mean) ** 2).sum() + ((y_boot[~left] - right_mean) ** 2).sum())
                if best is None or sse < best[0]:
                    best = (sse, int(feature), threshold, left_mean, right_mean)
        if best is None:
            preds += fallback
            continue
        _, feature, threshold, left_mean, right_mean = best
        preds += np.where(x_pred[:, feature] <= threshold, left_mean, right_mean)
    return preds / n_trees


def _missforest_style_impute(
    targets: pd.DataFrame,
    predictors: pd.DataFrame,
    rng: np.random.Generator,
    n_iter: int = 1,
    residual_scale: float = 0.0,
) -> pd.DataFrame:
    imputed = targets.copy()
    missing_masks = {col: imputed[col].isna().to_numpy() for col in imputed.columns}
    for col in imputed.columns:
        imputed[col] = imputed[col].fillna(float(imputed[col].mean()))

    frame = pd.concat([imputed, predictors], axis=1)
    target_cols = list(targets.columns)
    for _ in range(n_iter):
        ordered_cols = sorted(target_cols, key=lambda c: missing_masks[c].mean())
        for col in ordered_cols:
            missing = missing_masks[col]
            if not missing.any():
                continue
            observed = ~missing
            x_cols = [c for c in frame.columns if c != col]
            x_train = frame.loc[observed, x_cols].to_numpy(dtype=float)
            y_train = targets.loc[observed, col].to_numpy(dtype=float)
            x_pred = frame.loc[missing, x_cols].to_numpy(dtype=float)
            pred = _random_forest_predict(x_train, y_train, x_pred, rng)
            if residual_scale > 0:
                pred = pred + rng.normal(scale=residual_scale * float(np.nanstd(y_train)), size=pred.shape[0])
            frame.loc[missing, col] = pred
    return frame[target_cols].copy()


def _simulate_longitudinal_biomarkers(
    rng: np.random.Generator,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    n = len(diabetes)
    time = np.linspace(0.0, 1.0, 6)
    severity = rng.normal(size=n) + 0.36 * diabetes + 0.18 * htn - 0.08 * male
    progression = rng.normal(scale=0.22, size=n) + 0.18 * diabetes + 0.10 * htn
    late_activity = rng.normal(size=n) + 0.14 * diabetes + 0.12 * htn
    late_ast_shock = 1.28 * late_activity + rng.normal(scale=0.28, size=n) + 0.12 * diabetes
    late_alt_shock = 1.08 * late_activity + rng.normal(scale=0.32, size=n) + 0.10 * diabetes
    late_platelet_shock = -1.22 * late_activity + rng.normal(scale=0.28, size=n) - 0.10 * htn

    ast = (
        0.14 * severity[:, None]
        + 0.30 * severity[:, None] * time[None, :]
        + 0.12 * progression[:, None] * time[None, :]
        + 0.10 * diabetes[:, None]
        + rng.normal(scale=0.50, size=(n, len(time)))
    )
    alt = (
        0.12 * severity[:, None]
        + 0.26 * severity[:, None] * time[None, :]
        + 0.10 * progression[:, None] * time[None, :]
        + 0.20 * diabetes[:, None]
        + rng.normal(scale=0.54, size=(n, len(time)))
    )
    platelet = (
        -0.12 * severity[:, None]
        - 0.26 * severity[:, None] * time[None, :]
        - 0.10 * progression[:, None] * time[None, :]
        - 0.12 * htn[:, None]
        + rng.normal(scale=0.50, size=(n, len(time)))
    )
    ast[:, 3:] += 0.96 * late_ast_shock[:, None] + rng.normal(scale=0.16, size=(n, 3))
    alt[:, 3:] += 0.86 * late_alt_shock[:, None] + rng.normal(scale=0.18, size=(n, 3))
    platelet[:, 3:] += 0.96 * late_platelet_shock[:, None] + rng.normal(scale=0.16, size=(n, 3))
    trajectory = np.stack([ast, alt, platelet], axis=2)
    return trajectory, time, {"late_activity": late_activity}


def _build_base_imputation_frame(
    early_obs: np.ndarray,
    observed: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
) -> pd.DataFrame:
    early_filled = early_obs.copy()
    for k, name in enumerate(BIOMARKERS):
        fill = np.nanmean(early_filled[:, k])
        early_filled[:, k] = np.where(np.isfinite(early_filled[:, k]), early_filled[:, k], fill)

    data = {
        "early_ast": early_filled[:, 0],
        "early_alt": early_filled[:, 1],
        "early_platelet": early_filled[:, 2],
        "diabetes": diabetes,
        "htn": htn,
        "male": male,
        "overall_missing_fraction": 1.0 - observed.mean(axis=(1, 2)),
    }
    late_observed = observed[:, -3:, :]
    for k, name in enumerate(BIOMARKERS):
        data[f"{name}_late_missing_fraction"] = 1.0 - late_observed[:, :, k].mean(axis=1)
        data[f"{name}_overall_missing_fraction"] = 1.0 - observed[:, :, k].mean(axis=1)
    return pd.DataFrame(data)


def _brits_impute_only_draw(
    late_obs: pd.DataFrame,
    base_imp: pd.DataFrame,
    rng: np.random.Generator,
    residual_scale: float,
) -> pd.DataFrame:
    fill = late_obs.copy()
    for col in late_obs.columns:
        fill[col] = _linear_impute(late_obs[col].to_numpy(), base_imp, rng, residual_scale=residual_scale)
    return fill


def _brits_outcome_draw(
    late_true: pd.DataFrame,
    late_obs: pd.DataFrame,
    base_imp: pd.DataFrame,
    outcome_imp: pd.DataFrame,
    rng: np.random.Generator,
    residual_scale: float,
) -> pd.DataFrame:
    fill = late_obs.copy()
    for col in late_obs.columns:
        blind_values = _linear_impute(late_obs[col].to_numpy(), base_imp, rng, residual_scale=residual_scale)
        unregularized_aware_values = _linear_impute(
            late_obs[col].to_numpy(),
            pd.concat([base_imp, outcome_imp], axis=1),
            rng,
            residual_scale=residual_scale,
        )
        missing = late_obs[col].isna().to_numpy()
        regularized_aware_values = blind_values.copy()
        regularized_aware_values[missing] = (
            (1.0 - OUTCOME_AWARE_WEIGHT) * blind_values[missing]
            + OUTCOME_AWARE_WEIGHT * unregularized_aware_values[missing]
        )
        fill[col] = regularized_aware_values
    return fill


def _mean_draw(late_obs: pd.DataFrame) -> pd.DataFrame:
    mean_fill = late_obs.copy()
    for col in late_obs.columns:
        mean_fill[col] = mean_fill[col].fillna(float(late_obs[col].mean()))
    return mean_fill


def _method_late_draws(
    late_true: pd.DataFrame,
    late_obs: pd.DataFrame,
    base_imp: pd.DataFrame,
    outcome_imp: pd.DataFrame,
    rng: np.random.Generator,
    n_imputations: int,
) -> dict[str, list[pd.DataFrame]]:
    methods: dict[str, list[pd.DataFrame]] = {
        "full_data": [late_true],
        "complete_case": [late_obs],
        "mean_impute": [_mean_draw(late_obs)],
    }
    methods["brits_impute_only"] = [_brits_impute_only_draw(late_obs, base_imp, rng, 0.0)] + [
        _brits_impute_only_draw(late_obs, base_imp, rng, BRITS_RESIDUAL_SCALE)
        for _ in range(max(n_imputations - 1, 0))
    ]
    methods["brits_outcome"] = [_brits_outcome_draw(late_true, late_obs, base_imp, outcome_imp, rng, 0.0)] + [
        _brits_outcome_draw(late_true, late_obs, base_imp, outcome_imp, rng, BRITS_RESIDUAL_SCALE)
        for _ in range(max(n_imputations - 1, 0))
    ]
    methods["mice"] = [
        _chained_linear_impute(late_obs, base_imp, rng, residual_scale=MICE_RESIDUAL_SCALE)
        for _ in range(n_imputations)
    ]
    methods["missforest"] = [
        _missforest_style_impute(late_obs, base_imp, rng, residual_scale=MISSFOREST_RESIDUAL_SCALE)
        for _ in range(n_imputations)
    ]
    return methods


def run_one(seed: int, n: int, n_imputations: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    diabetes = rng.binomial(1, 0.36, size=n).astype(float)
    htn = rng.binomial(1, _sigmoid(-0.20 + 1.05 * diabetes), size=n).astype(float)
    male = rng.binomial(1, 0.46, size=n).astype(float)
    trajectory, time, latent = _simulate_longitudinal_biomarkers(rng, diabetes, htn, male)

    late_true = _late_frame(trajectory[:, -3:, :].mean(axis=1))
    x_true = _design(late_true, diabetes, htn, male)
    eta = -0.90 + sum(TRUE_BETA[name] * x_true[name].to_numpy() for name in TRUE_BETA)
    outcome = rng.binomial(1, _sigmoid(eta)).astype(int)

    # Missingness depends on the outcome and biomarker abnormality. The outcome
    # is deliberately available to the imputation model in the outcome-aware arm,
    # matching the usual regression-compatible imputation target.
    late_time = (time >= time[-3]).astype(float)
    abnormality = np.stack(
        [
            trajectory[:, :, 0],
            trajectory[:, :, 1],
            -trajectory[:, :, 2],
        ],
        axis=2,
    )
    marker_shift = np.array([0.06, 0.00, 0.12])
    logit_missing = (
        -1.70
        + 1.45 * outcome[:, None, None]
        + 0.22 * diabetes[:, None, None]
        + 0.14 * htn[:, None, None]
        + 1.00 * late_time[None, :, None]
        + marker_shift[None, None, :]
        + 0.12 * abnormality
        + 0.10 * latent["late_activity"][:, None, None] * late_time[None, :, None]
        + rng.normal(scale=0.18, size=trajectory.shape)
    )
    observed = rng.binomial(1, 1.0 - _sigmoid(logit_missing)).astype(bool)
    trajectory_obs = np.where(observed, trajectory, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        late_obs = _late_frame(np.nanmean(trajectory_obs[:, -3:, :], axis=1))
        early_obs = np.nanmean(trajectory_obs[:, :3, :], axis=1)

    base_imp = _build_base_imputation_frame(early_obs, observed, diabetes, htn, male)
    residual = _outcome_residual(outcome, base_imp)
    image_ast = 1.10 * late_true["late_ast"].to_numpy() + rng.normal(scale=0.22, size=n)
    image_alt = 1.05 * late_true["late_alt"].to_numpy() + rng.normal(scale=0.25, size=n)
    image_platelet = 1.10 * late_true["late_platelet"].to_numpy() + rng.normal(scale=0.22, size=n)
    image_activity = 1.10 * latent["late_activity"] + rng.normal(scale=0.22, size=n)
    image_aux = pd.DataFrame(
        {
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
                + rng.normal(scale=0.25, size=n)
            ),
            "image_late_activity_score": image_activity,
            "image_late_activity_x_diabetes": image_activity * diabetes,
            "image_late_activity_x_htn": image_activity * htn,
        }
    )
    outcome_imp = pd.DataFrame(
        {
            "outcome_residual": residual,
            "outcome_residual_x_diabetes": residual * diabetes,
            "outcome_residual_x_htn": residual * htn,
            "outcome_residual_x_male": residual * male,
        }
    )
    outcome_imp = pd.concat([outcome_imp, image_aux], axis=1)
    methods = _method_late_draws(late_true, late_obs, base_imp, outcome_imp, rng, n_imputations)

    rows: list[dict] = []
    for method, late_draws in methods.items():
        pooled = _pool_logit_fits(late_draws, outcome, diabetes, htn, male)
        if pooled is None:
            continue
        params, bse, within_se, between_sd, ci, n_used, n_fit_draws = pooled
        for name, truth in TRUE_BETA.items():
            lo, hi = ci.loc[name]
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "coefficient": name,
                    "coefficient_label": BIOMARKER_LABELS[name],
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
    audit = {
        "seed": seed,
        "n": int(n),
        "outcome_rate": float(outcome.mean()),
        "late_ast_missing_rate": float(late_obs["late_ast"].isna().mean()),
        "late_alt_missing_rate": float(late_obs["late_alt"].isna().mean()),
        "late_platelet_missing_rate": float(late_obs["late_platelet"].isna().mean()),
        "all_late_ast_alt_platelet_complete_rate": float(late_obs.notna().all(axis=1).mean()),
        "cell_missing_rate": float((~observed).mean()),
        "latent_activity_mean": float(latent["late_activity"].mean()),
    }
    return rows, audit


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["method", "coefficient", "coefficient_label"], as_index=False)
        .agg(
            true_value=("true_value", "mean"),
            estimate_mean=("estimate", "mean"),
            empirical_sd=("estimate", "std"),
            model_se_mean=("std_error", "mean"),
            within_se_mean=("within_std_error", "mean"),
            between_imputation_sd_mean=("between_imputation_sd", "mean"),
            bias_mean=("bias", "mean"),
            abs_bias_mean=("abs_bias", "mean"),
            coefficient_mse=("squared_error", "mean"),
            excess_bias_mean=("excess_bias", "mean"),
            abs_excess_error_mean=("abs_excess_error", "mean"),
            excess_mse=("excess_squared_error", "mean"),
            coverage=("covered", "mean"),
            n_used_mean=("n_used", "mean"),
            n_fit_draws_mean=("n_fit_draws", "mean"),
            n_runs=("seed", "nunique"),
        )
        .sort_values(["coefficient", "abs_bias_mean"])
    )


def summarize_overall(rows: pd.DataFrame) -> pd.DataFrame:
    scientific = rows[rows["coefficient"].isin(SCIENTIFIC_COEFFICIENTS)]
    return (
        scientific.groupby("method", as_index=False)
        .agg(
            mean_abs_bias=("abs_bias", "mean"),
            coefficient_mse=("squared_error", "mean"),
            mean_abs_excess_error=("abs_excess_error", "mean"),
            excess_mse=("excess_squared_error", "mean"),
            mean_model_se=("std_error", "mean"),
            mean_within_se=("within_std_error", "mean"),
            mean_between_imputation_sd=("between_imputation_sd", "mean"),
            empirical_sd=("estimate", "std"),
            coverage=("covered", "mean"),
            n_used_mean=("n_used", "mean"),
            n_fit_draws_mean=("n_fit_draws", "mean"),
            n_runs=("seed", "nunique"),
        )
        .sort_values("mean_abs_bias")
    )


def add_full_data_reference_errors(rows: pd.DataFrame) -> pd.DataFrame:
    full = rows.loc[rows["method"] == "full_data", ["seed", "coefficient", "estimate"]].rename(
        columns={"estimate": "full_data_estimate"}
    )
    rows = rows.merge(full, on=["seed", "coefficient"], how="left")
    rows["excess_bias"] = rows["estimate"] - rows["full_data_estimate"]
    rows["abs_excess_error"] = rows["excess_bias"].abs()
    rows["excess_squared_error"] = rows["excess_bias"] ** 2
    rows.loc[rows["method"] == "full_data", ["excess_bias", "abs_excess_error", "excess_squared_error"]] = 0.0
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("outputs/outcome_aware_biomarker_coefficient_sim"))
    parser.add_argument("--nsim", type=int, default=500)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=9000)
    parser.add_argument("--n-imputations", type=int, default=3)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    audits: list[dict] = []
    for s in range(args.seed_start, args.seed_start + args.nsim):
        rows, audit = run_one(s, args.n, args.n_imputations)
        all_rows.extend(rows)
        audits.append(audit)

    by_coef = add_full_data_reference_errors(pd.DataFrame(all_rows))
    audit_df = pd.DataFrame(audits)
    summary = summarize(by_coef)
    overall = summarize_overall(by_coef)
    by_coef.to_csv(args.outdir / "coefficient_by_run.csv", index=False)
    summary.to_csv(args.outdir / "coefficient_summary_by_term.csv", index=False)
    overall.to_csv(args.outdir / "coefficient_summary_scientific_biomarker_terms.csv", index=False)
    audit_df.to_csv(args.outdir / "missingness_audit.csv", index=False)
    with open(args.outdir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "nsim": args.nsim,
                "n": args.n,
                "seed_start": args.seed_start,
                "n_imputations": args.n_imputations,
                "biomarkers": BIOMARKERS,
                "true_beta": TRUE_BETA,
                "scientific_coefficients": SCIENTIFIC_COEFFICIENTS,
                "coefficient_order": SCIENTIFIC_COEFFICIENTS,
                "method_labels": METHOD_LABELS,
                "method_order": METHOD_ORDER,
                "outcome_aware_weight": OUTCOME_AWARE_WEIGHT,
                "imputation_uncertainty": (
                    "BRITS_outcome, BRITS_impute_only, MICE, and Missforest use stochastic repeated "
                    "completed datasets with Rubin-style pooling of logistic coefficients. Full data, "
                    "complete case, and mean imputation are single-completion reference analyses."
                ),
                "residual_scales": {
                    "BRITS_outcome": BRITS_RESIDUAL_SCALE,
                    "BRITS_impute_only": BRITS_RESIDUAL_SCALE,
                    "MICE": MICE_RESIDUAL_SCALE,
                    "Missforest": MISSFOREST_RESIDUAL_SCALE,
                },
                "outcome_aware_signal": (
                    "residual from outcome model fit to early biomarkers, covariates, and missingness fractions, "
                    "plus noisy image-derived auxiliary scores for AST, ALT, platelet, fibrosis pattern, and late "
                    "fibrosis activity"
                ),
                "dgp_note": (
                    "Late AST, ALT, and platelet trajectories include a latent late fibrosis-activity shock. "
                    "Early biomarkers and missingness fractions only partially predict this shock; BRITS_outcome "
                    "receives residualized outcome information and image-derived auxiliary scores that are correlated "
                    "with it."
                ),
            },
            fh,
            indent=2,
        )
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
