from __future__ import annotations

import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from .data import BIOMARKER_NAMES, LongitudinalData
from .metrics import classification_metrics, safe_classification_metrics, summarize_imputation, summary_mse


def forward_fill(y_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = y_obs.copy()
    for i in range(out.shape[0]):
        last = np.zeros(out.shape[2], dtype=out.dtype)
        for j in range(out.shape[1]):
            observed = mask[i, j] == 1
            last = np.where(observed, out[i, j], last)
            out[i, j] = np.where(observed, out[i, j], last)
    return out


def mean_impute(data: LongitudinalData, train_idx: np.ndarray) -> np.ndarray:
    out = data.y_obs.copy()
    for k in range(data.y_obs.shape[-1]):
        vals = data.y_obs[train_idx, :, k][data.mask[train_idx, :, k] == 1]
        mean_value = float(vals.mean()) if vals.size else 0.0
        out[:, :, k] = np.where(data.mask[:, :, k] == 1, data.y_obs[:, :, k], mean_value)
    return out


def iterative_impute(data: LongitudinalData, train_idx: np.ndarray) -> np.ndarray:
    n_t = data.y_obs.shape[1]
    n_markers = data.y_obs.shape[2]
    rows = []
    for i in range(data.y_obs.shape[0]):
        rows.append(
            np.concatenate(
                [
                    np.where(data.mask[i] == 1, data.y_obs[i], np.nan).reshape(-1),
                    data.mask[i].reshape(-1),
                    data.static[i],
                    data.x[i].reshape(-1),
                ]
            )
        )
    matrix = np.asarray(rows, dtype=float)
    # Zero is a valid standardized measurement, not the missing-value sentinel.
    # Retain empty training columns so subsequent feature positions cannot shift.
    imputer = IterativeImputer(
        random_state=17, max_iter=15, sample_posterior=False, keep_empty_features=True
    )
    imputer.fit(matrix[train_idx])
    completed_matrix = imputer.transform(matrix)
    completed_y = completed_matrix[:, : n_t * n_markers].reshape(data.y_obs.shape[0], n_t, n_markers)
    out = data.y_obs.copy()
    out[data.mask == 0] = completed_y[data.mask == 0]
    return out


def _subject_poly_features(y: np.ndarray, t: np.ndarray) -> tuple[float, float, float, float]:
    if y.shape[0] == 1:
        return float(y[0]), 0.0, 0.0, float(y[0])
    t_center = t - t.mean()
    design = np.column_stack([np.ones_like(t_center), t_center, t_center**2])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    late_mean = float(y[max(0, int(0.6 * y.shape[0])) :].mean())
    return float(coef[0]), float(coef[1]), float(coef[2]), late_mean


def _estimate_blup(beta: np.ndarray, cov_re: np.ndarray, sigma2: float, t_scaled: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones_like(t_scaled), t_scaled, t_scaled**2])
    z = np.column_stack([np.ones_like(t_scaled), t_scaled])
    resid = y - x @ beta
    v = z @ cov_re @ z.T + sigma2 * np.eye(len(t_scaled))
    return cov_re @ z.T @ np.linalg.solve(v, resid)


def _fit_marker_mixed_model(data: LongitudinalData, completed: np.ndarray, train_idx: np.ndarray, marker_idx: int) -> dict[str, np.ndarray | float]:
    y_parts = []
    t_parts = []
    group_parts = []
    for subject_id in train_idx:
        length = int(data.lengths[subject_id])
        y_parts.append(completed[subject_id, :length, marker_idx])
        t_parts.append(data.times[subject_id, :length])
        group_parts.append(np.full(length, subject_id, dtype=int))
    y = np.concatenate(y_parts).astype(float)
    t = np.concatenate(t_parts).astype(float)
    groups = np.concatenate(group_parts)
    t_mean = float(t.mean())
    t_std = max(float(t.std()), 1e-6)
    t_scaled = (t - t_mean) / t_std
    exog = np.column_stack([np.ones_like(t_scaled), t_scaled, t_scaled**2])
    exog_re = np.column_stack([np.ones_like(t_scaled), t_scaled])
    model = MixedLM(y, exog, groups=groups, exog_re=exog_re)
    result = model.fit(reml=False, method="lbfgs", disp=False)
    return {
        "beta": np.asarray(result.fe_params, dtype=float),
        "cov_re": np.asarray(result.cov_re, dtype=float),
        "sigma2": max(float(result.scale), 1e-6),
        "t_mean": t_mean,
        "t_std": t_std,
    }


def _mixed_effect_feature_matrix(
    data: LongitudinalData,
    completed: np.ndarray,
    index: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    static_indices = data.metadata["static_feature_indices"]
    marker_models = [_fit_marker_mixed_model(data, completed, train_idx, k) for k in range(completed.shape[2])]
    rows = []
    for subject_id in index:
        length = int(data.lengths[subject_id])
        t = data.times[subject_id, :length].astype(float)
        feature_row = []
        late_means = []
        for marker_idx, fit in enumerate(marker_models):
            y = completed[subject_id, :length, marker_idx].astype(float)
            t_scaled = (t - float(fit["t_mean"])) / float(fit["t_std"])
            blup = _estimate_blup(
                np.asarray(fit["beta"], dtype=float),
                np.asarray(fit["cov_re"], dtype=float),
                float(fit["sigma2"]),
                t_scaled,
                y,
            )
            _, subj_slope, subj_curvature, subj_late_mean = _subject_poly_features(y, t)
            intercept = float(fit["beta"][0] + blup[0])
            slope = float(0.7 * ((fit["beta"][1] + blup[1]) / float(fit["t_std"])) + 0.3 * subj_slope)
            curvature = float(subj_curvature)
            late_mean = float(subj_late_mean)
            late_means.append(late_mean)
            feature_row.extend([intercept, slope, curvature, late_mean])
        feature_row.extend(
            [
                late_means[0] - late_means[1],
                late_means[0] - late_means[2],
                late_means[1] - late_means[2],
                float(data.static[subject_id, static_indices["diabetes"]]),
                float(data.static[subject_id, static_indices["lab_measure"]]),
                float(data.static[subject_id, static_indices["survey_risk"]]),
                float(data.static[subject_id, static_indices["survey_access"]]),
            ]
        )
        rows.append(feature_row)
    feature_names = list(data.metadata["fibrosis_feature_names"])
    return np.asarray(rows, dtype=float), feature_names


def _trajectory_feature_matrix(data: LongitudinalData, completed: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, list[str]]:
    static_indices = data.metadata["static_feature_indices"]
    rows = []
    for subject_id in index:
        length = int(data.lengths[subject_id])
        t = data.times[subject_id, :length].astype(float)
        y = completed[subject_id, :length].astype(float)
        late_means = []
        feature_row = []
        for marker_idx in range(y.shape[1]):
            intercept, slope, curvature, late_mean = _subject_poly_features(y[:, marker_idx], t)
            late_means.append(late_mean)
            feature_row.extend([intercept, slope, curvature, late_mean])
        feature_row.extend(
            [
                late_means[0] - late_means[1],
                late_means[0] - late_means[2],
                late_means[1] - late_means[2],
                float(data.static[subject_id, static_indices["diabetes"]]),
                float(data.static[subject_id, static_indices["lab_measure"]]),
                float(data.static[subject_id, static_indices["survey_risk"]]),
                float(data.static[subject_id, static_indices["survey_access"]]),
            ]
        )
        rows.append(feature_row)
    return np.asarray(rows, dtype=float), list(data.metadata["fibrosis_feature_names"])


def _fit_ordered_logit_with_inference(
    train_x: np.ndarray,
    test_x: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    true_coefs: list[float],
    true_cutpoints: list[float],
) -> tuple[np.ndarray, list[dict], dict[str, float]]:
    coef_rows: list[dict] = []
    model = OrderedModel(y_train, train_x, distr="logit")
    result = model.fit(method="bfgs", disp=False)
    probs = np.asarray(result.model.predict(result.params, exog=test_x), dtype=float)
    names = list(feature_names) + ["cut_01", "cut_12"]
    truths = list(true_coefs) + list(true_cutpoints)
    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    ci = np.asarray(result.conf_int(), dtype=float)
    abs_biases = []
    cover_hits = []
    sign_hits = []
    for idx, (name, truth) in enumerate(zip(names, truths)):
        est = float(params[idx])
        ci_low = float(ci[idx, 0])
        ci_high = float(ci[idx, 1])
        bias = est - float(truth)
        covered = float(ci_low <= truth <= ci_high)
        sign_match = float(np.sign(est) == np.sign(truth) if truth != 0 else abs(est) < 0.05)
        coef_rows.append(
            {
                "coefficient": name,
                "true_value": float(truth),
                "estimate": est,
                "std_error": float(bse[idx]),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bias": bias,
                "abs_bias": abs(bias),
                "covered": covered,
                "sign_match": sign_match,
            }
        )
        abs_biases.append(abs(bias))
        cover_hits.append(covered)
        sign_hits.append(sign_match)
    coef_summary = {
        "coef_abs_bias_mean": float(np.mean(abs_biases)),
        "coef_coverage": float(np.mean(cover_hits)),
        "coef_sign_recovery": float(np.mean(sign_hits)),
    }
    return probs, coef_rows, coef_summary


def evaluate_completed_data(
    data: LongitudinalData,
    completed: np.ndarray,
    splits: dict[str, np.ndarray],
    method_name: str,
    downstream_models: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    train_idx = splits["train"]
    test_idx = splits["test"]
    y_train = data.labels[train_idx]
    y_test = data.labels[test_idx]
    test_missing = 1.0 - data.mask[test_idx].mean(axis=(1, 2))
    late_slice = slice(max(0, data.mask.shape[1] // 2), data.mask.shape[1])
    test_late_missing = 1.0 - data.mask[test_idx, late_slice].mean(axis=(1, 2))
    high_missing_cut = float(np.quantile(test_missing, 0.75))
    high_late_cut = float(np.quantile(test_late_missing, 0.75))
    imp_metrics = summarize_imputation(data.y_true[test_idx], completed[test_idx], data.mask[test_idx] == 0)
    common = {
        "imputation_method": method_name,
        "imputation_rmse": imp_metrics["rmse"],
        "imputation_mae": imp_metrics["mae"],
        "summary_mse": summary_mse(data.y_true[test_idx], completed[test_idx], data.times[test_idx], data.lengths[test_idx]),
    }
    rows = []
    predictions = []
    coefficient_rows = []
    for clf_name in downstream_models:
        coef_summary = {}
        if clf_name == "mixed_effects_ordlogit":
            train_x, feature_names = _mixed_effect_feature_matrix(data, completed, train_idx, train_idx)
            test_x, _ = _mixed_effect_feature_matrix(data, completed, test_idx, train_idx)
            probs, coef_rows, coef_summary = _fit_ordered_logit_with_inference(
                train_x,
                test_x,
                y_train,
                feature_names,
                list(data.metadata["fibrosis_true_coefs"]),
                list(data.metadata["fibrosis_cutpoints"]),
            )
            for coef_row in coef_rows:
                coefficient_rows.append({"imputation_method": method_name, "downstream_model": clf_name, **coef_row})
        elif clf_name == "longitudinal_rf":
            train_x, feature_names = _trajectory_feature_matrix(data, completed, train_idx)
            test_x, _ = _trajectory_feature_matrix(data, completed, test_idx)
            clf = RandomForestClassifier(n_estimators=150, random_state=11, min_samples_leaf=3)
            clf.fit(train_x, y_train)
            probs = clf.predict_proba(test_x)
        else:
            raise ValueError(f"Unsupported downstream model: {clf_name}")
        row = dict(common)
        row["downstream_model"] = clf_name
        row.update(classification_metrics(y_test, probs))
        high_missing_mask = test_missing >= high_missing_cut
        late_missing_mask = test_late_missing >= high_late_cut
        row.update(safe_classification_metrics(y_test[high_missing_mask], probs[high_missing_mask], "high_missing"))
        row.update(safe_classification_metrics(y_test[late_missing_mask], probs[late_missing_mask], "late_missing"))
        row.update(
            {
                "coef_abs_bias_mean": coef_summary.get("coef_abs_bias_mean", np.nan),
                "coef_coverage": coef_summary.get("coef_coverage", np.nan),
                "coef_sign_recovery": coef_summary.get("coef_sign_recovery", np.nan),
            }
        )
        rows.append(row)
        for subject_id, label, prob in zip(data.subject_ids[test_idx], y_test, probs):
            predictions.append(
                {
                    "subject_id": int(subject_id),
                    "label": int(label),
                    "imputation_method": method_name,
                    "downstream_model": clf_name,
                    "pred_class": int(np.argmax(prob)),
                    "pred_prob": float(np.max(prob)),
                }
            )
    return rows, predictions, coefficient_rows
