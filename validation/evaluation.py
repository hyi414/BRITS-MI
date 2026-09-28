"""Source-relative association, reconstruction, and prediction evaluation.

Extracted from the manuscript masking analysis. Public inputs are analysis-ready
arrays; this module has no patient-data loader or patient-level file writer.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

MARKERS = ["a1c", "sbp", "dbp", "ast", "alt", "platelet"]


DISPLAY_TERMS = [
    "overall",
    "fib4_log",
    "diabetes",
    "hypertension",
    "male",
    "fib4_log_x_diabetes",
    "hypertension_x_male",
    "late_ast",
    "late_alt",
    "late_platelet",
    "slope_ast",
    "slope_alt",
    "slope_platelet",
]


TERM_LABELS = {
    "overall": "Overall",
    "fib4_log": "FIB-4 log",
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "male": "Male sex",
    "fib4_log_x_diabetes": "FIB-4 log x diabetes",
    "hypertension_x_male": "Hypertension x male sex",
    "late_ast": "Late aspartate aminotransferase",
    "late_alt": "Late alanine aminotransferase",
    "late_platelet": "Late platelet count",
    "slope_ast": "Aspartate aminotransferase slope",
    "slope_alt": "Alanine aminotransferase slope",
    "slope_platelet": "Platelet count slope",
}


def _yes(series: pd.Series) -> np.ndarray:
    s = series.astype("string").str.lower().fillna("")
    return s.str.contains("yes|true|1").to_numpy(dtype=float)


def _male(series: pd.Series) -> np.ndarray:
    s = series.astype("string").str.lower().str.strip().fillna("")
    return (s == "male").to_numpy(dtype=float)


def _fit_logit(y: pd.Series, x: pd.DataFrame) -> pd.DataFrame | None:
    scaler = StandardScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    x_scaled = sm.add_constant(x_scaled, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(y.to_numpy(dtype=float), x_scaled, family=sm.families.Binomial()).fit(
                maxiter=200, disp=0
            )
    except Exception:
        return None
    rows = []
    for term in x.columns:
        est = float(fit.params[term])
        se = float(fit.bse[term])
        rows.append(
            {
                "term": term,
                "term_label": TERM_LABELS.get(term, term.replace("_", " ")),
                "estimate": est,
                "std_error": se,
                "ci_low": est - 1.96 * se,
                "ci_high": est + 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


def balanced_accuracy_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in np.unique(y_true):
        idx = y_true == cls
        recalls.append(float((y_pred[idx] == cls).mean()))
    return float(np.mean(recalls))


def macro_ovr_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    classes = np.unique(y_true)
    aucs = []
    for j, cls in enumerate(classes):
        binary = (y_true == cls).astype(int)
        if binary.min() == binary.max():
            continue
        aucs.append(roc_auc_score(binary, prob[:, j]))
    return float(np.mean(aucs))


def macro_auprc(y_true: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    classes = np.unique(y_true)
    scores = []
    for j, cls in enumerate(classes):
        binary = (y_true == cls).astype(int)
        if binary.min() == binary.max():
            continue
        scores.append(average_precision_score(binary, prob[:, j]))
    return float(np.mean(scores))


def summarize_sequence_features(seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mean = np.mean(seq, axis=1)
    last = np.mean(seq[:, -2:, :], axis=1)
    slope = seq[:, -1, :] - seq[:, 0, :]
    miss = 1.0 - mask.mean(axis=1)
    return np.concatenate([mean, last, slope, miss], axis=1)


def downstream_features(
    static: np.ndarray, completed_seq: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    seq_feat = summarize_sequence_features(completed_seq, mask)
    return np.concatenate([static, seq_feat], axis=1)


def fit_downstream_classifier(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> tuple[LogisticRegression, np.ndarray]:
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
    )
    clf.fit(x_train, y_train)
    return clf, clf.predict_proba(x_test)


def _hide_observed_cells(
    mask: np.ndarray,
    rate: float,
    rng: np.random.Generator,
    channel_idx: list[int] | None = None,
) -> np.ndarray:
    eligible = mask > 0.5
    if channel_idx is not None:
        channel_selector = np.zeros(mask.shape[2], dtype=bool)
        channel_selector[channel_idx] = True
        eligible = eligible & channel_selector[None, None, :]
    hide = (rng.random(mask.shape) < rate) & eligible
    return hide


def _masked_metrics(
    completed: np.ndarray, y_true: np.ndarray, hide: np.ndarray
) -> dict[str, float]:
    err = completed[hide] - y_true[hide]
    true = y_true[hide]
    sd = float(np.std(true))
    return {
        "masked_cell_bias": float(np.mean(err)),
        "masked_cell_rmse": float(np.sqrt(np.mean(err**2))),
        "masked_cell_standardized_rmse": float(np.sqrt(np.mean(err**2)) / max(sd, 1e-8)),
        "masked_cell_n": float(hide.sum()),
    }


def observed_summary_features(
    sequence: np.ndarray,
    source_mask: np.ndarray,
    channel_idx: list[int],
) -> np.ndarray:
    """Return source-observed mean, late mean, and terminal change by marker."""
    n_subjects = sequence.shape[0]
    pieces: list[np.ndarray] = []
    for marker in channel_idx:
        values = np.where(source_mask[:, :, marker] > 0.5, sequence[:, :, marker], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(values, axis=1)
            late_mean = np.nanmean(values[:, -2:], axis=1)
        terminal_change = np.full(n_subjects, np.nan, dtype=float)
        for subject in range(n_subjects):
            observed = np.flatnonzero(np.isfinite(values[subject]))
            if observed.size >= 2:
                terminal_change[subject] = (
                    values[subject, observed[-1]] - values[subject, observed[0]]
                )
        pieces.extend([mean, late_mean, terminal_change])
    return np.column_stack(pieces)


def trajectory_summary_metrics(
    completed_raw: np.ndarray,
    truth_raw: np.ndarray,
    source_mask: np.ndarray,
    hidden_targets: np.ndarray,
    channel_idx: list[int],
) -> dict[str, float]:
    """Evaluate summaries after replacing only artificially hidden measured cells."""
    hybrid = truth_raw.copy()
    hybrid[hidden_targets] = completed_raw[hidden_targets]
    reference = observed_summary_features(truth_raw, source_mask, channel_idx)
    estimated = observed_summary_features(hybrid, source_mask, channel_idx)
    valid = np.isfinite(reference) & np.isfinite(estimated)
    if not valid.any():
        return {
            "trajectory_summary_rmse": np.nan,
            "trajectory_summary_nrmse": np.nan,
            "trajectory_summary_n": 0,
        }
    error = estimated[valid] - reference[valid]
    reference_values = reference[valid]
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "trajectory_summary_rmse": rmse,
        "trajectory_summary_nrmse": rmse / max(float(np.std(reference_values)), 1e-8),
        "trajectory_summary_n": int(valid.sum()),
    }


def observed_trajectory_summary(
    completed: np.ndarray,
    source_mask: np.ndarray,
) -> pd.DataFrame:
    rows: dict[str, np.ndarray] = {}
    for marker_index, marker in enumerate(MARKERS[: completed.shape[2]]):
        values = np.where(
            source_mask[:, :, marker_index] > 0.5,
            completed[:, :, marker_index],
            np.nan,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(values, axis=1)
            late = np.nanmean(values[:, -2:], axis=1)
        late = np.where(np.isfinite(late), late, mean)
        slope = np.zeros(len(completed), dtype=float)
        for subject in range(len(completed)):
            observed = np.flatnonzero(np.isfinite(values[subject]))
            if len(observed) >= 2:
                slope[subject] = values[subject, observed[-1]] - values[subject, observed[0]]
        rows[f"mean_{marker}"] = mean
        rows[f"late_{marker}"] = late
        rows[f"slope_{marker}"] = slope
        rows[f"missing_fraction_{marker}"] = 1.0 - source_mask[:, :, marker_index].mean(axis=1)
    return pd.DataFrame(rows)


def effect_design_observed(
    df: pd.DataFrame,
    completed: np.ndarray,
    source_mask: np.ndarray,
) -> tuple[pd.Series, pd.DataFrame]:
    y = (df["outcome_3class"].astype(str) == "F3-F4").astype(int)
    seq = observed_trajectory_summary(completed, source_mask)
    fib4_log = pd.to_numeric(df["FIB4_log"], errors="coerce")
    age = pd.to_numeric(df["FIB4_Age"], errors="coerce")
    bmi = pd.to_numeric(df["bmi_consolidated"], errors="coerce")
    fib4_log = fib4_log.fillna(fib4_log.median())
    age = age.fillna(age.median())
    bmi = bmi.fillna(bmi.median())
    diabetes = _yes(df["Diabetes_RF"])
    hypertension = _yes(df["HTN_RF"])
    hyperlipidemia = _yes(df["HLD_RF"])
    male = _male(df["Sex"])
    study_mre = (
        df["Study_Type"]
        .astype("string")
        .str.upper()
        .fillna("")
        .str.contains("MRE")
        .to_numpy(dtype=float)
    )
    x = pd.DataFrame(
        {
            "fib4_log": fib4_log.to_numpy(dtype=float),
            "age": age.to_numpy(dtype=float),
            "bmi": bmi.to_numpy(dtype=float),
            "diabetes": diabetes,
            "hypertension": hypertension,
            "hyperlipidemia": hyperlipidemia,
            "male": male,
            "study_mre": study_mre,
        }
    )
    x["fib4_log_x_diabetes"] = x["fib4_log"] * x["diabetes"]
    x["hypertension_x_male"] = x["hypertension"] * x["male"]
    # Keep the downstream estimand prespecified and estimable. Including every
    # mean, late value, slope, and missingness fraction creates near-duplicate
    # columns and complete separation after deterministic completion.
    trajectory_terms = [
        "late_ast",
        "late_alt",
        "late_platelet",
        "slope_ast",
        "slope_alt",
        "slope_platelet",
        "missing_fraction_ast",
        "missing_fraction_alt",
        "missing_fraction_platelet",
    ]
    x = pd.concat([x, seq[trajectory_terms]], axis=1)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    return y, x


def reference_coefficients(
    df: pd.DataFrame,
    truth_raw: np.ndarray,
    source_mask: np.ndarray,
) -> dict[str, float]:
    y, x = effect_design_observed(df, truth_raw, source_mask)
    fit = _fit_logit(y, x)
    if fit is None:
        raise RuntimeError("The common observed-cell reference outcome model failed.")
    return dict(zip(fit["term"], fit["estimate"]))


def pooled_effect_rows(
    df: pd.DataFrame,
    draws_raw: list[np.ndarray],
    truth_raw: np.ndarray,
    source_mask: np.ndarray,
    hidden_targets: np.ndarray,
    reference: dict[str, float],
    seed: int,
    repeat: int,
    missing_rate: float,
    method: str,
    method_label: str,
) -> tuple[list[dict], dict[str, float]]:
    fits: list[pd.DataFrame] = []
    for draw in draws_raw:
        hybrid = truth_raw.copy()
        hybrid[hidden_targets] = draw[hidden_targets]
        y, x = effect_design_observed(df, hybrid, source_mask)
        fit = _fit_logit(y, x)
        if fit is not None:
            fits.append(fit.set_index("term"))
    if not fits:
        return [], {
            "effect_abs_bias": np.nan,
            "effect_mse": np.nan,
            "effect_coverage": np.nan,
            "effect_terms": 0,
        }
    terms = sorted(set(reference).intersection(*(set(frame.index) for frame in fits)))
    estimates = np.stack([frame.loc[terms, "estimate"].to_numpy(float) for frame in fits])
    variances = np.stack([frame.loc[terms, "std_error"].to_numpy(float) ** 2 for frame in fits])
    pooled = estimates.mean(axis=0)
    within = variances.mean(axis=0)
    between = estimates.var(axis=0, ddof=1) if len(fits) > 1 else np.zeros(len(terms))
    total = within + (1.0 + 1.0 / len(fits)) * between
    standard_error = np.sqrt(np.maximum(total, 0.0))
    rows: list[dict] = []
    for term_index, term in enumerate(terms):
        truth = float(reference[term])
        estimate = float(pooled[term_index])
        se = float(standard_error[term_index])
        rows.append(
            {
                "seed": seed,
                "repeat": repeat,
                "missing_rate": missing_rate,
                "masking_percent": int(round(100 * missing_rate)),
                "method": method,
                "method_label": method_label,
                "term": term,
                "term_label": TERM_LABELS.get(term, term.replace("_", " ")),
                "reference": truth,
                "estimate": estimate,
                "std_error": se,
                "bias": estimate - truth,
                "abs_bias": abs(estimate - truth),
                "squared_error": (estimate - truth) ** 2,
                "covered": float(estimate - 1.96 * se <= truth <= estimate + 1.96 * se),
                "n_fit_draws": len(fits),
            }
        )
    selected = [row for row in rows if row["term"] in DISPLAY_TERMS[1:]]
    return rows, {
        "effect_abs_bias": float(np.mean([row["abs_bias"] for row in selected])),
        "effect_mse": float(np.mean([row["squared_error"] for row in selected])),
        "effect_coverage": float(np.mean([row["covered"] for row in selected])),
        "effect_terms": len(selected),
    }


def build_frame(
    input_values: np.ndarray,
    input_mask: np.ndarray,
    static: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    n, length, markers = input_values.shape
    targets = [f"y_t{time + 1}_{marker + 1}" for time in range(length) for marker in range(markers)]
    y = input_values.reshape(n, -1).astype(float)
    mask = input_mask.reshape(n, -1).astype(float)
    y[mask < 0.5] = np.nan
    frame = pd.DataFrame(y, columns=targets)
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(mask, columns=[f"mask_{j + 1}" for j in range(mask.shape[1])]),
            pd.DataFrame(static, columns=[f"static_{j + 1}" for j in range(static.shape[1])]),
        ],
        axis=1,
    )
    return frame, targets


def completed_array(
    input_values: np.ndarray,
    input_mask: np.ndarray,
    draw: pd.DataFrame,
) -> np.ndarray:
    candidate = draw.to_numpy(dtype=float).reshape(input_values.shape)
    completed = input_values.copy().astype(float)
    completed[input_mask < 0.5] = candidate[input_mask < 0.5]
    return completed.astype(np.float32)


def pooled_apparent_prediction(
    completions_raw: list[np.ndarray],
    input_mask: np.ndarray,
    static: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    probabilities = []
    for completed in completions_raw:
        features = downstream_features(static, completed, input_mask)
        features = StandardScaler().fit_transform(features)
        _, probability = fit_downstream_classifier(
            features,
            labels,
            features,
        )
        probabilities.append(probability)
    probability = np.mean(np.stack(probabilities, axis=0), axis=0)
    prediction = probability.argmax(axis=1)
    return {
        "auroc": macro_ovr_auc(labels, probability),
        "auprc": macro_auprc(labels, probability),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": balanced_accuracy_score_manual(labels, prediction),
        "log_loss": float(log_loss(labels, probability, labels=np.arange(probability.shape[1]))),
    }
