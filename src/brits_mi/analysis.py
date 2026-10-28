"""Downstream refitting, Rubin pooling, and recovery metrics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning

COEFFICIENT_ORDER = (
    "late_ast",
    "late_alt",
    "late_platelet",
    "slope_ast",
    "slope_alt",
    "slope_platelet",
    "diabetes",
    "hypertension",
    "male",
    "late_ast_x_diabetes",
    "slope_ast_x_diabetes",
    "slope_platelet_x_hypertension",
    "hypertension_x_male",
)


@dataclass
class PooledLogisticResult:
    estimate: pd.Series
    standard_error: pd.Series
    within_variance: pd.Series
    between_variance: pd.Series
    confidence_interval: pd.DataFrame
    n_imputations_used: int
    n_subjects_used: int
    degrees_of_freedom: pd.Series | None = None


def trajectory_features(trajectory: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    """Return prespecified late exposure and terminal slope for three markers."""

    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 3 or trajectory.shape[2] < 3:
        raise ValueError("trajectory must have shape (subjects, visits, at least 3 markers)")
    times = np.asarray(times, dtype=float)
    if times.ndim == 1:
        times = np.broadcast_to(times, trajectory.shape[:2])
    if times.shape != trajectory.shape[:2] or trajectory.shape[1] < 3:
        raise ValueError("times must match subjects/visits, with at least three visits")
    if not np.isfinite(times).all() or np.any(np.diff(times, axis=1) <= 0):
        raise ValueError("visit times must be finite and strictly increasing")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        late_mean = np.nanmean(trajectory[:, -3:, :3], axis=1)
    denominator = (times[:, -1] - times[:, -3])[:, None]
    slope = (trajectory[:, -1, :3] - trajectory[:, -3, :3]) / denominator
    return pd.DataFrame(
        {
            "late_ast": late_mean[:, 0],
            "late_alt": late_mean[:, 1],
            "late_platelet": late_mean[:, 2],
            "slope_ast": slope[:, 0],
            "slope_alt": slope[:, 1],
            "slope_platelet": slope[:, 2],
        }
    )


def design_matrix(
    features: pd.DataFrame,
    diabetes: np.ndarray,
    hypertension: np.ndarray,
    male: np.ndarray,
) -> pd.DataFrame:
    """Construct the prespecified clinical association model matrix."""

    design = features.copy()
    design["diabetes"] = np.asarray(diabetes, dtype=float)
    design["hypertension"] = np.asarray(hypertension, dtype=float)
    design["male"] = np.asarray(male, dtype=float)
    design["late_ast_x_diabetes"] = design["late_ast"] * design["diabetes"]
    design["slope_ast_x_diabetes"] = design["slope_ast"] * design["diabetes"]
    design["slope_platelet_x_hypertension"] = (
        design["slope_platelet"] * design["hypertension"]
    )
    design["hypertension_x_male"] = design["hypertension"] * design["male"]
    return design[list(COEFFICIENT_ORDER)]


def pool_logistic_regressions(
    completed_trajectories: list[np.ndarray],
    times: np.ndarray,
    static: np.ndarray,
    outcome: np.ndarray,
    confidence: float = 0.95,
    reference_scale: np.ndarray | None = None,
) -> PooledLogisticResult:
    """Refit the logistic outcome model in every completion and apply Rubin's rules."""

    if not completed_trajectories:
        raise ValueError("at least one completed trajectory is required")
    static = np.asarray(static, dtype=float)
    outcome = np.asarray(outcome, dtype=int)
    estimates: list[pd.Series] = []
    variances: list[pd.Series] = []
    sample_sizes: list[int] = []
    for trajectory in completed_trajectories:
        features = trajectory_features(trajectory, times)
        keep = features.notna().all(axis=1).to_numpy()
        if int(keep.sum()) < len(COEFFICIENT_ORDER) + 5:
            continue
        design = design_matrix(
            features.loc[keep].reset_index(drop=True),
            static[keep, 0],
            static[keep, 1],
            static[keep, 2],
        )
        if reference_scale is not None:
            design = design / np.asarray(reference_scale)
        fit = validated_logistic_fit(outcome[keep], design)
        estimates.append(fit.params.drop("const"))
        variances.append(fit.bse.drop("const") ** 2)
        sample_sizes.append(int(keep.sum()))
    if not estimates:
        raise RuntimeError("all downstream logistic fits failed")

    estimate_frame = pd.DataFrame(estimates)
    variance_frame = pd.DataFrame(variances)
    pooled = estimate_frame.mean(axis=0)
    within = variance_frame.mean(axis=0)
    if len(estimate_frame) > 1:
        between = estimate_frame.var(axis=0, ddof=1).fillna(0.0)
    else:
        between = pd.Series(0.0, index=pooled.index)
    total = within + (1.0 + 1.0 / len(estimate_frame)) * between
    standard_error = np.sqrt(np.maximum(total, 1e-12))
    df = barnard_rubin_df(within.to_numpy(), between.to_numpy(), len(estimates),
                         min(sample_sizes) - len(COEFFICIENT_ORDER) - 1)
    critical = t.ppf(0.5 + confidence / 2.0, df)
    interval = pd.DataFrame(
        {
            "lower": pooled - critical * standard_error,
            "upper": pooled + critical * standard_error,
        }
    )
    return PooledLogisticResult(
        estimate=pooled,
        standard_error=standard_error,
        within_variance=within,
        between_variance=between,
        confidence_interval=interval,
        n_imputations_used=len(estimate_frame),
        n_subjects_used=round(float(np.mean(sample_sizes))),
        degrees_of_freedom=pd.Series(df, index=pooled.index),
    )


def validated_logistic_fit(outcome, design):
    """Reject non-identifiable fits rather than counting huge intervals as recovery."""
    x = sm.add_constant(pd.DataFrame(design), has_constant="add")
    y = np.asarray(outcome, dtype=float)
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite outcome-model input")
    if len(np.unique(y)) != 2 or len(y) <= x.shape[1] + 1:
        raise ValueError("insufficient observations or outcome variation")
    if np.linalg.matrix_rank(x.to_numpy()) != x.shape[1]:
        raise ValueError("rank-deficient outcome design")
    # A finite MLE need not exist under quasi-separation, even when IRLS reports
    # convergence. Test whether a nonzero nonnegative signed margin is feasible.
    from scipy.optimize import linprog
    matrix = x.to_numpy()
    signed = (2 * y - 1)[:, None] * matrix / np.maximum(np.linalg.norm(matrix, axis=0), 1e-12)
    separation = linprog(-signed.sum(0), A_ub=-signed, b_ub=np.zeros(len(y)),
                         bounds=[(-1., 1.)] * matrix.shape[1], method="highs")
    if not separation.success:
        raise ValueError("logistic separation check did not converge")
    if -separation.fun > 1e-7:
        raise ValueError("complete or quasi-separated logistic fit")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerfectSeparationWarning)
        warnings.simplefilter("error", RuntimeWarning)
        try:
            fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=200, disp=False)
        except (PerfectSeparationError, PerfectSeparationWarning, RuntimeWarning) as exc:
            raise ValueError("separated or numerically unstable logistic fit") from exc
    covariance = np.asarray(fit.cov_params())
    if (not fit.converged or not np.isfinite(fit.params).all()
            or not np.isfinite(covariance).all()
            or np.min(np.linalg.eigvalsh(covariance)) <= 0):
        raise ValueError("invalid convergence or covariance")
    probability = np.asarray(fit.fittedvalues)
    information = x.to_numpy().T @ ((probability * (1 - probability))[:, None] * x.to_numpy())
    if np.linalg.cond(information) > 1e12:
        raise ValueError("near-singular logistic information")
    if np.all((probability >= 0.5) == y) and np.max(np.abs(probability - y)) < 1e-6:
        raise ValueError("perfectly separated logistic fit")
    return fit


def barnard_rubin_df(within, between, m, df_complete):
    """Small-sample MI degrees of freedom, including single-imputation limit."""
    within, between = np.asarray(within), np.asarray(between)
    if m == 1:
        return np.full_like(within, float(df_complete))
    total = within + (1 + 1 / m) * between
    fraction = np.clip((1 + 1 / m) * between / np.maximum(total, 1e-30), 0, 1)
    old = (m - 1) / np.maximum(fraction, 1e-15) ** 2
    observed = (df_complete + 1) / (df_complete + 3) * df_complete * (1 - fraction)
    return 1 / (1 / old + 1 / np.maximum(observed, 1e-12))


def coefficient_recovery(
    result: PooledLogisticResult,
    truth: dict[str, float],
) -> pd.DataFrame:
    """Return single-run coefficient errors and interval-inclusion indicators.

    Bias is obtained by averaging signed errors across independent runs, not
    by averaging their absolute values.
    """

    rows: list[dict[str, float | str | int]] = []
    for term, true_value in truth.items():
        estimate = float(result.estimate[term])
        lower = float(result.confidence_interval.loc[term, "lower"])
        upper = float(result.confidence_interval.loc[term, "upper"])
        bias = estimate - true_value
        rows.append(
            {
                "term": term,
                "truth": true_value,
                "estimate": estimate,
                "signed_error": bias,
                "absolute_error": abs(bias),
                "squared_error": bias * bias,
                "standard_error": float(result.standard_error[term]),
                "lower": lower,
                "upper": upper,
                "covered": int(lower <= true_value <= upper),
                "n_imputations_used": result.n_imputations_used,
                "n_subjects_used": result.n_subjects_used,
            }
        )
    return pd.DataFrame(rows)


def imputation_recovery(
    completed_trajectories: list[np.ndarray],
    truth: np.ndarray,
    observed_mask: np.ndarray,
    times: np.ndarray,
) -> pd.DataFrame:
    """Compute missing-cell and trajectory-summary normalized RMSE by draw."""

    truth = np.asarray(truth, dtype=float)
    missing = ~np.asarray(observed_mask, dtype=bool)
    marker_scale = np.nanstd(truth, axis=(0, 1), ddof=1)
    marker_scale = np.where(marker_scale > 1e-8, marker_scale, 1.0)
    true_summary = trajectory_features(truth, times)
    summary_scale = true_summary.std(axis=0, ddof=1).replace(0.0, 1.0)
    rows: list[dict[str, float | int]] = []
    for index, completed in enumerate(completed_trajectories, start=1):
        standardized_error = (completed - truth) / marker_scale.reshape(1, 1, -1)
        missing_nrmse = float(np.sqrt(np.mean(standardized_error[missing] ** 2)))
        completed_summary = trajectory_features(completed, times)
        summary_error = (completed_summary - true_summary) / summary_scale
        trajectory_nrmse = float(np.sqrt(np.nanmean(summary_error.to_numpy() ** 2)))
        rows.append(
            {
                "imputation": index,
                "missing_cell_nrmse": missing_nrmse,
                "trajectory_summary_nrmse": trajectory_nrmse,
            }
        )
    return pd.DataFrame(rows)
