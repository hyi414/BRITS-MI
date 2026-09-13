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
HIST_PATH = ROOT / "scripts" / "run_trajectory_history_imputation_sim.py"
SPEC = importlib.util.spec_from_file_location("hist_sim", HIST_PATH)
hist_sim = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(hist_sim)
coefsim = hist_sim.coefsim

BIOMARKERS = ("ast", "alt", "platelet")
METHOD_ORDER = hist_sim.METHOD_ORDER
METHOD_LABELS = hist_sim.METHOD_LABELS
NORMAL_CRITICAL_VALUE = coefsim.NORMAL_CRITICAL_VALUE

TRUE_BETA = {
    "late_ast": 0.32,
    "late_alt": 0.24,
    "late_platelet": -0.34,
    "slope_ast": 0.34,
    "slope_alt": 0.22,
    "slope_platelet": -0.28,
    "diabetes": 0.22,
    "htn": 0.16,
    "male": 0.04,
    "late_ast_x_diabetes": 0.24,
    "slope_ast_x_diabetes": 0.20,
    "slope_platelet_x_htn": -0.18,
    "htn_x_male": -0.08,
}

COEFFICIENT_LABELS = {
    "late_ast": "Late AST exposure",
    "late_alt": "Late ALT exposure",
    "late_platelet": "Late platelet exposure",
    "slope_ast": "AST terminal slope",
    "slope_alt": "ALT terminal slope",
    "slope_platelet": "Platelet terminal slope",
    "diabetes": "Diabetes",
    "htn": "Hypertension",
    "male": "Male sex",
    "late_ast_x_diabetes": "Late AST x diabetes",
    "slope_ast_x_diabetes": "AST slope x diabetes",
    "slope_platelet_x_htn": "Platelet slope x hypertension",
    "htn_x_male": "Hypertension x male sex",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _simulate_irregular_trajectory(
    rng: np.random.Generator,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    n = len(diabetes)
    time = np.array([0.00, 0.12, 0.29, 0.51, 0.76, 1.00])
    severity = rng.normal(size=n) + 0.36 * diabetes + 0.22 * htn - 0.08 * male
    chronic_slope = rng.normal(scale=0.32, size=n) + 0.22 * diabetes + 0.15 * htn
    late_activity = rng.normal(size=n) + 0.18 * diabetes + 0.12 * htn
    late_acceleration = rng.normal(size=n) + 0.70 * late_activity + 0.22 * diabetes + 0.18 * htn
    pulse_center = rng.uniform(0.58, 0.88, size=n)
    pulse_width = rng.uniform(0.08, 0.15, size=n)

    t = time[None, :]
    pulse = np.exp(-0.5 * ((t - pulse_center[:, None]) / pulse_width[:, None]) ** 2)
    bend = np.maximum(t - 0.45, 0.0) ** 1.35
    ast = (
        0.16 * severity[:, None]
        + 0.42 * chronic_slope[:, None] * t
        + 0.96 * late_acceleration[:, None] * bend
        + 0.42 * late_activity[:, None] * pulse
        + 0.12 * diabetes[:, None]
        + rng.normal(scale=0.40, size=(n, len(time)))
    )
    alt = (
        0.12 * severity[:, None]
        + 0.34 * chronic_slope[:, None] * t
        + 0.78 * late_acceleration[:, None] * bend
        + 0.34 * late_activity[:, None] * pulse
        + 0.18 * diabetes[:, None]
        + rng.normal(scale=0.44, size=(n, len(time)))
    )
    platelet = (
        -0.14 * severity[:, None]
        - 0.38 * chronic_slope[:, None] * t
        - 0.88 * late_acceleration[:, None] * bend
        - 0.38 * late_activity[:, None] * pulse
        - 0.12 * htn[:, None]
        + rng.normal(scale=0.40, size=(n, len(time)))
    )
    trajectory = np.stack([ast, alt, platelet], axis=2)
    return trajectory, time, {
        "severity": severity,
        "late_activity": late_activity,
        "late_acceleration": late_acceleration,
        "pulse_center": pulse_center,
    }


def _trajectory_features(trajectory: np.ndarray, time: np.ndarray) -> pd.DataFrame:
    late = trajectory[:, -3:, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        late_mean = np.nanmean(late, axis=1)
    dt = max(float(time[-1] - time[-3]), 1e-6)
    slope = (trajectory[:, -1, :] - trajectory[:, -3, :]) / dt
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


def _design(features: pd.DataFrame, diabetes: np.ndarray, htn: np.ndarray, male: np.ndarray) -> pd.DataFrame:
    out = features.copy()
    out["diabetes"] = diabetes
    out["htn"] = htn
    out["male"] = male
    out["late_ast_x_diabetes"] = out["late_ast"].to_numpy() * diabetes
    out["slope_ast_x_diabetes"] = out["slope_ast"].to_numpy() * diabetes
    out["slope_platelet_x_htn"] = out["slope_platelet"].to_numpy() * htn
    out["htn_x_male"] = htn * male
    return out[list(TRUE_BETA)]


def _pool_logit_fits(
    feature_draws: list[pd.DataFrame],
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame, int, int] | None:
    params: list[pd.Series] = []
    variances: list[pd.Series] = []
    n_used: list[int] = []
    for features in feature_draws:
        keep = features.notna().all(axis=1).to_numpy()
        if keep.sum() < 25:
            continue
        x = _design(features.loc[keep].reset_index(drop=True), diabetes[keep], htn[keep], male[keep])
        try:
            fit = sm.GLM(outcome[keep], sm.add_constant(x, has_constant="add"), family=sm.families.Binomial()).fit(
                maxiter=100, disp=False
            )
        except Exception:
            continue
        params.append(fit.params.drop("const"))
        variances.append(fit.bse.drop("const") ** 2)
        n_used.append(int(keep.sum()))
    if not params:
        return None
    param_df = pd.DataFrame(params)
    variance_df = pd.DataFrame(variances)
    qbar = param_df.mean(axis=0)
    within_var = variance_df.mean(axis=0)
    between_var = param_df.var(axis=0, ddof=1).fillna(0.0) if len(param_df) > 1 else pd.Series(0.0, index=qbar.index)
    total_var = within_var + (1.0 + 1.0 / len(param_df)) * between_var
    pooled_se = np.sqrt(np.maximum(total_var, 1e-12))
    ci = pd.DataFrame({0: qbar - NORMAL_CRITICAL_VALUE * pooled_se, 1: qbar + NORMAL_CRITICAL_VALUE * pooled_se})
    return qbar, pooled_se, np.sqrt(np.maximum(within_var, 0.0)), np.sqrt(np.maximum(between_var, 0.0)), ci, int(np.mean(n_used)), len(param_df)


def _make_missingness(
    trajectory: np.ndarray,
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    latent: dict[str, np.ndarray],
    time: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    features = _trajectory_features(trajectory, time)
    slope_pressure = (
        0.30 * features["slope_ast"].to_numpy()
        + 0.22 * features["slope_alt"].to_numpy()
        - 0.28 * features["slope_platelet"].to_numpy()
    )
    visit_shift = np.array([-0.30, -0.06, -0.16, -0.50, 0.12, 0.62])
    marker_shift = np.array([0.05, -0.02, 0.14])
    abnormality = np.stack([trajectory[:, :, 0], trajectory[:, :, 1], -trajectory[:, :, 2]], axis=2)
    logit_missing = (
        -1.92
        + 1.05 * outcome[:, None, None]
        + 0.28 * diabetes[:, None, None]
        + 0.16 * htn[:, None, None]
        + visit_shift[None, :, None]
        + marker_shift[None, None, :]
        + 0.18 * abnormality
        + 0.26 * slope_pressure[:, None, None] * (time[None, :, None] > 0.50)
        + 0.08 * latent["late_activity"][:, None, None]
        + rng.normal(scale=0.24, size=trajectory.shape)
    )
    return rng.binomial(1, 1.0 - _sigmoid(logit_missing)).astype(bool)


def _build_base_frame(
    trajectory_obs: np.ndarray,
    observed: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    time: np.ndarray,
) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        early = np.nanmean(trajectory_obs[:, :3, :], axis=1)
    base = coefsim._build_base_imputation_frame(early, observed, diabetes, htn, male)
    first_half_obs = observed[:, :3, :]
    for k, name in enumerate(BIOMARKERS):
        count = first_half_obs[:, :, k].sum(axis=1)
        last_gap = np.full(len(diabetes), time[-1] + 1.0)
        for i in range(len(diabetes)):
            idx = np.flatnonzero(first_half_obs[i, :, k])
            if idx.size:
                last_gap[i] = time[3] - time[idx[-1]]
        base[f"{name}_early_count"] = count
        base[f"{name}_early_gap_to_late"] = last_gap
    return base


def _outcome_features(
    true_features: pd.DataFrame,
    latent: dict[str, np.ndarray],
    outcome: np.ndarray,
    base_imp: pd.DataFrame,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    residual = coefsim._outcome_residual(outcome, base_imp)
    image_fibrosis = (
        0.50 * true_features["late_ast"].to_numpy()
        + 0.38 * true_features["late_alt"].to_numpy()
        - 0.46 * true_features["late_platelet"].to_numpy()
        + 0.45 * latent["late_acceleration"]
        + rng.normal(scale=0.65, size=len(outcome))
    )
    image_slope = (
        0.46 * true_features["slope_ast"].to_numpy()
        + 0.34 * true_features["slope_alt"].to_numpy()
        - 0.42 * true_features["slope_platelet"].to_numpy()
        + rng.normal(scale=0.60, size=len(outcome))
    )
    return pd.DataFrame(
        {
            "outcome_residual": residual,
            "outcome_residual_x_diabetes": residual * diabetes,
            "outcome_residual_x_htn": residual * htn,
            "outcome_residual_x_male": residual * male,
            "image_fibrosis_score": image_fibrosis,
            "image_late_activity_score": 0.65 * latent["late_activity"] + 0.50 * latent["late_acceleration"] + rng.normal(scale=0.55, size=len(outcome)),
            "image_ast_texture_score": 0.58 * true_features["late_ast"].to_numpy() + rng.normal(scale=0.62, size=len(outcome)),
            "image_alt_texture_score": 0.55 * true_features["late_alt"].to_numpy() + rng.normal(scale=0.64, size=len(outcome)),
            "image_platelet_texture_score": 0.58 * true_features["late_platelet"].to_numpy() + rng.normal(scale=0.62, size=len(outcome)),
            "image_slope_score": image_slope,
            "image_slope_x_diabetes": image_slope * diabetes,
            "image_slope_x_htn": image_slope * htn,
        }
    )


def _classification_metrics(feature_draws: list[pd.DataFrame], outcome: np.ndarray, diabetes: np.ndarray, htn: np.ndarray, male: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed + 911)
    idx = np.arange(len(outcome))
    rng.shuffle(idx)
    test_size = max(100, int(0.30 * len(idx)))
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    probs = []
    y_ref = None
    for features in feature_draws:
        keep_train = features.iloc[train_idx].notna().all(axis=1).to_numpy()
        keep_test = features.iloc[test_idx].notna().all(axis=1).to_numpy()
        if keep_train.sum() < 50 or keep_test.sum() < 50:
            continue
        train_rows = train_idx[keep_train]
        test_rows = test_idx[keep_test]
        x_train = _design(features.iloc[train_rows].reset_index(drop=True), diabetes[train_rows], htn[train_rows], male[train_rows])
        x_test = _design(features.iloc[test_rows].reset_index(drop=True), diabetes[test_rows], htn[test_rows], male[test_rows])
        try:
            result = sm.GLM(outcome[train_rows], sm.add_constant(x_train, has_constant="add"), family=sm.families.Binomial()).fit(
                maxiter=100, disp=False
            )
        except Exception:
            continue
        prob = np.asarray(result.predict(sm.add_constant(x_test, has_constant="add")), dtype=float)
        if y_ref is None:
            y_ref = outcome[test_rows]
            probs.append(prob)
        elif len(test_rows) == len(y_ref) and np.array_equal(outcome[test_rows], y_ref):
            probs.append(prob)
    if y_ref is None or not probs:
        return {"auroc": np.nan, "auprc": np.nan, "accuracy": np.nan, "balanced_accuracy": np.nan, "log_loss": np.nan, "n_test": 0.0}
    return hist_sim._classification_metrics(y_ref, np.mean(np.vstack(probs), axis=0))


def _add_full_data_reference_errors(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    full = out[out["method"] == "full_data"][["seed", "coefficient", "estimate"]].rename(columns={"estimate": "full_data_estimate"})
    out = out.merge(full, on=["seed", "coefficient"], how="left")
    out["excess_error"] = out["estimate"] - out["full_data_estimate"]
    out["abs_excess_error"] = out["excess_error"].abs()
    out["squared_excess_error"] = out["excess_error"] ** 2
    out.loc[out["method"] == "full_data", ["excess_error", "abs_excess_error", "squared_excess_error"]] = 0.0
    return out


def _summarize_coef(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("method", as_index=False)
        .agg(
            mean_abs_bias=("abs_bias", "mean"),
            coefficient_mse=("squared_error", "mean"),
            mean_abs_excess_error=("abs_excess_error", "mean"),
            excess_mse=("squared_excess_error", "mean"),
            mean_model_se=("std_error", "mean"),
            coverage=("covered", "mean"),
            n_used_mean=("n_used", "mean"),
            n_runs=("seed", "nunique"),
        )
        .sort_values("mean_abs_excess_error")
    )


def _draw_coef_plot(summary: pd.DataFrame, outdir: Path) -> None:
    order = [m for m in METHOD_ORDER if m in set(summary["method"])]
    labels = [METHOD_LABELS.get(m, m) for m in order]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=220)
    specs = [
        ("mean_abs_excess_error", "Excess coefficient error"),
        ("excess_mse", "Excess coefficient MSE"),
        ("coverage", "95% CI coverage"),
    ]
    colors = ["#74c7b0" if m == "full_data" else "#d95f02" if m == "brits_outcome_history" else "#f3a343" if m == "brits_outcome" else "#8f8f8f" for m in order]
    for ax, (metric, title) in zip(axes, specs):
        vals = [float(summary.loc[summary["method"] == m, metric].iloc[0]) for m in order]
        ax.barh(labels, vals, color=colors)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(axis="x", color="#d8d8d8", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Irregular trajectory DGP: coefficient recovery", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outdir / "irregular_trajectory_coefficient_recovery.png", bbox_inches="tight")
    plt.close(fig)


def run_one(seed: int, n: int, n_imputations: int) -> tuple[list[dict], list[dict], dict]:
    rng = np.random.default_rng(seed)
    diabetes = rng.binomial(1, 0.36, size=n).astype(float)
    htn = rng.binomial(1, _sigmoid(-0.20 + 1.05 * diabetes), size=n).astype(float)
    male = rng.binomial(1, 0.46, size=n).astype(float)
    trajectory, time, latent = _simulate_irregular_trajectory(rng, diabetes, htn, male)
    true_features = _trajectory_features(trajectory, time)
    x_true = _design(true_features, diabetes, htn, male)
    eta = -0.84 + sum(TRUE_BETA[name] * x_true[name].to_numpy() for name in TRUE_BETA)
    outcome = rng.binomial(1, _sigmoid(eta)).astype(int)

    observed = _make_missingness(trajectory, outcome, diabetes, htn, latent, time, rng)
    trajectory_obs = np.where(observed, trajectory, np.nan)
    base_imp = _build_base_frame(trajectory_obs, observed, diabetes, htn, male, time)
    outcome_imp = _outcome_features(true_features, latent, outcome, base_imp, diabetes, htn, male, rng)
    late_true_cells = hist_sim._cell_frame(trajectory)
    late_obs_cells = hist_sim._cell_frame(trajectory_obs)
    history_by_cell = {}
    for local_j, j in enumerate(range(3, 6), start=4):
        for k, name in enumerate(BIOMARKERS):
            history_by_cell[f"{name}_v{local_j}"] = hist_sim._history_features(trajectory_obs, observed, time, j, k)
    methods = hist_sim._method_cell_draws(
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
    methods["tap_brits_outcome_history"] = hist_sim._targeted_association_fluctuation(
        methods["brits_outcome_history"],
        trajectory_obs,
        observed,
        time,
        outcome,
        diabetes,
        htn,
        male,
        _trajectory_features,
        _design,
        rng,
    )
    methods["tap_saits_outcome_history"] = hist_sim._targeted_association_fluctuation(
        methods["saits_outcome_history"],
        trajectory_obs,
        observed,
        time,
        outcome,
        diabetes,
        htn,
        male,
        _trajectory_features,
        _design,
        rng,
    )
    methods["cf_tap_brits_outcome_history"] = hist_sim._crossfit_targeted_association_fluctuation(
        methods["brits_outcome_history"],
        trajectory_obs,
        observed,
        time,
        outcome,
        diabetes,
        htn,
        male,
        _trajectory_features,
        _design,
        rng,
    )
    methods["cf_tap_saits_outcome_history"] = hist_sim._crossfit_targeted_association_fluctuation(
        methods["saits_outcome_history"],
        trajectory_obs,
        observed,
        time,
        outcome,
        diabetes,
        htn,
        male,
        _trajectory_features,
        _design,
        rng,
    )

    coef_rows: list[dict] = []
    metric_rows: list[dict] = []
    missing_late = ~observed[:, -3:, :]
    true_late_cells = trajectory[:, -3:, :]
    for method, cell_draws in methods.items():
        completed_trajectories = [hist_sim._frame_to_late_trajectory(draw, trajectory_obs) for draw in cell_draws]
        feature_draws = [_trajectory_features(comp, time) for comp in completed_trajectories]
        pooled = _pool_logit_fits(feature_draws, outcome, diabetes, htn, male)
        if pooled is not None:
            params, bse, within_se, between_sd, ci, n_used, n_fit_draws = pooled
            for name, truth in TRUE_BETA.items():
                lo, hi = ci.loc[name]
                coef_rows.append(
                    {
                        "seed": seed,
                        "method": method,
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
                    }
                )

        class_metrics = _classification_metrics(feature_draws, outcome, diabetes, htn, male, seed)
        completed_mean = np.mean(np.stack([comp[:, -3:, :] for comp in completed_trajectories], axis=0), axis=0)
        impute_rmse = 0.0 if method == "full_data" else np.nan
        if method not in {"full_data", "complete_case"}:
            impute_rmse = float(np.sqrt(np.nanmean((completed_mean[missing_late] - true_late_cells[missing_late]) ** 2)))
        feature_mse = float(np.nanmean((feature_draws[0].to_numpy() - true_features.to_numpy()) ** 2))
        metric_rows.append(
            {
                "seed": seed,
                "method": method,
                "imputation_rmse": impute_rmse,
                "summary_mse": feature_mse,
                **class_metrics,
            }
        )

    audit = {
        "seed": seed,
        "n": int(n),
        "outcome_rate": float(outcome.mean()),
        "late_cell_missing_rate": float(missing_late.mean()),
        "cell_missing_rate": float((~observed).mean()),
        "complete_late_rate": float(observed[:, -3:, :].all(axis=(1, 2)).mean()),
    }
    return coef_rows, metric_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("outputs/irregular_predictor_trajectory_dgp"))
    parser.add_argument("--nsim", type=int, default=200)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed-start", type=int, default=52000)
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

    coef_df = _add_full_data_reference_errors(pd.DataFrame(coef_rows))
    coef_summary = _summarize_coef(coef_df)
    metrics = pd.DataFrame(metric_rows)
    metrics_summary = hist_sim.summarize_metrics(metrics)
    audit_df = pd.DataFrame(audits)
    coef_df.to_csv(args.outdir / "coefficient_by_run.csv", index=False)
    coef_summary.to_csv(args.outdir / "coefficient_summary_all_terms.csv", index=False)
    metrics.to_csv(args.outdir / "metrics_by_seed.csv", index=False)
    metrics_summary.to_csv(args.outdir / "metrics_aggregate.csv", index=False)
    audit_df.to_csv(args.outdir / "missingness_audit.csv", index=False)
    hist_sim.draw_old_style_plots(metrics, args.outdir)
    _draw_coef_plot(coef_summary, args.outdir)
    with open(args.outdir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "nsim": args.nsim,
                "n": args.n,
                "seed_start": args.seed_start,
                "n_imputations": args.n_imputations,
                "true_beta": TRUE_BETA,
                "dgp": (
                    "Irregularly observed late biomarker trajectories affect the outcome through late exposure "
                    "and terminal slopes. Missingness depends on outcome, diabetes, hypertension, biomarker "
                    "abnormality, and latent late trajectory acceleration."
                ),
            },
            fh,
            indent=2,
        )
    print(coef_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nClassification/imputation metrics:")
    print(metrics_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMissingness audit:")
    print(audit_df.mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
