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


ROOT = Path(__file__).resolve().parents[1]
HIST_PATH = ROOT / "scripts" / "run_trajectory_history_imputation_sim.py"
DGP1_PATH = ROOT / "scripts" / "run_irregular_predictor_trajectory_sim.py"

SPEC = importlib.util.spec_from_file_location("hist_sim", HIST_PATH)
hist_sim = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(hist_sim)

DGP_SPEC = importlib.util.spec_from_file_location("dgp1", DGP1_PATH)
dgp1 = importlib.util.module_from_spec(DGP_SPEC)
assert DGP_SPEC and DGP_SPEC.loader
DGP_SPEC.loader.exec_module(dgp1)

coefsim = hist_sim.coefsim
BIOMARKERS = dgp1.BIOMARKERS
TRUE_BETA = dgp1.TRUE_BETA
COEFFICIENT_LABELS = dgp1.COEFFICIENT_LABELS

METHOD_LABELS = {
    "full_data": "Full data",
    "role_vc_brits_mi": "Role-aware VC-BRITS MI",
    "role_vc_saits_mi": "Role-aware VC-SAITS MI",
    "avc_smc_saits_mi": "Adaptive VC-SMC-SAITS MI",
    "avc_smc_brits_mi": "Adaptive VC-SMC-BRITS MI",
    "vc_smc_saits_mi": "VC-SMC-SAITS multiple imputation",
    "vc_smc_brits_mi": "VC-SMC-BRITS multiple imputation",
    "smc_saits_mi": "SMC-SAITS multiple imputation",
    "smc_brits_mi": "SMC-BRITS multiple imputation",
    "gm_saits_stochastic_mi": "GM-SAITS stochastic MI",
    "gm_brits_stochastic_mi": "GM-BRITS stochastic MI",
    "saits_outcome_history": "SAITS outcome + history",
    "brits_outcome_history": "BRITS outcome + history",
    "mice": "MICE",
    "missforest": "Missforest",
    "mean_impute": "Mean imputation",
    "brits_impute_only": "BRITS impute only",
}

METHOD_ORDER = tuple(METHOD_LABELS)
COLORS = {
    "full_data": "#74c7b0",
    "role_vc_brits_mi": "#3b0f00",
    "role_vc_saits_mi": "#001f10",
    "avc_smc_saits_mi": "#002d16",
    "avc_smc_brits_mi": "#4d1600",
    "vc_smc_saits_mi": "#00441b",
    "vc_smc_brits_mi": "#7f2704",
    "smc_saits_mi": "#005a32",
    "smc_brits_mi": "#8c2d04",
    "gm_saits_stochastic_mi": "#018571",
    "gm_brits_stochastic_mi": "#a6611a",
    "saits_outcome_history": "#1b9e77",
    "brits_outcome_history": "#d95f02",
    "mice": "#f2d263",
    "missforest": "#66a61e",
    "mean_impute": "#c4a76b",
    "brits_impute_only": "#9b95d1",
}

SCENARIOS = {
    "baseline_irregular": {
        "label": "Baseline irregularity",
        "time": np.array([0.00, 0.12, 0.29, 0.51, 0.76, 1.00]),
        "late_accel_mult": 1.00,
        "pulse_mult": 1.00,
        "noise_mult": 1.00,
        "missing_abnormality_mult": 1.00,
        "missing_slope_mult": 1.00,
    },
    "strong_irregular": {
        "label": "Strong irregularity",
        "time": np.array([0.00, 0.05, 0.18, 0.43, 0.72, 1.00]),
        "late_accel_mult": 1.28,
        "pulse_mult": 1.20,
        "noise_mult": 1.10,
        "missing_abnormality_mult": 1.25,
        "missing_slope_mult": 1.35,
    },
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _simulate_irregular_trajectory(
    rng: np.random.Generator,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    cfg = SCENARIOS[scenario]
    time = np.asarray(cfg["time"], dtype=float)
    n = len(diabetes)
    severity = rng.normal(size=n) + 0.36 * diabetes + 0.22 * htn - 0.08 * male
    chronic_slope = rng.normal(scale=0.32, size=n) + 0.22 * diabetes + 0.15 * htn
    late_activity = rng.normal(size=n) + 0.18 * diabetes + 0.12 * htn
    late_acceleration = rng.normal(size=n) + 0.70 * late_activity + 0.22 * diabetes + 0.18 * htn
    pulse_center = rng.uniform(0.58, 0.88, size=n)
    pulse_width = rng.uniform(0.08, 0.15, size=n)

    t = time[None, :]
    pulse = np.exp(-0.5 * ((t - pulse_center[:, None]) / pulse_width[:, None]) ** 2)
    bend = np.maximum(t - 0.45, 0.0) ** 1.35
    late_mult = float(cfg["late_accel_mult"])
    pulse_mult = float(cfg["pulse_mult"])
    noise_mult = float(cfg["noise_mult"])
    ast = (
        0.16 * severity[:, None]
        + 0.42 * chronic_slope[:, None] * t
        + 0.96 * late_mult * late_acceleration[:, None] * bend
        + 0.42 * pulse_mult * late_activity[:, None] * pulse
        + 0.12 * diabetes[:, None]
        + rng.normal(scale=0.40 * noise_mult, size=(n, len(time)))
    )
    alt = (
        0.12 * severity[:, None]
        + 0.34 * chronic_slope[:, None] * t
        + 0.78 * late_mult * late_acceleration[:, None] * bend
        + 0.34 * pulse_mult * late_activity[:, None] * pulse
        + 0.18 * diabetes[:, None]
        + rng.normal(scale=0.44 * noise_mult, size=(n, len(time)))
    )
    platelet = (
        -0.14 * severity[:, None]
        - 0.38 * chronic_slope[:, None] * t
        - 0.88 * late_mult * late_acceleration[:, None] * bend
        - 0.38 * pulse_mult * late_activity[:, None] * pulse
        - 0.12 * htn[:, None]
        + rng.normal(scale=0.40 * noise_mult, size=(n, len(time)))
    )
    trajectory = np.stack([ast, alt, platelet], axis=2)
    return trajectory, time, {
        "severity": severity,
        "late_activity": late_activity,
        "late_acceleration": late_acceleration,
        "pulse_center": pulse_center,
    }


def _make_missingness_targeted(
    trajectory: np.ndarray,
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    latent: dict[str, np.ndarray],
    time: np.ndarray,
    rng: np.random.Generator,
    target_late_missing: float,
    scenario: str,
) -> np.ndarray:
    cfg = SCENARIOS[scenario]
    features = dgp1._trajectory_features(trajectory, time)
    slope_pressure = (
        0.30 * features["slope_ast"].to_numpy()
        + 0.22 * features["slope_alt"].to_numpy()
        - 0.28 * features["slope_platelet"].to_numpy()
    )
    visit_shift = np.array([-0.30, -0.06, -0.16, -0.50, 0.12, 0.62])
    marker_shift = np.array([0.05, -0.02, 0.14])
    abnormality = np.stack([trajectory[:, :, 0], trajectory[:, :, 1], -trajectory[:, :, 2]], axis=2)
    component = (
        1.05 * outcome[:, None, None]
        + 0.28 * diabetes[:, None, None]
        + 0.16 * htn[:, None, None]
        + visit_shift[None, :, None]
        + marker_shift[None, None, :]
        + float(cfg["missing_abnormality_mult"]) * 0.18 * abnormality
        + float(cfg["missing_slope_mult"]) * 0.26 * slope_pressure[:, None, None] * (time[None, :, None] > 0.50)
        + 0.08 * latent["late_activity"][:, None, None]
        + rng.normal(scale=0.24, size=trajectory.shape)
    )
    late = np.zeros_like(component, dtype=bool)
    late[:, -3:, :] = True
    lo, hi = -8.0, 8.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        expected = float(_sigmoid(mid + component)[late].mean())
        if expected < target_late_missing:
            lo = mid
        else:
            hi = mid
    p_missing = _sigmoid(0.5 * (lo + hi) + component)
    return rng.binomial(1, 1.0 - p_missing).astype(bool)


def _make_core_methods(
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
        "mean_impute": [mean_fill],
    }
    methods["brits_impute_only"] = [
        hist_sim._linear_cell_impute(
            late_obs_cells,
            base_imp,
            history_by_cell,
            None,
            rng,
            0.0 if j == 0 else hist_sim.BRITS_RESIDUAL_SCALE,
            use_history=False,
        )
        for j in range(n_imputations)
    ]
    methods["brits_outcome_history"] = [
        hist_sim._sequential_history_cell_impute(
            late_obs_cells,
            base_imp,
            outcome_imp,
            trajectory_obs,
            observed,
            time,
            rng,
            0.0 if j == 0 else hist_sim.BRITS_RESIDUAL_SCALE,
        )
        for j in range(n_imputations)
    ]
    methods["saits_outcome_history"] = [
        hist_sim._saits_style_cell_impute(
            late_obs_cells,
            base_imp,
            history_by_cell,
            outcome_imp,
            time,
            rng,
            0.0 if j == 0 else hist_sim.BRITS_RESIDUAL_SCALE,
            use_history=True,
        )
        for j in range(n_imputations)
    ]
    methods["mice"] = [
        coefsim._chained_linear_impute(
            late_obs_cells,
            base_imp,
            rng,
            n_iter=2,
            residual_scale=hist_sim.MICE_RESIDUAL_SCALE,
        )
        for _ in range(n_imputations)
    ]
    # Missforest is expensive and does not benefit much from 50 repeated draws;
    # use stochastic bootstrap-style draws but cap at 20 and recycle for M=50.
    mf_draws = [
        coefsim._missforest_style_impute(
            late_obs_cells,
            base_imp,
            rng,
            n_iter=1,
            residual_scale=hist_sim.MISSFOREST_RESIDUAL_SCALE,
        )
        for _ in range(min(n_imputations, 20))
    ]
    while len(mf_draws) < n_imputations:
        mf_draws.append(mf_draws[len(mf_draws) % len(mf_draws)].copy())
    methods["missforest"] = mf_draws
    methods["gm_brits_stochastic_mi"] = hist_sim._missingness_calibrated_stochastic_draws(
        methods["brits_outcome_history"],
        late_obs_cells,
        base_imp,
        history_by_cell,
        outcome_imp,
        time,
        rng,
        use_attention=False,
    )
    methods["gm_saits_stochastic_mi"] = hist_sim._missingness_calibrated_stochastic_draws(
        methods["saits_outcome_history"],
        late_obs_cells,
        base_imp,
        history_by_cell,
        outcome_imp,
        time,
        rng,
        use_attention=True,
    )
    return methods


def _fit_substantive_model(
    draw: pd.DataFrame,
    trajectory_obs: np.ndarray,
    time: np.ndarray,
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
) -> tuple[float, pd.Series] | None:
    """Fit the downstream fibrosis model implied by the planned analysis.

    The score is the average outcome log likelihood for the completed data.
    This makes the imputation draw compatible with the same late exposure,
    slope, and interaction terms that will be used after imputation.
    """
    completed = hist_sim._frame_to_late_trajectory(draw, trajectory_obs)
    features = dgp1._trajectory_features(completed, time)
    design = dgp1._design(features, diabetes, htn, male)
    keep = design.notna().all(axis=1).to_numpy()
    if keep.sum() < 50:
        return None
    try:
        fit = dgp1.sm.GLM(
            outcome[keep],
            dgp1.sm.add_constant(design.loc[keep].reset_index(drop=True), has_constant="add"),
            family=dgp1.sm.families.Binomial(),
        ).fit(maxiter=80, disp=False)
    except Exception:
        return None
    params = fit.params.drop("const")
    return float(fit.llf / keep.sum()), params


def _role_scaled_candidate(
    seed_draw: pd.DataFrame,
    late_obs_cells: pd.DataFrame,
    params: pd.Series | None,
    time: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    rng: np.random.Generator,
    base_noise: float,
) -> pd.DataFrame:
    """Perturb missing cells with scale weighted by each cell's role in the outcome model."""
    out = seed_draw.copy()
    late_missing = late_obs_cells.isna()
    for local_j, visit in enumerate(range(4, 7)):
        for marker in BIOMARKERS:
            col = f"{marker}_v{visit}"
            missing = late_missing[col].to_numpy()
            if not missing.any():
                continue
            values = out[col].to_numpy(dtype=float).copy()
            sd = float(np.nanstd(values))
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0
            role_scale = np.ones(len(values), dtype=float)
            if params is not None:
                sensitivity = np.abs(hist_sim._cell_score_sensitivity(params, marker, visit, time, diabetes, htn))
                denom = float(np.nanmedian(sensitivity[missing])) if np.isfinite(sensitivity[missing]).any() else 0.0
                if denom > 1e-8:
                    role_scale = np.clip(0.70 + 0.45 * sensitivity / denom, 0.55, 2.00)
            values[missing] = values[missing] + rng.normal(
                scale=base_noise * sd * role_scale[missing],
                size=int(missing.sum()),
            )
            out[col] = values
    return out


def _substantive_model_compatible_draws(
    seed_draws: list[pd.DataFrame],
    late_obs_cells: pd.DataFrame,
    trajectory_obs: np.ndarray,
    time: np.ndarray,
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    rng: np.random.Generator,
    n_imputations: int,
    base_noise: float = 0.11,
    n_candidates_per_draw: int = 4,
    temperature: float = 85.0,
    distance_penalty: float = 0.20,
) -> list[pd.DataFrame]:
    """Substantive-model-compatible deep MI.

    Starting from BRITS/SAITS stochastic draws, generate local candidate draws
    at originally missing cells. Candidate draws are weighted by compatibility
    with the downstream outcome model actually used after imputation. This is
    a stochastic MI analogue of substantive-model-compatible imputation rather
    than a deterministic outcome-targeted correction.
    """
    if not seed_draws:
        return []

    missing = late_obs_cells.isna().to_numpy()
    candidates: list[pd.DataFrame] = []
    scores: list[float] = []
    distances: list[float] = []
    params_cache: list[pd.Series | None] = []
    for seed_draw in seed_draws:
        fit = _fit_substantive_model(seed_draw, trajectory_obs, time, outcome, diabetes, htn, male)
        params = fit[1] if fit is not None else None
        params_cache.append(params)

    for seed_draw, params in zip(seed_draws, params_cache):
        base_fit = _fit_substantive_model(seed_draw, trajectory_obs, time, outcome, diabetes, htn, male)
        if base_fit is not None:
            candidates.append(seed_draw.copy())
            scores.append(base_fit[0])
            distances.append(0.0)
        seed_values = seed_draw.to_numpy(dtype=float)
        for _ in range(n_candidates_per_draw):
            cand = _role_scaled_candidate(
                seed_draw,
                late_obs_cells,
                params,
                time,
                diabetes,
                htn,
                rng,
                base_noise=base_noise,
            )
            fit = _fit_substantive_model(cand, trajectory_obs, time, outcome, diabetes, htn, male)
            if fit is None:
                continue
            cand_values = cand.to_numpy(dtype=float)
            diff = cand_values[missing] - seed_values[missing]
            dist = float(np.sqrt(np.nanmean(diff**2))) if diff.size else 0.0
            candidates.append(cand)
            scores.append(fit[0])
            distances.append(dist)

    if not candidates:
        return seed_draws[:n_imputations]

    score = np.asarray(scores, dtype=float)
    distance = np.asarray(distances, dtype=float)
    if np.isfinite(distance).any() and float(np.nanstd(distance)) > 1e-8:
        distance = (distance - float(np.nanmean(distance))) / float(np.nanstd(distance))
    else:
        distance = np.zeros_like(distance)
    logits = temperature * (score - np.nanmax(score)) - distance_penalty * distance
    logits = np.where(np.isfinite(logits), logits, -1e9)
    weights = np.exp(logits - np.max(logits))
    weights = weights / weights.sum()
    replace = len(candidates) < n_imputations
    chosen = rng.choice(len(candidates), size=n_imputations, replace=replace, p=weights)
    return [candidates[int(idx)].copy() for idx in chosen]


def _adaptive_smc_draws(
    base_draws: list[pd.DataFrame],
    trajectory_obs: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    rng: np.random.Generator,
    step_size: float = 0.026,
    perturbation_penalty: float = 0.0035,
) -> list[pd.DataFrame]:
    """Accept SMC updates only when outcome-model compatibility improves enough.

    This keeps the stochastic deep MI draw as the default and uses the planned
    outcome model to accept bounded association-preserving fluctuations. The
    penalty prevents large changes that improve apparent likelihood but degrade
    imputation plausibility.
    """
    candidates = hist_sim._targeted_association_fluctuation(
        base_draws,
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
        step_size=step_size,
    )
    late_missing = ~observed[:, -3:, :]
    accepted: list[pd.DataFrame] = []
    for base, cand in zip(base_draws, candidates):
        base_fit = _fit_substantive_model(base, trajectory_obs, time, outcome, diabetes, htn, male)
        cand_fit = _fit_substantive_model(cand, trajectory_obs, time, outcome, diabetes, htn, male)
        if base_fit is None or cand_fit is None:
            accepted.append(base.copy())
            continue
        base_vals = base.to_numpy(dtype=float)
        cand_vals = cand.to_numpy(dtype=float)
        diff = cand_vals[late_missing.reshape(late_missing.shape[0], -1)] - base_vals[
            late_missing.reshape(late_missing.shape[0], -1)
        ]
        distance = float(np.sqrt(np.nanmean(diff**2))) if diff.size else 0.0
        scale = float(np.nanstd(base_vals[late_missing.reshape(late_missing.shape[0], -1)]))
        rel_distance = distance / max(scale, 1e-6) if np.isfinite(scale) else distance
        if cand_fit[0] - base_fit[0] > perturbation_penalty * rel_distance:
            accepted.append(cand.copy())
        else:
            accepted.append(base.copy())
    return accepted


def _role_aware_variance_draws(
    seed_draws: list[pd.DataFrame],
    late_obs_cells: pd.DataFrame,
    trajectory_obs: np.ndarray,
    time: np.ndarray,
    outcome: np.ndarray,
    diabetes: np.ndarray,
    htn: np.ndarray,
    male: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float = 0.035,
) -> list[pd.DataFrame]:
    """Inflate MI uncertainty according to each missing cell's outcome-model role.

    This preserves the strong deep-imputation conditional mean but uses the
    planned downstream model to allocate more between-imputation variability to
    cells with larger derivative in the scientific outcome model.
    """
    late_missing = late_obs_cells.isna()
    out_draws: list[pd.DataFrame] = []
    for draw in seed_draws:
        fit = _fit_substantive_model(draw, trajectory_obs, time, outcome, diabetes, htn, male)
        params = fit[1] if fit is not None else None
        out = draw.copy()
        for local_j, visit in enumerate(range(4, 7)):
            for marker in BIOMARKERS:
                col = f"{marker}_v{visit}"
                missing = late_missing[col].to_numpy()
                if not missing.any():
                    continue
                values = out[col].to_numpy(dtype=float).copy()
                sd = float(np.nanstd(values))
                if not np.isfinite(sd) or sd <= 0:
                    sd = 1.0
                role = np.ones(len(values), dtype=float)
                if params is not None:
                    sensitivity = np.abs(hist_sim._cell_score_sensitivity(params, marker, visit, time, diabetes, htn))
                    med = float(np.nanmedian(sensitivity[missing])) if np.isfinite(sensitivity[missing]).any() else 0.0
                    if med > 1e-8:
                        role = np.clip(0.75 + 0.35 * sensitivity / med, 0.70, 1.75)
                values[missing] = values[missing] + rng.normal(
                    scale=noise_scale * sd * role[missing],
                    size=int(missing.sum()),
                )
                out[col] = values
        out_draws.append(out)
    return out_draws


def run_one(
    seed: int,
    n: int,
    n_imputations: int,
    target_missing: float,
    scenario: str,
) -> tuple[list[dict], list[dict], dict]:
    rng = np.random.default_rng(seed)
    diabetes = rng.binomial(1, 0.36, size=n).astype(float)
    htn = rng.binomial(1, _sigmoid(-0.20 + 1.05 * diabetes), size=n).astype(float)
    male = rng.binomial(1, 0.46, size=n).astype(float)
    trajectory, time, latent = _simulate_irregular_trajectory(rng, diabetes, htn, male, scenario)
    true_features = dgp1._trajectory_features(trajectory, time)
    x_true = dgp1._design(true_features, diabetes, htn, male)
    eta = -0.84 + sum(TRUE_BETA[name] * x_true[name].to_numpy() for name in TRUE_BETA)
    outcome = rng.binomial(1, _sigmoid(eta)).astype(int)

    observed = _make_missingness_targeted(trajectory, outcome, diabetes, htn, latent, time, rng, target_missing, scenario)
    trajectory_obs = np.where(observed, trajectory, np.nan)
    base_imp = dgp1._build_base_frame(trajectory_obs, observed, diabetes, htn, male, time)
    outcome_imp = dgp1._outcome_features(true_features, latent, outcome, base_imp, diabetes, htn, male, rng)
    late_true_cells = hist_sim._cell_frame(trajectory)
    late_obs_cells = hist_sim._cell_frame(trajectory_obs)
    history_by_cell = {}
    for local_j, j in enumerate(range(3, 6), start=4):
        for k, name in enumerate(BIOMARKERS):
            history_by_cell[f"{name}_v{local_j}"] = hist_sim._history_features(trajectory_obs, observed, time, j, k)
    methods = {"full_data": [late_true_cells]}
    methods.update(
        _make_core_methods(
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
    )
    methods["smc_brits_mi"] = hist_sim._targeted_association_fluctuation(
        methods["gm_brits_stochastic_mi"],
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
    methods["smc_saits_mi"] = hist_sim._targeted_association_fluctuation(
        methods["gm_saits_stochastic_mi"],
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
    methods["role_vc_brits_mi"] = _role_aware_variance_draws(
        methods["gm_brits_stochastic_mi"],
        late_obs_cells,
        trajectory_obs,
        time,
        outcome,
        diabetes,
        htn,
        male,
        rng,
        noise_scale=0.030,
    )
    methods["role_vc_saits_mi"] = _role_aware_variance_draws(
        methods["gm_saits_stochastic_mi"],
        late_obs_cells,
        trajectory_obs,
        time,
        outcome,
        diabetes,
        htn,
        male,
        rng,
        noise_scale=0.030,
    )
    adaptive_brits = _adaptive_smc_draws(
        methods["gm_brits_stochastic_mi"],
        trajectory_obs,
        observed,
        time,
        outcome,
        diabetes,
        htn,
        male,
        rng,
    )
    adaptive_saits = _adaptive_smc_draws(
        methods["gm_saits_stochastic_mi"],
        trajectory_obs,
        observed,
        time,
        outcome,
        diabetes,
        htn,
        male,
        rng,
    )
    methods["vc_smc_brits_mi"] = hist_sim._missingness_calibrated_stochastic_draws(
        methods["smc_brits_mi"],
        late_obs_cells,
        base_imp,
        history_by_cell,
        outcome_imp,
        time,
        rng,
        use_attention=False,
    )
    methods["vc_smc_saits_mi"] = hist_sim._missingness_calibrated_stochastic_draws(
        methods["smc_saits_mi"],
        late_obs_cells,
        base_imp,
        history_by_cell,
        outcome_imp,
        time,
        rng,
        use_attention=True,
    )
    methods["avc_smc_brits_mi"] = hist_sim._missingness_calibrated_stochastic_draws(
        adaptive_brits,
        late_obs_cells,
        base_imp,
        history_by_cell,
        outcome_imp,
        time,
        rng,
        use_attention=False,
    )
    methods["avc_smc_saits_mi"] = hist_sim._missingness_calibrated_stochastic_draws(
        adaptive_saits,
        late_obs_cells,
        base_imp,
        history_by_cell,
        outcome_imp,
        time,
        rng,
        use_attention=True,
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
                        "scenario_label": SCENARIOS[scenario]["label"],
                        "target_missing": target_missing,
                        "n_imputations": n_imputations,
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
                "scenario_label": SCENARIOS[scenario]["label"],
                "target_missing": target_missing,
                "n_imputations": n_imputations,
                "method": method,
                "imputation_rmse": impute_rmse,
                "summary_mse": summary_mse,
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
    }
    return coef_rows, metric_rows, audit


def _run_one_task(task: tuple[int, int, int, float, str]) -> tuple[list[dict], list[dict], dict]:
    seed, n, n_imputations, target_missing, scenario = task
    return run_one(seed, n, n_imputations, target_missing, scenario)


def add_reference_errors(rows: pd.DataFrame) -> pd.DataFrame:
    full = rows[rows["method"] == "full_data"][
        ["seed", "scenario", "target_missing", "n_imputations", "coefficient", "estimate"]
    ].rename(columns={"estimate": "full_data_estimate"})
    out = rows.merge(full, on=["seed", "scenario", "target_missing", "n_imputations", "coefficient"], how="left")
    out["excess_error"] = out["estimate"] - out["full_data_estimate"]
    out["abs_excess_error"] = out["excess_error"].abs()
    out["squared_excess_error"] = out["excess_error"] ** 2
    out.loc[out["method"] == "full_data", ["excess_error", "abs_excess_error", "squared_excess_error"]] = 0.0
    return out


def summarize(coef_df: pd.DataFrame, metric_df: pd.DataFrame) -> pd.DataFrame:
    coef_summary = (
        coef_df.groupby(["scenario", "scenario_label", "target_missing", "n_imputations", "method"], as_index=False)
        .agg(
            mean_abs_bias=("abs_bias", "mean"),
            coefficient_mse=("squared_error", "mean"),
            mean_abs_excess_error=("abs_excess_error", "mean"),
            excess_mse=("squared_excess_error", "mean"),
            mean_model_se=("std_error", "mean"),
            coverage=("covered", "mean"),
            n_runs=("seed", "nunique"),
        )
    )
    metric_summary = (
        metric_df.groupby(["scenario", "scenario_label", "target_missing", "n_imputations", "method"], as_index=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            log_loss_mean=("log_loss", "mean"),
            imputation_rmse_mean=("imputation_rmse", "mean"),
            summary_mse_mean=("summary_mse", "mean"),
        )
    )
    out = coef_summary.merge(
        metric_summary,
        on=["scenario", "scenario_label", "target_missing", "n_imputations", "method"],
        how="inner",
    )
    out["method_label"] = out["method"].map(METHOD_LABELS)
    return out


def plot_metric_grid(summary: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        ("mean_abs_excess_error", "Coefficient recovery error", "lower"),
        ("coverage", "95% coverage", "higher"),
        ("auroc_mean", "AUROC", "higher"),
        ("imputation_rmse_mean", "Imputation RMSE", "lower"),
    ]
    scenarios = list(SCENARIOS)
    m_values = sorted(summary["n_imputations"].unique())
    for metric, title, direction in metrics:
        fig, axes = plt.subplots(len(scenarios), len(m_values), figsize=(15, 8), dpi=220, sharex=True, squeeze=False)
        for r, scenario in enumerate(scenarios):
            for c, m in enumerate(m_values):
                ax = axes[r, c]
                data = summary[(summary["scenario"] == scenario) & (summary["n_imputations"] == m)]
                for method in METHOD_ORDER:
                    if method == "full_data" or method not in set(data["method"]):
                        continue
                    line = data[data["method"] == method].sort_values("target_missing")
                    ax.plot(
                        100 * line["target_missing"],
                        line[metric],
                        marker="o",
                        linewidth=1.6,
                        markersize=3.8,
                        color=COLORS.get(method, "#999999"),
                        label=METHOD_LABELS[method],
                    )
                if r == 0:
                    ax.set_title(f"M = {m}", fontsize=11, weight="bold")
                if c == 0:
                    ax.set_ylabel(f"{SCENARIOS[scenario]['label']}\n{title}", fontsize=9)
                ax.grid(color="#dddddd", linewidth=0.7)
                ax.set_axisbelow(True)
                ax.set_xlabel("Late-cell missingness target (%)", fontsize=8)
        fig.suptitle(f"DGP1 grid: {title} ({direction} is better)", fontsize=15, weight="bold")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.10, 1, 0.94))
        fig.savefig(outdir / f"dgp1_grid_{metric}.png", bbox_inches="tight")
        plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame, outdir: Path) -> None:
    scenarios = list(SCENARIOS)
    m_values = sorted(summary["n_imputations"].unique())
    fig, axes = plt.subplots(len(scenarios), len(m_values), figsize=(15, 8), dpi=220, squeeze=False)
    for r, scenario in enumerate(scenarios):
        for c, m in enumerate(m_values):
            ax = axes[r, c]
            data = summary[(summary["scenario"] == scenario) & (summary["n_imputations"] == m)]
            for method in METHOD_ORDER:
                if method == "full_data":
                    continue
                line = data[data["method"] == method].sort_values("target_missing")
                if line.empty:
                    continue
                ax.plot(
                    line["mean_abs_excess_error"],
                    line["auroc_mean"],
                    marker="o",
                    linewidth=1.5,
                    markersize=4,
                    color=COLORS.get(method, "#999999"),
                    label=METHOD_LABELS[method],
                    alpha=0.92,
                )
                for _, row in line.iterrows():
                    ax.text(
                        row["mean_abs_excess_error"],
                        row["auroc_mean"],
                        f"{int(round(100 * row['target_missing']))}",
                        fontsize=6.5,
                        ha="center",
                        va="center",
                        color="white" if method.startswith("gm_") else "#222222",
                    )
            if r == 0:
                ax.set_title(f"M = {m}", fontsize=11, weight="bold")
            if c == 0:
                ax.set_ylabel(f"{SCENARIOS[scenario]['label']}\nAUROC", fontsize=9)
            ax.set_xlabel("Recovery error; lower is better", fontsize=8)
            ax.grid(color="#dddddd", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.invert_xaxis()
    fig.suptitle("DGP1 accuracy-recovery trade-off grid\nPoint labels show missingness target (%)", fontsize=15, weight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(outdir / "dgp1_accuracy_recovery_tradeoff_grid.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("outputs/gm_mi_dgp1_grid"))
    parser.add_argument("--nsim", type=int, default=100)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--missing", type=float, nargs="+", default=[0.20, 0.40, 0.60])
    parser.add_argument("--m-values", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    coef_rows: list[dict] = []
    metric_rows: list[dict] = []
    audit_rows: list[dict] = []
    tasks: list[tuple[int, int, int, float, str]] = []
    job = 0
    for scenario in args.scenarios:
        for target_missing in args.missing:
            for m in args.m_values:
                for rep in range(args.nsim):
                    seed = args.seed_start + job * 1000 + rep
                    tasks.append((seed, args.n, m, target_missing, scenario))
                job += 1
    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            for c_rows, rows, audit in executor.map(_run_one_task, tasks, chunksize=1):
                coef_rows.extend(c_rows)
                metric_rows.extend(rows)
                audit_rows.append(audit)
    else:
        for task in tasks:
            c_rows, rows, audit = _run_one_task(task)
            coef_rows.extend(c_rows)
            metric_rows.extend(rows)
            audit_rows.append(audit)

    coef_df = add_reference_errors(pd.DataFrame(coef_rows))
    metric_df = pd.DataFrame(metric_rows)
    audit_df = pd.DataFrame(audit_rows)
    summary = summarize(coef_df, metric_df)

    coef_df.to_csv(args.outdir / "coefficient_by_run.csv", index=False)
    metric_df.to_csv(args.outdir / "metrics_by_run.csv", index=False)
    audit_df.to_csv(args.outdir / "missingness_audit.csv", index=False)
    summary.to_csv(args.outdir / "summary_accuracy_recovery.csv", index=False)
    plot_metric_grid(summary, args.outdir)
    plot_tradeoff(summary, args.outdir)
    with open(args.outdir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "nsim": args.nsim,
                "n": args.n,
                "seed_start": args.seed_start,
                "missing": args.missing,
                "m_values": args.m_values,
                "scenarios": {name: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cfg.items()} for name, cfg in SCENARIOS.items()},
                "methods": METHOD_LABELS,
                "n_workers": args.n_workers,
            },
            fh,
            indent=2,
        )
    display = summary.sort_values(["scenario", "target_missing", "n_imputations", "mean_abs_excess_error"])
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
