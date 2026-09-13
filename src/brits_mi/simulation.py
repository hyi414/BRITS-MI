"""Known-truth irregular longitudinal association-recovery simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import design_matrix, trajectory_features

BIOMARKERS = ("ast", "alt", "platelet")

TRUE_BETA = {
    "late_ast": 0.32,
    "late_alt": 0.24,
    "late_platelet": -0.34,
    "slope_ast": 0.34,
    "slope_alt": 0.22,
    "slope_platelet": -0.28,
    "diabetes": 0.22,
    "hypertension": 0.16,
    "male": 0.04,
    "late_ast_x_diabetes": 0.24,
    "slope_ast_x_diabetes": 0.20,
    "slope_platelet_x_hypertension": -0.18,
    "hypertension_x_male": -0.08,
}


SCENARIOS = {
    "mild_harder": {
        "times": np.array([0.00, 0.10, 0.27, 0.50, 0.77, 1.00]),
        "late_acceleration": 1.05,
        "pulse": 1.05,
        "noise": 1.05,
        "missing_abnormality": 1.10,
        "missing_slope": 1.15,
    },
    "baseline_harder": {
        "times": np.array([0.00, 0.07, 0.22, 0.46, 0.73, 1.00]),
        "late_acceleration": 1.30,
        "pulse": 1.25,
        "noise": 1.15,
        "missing_abnormality": 1.40,
        "missing_slope": 1.50,
    },
    "strong_harder": {
        "times": np.array([0.00, 0.03, 0.14, 0.37, 0.67, 1.00]),
        "late_acceleration": 1.65,
        "pulse": 1.45,
        "noise": 1.25,
        "missing_abnormality": 1.80,
        "missing_slope": 2.00,
    },
    "baseline_irregular": {
        "times": np.array([0.00, 0.12, 0.29, 0.51, 0.76, 1.00]),
        "late_acceleration": 1.00,
        "pulse": 1.00,
        "noise": 1.00,
        "missing_abnormality": 1.00,
        "missing_slope": 1.00,
    },
    "strong_irregular": {
        "times": np.array([0.00, 0.05, 0.18, 0.43, 0.72, 1.00]),
        "late_acceleration": 1.28,
        "pulse": 1.20,
        "noise": 1.10,
        "missing_abnormality": 1.25,
        "missing_slope": 1.35,
    },
}


@dataclass(frozen=True)
class SimulationConfig:
    n_subjects: int = 500
    target_missing: float = 0.20
    scenario: str = "baseline_harder"
    seed: int = 1
    outcome_intercept: float = -0.84


@dataclass
class SimulationData:
    true_values: np.ndarray
    observed_values: np.ndarray
    observed_mask: np.ndarray
    times: np.ndarray
    static_covariates: np.ndarray
    outcome: np.ndarray
    true_features: pd.DataFrame
    true_design: pd.DataFrame
    latent: dict[str, np.ndarray]
    config: SimulationConfig


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _simulate_trajectory(
    rng: np.random.Generator,
    diabetes: np.ndarray,
    hypertension: np.ndarray,
    male: np.ndarray,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    settings = SCENARIOS[scenario]
    times = np.asarray(settings["times"], dtype=float)
    n_subjects = len(diabetes)
    severity = rng.normal(size=n_subjects) + 0.36 * diabetes + 0.22 * hypertension - 0.08 * male
    chronic_slope = rng.normal(scale=0.32, size=n_subjects) + 0.22 * diabetes + 0.15 * hypertension
    late_activity = rng.normal(size=n_subjects) + 0.18 * diabetes + 0.12 * hypertension
    late_acceleration = (
        rng.normal(size=n_subjects)
        + 0.70 * late_activity
        + 0.22 * diabetes
        + 0.18 * hypertension
    )
    pulse_center = rng.uniform(0.58, 0.88, size=n_subjects)
    pulse_width = rng.uniform(0.08, 0.15, size=n_subjects)

    time_grid = times[None, :]
    pulse = np.exp(
        -0.5 * ((time_grid - pulse_center[:, None]) / pulse_width[:, None]) ** 2
    )
    bend = np.maximum(time_grid - 0.45, 0.0) ** 1.35
    late_multiplier = float(settings["late_acceleration"])
    pulse_multiplier = float(settings["pulse"])
    noise_multiplier = float(settings["noise"])
    ast = (
        0.16 * severity[:, None]
        + 0.42 * chronic_slope[:, None] * time_grid
        + 0.96 * late_multiplier * late_acceleration[:, None] * bend
        + 0.42 * pulse_multiplier * late_activity[:, None] * pulse
        + 0.12 * diabetes[:, None]
        + rng.normal(scale=0.40 * noise_multiplier, size=(n_subjects, len(times)))
    )
    alt = (
        0.12 * severity[:, None]
        + 0.34 * chronic_slope[:, None] * time_grid
        + 0.78 * late_multiplier * late_acceleration[:, None] * bend
        + 0.34 * pulse_multiplier * late_activity[:, None] * pulse
        + 0.18 * diabetes[:, None]
        + rng.normal(scale=0.44 * noise_multiplier, size=(n_subjects, len(times)))
    )
    platelet = (
        -0.14 * severity[:, None]
        - 0.38 * chronic_slope[:, None] * time_grid
        - 0.88 * late_multiplier * late_acceleration[:, None] * bend
        - 0.38 * pulse_multiplier * late_activity[:, None] * pulse
        - 0.12 * hypertension[:, None]
        + rng.normal(scale=0.40 * noise_multiplier, size=(n_subjects, len(times)))
    )
    return np.stack([ast, alt, platelet], axis=2), times, {
        "severity": severity,
        "late_activity": late_activity,
        "late_acceleration": late_acceleration,
        "pulse_center": pulse_center,
    }


def _generate_observation_mask(
    trajectory: np.ndarray,
    outcome: np.ndarray,
    diabetes: np.ndarray,
    hypertension: np.ndarray,
    latent: dict[str, np.ndarray],
    times: np.ndarray,
    rng: np.random.Generator,
    target_missing: float,
    scenario: str,
) -> np.ndarray:
    settings = SCENARIOS[scenario]
    features = trajectory_features(trajectory, times)
    slope_pressure = (
        0.30 * features["slope_ast"].to_numpy()
        + 0.22 * features["slope_alt"].to_numpy()
        - 0.28 * features["slope_platelet"].to_numpy()
    )
    visit_shift = np.array([-0.30, -0.06, -0.16, -0.50, 0.12, 0.62])
    marker_shift = np.array([0.05, -0.02, 0.14])
    abnormality = np.stack(
        [trajectory[:, :, 0], trajectory[:, :, 1], -trajectory[:, :, 2]],
        axis=2,
    )
    component = (
        1.05 * outcome[:, None, None]
        + 0.28 * diabetes[:, None, None]
        + 0.16 * hypertension[:, None, None]
        + visit_shift[None, :, None]
        + marker_shift[None, None, :]
        + float(settings["missing_abnormality"]) * 0.18 * abnormality
        + float(settings["missing_slope"])
        * 0.26
        * slope_pressure[:, None, None]
        * (times[None, :, None] > 0.50)
        + 0.08 * latent["late_activity"][:, None, None]
        + rng.normal(scale=0.24, size=trajectory.shape)
    )
    late_cells = np.zeros_like(component, dtype=bool)
    late_cells[:, -3:, :] = True
    lower, upper = -8.0, 8.0
    for _ in range(45):
        midpoint = 0.5 * (lower + upper)
        expected = float(_sigmoid(midpoint + component)[late_cells].mean())
        if expected < target_missing:
            lower = midpoint
        else:
            upper = midpoint
    missing_probability = _sigmoid(0.5 * (lower + upper) + component)
    return rng.binomial(1, 1.0 - missing_probability).astype(bool)


def simulate_clinical_association(config: SimulationConfig | None = None) -> SimulationData:
    """Generate one known-truth association-recovery replicate."""

    config = config or SimulationConfig()
    if config.scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {sorted(SCENARIOS)}")
    if not 0.0 < config.target_missing < 1.0:
        raise ValueError("target_missing must be between 0 and 1")
    if config.n_subjects < 20:
        raise ValueError("n_subjects must be at least 20")

    rng = np.random.default_rng(config.seed)
    diabetes = rng.binomial(1, 0.36, size=config.n_subjects).astype(float)
    hypertension = rng.binomial(
        1,
        _sigmoid(-0.20 + 1.05 * diabetes),
        size=config.n_subjects,
    ).astype(float)
    male = rng.binomial(1, 0.46, size=config.n_subjects).astype(float)
    trajectory, times, latent = _simulate_trajectory(
        rng, diabetes, hypertension, male, config.scenario
    )
    features = trajectory_features(trajectory, times)
    design = design_matrix(features, diabetes, hypertension, male)
    linear_predictor = config.outcome_intercept + sum(
        TRUE_BETA[name] * design[name].to_numpy() for name in TRUE_BETA
    )
    outcome = rng.binomial(1, _sigmoid(linear_predictor)).astype(int)
    observed_mask = _generate_observation_mask(
        trajectory,
        outcome,
        diabetes,
        hypertension,
        latent,
        times,
        rng,
        config.target_missing,
        config.scenario,
    )
    observed_values = np.where(observed_mask, trajectory, np.nan)
    return SimulationData(
        true_values=trajectory,
        observed_values=observed_values,
        observed_mask=observed_mask,
        times=times,
        static_covariates=np.column_stack([diabetes, hypertension, male]),
        outcome=outcome,
        true_features=features,
        true_design=design,
        latent=latent,
        config=config,
    )
