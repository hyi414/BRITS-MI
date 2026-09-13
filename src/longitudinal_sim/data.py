from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ExperimentConfig


N_BIOMARKERS = 3
BIOMARKER_NAMES = ["ast", "alt", "platelet"]


@dataclass
class LongitudinalData:
    y_true: np.ndarray
    y_obs: np.ndarray
    mask: np.ndarray
    x: np.ndarray
    x_mask: np.ndarray
    static: np.ndarray
    times: np.ndarray
    deltas: np.ndarray
    lengths: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    image: np.ndarray
    image_features: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _generate_static_features(n: int, ps: int, rng: np.random.Generator, info_profile: str) -> np.ndarray:
    survey_dim = max(2, ps // 2)
    clinical_dim = max(2, ps - survey_dim)
    shared = rng.normal(size=(n, 2)).astype(np.float32)
    survey = 0.65 * shared[:, [0]] + 0.2 * shared[:, [1]] + rng.normal(scale=0.7, size=(n, survey_dim)).astype(np.float32)
    clinical = 0.2 * shared[:, [0]] + 0.65 * shared[:, [1]] + rng.normal(scale=0.7, size=(n, clinical_dim)).astype(np.float32)

    if info_profile == "survey_rich":
        survey += rng.normal(scale=0.8, size=(n, survey_dim)).astype(np.float32)
        clinical *= 0.45
    elif info_profile == "clinical_rich":
        clinical += rng.normal(scale=0.8, size=(n, clinical_dim)).astype(np.float32)
        survey *= 0.45
    elif info_profile == "discordant":
        survey += 0.7 * shared[:, [0]]
        clinical -= 0.7 * shared[:, [0]]
    elif info_profile == "sparse_survey":
        survey = 0.25 * survey + rng.normal(scale=0.25, size=(n, survey_dim)).astype(np.float32)

    static = np.concatenate([survey, clinical], axis=1)
    if static.shape[1] < ps:
        static = np.concatenate([static, rng.normal(scale=0.35, size=(n, ps - static.shape[1])).astype(np.float32)], axis=1)
    return static[:, :ps].astype(np.float32)


def _selected_static_indices(ps: int) -> dict[str, int]:
    survey_end = max(1, ps // 2)
    clinical_start = survey_end
    return {
        "survey_risk": 0,
        "survey_access": min(survey_end - 1, 1),
        "diabetes": clinical_start if clinical_start < ps else ps - 1,
        "lab_measure": min(ps - 1, clinical_start + 1),
    }


def _survey_signal(static_row: np.ndarray, survey_idx: np.ndarray) -> float:
    return float(np.mean(static_row[survey_idx]) + 0.4 * np.std(static_row[survey_idx]))


def _clinical_signal(static_row: np.ndarray, clinical_idx: np.ndarray) -> float:
    return float(np.mean(static_row[clinical_idx]) + 0.4 * np.std(static_row[clinical_idx]))


def _quadratic_terms(y: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    if y.shape[0] <= 1:
        return float(y[0]), 0.0, 0.0
    t_center = t - t.mean()
    design = np.column_stack([np.ones_like(t_center), t_center, t_center**2])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def _trajectory_feature_names() -> list[str]:
    names: list[str] = []
    for marker in BIOMARKER_NAMES:
        names.extend(
            [
                f"{marker}_intercept",
                f"{marker}_slope",
                f"{marker}_curvature",
                f"{marker}_late_mean",
            ]
        )
    names.extend(
        [
            "ast_alt_gap",
            "ast_platelet_gap",
            "alt_platelet_gap",
            "diabetes",
            "lab_measure",
            "survey_risk",
            "survey_access",
        ]
    )
    return names


def _trajectory_feature_vector(
    yv: np.ndarray,
    times: np.ndarray,
    static_row: np.ndarray,
    static_indices: dict[str, int],
) -> np.ndarray:
    late_start = max(0, int(0.6 * yv.shape[0]))
    late = yv[late_start:]
    marker_features = []
    for marker_idx in range(yv.shape[1]):
        intercept, slope, curvature = _quadratic_terms(yv[:, marker_idx], times)
        late_mean = float(late[:, marker_idx].mean())
        marker_features.extend([intercept, slope, curvature, late_mean])
    cross_features = [
        float(late[:, 0].mean() - late[:, 1].mean()),
        float(late[:, 0].mean() - late[:, 2].mean()),
        float(late[:, 1].mean() - late[:, 2].mean()),
    ]
    static_part = [
        float(static_row[static_indices["diabetes"]]),
        float(static_row[static_indices["lab_measure"]]),
        float(static_row[static_indices["survey_risk"]]),
        float(static_row[static_indices["survey_access"]]),
    ]
    return np.asarray(marker_features + cross_features + static_part, dtype=float)


def _generate_liver_image(
    image_size: int,
    severity: float,
    survey_signal: float,
    clinical_signal: float,
    severity_scale: float,
    nuisance_scale: float,
    stage_hint: int,
    motif_strength: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-1.0, 1.0, image_size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    sev = float(np.tanh(severity_scale * severity / 2.5))
    nuisance_phase = rng.uniform(-np.pi, np.pi)
    nuisance_field = nuisance_scale * (
        0.06 * np.sin(2.2 * xx + nuisance_phase)
        + 0.05 * np.cos(1.7 * yy - 0.5 * nuisance_phase)
        + 0.04 * np.sin(2.8 * (xx + yy))
    )
    organ = 0.18 + 0.32 * np.exp(-((xx / 1.1) ** 2 + (yy / 0.7) ** 2))
    capsule = 0.05 * np.exp(-((yy + 0.55) / 0.08) ** 2)
    coarse_texture = (
        0.045 * np.sin((3.0 + 0.4 * sev) * xx + rng.uniform(-np.pi, np.pi))
        + 0.035 * np.cos((4.0 - 0.3 * sev) * yy + rng.uniform(-np.pi, np.pi))
    )
    attenuation = (0.04 + 0.12 * max(sev, 0.0)) * np.clip(yy + 0.05, 0.0, None)
    septa = 0.10 * max(sev, 0.0) * np.exp(-((yy - 0.1 - 0.22 * xx) / 0.14) ** 2)
    speckle = rng.normal(scale=0.025 + 0.02 * abs(clinical_signal) + 0.015 * nuisance_scale, size=(image_size, image_size))
    image2d = organ + capsule + coarse_texture + septa - attenuation + 0.015 * survey_signal + nuisance_field + speckle
    motif_strength = float(np.clip(motif_strength, 0.0, 2.5))
    if stage_hint == 0:
        smooth_arc = 0.05 * np.exp(-((yy + 0.05 + 0.18 * xx) / 0.22) ** 2)
        periportal = 0.04 * np.exp(-((xx + 0.45) ** 2 + (yy + 0.05) ** 2) / 0.10)
        image2d += smooth_arc + periportal
    elif stage_hint == 1:
        bridge_1 = 0.09 * np.exp(-((yy - 0.16 - 0.28 * xx) / 0.11) ** 2)
        bridge_2 = 0.07 * np.exp(-((yy + 0.20 + 0.20 * xx) / 0.10) ** 2)
        mottled = 0.045 * np.sin(5.8 * xx - 1.7 * yy + rng.uniform(-np.pi, np.pi))
        image2d += bridge_1 + bridge_2 + mottled
    else:
        bridge_1 = 0.11 * np.exp(-((yy - 0.18 - 0.30 * xx) / 0.10) ** 2)
        bridge_2 = 0.10 * np.exp(-((yy + 0.24 + 0.26 * xx) / 0.11) ** 2)
        rim = 0.08 * np.exp(-((np.sqrt((1.05 * xx) ** 2 + (0.85 * yy) ** 2) - 0.72) / 0.08) ** 2)
        coarse_nodular = 0.05 * np.sin(7.2 * xx + 0.8 * yy + rng.uniform(-np.pi, np.pi))
        image2d += bridge_1 + bridge_2 + rim + coarse_nodular
    swirl = 0.035 * motif_strength * np.sin((4.5 + 0.3 * stage_hint) * (xx - yy) + rng.uniform(-np.pi, np.pi))
    wedge = 0.045 * motif_strength * np.exp(-((yy - 0.30 * np.sin(2.4 * xx + rng.uniform(-np.pi, np.pi))) / 0.16) ** 2)
    image2d += swirl + wedge
    n_nodules = int(np.clip(np.round(1.0 + 3.0 * max(sev, 0.0) + 0.5 * abs(clinical_signal)), 1, 5))
    for _ in range(n_nodules):
        cx = rng.uniform(-0.55, 0.55)
        cy = rng.uniform(-0.35, 0.5)
        radius = rng.uniform(0.08, 0.18)
        amp = rng.uniform(0.03, 0.09) * (1.0 + max(sev, 0.0))
        image2d += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * radius**2))
    image = np.clip(image2d[np.newaxis, :, :], 0.0, 1.0).astype(np.float32)
    features = np.array(
        [
            float(image.mean()),
            float(image.std()),
            float(np.mean(np.abs(np.diff(image[0], axis=0)))),
            float(np.mean(np.abs(np.diff(image[0], axis=1)))),
            float((image[0] > np.quantile(image[0], 0.9)).mean()),
            float(n_nodules),
        ],
        dtype=np.float32,
    )
    return image, features


def _generate_liver_ultrasound_image(
    image_size: int,
    severity: float,
    survey_signal: float,
    clinical_signal: float,
    severity_scale: float,
    nuisance_scale: float,
    stage_hint: int,
    motif_strength: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-1.0, 1.0, image_size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(xx**2 + (yy + 1.05) ** 2)
    theta = np.arctan2(xx, yy + 1.05)
    sector = ((radius > 0.18) & (radius < 1.55) & (np.abs(theta) < 0.78)).astype(np.float32)
    sev = float(np.tanh(severity_scale * severity / 2.2))
    motif_strength = float(np.clip(motif_strength, 0.0, 2.5))

    base = 0.12 + 0.34 * np.exp(-((radius - 0.85) / 0.40) ** 2)
    fan_gain = 0.08 * np.cos(2.8 * theta + rng.uniform(-np.pi, np.pi))
    depth_atten = (0.05 + 0.10 * max(sev, 0.0)) * np.clip(radius - 0.56, 0.0, None)
    capsule = 0.11 * np.exp(-((radius - 0.54) / 0.030) ** 2)
    septal_arc = 0.15 * max(sev, 0.0) * np.exp(-((theta - 0.12) / 0.08) ** 2) * np.exp(-((radius - 0.92) / 0.22) ** 2)
    bridge_arc = 0.13 * max(sev, 0.0) * np.exp(-((theta + 0.18) / 0.09) ** 2) * np.exp(-((radius - 1.02) / 0.20) ** 2)
    periportal = 0.07 * np.exp(-((theta + 0.30) / 0.07) ** 2) * np.exp(-((radius - 0.76) / 0.10) ** 2)
    sweep = 0.04 * motif_strength * np.sin(10.0 * theta + 3.5 * radius + rng.uniform(-np.pi, np.pi))
    wedge = 0.07 * motif_strength * np.exp(-((theta - 0.06 * np.sin(6.0 * radius)) / 0.06) ** 2)

    shadow = np.zeros_like(base)
    n_shadows = 1 + int(stage_hint >= 1) + int(max(sev, 0.0) > 0.35)
    for _ in range(n_shadows):
        shadow_theta = rng.uniform(-0.32, 0.32)
        shadow_width = rng.uniform(0.03, 0.07)
        shadow_depth = rng.uniform(0.75, 1.20)
        shadow += 0.06 * np.exp(-((theta - shadow_theta) / shadow_width) ** 2) * np.clip(radius - shadow_depth, 0.0, None)

    if stage_hint == 0:
        stage_pattern = periportal + 0.04 * np.exp(-((theta + 0.10) / 0.12) ** 2) * np.exp(-((radius - 0.88) / 0.18) ** 2)
    elif stage_hint == 1:
        stage_pattern = periportal + septal_arc + 0.06 * np.sin(16.0 * theta + rng.uniform(-np.pi, np.pi)) * np.exp(-((radius - 0.95) / 0.20) ** 2)
    else:
        nodular_rim = 0.10 * np.exp(-((radius - 1.10) / 0.07) ** 2) * (1.0 + 0.5 * np.sin(14.0 * theta + rng.uniform(-np.pi, np.pi)))
        stage_pattern = periportal + septal_arc + bridge_arc + nodular_rim

    image2d = base + fan_gain + capsule + stage_pattern + sweep + wedge + 0.01 * survey_signal - depth_atten - shadow

    speckle_field = rng.rayleigh(scale=0.12 + 0.03 * abs(clinical_signal) + 0.02 * nuisance_scale, size=(image_size, image_size)).astype(np.float32)
    multiplicative = 1.0 + 0.26 * (speckle_field - speckle_field.mean())
    additive = rng.normal(scale=0.012 + 0.008 * nuisance_scale, size=(image_size, image_size))
    image2d = image2d * multiplicative + additive

    n_nodules = int(np.clip(np.round(0.8 + 2.4 * max(sev, 0.0) + 0.3 * abs(clinical_signal)), 1, 5))
    for _ in range(n_nodules):
        node_theta = rng.uniform(-0.30, 0.30)
        node_radius = rng.uniform(0.72, 1.18)
        node_scale_t = rng.uniform(0.03, 0.08)
        node_scale_r = rng.uniform(0.05, 0.10)
        node_amp = rng.uniform(0.05, 0.10) * (1.0 + 0.9 * max(sev, 0.0))
        image2d += node_amp * np.exp(-((theta - node_theta) / node_scale_t) ** 2 - ((radius - node_radius) / node_scale_r) ** 2)

    image2d = sector * image2d
    image = np.clip(image2d[np.newaxis, :, :], 0.0, 1.0).astype(np.float32)
    features = np.array(
        [
            float(image.mean()),
            float(image.std()),
            float(np.mean(np.abs(np.diff(image[0], axis=0)))),
            float(np.mean(np.abs(np.diff(image[0], axis=1)))),
            float((image[0] > np.quantile(image[0], 0.9)).mean()),
            float(n_nodules),
        ],
        dtype=np.float32,
    )
    return image, features


def generate_dataset(config: ExperimentConfig) -> LongitudinalData:
    rng = np.random.default_rng(config.seed)
    n = config.n_subjects
    max_t = config.max_visits
    p = config.n_timevarying
    ps = config.n_static

    y_true = np.zeros((n, max_t, N_BIOMARKERS), dtype=np.float32)
    y_obs = np.zeros((n, max_t, N_BIOMARKERS), dtype=np.float32)
    mask = np.zeros((n, max_t, N_BIOMARKERS), dtype=np.float32)
    x = np.zeros((n, max_t, p), dtype=np.float32)
    x_mask = np.ones((n, max_t, p), dtype=np.float32)
    times = np.zeros((n, max_t), dtype=np.float32)
    deltas = np.zeros((n, max_t), dtype=np.float32)
    lengths = rng.integers(config.min_visits, config.max_visits + 1, size=n)
    static = _generate_static_features(n, ps, rng, config.info_profile)
    labels = np.zeros(n, dtype=np.int64)
    image = np.zeros((n, 1, config.image_size, config.image_size), dtype=np.float32)
    image_features = np.zeros((n, 6), dtype=np.float32)

    survey_idx = np.arange(max(1, ps // 2))
    clinical_idx = np.arange(max(1, ps // 2), ps)
    if clinical_idx.size == 0:
        clinical_idx = np.array([ps - 1])
    static_indices = _selected_static_indices(ps)

    base_cross = np.array(
        [
            [0.58, 0.16, -0.08],
            [0.12, 0.54, 0.10],
            [-0.14, -0.18, 0.62],
        ],
        dtype=float,
    )
    delta_loadings = np.array(
        [
            [0.08, 0.06, -0.02],
            [0.04, 0.07, 0.03],
            [-0.06, -0.05, 0.05],
        ],
        dtype=float,
    )
    latent_loadings = np.array(
        [
            [0.32, 0.10],
            [0.18, 0.26],
            [-0.22, -0.18],
        ],
        dtype=float,
    )
    marker_bias = np.array([0.35, 0.15, -0.20], dtype=float)
    true_feature_names = _trajectory_feature_names()
    true_feature_coefs = np.array(
        [
            0.10, 0.24, 0.18, 0.34,
            0.08, 0.20, 0.16, 0.28,
            -0.06, -0.16, -0.12, -0.26,
            0.18, 0.26, 0.14,
            0.12, 0.10, 0.05, -0.03,
        ],
        dtype=float,
    )
    ordinal_cutpoints = np.array([-0.5, 1.0], dtype=float)

    for i in range(n):
        length = int(lengths[i])
        gaps = rng.gamma(shape=1.6, scale=1.05, size=length).astype(np.float32)
        gaps[0] = 0.0
        times_i = np.cumsum(gaps)
        deltas_i = np.diff(np.concatenate([[0.0], times_i])).astype(np.float32)
        times[i, :length] = times_i
        deltas[i, :length] = deltas_i

        survey_signal = _survey_signal(static[i], survey_idx)
        clinical_signal = _clinical_signal(static[i], clinical_idx)
        monitoring_propensity = rng.beta(2.5, 2.0)
        latent_severity = rng.normal(scale=0.95)
        latent_inflammation = rng.normal(scale=0.9)
        frailty = rng.normal(scale=0.8)
        regime_time = rng.uniform(times_i[max(1, length // 4)], times_i[-1])
        pulse_center = rng.uniform(times_i[0], times_i[-1])

        prev_y = rng.normal(scale=0.7, size=N_BIOMARKERS) + latent_loadings @ np.array([latent_severity, latent_inflammation])
        prev_x = rng.normal(scale=0.7, size=p)
        for j in range(length):
            t = times_i[j]
            late_phase = 1.0 / (1.0 + np.exp(-(t - times_i[max(1, length // 2)]) / 0.9))
            season = np.array(
                [
                    np.sin(0.7 * t + 0.2),
                    np.cos(0.55 * t - 0.4),
                    np.sin(0.35 * t + 0.8),
                ],
                dtype=float,
            )
            pulse = np.exp(-0.5 * ((t - pulse_center) / 0.85) ** 2)
            regime = 1.0 if t >= regime_time else -0.4
            cross_state = base_cross @ prev_y
            gap_state = delta_loadings @ prev_y * deltas_i[j]
            latent_state = latent_loadings @ np.array([latent_severity, latent_inflammation])
            nonlinear = np.array(
                [
                    np.tanh(prev_y[1] - 0.6 * prev_y[2]),
                    np.tanh(prev_y[0] + 0.3 * prev_y[2]),
                    np.tanh(-0.5 * prev_y[0] - 0.4 * prev_y[1]),
                ],
                dtype=float,
            )
            static_drive = np.array(
                [
                    0.18 * clinical_signal + 0.05 * survey_signal,
                    0.10 * clinical_signal + 0.08 * survey_signal,
                    -0.14 * clinical_signal + 0.03 * survey_signal,
                ],
                dtype=float,
            )
            severity_bridge = late_phase * np.array(
                [
                    0.45 * latent_severity + 0.12 * latent_inflammation,
                    0.36 * latent_severity + 0.10 * latent_inflammation,
                    -0.30 * latent_severity - 0.08 * latent_inflammation,
                ],
                dtype=float,
            )
            if config.dgp_profile in {"paper_britsadv", "paper_britsadv_img06", "paper_britsadv_img065", "paper_britsadv_img065_altus"}:
                severity_bridge = severity_bridge + late_phase * np.array(
                    [
                        0.34 * latent_severity + 0.12 * frailty + 0.18 * np.tanh(prev_y[0] - 0.8 * prev_y[2]),
                        0.28 * latent_severity + 0.10 * frailty + 0.14 * np.tanh(prev_y[1] - 0.5 * prev_y[2]),
                        -0.30 * latent_severity - 0.14 * frailty - 0.12 * np.tanh(prev_y[0] - prev_y[1]),
                    ],
                    dtype=float,
                )
            elif config.dgp_profile == "paper_targetedgap":
                severity_bridge = severity_bridge + late_phase * np.array(
                    [
                        0.42 * latent_severity + 0.18 * frailty + 0.10 * np.tanh(prev_y[0] - prev_y[2]),
                        0.30 * latent_severity + 0.14 * frailty + 0.08 * np.tanh(prev_y[1] - prev_y[2]),
                        -0.34 * latent_severity - 0.16 * frailty - 0.10 * np.tanh(prev_y[0] - prev_y[1]),
                    ],
                    dtype=float,
                )
            targeted_rebound = np.zeros(N_BIOMARKERS, dtype=float)
            if config.dgp_profile in {"paper_britsadv", "paper_britsadv_img06", "paper_britsadv_img065", "paper_britsadv_img065_altus"}:
                gap_trigger = max(deltas_i[j] - 1.0, 0.0)
                rebound_gate = late_phase * np.tanh(1.8 * gap_trigger)
                targeted_rebound = rebound_gate * np.array(
                    [
                        0.28 * np.sign(prev_y[0] - prev_y[2]) + 0.18 * np.sign(prev_y[1]),
                        0.26 * np.sign(prev_y[0] - prev_y[1]) + 0.10 * np.sign(prev_y[2]),
                        -0.24 * np.sign(prev_y[0] + 0.4 * prev_y[1]) + 0.16 * np.sign(prev_y[2]),
                    ],
                    dtype=float,
                )
            elif config.dgp_profile == "paper_targetedgap":
                gap_trigger = max(deltas_i[j] - 1.15, 0.0)
                rebound_gate = late_phase * np.tanh(1.6 * gap_trigger)
                targeted_rebound = rebound_gate * np.array(
                    [
                        -0.34 * np.sign(prev_y[0] - prev_y[1]) + 0.18 * np.sign(prev_y[2]),
                        0.28 * np.sign(prev_y[0] - prev_y[2]),
                        0.26 * np.sign(prev_y[0] + prev_y[1]) - 0.18 * np.sign(prev_y[2]),
                    ],
                    dtype=float,
                )
            y_now = (
                marker_bias
                + 0.42 * cross_state
                + 0.10 * gap_state
                + 0.28 * season
                + 0.22 * pulse * np.array([1.0, 0.7, -0.6])
                + 0.16 * regime * np.array([0.8, 0.3, -0.5])
                + 0.22 * latent_state
                + 0.10 * nonlinear
                + static_drive
                + severity_bridge
                + targeted_rebound
                + 0.12 * frailty
                + rng.normal(scale=[0.65, 0.65, 0.45], size=N_BIOMARKERS)
            )
            prev_y = y_now
            y_true[i, j] = y_now.astype(np.float32)

            x_prev_mean = prev_y.mean()
            prev_x = (
                0.45 * prev_x
                + 0.18 * np.pad(prev_y, (0, max(0, p - N_BIOMARKERS)), mode="edge")[:p]
                + 0.08 * survey_signal
                + 0.10 * clinical_signal
                + 0.06 * x_prev_mean
                + rng.normal(scale=0.7, size=p)
            )
            x[i, j] = prev_x.astype(np.float32)

        yv = y_true[i, :length].astype(float)
        trajectory_features = _trajectory_feature_vector(yv, times_i.astype(float), static[i], static_indices)
        cross_diff = yv[:, 0] - yv[:, 1]
        platelet_drop = -np.diff(yv[:, 2]) if length > 1 else np.zeros(1)
        ast_alt_ratio = yv[:, 0] / np.maximum(np.abs(yv[:, 1]), 0.6)
        local_curvature = np.diff(yv[:, 0] - 0.7 * yv[:, 2], n=2) if length > 2 else np.zeros(1)
        oscillation = np.std(np.diff(yv, axis=0), axis=0).mean() if length > 1 else 0.0
        cross_instability = np.std(cross_diff) + np.mean(np.abs(np.diff(ast_alt_ratio))) if length > 1 else 0.0
        platelet_signal = np.mean(platelet_drop.clip(min=0.0)) if platelet_drop.size else 0.0
        late_idx = max(1, int(0.6 * length))
        early_idx = max(1, int(0.35 * length))
        late_burden = float(np.mean(yv[late_idx:, 0] + 0.8 * yv[late_idx:, 1] - 0.7 * yv[late_idx:, 2]))
        fibrosis_signature = float(
            np.mean(yv[late_idx:, 0] - 0.6 * yv[late_idx:, 2])
            + 0.7 * np.mean(np.maximum(np.diff(yv[late_idx:, 0], axis=0), 0.0)) if length - late_idx > 1 else np.mean(yv[late_idx:, 0] - 0.6 * yv[late_idx:, 2])
        )
        trajectory_score = float(trajectory_features @ true_feature_coefs)
        shape_score = 0.25 * oscillation + 0.28 * cross_instability + 0.22 * np.mean(np.abs(local_curvature)) + 0.18 * platelet_signal
        early_mean = np.mean(yv[:early_idx], axis=0)
        late_mean = np.mean(yv[late_idx:], axis=0)
        mean_gap = float(np.mean(np.abs(late_mean - early_mean)))
        centered_yv = yv - np.mean(yv, axis=0, keepdims=True)
        centered_diffs = np.diff(centered_yv, axis=0) if length > 1 else np.zeros((1, yv.shape[1]))
        phase_mismatch = float(np.mean(np.sign(np.diff(yv[:, 0], prepend=yv[0, 0])) != np.sign(np.diff(yv[:, 2], prepend=yv[0, 2])))) if length > 1 else 0.0
        rebound_pattern = float(np.mean(np.maximum(np.diff(yv[late_idx:, 1], axis=0), 0.0))) if length - late_idx > 1 else 0.0
        gap_surge = float(
            np.mean(
                np.maximum(np.diff(times_i, prepend=times_i[0]) - np.median(np.diff(times_i, prepend=times_i[0])), 0.0)
                * np.maximum(np.diff(yv[:, 0], prepend=yv[0, 0]), 0.0)
            )
        ) if length > 1 else 0.0
        motif_score = (
            0.34 * np.mean(np.abs(local_curvature))
            + 0.28 * oscillation
            + 0.22 * phase_mismatch
            + 0.24 * rebound_pattern
            + 0.08 * mean_gap
        )
        shared_severity = 0.75 * latent_severity + 0.28 * latent_inflammation + 0.18 * late_burden + 0.15 * shape_score
        if config.dgp_profile in {"paper_britsadv", "paper_britsadv_img06", "paper_britsadv_img065", "paper_britsadv_img065_altus"}:
            centered_curv = np.diff(centered_yv, n=2, axis=0) if length > 2 else np.zeros((1, yv.shape[1]))
            centered_shape = float(
                0.52 * np.mean(np.abs(centered_diffs))
                + 0.64 * np.mean(np.abs(centered_curv))
                + 0.30 * np.std(centered_yv[:, 0] - centered_yv[:, 2])
            )
            centered_late_shift = float(np.mean(np.abs(np.mean(centered_yv[late_idx:], axis=0) - np.mean(centered_yv[:early_idx], axis=0))))
            centered_late_gap = centered_yv[late_idx:] if late_idx < length else centered_yv[-1:]
            late_motif_gap = float(
                np.mean(
                    np.maximum(centered_late_gap[:, 0] - 0.45 * centered_late_gap[:, 2], 0.0)
                    + 0.9 * np.maximum(centered_late_gap[:, 1] - 0.15 * centered_late_gap[:, 2], 0.0)
                )
            )
            gap_rebound = float(
                np.mean(
                    np.maximum(np.diff(centered_yv[late_idx:, 1], prepend=centered_yv[late_idx, 1]), 0.0)
                    * np.maximum(np.diff(times_i[late_idx:], prepend=times_i[late_idx]) - np.median(np.diff(times_i, prepend=times_i[0])), 0.0)
                )
            ) if length - late_idx > 1 else 0.0
            shared_severity = 0.10 * latent_severity + 0.08 * latent_inflammation + 0.70 * motif_score + 0.48 * centered_shape
            latent_score = (
                -2.95
                + 0.52 * shape_score
                + 1.38 * motif_score
                + 1.24 * centered_shape
                + 0.88 * centered_late_shift
                + 0.96 * late_motif_gap
                + 0.62 * gap_surge
                + 0.70 * gap_rebound
                + 0.02 * latent_severity
                + rng.normal(scale=0.82)
            )
            image_severity_scale = 0.06
            image_nuisance_scale = 1.72
            if config.dgp_profile == "paper_britsadv_img06":
                image_severity_scale = 0.16
                image_nuisance_scale = 1.18
            elif config.dgp_profile in {"paper_britsadv_img065", "paper_britsadv_img065_altus"}:
                image_severity_scale = 0.34
                image_nuisance_scale = 0.80
        elif config.dgp_profile == "paper_triclass":
            centered_shape = (
                0.42 * np.mean(np.abs(centered_diffs))
                + 0.36 * np.mean(np.abs(np.diff(centered_yv, n=2, axis=0))) if length > 2 else 0.42 * np.mean(np.abs(centered_diffs))
            )
            centered_late_shift = float(np.mean(np.abs(np.mean(centered_yv[late_idx:], axis=0) - np.mean(centered_yv[:early_idx], axis=0))))
            late_motif_gap = float(
                np.mean(
                    np.maximum(yv[late_idx:, 0] - 0.55 * yv[late_idx:, 2], 0.0)
                    + 0.8 * np.maximum(yv[late_idx:, 1] - 0.25 * yv[late_idx:, 2], 0.0)
                )
            )
            shared_severity = 0.16 * latent_severity + 0.10 * latent_inflammation + 0.82 * motif_score + 0.48 * centered_shape
            latent_score = (
                -3.05
                + 1.06 * shape_score
                + 1.30 * motif_score
                + 0.82 * centered_shape
                + 0.46 * centered_late_shift
                + 0.54 * late_motif_gap
                + 0.04 * latent_severity
                + rng.normal(scale=0.86)
            )
            image_severity_scale = 0.08
            image_nuisance_scale = 1.62
        elif config.dgp_profile == "paper_targetedgap":
            latent_score = (
                -0.24
                + 0.08 * trajectory_score
                + 0.42 * late_burden
                + 0.48 * shape_score
                + 0.58 * fibrosis_signature
                + 0.02 * latent_severity
                + 0.02 * latent_inflammation
                + rng.normal(scale=2.15)
            )
            image_severity_scale = 0.10
            image_nuisance_scale = 1.8
        elif config.dgp_profile == "paper_randomgap":
            latent_score = (
                -0.40
                + 0.24 * trajectory_score
                + 0.95 * late_burden
                + 0.90 * shape_score
                + 0.95 * fibrosis_signature
                + 0.12 * latent_severity
                + 0.06 * latent_inflammation
                + rng.normal(scale=0.8)
            )
            image_severity_scale = 0.35
            image_nuisance_scale = 1.0
        elif config.dgp_profile == "paper_gap":
            latent_score = (
                -0.45
                + 0.26 * trajectory_score
                + 1.10 * late_burden
                + 1.05 * shape_score
                + 1.15 * fibrosis_signature
                + 0.10 * latent_severity
                + 0.04 * latent_inflammation
                + rng.normal(scale=0.72)
            )
            image_severity_scale = 0.28
            image_nuisance_scale = 0.8
        elif config.dgp_profile == "paper_signal":
            latent_score = (
                -0.35
                + 0.22 * trajectory_score
                + 0.90 * late_burden
                + 0.85 * shape_score
                + 0.75 * fibrosis_signature
                + 0.18 * latent_severity
                + 0.08 * latent_inflammation
                + rng.normal(scale=0.8)
            )
            image_severity_scale = 0.55
            image_nuisance_scale = 0.5
        else:
            latent_score = (
                -0.25
                + 0.20 * trajectory_score
                + 0.55 * late_burden
                + 0.35 * shape_score
                + 0.32 * latent_severity
                + 0.18 * latent_inflammation
                + rng.normal(scale=1.05)
            )
            image_severity_scale = 1.0
            image_nuisance_scale = 0.3
        if latent_score < ordinal_cutpoints[0]:
            labels[i] = 0
        elif latent_score < ordinal_cutpoints[1]:
            labels[i] = 1
        else:
            labels[i] = 2

        if config.missingness == "mcar":
            prob = np.full((length, N_BIOMARKERS), config.mask_rate, dtype=float)
        else:
            dy = np.vstack([np.zeros((1, N_BIOMARKERS)), np.diff(yv, axis=0)]) if length > 1 else np.zeros((length, N_BIOMARKERS))
            discordance = np.abs(yv[:, [0, 1]] - yv[:, [1, 2]])
            discordance = np.column_stack([discordance, np.abs(yv[:, 0] - yv[:, 2])])
            x_scale = np.sqrt(np.mean(x[i, :length, : min(p, 3)] ** 2, axis=1))
            turning = np.zeros((length, N_BIOMARKERS), dtype=float)
            if length > 2:
                turning[1:-1] = (np.sign(dy[1:-1]) * np.sign(dy[2:]) < 0).astype(float)
            anchor = np.exp(-0.5 * ((times_i - times_i[length // 2]) / 0.8) ** 2) + np.exp(-0.5 * ((times_i - times_i[-1]) / 0.8) ** 2)
            late_anchor = np.clip((times_i - times_i[max(1, length // 2)]) / max(times_i[-1] - times_i[0], 1e-6), 0.0, 1.0)
            severity_proxy = 0.42 * shared_severity + 0.12 * shape_score
            biomarker_driver = (
                0.22 * np.abs(dy)
                + 0.16 * np.abs(yv)
                + 0.12 * discordance
                + 0.18 * turning
                + 0.14 * anchor[:, None]
                + 0.22 * late_anchor[:, None] * np.array([[1.2, 1.0, 0.9]])
                + 0.08 * x_scale[:, None]
                + 0.16 * severity_proxy
            )
            survey_driver = (
                0.18 * survey_signal
                + 0.08 * np.cos(times_i)[:, None]
                + 0.10 * anchor[:, None]
                + 0.10 * severity_proxy
            )
            clinical_driver = 0.08 * clinical_signal + 0.06 * np.abs(yv[:, [0, 2]])
            if clinical_driver.ndim == 2:
                clinical_driver = np.column_stack([clinical_driver[:, 0], 0.5 * clinical_driver[:, 0] + 0.5 * clinical_driver[:, 1], clinical_driver[:, 1]])
            else:
                clinical_driver = np.repeat(clinical_driver[:, None], N_BIOMARKERS, axis=1)

            base_logit = -1.75 + 0.35 * config.mask_rate + rng.normal(scale=0.28, size=(length, N_BIOMARKERS))
            if config.dgp_profile in {"paper_britsadv", "paper_britsadv_img06", "paper_britsadv_img065", "paper_britsadv_img065_altus"}:
                visit_propensity = rng.normal(scale=0.46, size=length)
                marker_bias_vec = rng.normal(scale=0.18, size=N_BIOMARKERS)
                reentry_wave = np.sin(times_i[:, None] * np.array([[1.2, 0.95, 1.5]]) + rng.uniform(-np.pi, np.pi, size=(1, N_BIOMARKERS)))
                local_instability = 0.78 * np.abs(dy) + 0.66 * discordance + 0.44 * turning + 0.12 * np.abs(yv)
                shock = rng.normal(scale=0.22, size=(length, N_BIOMARKERS))
                random_anchor = np.exp(-0.5 * ((times_i - rng.choice(times_i)) / 0.6) ** 2)[:, None]
                gap_indicator = np.maximum(np.diff(times_i, prepend=times_i[0]) - np.median(np.diff(times_i, prepend=times_i[0])), 0.0)[:, None]
                motif_window = np.column_stack(
                    [
                        np.maximum(np.abs(np.diff(yv[:, 0], prepend=yv[0, 0])) - 0.30, 0.0),
                        np.maximum(np.abs(np.diff(yv[:, 1], prepend=yv[0, 1])) - 0.26, 0.0),
                        np.maximum(np.abs(np.diff(yv[:, 2], prepend=yv[0, 2])) - 0.24, 0.0),
                    ]
                )
                motif_driver = (0.12 + 1.32 * late_anchor[:, None]) * (0.82 * motif_window + 0.34 * np.abs(dy) + 0.44 * gap_indicator)
                revisit = 0.18 * reentry_wave
                subject_shift = 0.02 * (1.0 - monitoring_propensity)
                if config.info_profile == "clinical_rich":
                    logit_missing = (
                        base_logit
                        + subject_shift
                        + 0.16 * local_instability
                        + 1.48 * motif_driver
                        + 0.10 * clinical_driver
                        + 0.12 * visit_propensity[:, None]
                        + 0.06 * shock
                        + 0.24 * random_anchor
                        + revisit
                        + marker_bias_vec
                    )
                elif config.info_profile == "survey_rich":
                    logit_missing = (
                        base_logit
                        + subject_shift
                        + 0.14 * local_instability
                        + 1.42 * motif_driver
                        + 0.10 * survey_driver
                        + 0.12 * visit_propensity[:, None]
                        + 0.06 * shock
                        + 0.24 * random_anchor
                        + revisit
                        + marker_bias_vec
                    )
                else:
                    logit_missing = (
                        base_logit
                        + subject_shift
                        + 0.16 * local_instability
                        + 1.46 * motif_driver
                        + 0.08 * survey_driver
                        + 0.12 * visit_propensity[:, None]
                        + 0.06 * shock
                        + 0.24 * random_anchor
                        + revisit
                        + marker_bias_vec
                    )
            elif config.dgp_profile == "paper_triclass":
                visit_propensity = rng.normal(scale=0.52, size=length)
                marker_bias_vec = rng.normal(scale=0.22, size=N_BIOMARKERS)
                reentry_wave = np.sin(times_i[:, None] * np.array([[1.4, 1.1, 1.7]]) + rng.uniform(-np.pi, np.pi, size=(1, N_BIOMARKERS)))
                local_instability = 0.80 * np.abs(dy) + 0.70 * discordance + 0.48 * turning + 0.14 * np.abs(yv)
                shock = rng.normal(scale=0.26, size=(length, N_BIOMARKERS))
                random_anchor = np.exp(-0.5 * ((times_i - rng.choice(times_i)) / 0.55) ** 2)[:, None]
                motif_window = np.column_stack(
                    [
                        np.maximum(np.abs(np.diff(yv[:, 0], prepend=yv[0, 0])) - 0.35, 0.0),
                        np.maximum(np.abs(np.diff(yv[:, 1], prepend=yv[0, 1])) - 0.30, 0.0),
                        np.maximum(np.abs(np.diff(yv[:, 2], prepend=yv[0, 2])) - 0.28, 0.0),
                    ]
                )
                motif_driver = (0.16 + 1.18 * late_anchor[:, None]) * (0.70 * motif_window + 0.30 * np.abs(dy))
                revisit = 0.22 * reentry_wave
                subject_shift = 0.05 * (1.0 - monitoring_propensity)
                if config.info_profile == "clinical_rich":
                    logit_missing = (
                        base_logit
                        + subject_shift
                        + 0.30 * local_instability
                        + 1.18 * motif_driver
                        + 0.12 * clinical_driver
                        + 0.14 * visit_propensity[:, None]
                        + 0.08 * shock
                        + 0.28 * random_anchor
                        + revisit
                        + marker_bias_vec
                    )
                elif config.info_profile == "survey_rich":
                    logit_missing = (
                        base_logit
                        + subject_shift
                        + 0.24 * local_instability
                        + 1.10 * motif_driver
                        + 0.12 * survey_driver
                        + 0.14 * visit_propensity[:, None]
                        + 0.08 * shock
                        + 0.28 * random_anchor
                        + revisit
                        + marker_bias_vec
                    )
                else:
                    logit_missing = (
                        base_logit
                        + subject_shift
                        + 0.28 * local_instability
                        + 1.16 * motif_driver
                        + 0.08 * survey_driver
                        + 0.14 * visit_propensity[:, None]
                        + 0.08 * shock
                        + 0.28 * random_anchor
                        + revisit
                        + marker_bias_vec
                    )
            elif config.dgp_profile == "paper_targetedgap":
                visit_propensity = rng.normal(scale=0.45, size=length)
                marker_bias_vec = rng.normal(scale=0.18, size=N_BIOMARKERS)
                reentry_wave = np.sin(times_i[:, None] * np.array([[1.1, 1.6, 0.8]]) + rng.uniform(-np.pi, np.pi, size=(1, N_BIOMARKERS)))
                local_instability = 0.85 * np.abs(dy) + 0.72 * discordance + 0.30 * turning + 0.22 * np.abs(yv)
                shock = rng.normal(scale=0.45, size=(length, N_BIOMARKERS))
                random_anchor = np.exp(-0.5 * ((times_i - rng.choice(times_i)) / 0.7) ** 2)[:, None]
                motif_base = np.column_stack(
                    [
                        np.maximum(yv[:, 0] - 0.55 * yv[:, 2], 0.0),
                        np.maximum(yv[:, 1] - 0.30 * yv[:, 2], 0.0),
                        np.maximum(0.70 * yv[:, 0] - yv[:, 2], 0.0),
                    ]
                )
                motif_driver = (0.35 + 0.95 * late_anchor[:, None]) * motif_base
                revisit = 0.34 * reentry_wave
                subject_shift = 0.10 * (1.0 - monitoring_propensity)
                if config.info_profile == "clinical_rich":
                    logit_missing = (
                        base_logit
                        + subject_shift
                        + 0.70 * local_instability
                        + 1.28 * motif_driver
                        + 0.16 * clinical_driver
                        + 0.24 * visit_propensity[:, None]
                        + 0.18 * shock
                        + 0.34 * random_anchor
                        + 0.16 * revisit
                        + marker_bias_vec
                    )
                elif config.info_profile == "survey_rich":
                    logit_missing = (
                        base_logit
                        + subject_shift
                        + 0.42 * local_instability
                        + 1.18 * motif_driver
                        + 0.20 * survey_driver
                        + 0.22 * visit_propensity[:, None]
                        + 0.18 * shock
                        + 0.32 * random_anchor
                        + 0.14 * revisit
                        + marker_bias_vec
                    )
                elif config.info_profile == "balanced":
                    logit_missing = (
                        base_logit
                        + subject_shift
                        + 0.60 * local_instability
                        + 1.24 * motif_driver
                        + 0.10 * survey_driver
                        + 0.22 * visit_propensity[:, None]
                        + 0.18 * shock
                        + 0.34 * random_anchor
                        + 0.16 * revisit
                        + marker_bias_vec
                    )
                elif config.info_profile == "discordant":
                    logit_missing = (
                        base_logit
                        + subject_shift
                        + 0.54 * local_instability
                        + 1.20 * motif_driver
                        + 0.08 * (survey_signal - clinical_signal)
                        + 0.22 * visit_propensity[:, None]
                        + 0.18 * shock
                        + 0.36 * random_anchor
                        + 0.16 * revisit
                        + marker_bias_vec
                    )
                else:
                    logit_missing = (
                        base_logit
                        + subject_shift
                        + 0.66 * local_instability
                        + 1.24 * motif_driver
                        + 0.22 * visit_propensity[:, None]
                        + 0.18 * shock
                        + 0.34 * random_anchor
                        + 0.16 * revisit
                        + marker_bias_vec
                    )
            elif config.dgp_profile == "paper_randomgap":
                visit_propensity = rng.normal(scale=0.5, size=length)
                marker_bias_vec = rng.normal(scale=0.25, size=N_BIOMARKERS)
                reentry_wave = np.sin(times_i[:, None] * np.array([[1.3, 0.9, 1.7]]) + rng.uniform(-np.pi, np.pi, size=(1, N_BIOMARKERS)))
                local_instability = 0.9 * np.abs(dy) + 0.7 * discordance + 0.35 * turning + 0.25 * np.abs(yv)
                shock = rng.normal(scale=0.7, size=(length, N_BIOMARKERS))
                revisit = 0.45 * reentry_wave - 0.20 * np.sign(reentry_wave) * np.abs(reentry_wave)
                random_anchor = np.exp(-0.5 * ((times_i - rng.choice(times_i)) / 0.6) ** 2)[:, None]
                subject_shift = -0.18 * monitoring_propensity
                if config.info_profile == "clinical_rich":
                    logit_missing = base_logit + subject_shift + 0.72 * local_instability + 0.22 * clinical_driver + 0.18 * visit_propensity[:, None] + 0.16 * shock + 0.20 * random_anchor + revisit + marker_bias_vec
                elif config.info_profile == "survey_rich":
                    logit_missing = base_logit + subject_shift + 0.48 * local_instability + 0.34 * survey_driver + 0.18 * visit_propensity[:, None] + 0.16 * shock + 0.20 * random_anchor + revisit + marker_bias_vec
                elif config.info_profile == "balanced":
                    logit_missing = base_logit + subject_shift + 0.62 * local_instability + 0.18 * survey_driver + 0.18 * visit_propensity[:, None] + 0.16 * shock + 0.20 * random_anchor + revisit + marker_bias_vec
                elif config.info_profile == "discordant":
                    logit_missing = base_logit + subject_shift + 0.58 * local_instability + 0.10 * (survey_signal - clinical_signal) + 0.18 * visit_propensity[:, None] + 0.18 * shock + 0.22 * random_anchor + revisit + marker_bias_vec
                else:
                    logit_missing = base_logit + subject_shift + 0.65 * local_instability + 0.18 * visit_propensity[:, None] + 0.16 * shock + 0.20 * random_anchor + revisit + marker_bias_vec
            elif config.dgp_profile == "paper_gap":
                # Strongly target late, label-relevant biomarker structure while preserving some observability.
                marker_weights = np.array([1.55, 1.30, 1.10], dtype=float)
                late_driver = late_anchor[:, None] * marker_weights[None, :] * (
                    0.95 * np.abs(dy) + 0.80 * discordance + 0.65 * np.abs(yv)
                )
                motif_driver = late_anchor[:, None] * np.array([[1.4, 1.2, 1.0]]) * (
                    np.maximum(yv[:, [0, 1, 0]] - np.column_stack([0.55 * yv[:, 2], 0.25 * yv[:, 2], 0.75 * yv[:, 2]]), 0.0)
                )
                subject_shift = -0.40 * monitoring_propensity
                if config.info_profile == "clinical_rich":
                    logit_missing = base_logit + subject_shift + 1.20 * biomarker_driver + 0.95 * late_driver + 0.55 * motif_driver + 0.20 * clinical_driver
                elif config.info_profile == "survey_rich":
                    logit_missing = base_logit + subject_shift + 0.70 * biomarker_driver + 0.90 * late_driver + 0.40 * motif_driver + 0.35 * survey_driver
                elif config.info_profile == "balanced":
                    logit_missing = base_logit + subject_shift + 1.00 * biomarker_driver + 0.92 * late_driver + 0.48 * motif_driver + 0.25 * survey_driver
                elif config.info_profile == "discordant":
                    logit_missing = base_logit + subject_shift + 0.82 * biomarker_driver + 0.82 * late_driver + 0.42 * motif_driver + 0.18 * (survey_signal - clinical_signal)
                else:
                    logit_missing = base_logit + subject_shift + 1.05 * biomarker_driver + 0.88 * late_driver + 0.50 * motif_driver
            elif config.dgp_profile == "paper_signal":
                # Make late biomarker shape strongly informative for both missingness and class.
                marker_weights = np.array([1.35, 1.15, 0.95], dtype=float)
                late_driver = late_anchor[:, None] * marker_weights[None, :] * (
                    0.75 * np.abs(dy) + 0.55 * discordance + 0.45 * np.abs(yv)
                )
                subject_shift = -0.65 * monitoring_propensity
                if config.info_profile == "clinical_rich":
                    logit_missing = base_logit + subject_shift + 1.15 * biomarker_driver + 0.55 * late_driver + 0.30 * clinical_driver
                elif config.info_profile == "survey_rich":
                    logit_missing = base_logit + subject_shift + 0.55 * biomarker_driver + 0.65 * late_driver + 0.70 * survey_driver
                elif config.info_profile == "balanced":
                    logit_missing = base_logit + subject_shift + 0.95 * biomarker_driver + 0.80 * late_driver + 0.45 * survey_driver
                elif config.info_profile == "discordant":
                    logit_missing = base_logit + subject_shift + 0.75 * biomarker_driver + 0.70 * late_driver + 0.25 * (survey_signal - clinical_signal)
                else:
                    logit_missing = base_logit + subject_shift + 0.95 * biomarker_driver + 0.85 * late_driver - 0.06 * np.abs(survey_signal)
            elif config.info_profile == "clinical_rich":
                logit_missing = base_logit + 1.05 * biomarker_driver + 0.35 * clinical_driver
            elif config.info_profile == "survey_rich":
                logit_missing = base_logit + 0.95 * survey_driver
            elif config.info_profile == "balanced":
                logit_missing = base_logit + 0.82 * biomarker_driver + 0.82 * survey_driver
            elif config.info_profile == "discordant":
                logit_missing = base_logit + 0.6 * biomarker_driver + 0.25 * (survey_signal - clinical_signal)
            else:
                logit_missing = base_logit + 0.95 * biomarker_driver - 0.06 * np.abs(survey_signal)
            if config.missingness == "label_dependent":
                if config.dgp_profile in {"paper_britsadv", "paper_britsadv_img06", "paper_britsadv_img065", "paper_britsadv_img065_altus"}:
                    logit_missing = logit_missing + 0.01 * severity_proxy + 0.34 * motif_driver + 0.06 * late_anchor[:, None]
                elif config.dgp_profile == "paper_triclass":
                    logit_missing = logit_missing + 0.02 * severity_proxy + 0.28 * motif_driver + 0.03 * np.abs(np.sin(1.3 * times_i))[:, None]
                elif config.dgp_profile == "paper_targetedgap":
                    logit_missing = logit_missing + 0.04 * severity_proxy + 0.42 * motif_driver + 0.12 * np.sign(fibrosis_signature) * np.abs(np.sin(times_i))[:, None]
                elif config.dgp_profile == "paper_randomgap":
                    logit_missing = logit_missing + 0.10 * severity_proxy + 0.08 * np.sign(fibrosis_signature) * np.abs(np.sin(times_i))[:, None]
                elif config.dgp_profile == "paper_gap":
                    logit_missing = logit_missing + 0.01 * latent_score + 0.36 * severity_proxy + 0.30 * fibrosis_signature * late_anchor[:, None]
                elif config.dgp_profile == "paper_signal":
                    logit_missing = logit_missing + 0.03 * latent_score + 0.32 * severity_proxy + 0.18 * fibrosis_signature * late_anchor[:, None]
                else:
                    logit_missing = logit_missing + 0.06 * latent_score + 0.24 * severity_proxy
            prob = _sigmoid(logit_missing)

        if config.dgp_profile in {"paper_britsadv", "paper_britsadv_img06", "paper_britsadv_img065", "paper_britsadv_img065_altus"}:
            prob = np.clip(0.82 * prob + 0.18 * config.mask_rate, 0.10, 0.995)
        elif config.dgp_profile == "paper_targetedgap":
            prob = np.clip(0.78 * prob + 0.22 * config.mask_rate, 0.10, 0.995)
        else:
            prob = np.clip(0.35 * prob + 0.65 * config.mask_rate, 0.04, 0.97)
        m = rng.binomial(1, 1.0 - prob).astype(np.float32)
        for k in range(N_BIOMARKERS):
            if np.all(m[:, k] == 0):
                m[rng.integers(0, length), k] = 1.0
        mask[i, :length] = m
        y_obs[i, :length] = y_true[i, :length] * m

        image_stage_hint = int(labels[i])
        if config.dgp_profile == "paper_britsadv":
            noisy_image_score = (
                0.32 * latent_severity
                + 0.20 * latent_inflammation
                + 0.10 * survey_signal
                + 0.06 * clinical_signal
                + rng.normal(scale=1.1)
            )
            if noisy_image_score < -0.35:
                image_stage_hint = 0
            elif noisy_image_score < 0.75:
                image_stage_hint = 1
            else:
                image_stage_hint = 2
        elif config.dgp_profile == "paper_britsadv_img06":
            noisy_image_score = (
                0.52 * latent_severity
                + 0.28 * latent_inflammation
                + 0.16 * survey_signal
                + 0.10 * clinical_signal
                + 0.18 * motif_score
                + rng.normal(scale=0.72)
            )
            if noisy_image_score < -0.35:
                image_stage_hint = 0
            elif noisy_image_score < 0.75:
                image_stage_hint = 1
            else:
                image_stage_hint = 2
        elif config.dgp_profile in {"paper_britsadv_img065", "paper_britsadv_img065_altus"}:
            noisy_image_score = (
                0.76 * latent_severity
                + 0.42 * latent_inflammation
                + 0.24 * survey_signal
                + 0.16 * clinical_signal
                + 0.32 * motif_score
                + rng.normal(scale=0.36)
            )
            if noisy_image_score < -0.35:
                image_stage_hint = 0
            elif noisy_image_score < 0.75:
                image_stage_hint = 1
            else:
                image_stage_hint = 2
        image_fn = _generate_liver_image
        if config.dgp_profile == "paper_britsadv_img065_altus":
            image_fn = _generate_liver_ultrasound_image
        liver_image, liver_image_features = image_fn(
            config.image_size,
            severity=shared_severity,
            survey_signal=survey_signal,
            clinical_signal=clinical_signal,
            severity_scale=image_severity_scale,
            nuisance_scale=image_nuisance_scale,
            stage_hint=image_stage_hint,
            motif_strength=motif_score,
            rng=rng,
        )
        image[i] = liver_image
        image_features[i] = liver_image_features

    return LongitudinalData(
        y_true=y_true,
        y_obs=y_obs,
        mask=mask,
        x=x,
        x_mask=x_mask,
        static=static,
        times=times,
        deltas=deltas,
        lengths=lengths.astype(np.int64),
        labels=labels.astype(np.int64),
        subject_ids=np.arange(n, dtype=np.int64),
        image=image,
        image_features=image_features,
        metadata={
            "n_biomarkers": N_BIOMARKERS,
            "biomarker_names": BIOMARKER_NAMES,
            "fibrosis_feature_names": true_feature_names,
            "fibrosis_true_coefs": true_feature_coefs,
            "fibrosis_cutpoints": ordinal_cutpoints.tolist(),
            "static_feature_indices": static_indices,
            "image_feature_names": [
                "img_mean_intensity",
                "img_texture_sd",
                "img_edge_energy_v",
                "img_edge_energy_h",
                "img_high_echo_fraction",
                "img_nodule_score",
            ],
        },
    )


def train_val_test_split(data: LongitudinalData, config: ExperimentConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    idx = np.arange(len(data.subject_ids))
    y = data.labels.astype(int)
    train_parts = []
    val_parts = []
    test_parts = []
    for klass in np.unique(y):
        arr = idx[y == klass]
        rng.shuffle(arr)
        n_train = int(len(arr) * config.train_fraction)
        n_val = int(len(arr) * config.val_fraction)
        train_parts.append(arr[:n_train])
        val_parts.append(arr[n_train : n_train + n_val])
        test_parts.append(arr[n_train + n_val :])
    train = np.concatenate(train_parts)
    val = np.concatenate(val_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def compute_standardization(data: LongitudinalData, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    obs_train = data.y_obs[train_idx]
    obs_mask = data.mask[train_idx] == 1
    y_mean = np.zeros(data.y_obs.shape[-1], dtype=np.float32)
    y_std = np.ones(data.y_obs.shape[-1], dtype=np.float32)
    for k in range(data.y_obs.shape[-1]):
        vals = obs_train[..., k][obs_mask[..., k]]
        if vals.size:
            y_mean[k] = float(vals.mean())
            y_std[k] = max(float(vals.std()), 1e-6)
    x_train = data.x[train_idx].reshape(-1, data.x.shape[-1])
    x_mean = x_train.mean(axis=0)
    x_std = np.maximum(x_train.std(axis=0), 1e-6)
    s_train = data.static[train_idx]
    s_mean = s_train.mean(axis=0)
    s_std = np.maximum(s_train.std(axis=0), 1e-6)
    return {"y_mean": y_mean, "y_std": y_std, "x_mean": x_mean, "x_std": x_std, "s_mean": s_mean, "s_std": s_std}


def standardize_data(data: LongitudinalData, stats: dict[str, np.ndarray]) -> LongitudinalData:
    y_true = ((data.y_true - stats["y_mean"]) / stats["y_std"]).astype(np.float32)
    y_scaled = (data.y_obs - stats["y_mean"]) / stats["y_std"]
    y_obs = np.where(data.mask == 1, y_scaled, 0.0).astype(np.float32)
    x = ((data.x - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
    static = ((data.static - stats["s_mean"]) / stats["s_std"]).astype(np.float32)
    return LongitudinalData(
        y_true=y_true,
        y_obs=y_obs,
        mask=data.mask.astype(np.float32),
        x=x,
        x_mask=data.x_mask.astype(np.float32),
        static=static,
        times=data.times.astype(np.float32),
        deltas=data.deltas.astype(np.float32),
        lengths=data.lengths.astype(np.int64),
        labels=data.labels.astype(np.int64),
        subject_ids=data.subject_ids.astype(np.int64),
        image=data.image.astype(np.float32),
        image_features=data.image_features.astype(np.float32),
        metadata=dict(data.metadata),
    )
