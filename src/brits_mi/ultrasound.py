"""Sector-style synthetic ultrasound images and deterministic summaries."""

from __future__ import annotations

import numpy as np


def render_ultrasound(
    severity: float,
    activity: float,
    stage: int,
    rng: np.random.Generator,
    image_size: int = 24,
) -> np.ndarray:
    """Render one stylized liver ultrasound image for simulation experiments.

    The renderer represents a fan-shaped sector, depth attenuation, Rayleigh
    speckle, capsule and periportal structure, septal bands, acoustic shadows,
    and stage-dependent nodularity. It is a controlled DGP, not a physical
    scanner simulator.
    """

    if stage not in (0, 1, 2):
        raise ValueError("stage must be 0, 1, or 2")
    coordinates = np.linspace(-1.0, 1.0, image_size, dtype=np.float32)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.sqrt(xx**2 + (yy + 1.05) ** 2)
    angle = np.arctan2(xx, yy + 1.05)
    sector = ((radius > 0.18) & (radius < 1.55) & (np.abs(angle) < 0.78)).astype(float)
    fibrosis = float(np.tanh(severity / 2.2))
    activity = float(np.clip(activity, -2.5, 2.5))

    base = 0.12 + 0.34 * np.exp(-((radius - 0.85) / 0.40) ** 2)
    fan_gain = 0.08 * np.cos(2.8 * angle + rng.uniform(-np.pi, np.pi))
    attenuation = (0.05 + 0.10 * max(fibrosis, 0.0)) * np.clip(radius - 0.56, 0.0, None)
    capsule = 0.11 * np.exp(-((radius - 0.54) / 0.030) ** 2)
    periportal = (
        0.07
        * np.exp(-((angle + 0.30) / 0.07) ** 2)
        * np.exp(-((radius - 0.76) / 0.10) ** 2)
    )
    septal = (
        0.15
        * max(fibrosis, 0.0)
        * np.exp(-((angle - 0.12) / 0.08) ** 2)
        * np.exp(-((radius - 0.92) / 0.22) ** 2)
    )
    bridge = (
        0.13
        * max(fibrosis, 0.0)
        * np.exp(-((angle + 0.18) / 0.09) ** 2)
        * np.exp(-((radius - 1.02) / 0.20) ** 2)
    )
    sweep = 0.04 * abs(activity) * np.sin(
        10.0 * angle + 3.5 * radius + rng.uniform(-np.pi, np.pi)
    )

    shadow = np.zeros_like(base)
    for _ in range(1 + int(stage >= 1) + int(fibrosis > 0.35)):
        center = rng.uniform(-0.32, 0.32)
        width = rng.uniform(0.03, 0.07)
        depth = rng.uniform(0.75, 1.20)
        shadow += (
            0.06
            * np.exp(-((angle - center) / width) ** 2)
            * np.clip(radius - depth, 0.0, None)
        )

    if stage == 0:
        stage_pattern = periportal
    elif stage == 1:
        stage_pattern = periportal + septal
    else:
        nodular_rim = (
            0.10
            * np.exp(-((radius - 1.10) / 0.07) ** 2)
            * (1.0 + 0.5 * np.sin(14.0 * angle + rng.uniform(-np.pi, np.pi)))
        )
        stage_pattern = periportal + septal + bridge + nodular_rim

    image = base + fan_gain + capsule + stage_pattern + sweep - attenuation - shadow
    speckle = rng.rayleigh(
        scale=0.12 + 0.03 * abs(activity),
        size=(image_size, image_size),
    )
    image *= 1.0 + 0.26 * (speckle - speckle.mean())
    image += rng.normal(scale=0.015, size=image.shape)

    n_nodules = int(np.clip(round(0.8 + 2.4 * max(fibrosis, 0.0) + 0.3 * abs(activity)), 1, 5))
    for _ in range(n_nodules):
        node_angle = rng.uniform(-0.30, 0.30)
        node_radius = rng.uniform(0.72, 1.18)
        node_angle_scale = rng.uniform(0.03, 0.08)
        node_radius_scale = rng.uniform(0.05, 0.10)
        amplitude = rng.uniform(0.05, 0.10) * (1.0 + 0.9 * max(fibrosis, 0.0))
        image += amplitude * np.exp(
            -((angle - node_angle) / node_angle_scale) ** 2
            - ((radius - node_radius) / node_radius_scale) ** 2
        )
    return np.clip(sector * image, 0.0, 1.0).astype(np.float32)


def image_features(images: np.ndarray) -> np.ndarray:
    """Extract the 17 deterministic intensity, gradient, region, and quantile summaries."""

    images = np.asarray(images, dtype=float)
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim != 3:
        raise ValueError("images must have shape (subjects, height, width) or include one channel")
    n_subjects, height, width = images.shape
    quantiles = np.quantile(
        images.reshape(n_subjects, -1),
        [0.10, 0.25, 0.50, 0.75, 0.90],
        axis=1,
    ).T
    gradient_y, gradient_x = np.gradient(images, axis=(1, 2))
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((yy - (height - 1) / 2) ** 2 + (xx - (width - 1) / 2) ** 2)
    center_cut = np.quantile(distance, 0.35)
    ring_cut = np.quantile(distance, 0.70)
    center = distance <= center_cut
    ring = (distance > center_cut) & (distance <= ring_cut)
    outer = distance > ring_cut
    row_profile = images.mean(axis=2)
    column_profile = images.mean(axis=1)
    features = [
        images.reshape(n_subjects, -1).mean(axis=1),
        images.reshape(n_subjects, -1).std(axis=1),
        gradient.reshape(n_subjects, -1).mean(axis=1),
        gradient.reshape(n_subjects, -1).std(axis=1),
        images[:, center].mean(axis=1),
        images[:, ring].mean(axis=1),
        images[:, outer].mean(axis=1),
        images[:, center].mean(axis=1) - images[:, outer].mean(axis=1),
        row_profile.std(axis=1),
        column_profile.std(axis=1),
        row_profile[:, : height // 2].mean(axis=1)
        - row_profile[:, height // 2 :].mean(axis=1),
        column_profile[:, : width // 2].mean(axis=1)
        - column_profile[:, width // 2 :].mean(axis=1),
    ]
    return np.column_stack(features + [quantiles])


def simulate_ultrasound_context(
    severity: np.ndarray,
    activity: np.ndarray,
    stages: np.ndarray,
    seed: int = 1,
    image_size: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a cohort and return images plus 17 context features."""

    severity = np.asarray(severity, dtype=float)
    activity = np.asarray(activity, dtype=float)
    stages = np.asarray(stages, dtype=int)
    if not (len(severity) == len(activity) == len(stages)):
        raise ValueError("severity, activity, and stages must have equal length")
    rng = np.random.default_rng(seed)
    images = np.stack(
        [
            render_ultrasound(severity[i], activity[i], int(stages[i]), rng, image_size)
            for i in range(len(stages))
        ]
    )
    return images[:, None, :, :], image_features(images)
