"""Optional estimating-score calibration for completed trajectories.

This module implements the analysis-calibration unit described in the
BRITS-MI supplement. It is intentionally separate from label-free deployment:
the outcome and a reference coefficient vector are required only when an
analyst elects to calibrate imputations for an explanatory association model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import BRITSMI


@dataclass(frozen=True)
class AssociationCalibrationConfig:
    """Settings for bounded, preconditioned estimating-score updates."""

    step_size: float = 0.08
    n_steps: int = 1
    max_update_sd: float = 0.35
    max_backtracking: int = 8
    epsilon: float = 1e-8


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Loss and update summaries for one calibrated completed trajectory."""

    loss_before: float
    loss_after: float
    accepted_steps: int
    maximum_absolute_update: float


def clinical_association_score_loss(
    completed: torch.Tensor,
    times: torch.Tensor,
    static: torch.Tensor,
    outcome: torch.Tensor,
    coefficients: torch.Tensor,
    intercept: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Return a scaled logistic estimating-score discrepancy.

    ``coefficients`` follows the 13-term order returned by
    :meth:`BRITSMI.clinical_design`. The reference coefficients are held fixed;
    gradients pass only through the completed trajectory and its summaries.
    """

    design = BRITSMI.clinical_design(completed, times, static)
    if coefficients.shape != (design.shape[1],):
        raise ValueError(f"coefficients must have shape ({design.shape[1]},)")
    linear_predictor = torch.as_tensor(
        intercept, dtype=completed.dtype, device=completed.device
    ) + design @ coefficients
    residual = outcome.to(completed.dtype) - torch.sigmoid(linear_predictor)
    augmented = torch.cat([torch.ones_like(residual[:, None]), design], dim=1)
    score = (augmented * residual[:, None]).mean(dim=0)
    scale = torch.sqrt(augmented.square().mean(dim=0)).clamp_min(1e-4)
    return 0.5 * (score / scale).square().mean()


def calibrate_completed_trajectory(
    completed: np.ndarray,
    observed_values: np.ndarray,
    observed_mask: np.ndarray,
    times: np.ndarray,
    static: np.ndarray,
    outcome: np.ndarray,
    coefficients: np.ndarray,
    intercept: float = 0.0,
    config: AssociationCalibrationConfig | None = None,
) -> tuple[np.ndarray, CalibrationDiagnostics]:
    """Apply bounded score-descent updates to missing cells only.

    The update is preconditioned by marker scale and gradient root mean square.
    Backtracking accepts a step only when it does not increase the prespecified
    score discrepancy. Source-observed cells are restored after every proposal.
    """

    config = config or AssociationCalibrationConfig()
    completed = np.asarray(completed, dtype=np.float32)
    observed_values = np.asarray(observed_values, dtype=np.float32)
    observed_mask = np.asarray(observed_mask, dtype=bool)
    static = np.asarray(static, dtype=np.float32)
    outcome = np.asarray(outcome, dtype=np.float32)
    coefficients = np.asarray(coefficients, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    if times.ndim == 1:
        times = np.broadcast_to(times, completed.shape[:2]).copy()
    if completed.shape != observed_values.shape or completed.shape != observed_mask.shape:
        raise ValueError("completed, observed_values, and observed_mask must share shape")
    if static.shape[0] != completed.shape[0] or outcome.shape != (completed.shape[0],):
        raise ValueError("static and outcome must have one row or value per subject")

    missing = ~observed_mask
    marker_scale = np.nanstd(
        np.where(observed_mask, observed_values, np.nan), axis=(0, 1), ddof=1
    )
    marker_scale = np.where(np.isfinite(marker_scale) & (marker_scale > 1e-6), marker_scale, 1.0)
    candidate = torch.tensor(completed, dtype=torch.float32)
    mask_tensor = torch.tensor(observed_mask)
    observed_tensor = torch.tensor(np.nan_to_num(observed_values, nan=0.0))
    times_tensor = torch.tensor(times)
    static_tensor = torch.tensor(static)
    outcome_tensor = torch.tensor(outcome)
    coefficient_tensor = torch.tensor(coefficients)

    with torch.no_grad():
        initial_loss = float(
            clinical_association_score_loss(
                candidate,
                times_tensor,
                static_tensor,
                outcome_tensor,
                coefficient_tensor,
                intercept,
            )
        )
    current_loss = initial_loss
    accepted_steps = 0
    original = candidate.clone()

    for _ in range(config.n_steps):
        candidate = candidate.detach().requires_grad_(True)
        loss = clinical_association_score_loss(
            candidate,
            times_tensor,
            static_tensor,
            outcome_tensor,
            coefficient_tensor,
            intercept,
        )
        gradient = torch.autograd.grad(loss, candidate)[0]
        gradient = torch.where(mask_tensor, torch.zeros_like(gradient), gradient)
        direction = torch.zeros_like(gradient)
        for marker in range(completed.shape[2]):
            marker_missing = torch.tensor(missing[:, :, marker])
            if not bool(marker_missing.any()):
                continue
            marker_gradient = gradient[:, :, marker][marker_missing]
            rms = torch.sqrt(marker_gradient.square().mean()).clamp_min(config.epsilon)
            direction[:, :, marker] = gradient[:, :, marker] / rms * float(marker_scale[marker])

        accepted = False
        trial_step = config.step_size
        for _ in range(config.max_backtracking):
            maximum = torch.tensor(
                marker_scale * config.max_update_sd, dtype=candidate.dtype
            ).reshape(1, 1, -1)
            raw_update = -trial_step * direction
            update = torch.maximum(torch.minimum(raw_update, maximum), -maximum)
            proposal = candidate.detach() + update
            proposal = torch.where(mask_tensor, observed_tensor, proposal)
            with torch.no_grad():
                proposal_loss = float(
                    clinical_association_score_loss(
                        proposal,
                        times_tensor,
                        static_tensor,
                        outcome_tensor,
                        coefficient_tensor,
                        intercept,
                    )
                )
            if proposal_loss <= current_loss + config.epsilon:
                candidate = proposal
                current_loss = proposal_loss
                accepted_steps += 1
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            candidate = candidate.detach()
            break

    calibrated = candidate.detach().cpu().numpy()
    calibrated[observed_mask] = observed_values[observed_mask]
    maximum_update = float(np.max(np.abs(calibrated - original.cpu().numpy())))
    return calibrated, CalibrationDiagnostics(
        loss_before=initial_loss,
        loss_after=current_loss,
        accepted_steps=accepted_steps,
        maximum_absolute_update=maximum_update,
    )
