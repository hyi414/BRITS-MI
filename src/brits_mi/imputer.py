"""High-level training and stochastic multiple-completion interface."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from .calibration import (
    AssociationCalibrationConfig,
    CalibrationDiagnostics,
    calibrate_completed_trajectory,
)
from .config import TrainingConfig
from .model import BRITSMI, marker_gaps, torch_marker_gaps


@dataclass(frozen=True)
class FitSummary:
    """Training metadata retained with a fitted imputer."""

    best_epoch: int
    best_validation_loss: float
    n_train: int
    n_validation: int
    residual_sd: tuple[float, ...]


class BRITSMultipleImputer:
    """Fit BRITS-MI and draw completed longitudinal biomarker trajectories."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.model_: BRITSMI | None = None
        self.marker_mean_: np.ndarray | None = None
        self.marker_sd_: np.ndarray | None = None
        self.residual_sd_: np.ndarray | None = None
        self.validation_indices_: np.ndarray | None = None
        self.fit_summary_: FitSummary | None = None
        self.training_history_: list[dict[str, float]] = []
        self._fit_values: np.ndarray | None = None
        self._fit_mask: np.ndarray | None = None
        self._fit_times: np.ndarray | None = None
        self._fit_static: np.ndarray | None = None
        self._fit_context: np.ndarray | None = None
        self.device_: torch.device | None = None

    @staticmethod
    def _validate_arrays(
        values: np.ndarray,
        mask: np.ndarray,
        times: np.ndarray,
        static: np.ndarray,
        context: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        static = np.asarray(static, dtype=float)
        if values.ndim != 3 or mask.shape != values.shape:
            raise ValueError("values and mask must share shape (subjects, visits, markers)")
        if static.ndim != 2 or static.shape[0] != values.shape[0]:
            raise ValueError("static must have shape (subjects, static_features)")
        times = np.asarray(times, dtype=float)
        if times.ndim == 1:
            times = np.broadcast_to(times, values.shape[:2]).copy()
        if times.shape != values.shape[:2]:
            raise ValueError("times must have shape (visits,) or (subjects, visits)")
        if not np.all(np.diff(times, axis=1) >= 0):
            raise ValueError("times must be nondecreasing within subject")
        if context is None:
            context = np.zeros((values.shape[0], 0), dtype=float)
        context = np.asarray(context, dtype=float)
        if context.ndim != 2 or context.shape[0] != values.shape[0]:
            raise ValueError("context must have shape (subjects, context_features)")
        if np.any(mask & ~np.isfinite(values)):
            raise ValueError("every measured cell must contain a finite value")
        return values, mask, times, static, context

    def _resolve_device(self) -> torch.device:
        requested = self.config.device.lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(requested)

    @staticmethod
    def _downstream_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_classes: int,
    ) -> torch.Tensor:
        if n_classes == 2:
            return nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return nn.functional.cross_entropy(logits, labels.long())

    def fit(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        times: np.ndarray,
        static: np.ndarray,
        outcome: np.ndarray,
        context: np.ndarray | None = None,
    ) -> BRITSMultipleImputer:
        """Fit using measured cells and training outcomes.

        Artificially hidden measured cells identify reconstruction and
        trajectory-summary losses. Genuine missing cells are never treated as
        known targets. The outcome contributes only through the downstream
        training loss.
        """

        values, mask, times, static, context = self._validate_arrays(
            values, mask, times, static, context
        )
        outcome = np.asarray(outcome)
        if outcome.ndim != 1 or len(outcome) != len(values):
            raise ValueError("outcome must have one value per subject")
        classes = np.unique(outcome)
        if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError("outcome classes must be consecutive integers beginning at 0")
        n_classes = len(classes)

        rng = np.random.default_rng(self.config.seed + 4401)
        order = rng.permutation(len(values))
        n_validation = max(16, round(self.config.validation_fraction * len(values)))
        n_validation = min(n_validation, len(values) - 8)
        if n_validation < 1:
            raise ValueError("at least 10 subjects are required")
        validation_indices = order[:n_validation]
        training_indices = order[n_validation:]

        with np.errstate(invalid="ignore"):
            marker_mean = np.nanmean(
                np.where(mask[training_indices], values[training_indices], np.nan),
                axis=(0, 1),
            )
            marker_sd = np.nanstd(
                np.where(mask[training_indices], values[training_indices], np.nan),
                axis=(0, 1),
            )
        marker_mean = np.nan_to_num(marker_mean, nan=0.0)
        marker_sd = np.where(np.isfinite(marker_sd) & (marker_sd > 1e-5), marker_sd, 1.0)
        standardized = (values - marker_mean.reshape(1, 1, -1)) / marker_sd.reshape(1, 1, -1)
        input_values = np.nan_to_num(standardized, nan=0.0).astype(np.float32)
        input_mask = mask.astype(np.float32)
        input_times = times.astype(np.float32)
        input_static = static.astype(np.float32)
        input_context = context.astype(np.float32)
        labels = outcome.astype(np.int64 if n_classes > 2 else np.float32)

        tensors = [
            torch.from_numpy(array)
            for array in [
                input_values,
                input_mask,
                input_times,
                input_static,
                input_context,
                labels,
            ]
        ]
        dataset = TensorDataset(*tensors)
        generator = torch.Generator().manual_seed(self.config.seed + 77)
        loader = DataLoader(
            Subset(dataset, training_indices.tolist()),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )
        device = self._resolve_device()
        validation = [tensor[validation_indices].to(device) for tensor in tensors]

        torch.manual_seed(self.config.seed)
        model = BRITSMI(
            n_markers=values.shape[2],
            n_static=static.shape[1],
            n_context=context.shape[1],
            hidden_size=self.config.hidden_size,
            n_classes=n_classes,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        best_loss = float("inf")
        best_epoch = -1
        best_state = copy.deepcopy(model.state_dict())
        weights = self.config.losses

        for epoch in range(self.config.epochs):
            model.train()
            epoch_losses: list[float] = []
            for batch in loader:
                y_batch, mask_batch, time_batch, static_batch, context_batch, label_batch = [
                    item.to(device) for item in batch
                ]
                artificial = (
                    (torch.rand_like(mask_batch) < self.config.train_holdout_fraction)
                    & (mask_batch > 0.5)
                ).float()
                fit_mask = mask_batch * (1.0 - artificial)
                fit_values = torch.where(fit_mask > 0.5, y_batch, torch.zeros_like(y_batch))
                fit_gaps = torch_marker_gaps(fit_mask, time_batch)
                output = model(
                    fit_values,
                    fit_mask,
                    fit_gaps,
                    time_batch,
                    static_batch,
                    context_batch,
                )
                reconstruction = (
                    (output["mean"] - y_batch).square() * artificial
                ).sum() / artificial.sum().clamp_min(1.0)
                consistency = (
                    (output["forward_mean"] - output["backward_mean"]).square()
                    * (1.0 - fit_mask)
                ).mean()
                with torch.no_grad():
                    reference = model(
                        y_batch,
                        mask_batch,
                        torch_marker_gaps(mask_batch, time_batch),
                        time_batch,
                        static_batch,
                        context_batch,
                    )["design"]
                summary = (output["design"] - reference).square().mean()
                downstream = self._downstream_loss(output["logits"], label_batch, n_classes)
                loss = (
                    weights.reconstruction * reconstruction
                    + weights.consistency * consistency
                    + weights.trajectory_summary * summary
                    + weights.downstream * downstream
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            model.eval()
            with torch.no_grad():
                y_val, mask_val, time_val, static_val, context_val, label_val = validation
                validation_generator = torch.Generator(device=device).manual_seed(
                    self.config.seed + epoch + 8801
                )
                artificial = (
                    torch.rand(
                        mask_val.shape,
                        generator=validation_generator,
                        device=device,
                    )
                    < self.config.validation_holdout_fraction
                ) & (mask_val > 0.5)
                artificial = artificial.float()
                fit_mask = mask_val * (1.0 - artificial)
                fit_values = torch.where(fit_mask > 0.5, y_val, torch.zeros_like(y_val))
                output = model(
                    fit_values,
                    fit_mask,
                    torch_marker_gaps(fit_mask, time_val),
                    time_val,
                    static_val,
                    context_val,
                )
                reconstruction = (
                    (output["mean"] - y_val).square() * artificial
                ).sum() / artificial.sum().clamp_min(1.0)
                consistency = (
                    (output["forward_mean"] - output["backward_mean"]).square()
                    * (1.0 - fit_mask)
                ).mean()
                reference = model(
                    y_val,
                    mask_val,
                    torch_marker_gaps(mask_val, time_val),
                    time_val,
                    static_val,
                    context_val,
                )["design"]
                summary = (output["design"] - reference).square().mean()
                downstream = self._downstream_loss(output["logits"], label_val, n_classes)
                validation_loss = float(
                    (
                        weights.reconstruction * reconstruction
                        + weights.consistency * consistency
                        + weights.trajectory_summary * summary
                        + weights.downstream * downstream
                    ).cpu()
                )
            self.training_history_.append(
                {
                    "epoch": float(epoch),
                    "training_loss": float(np.mean(epoch_losses)),
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        model.eval()
        residual_sd = self._calibrate_residual_sd(
            model,
            input_values,
            input_mask,
            input_times,
            input_static,
            input_context,
            validation_indices,
            device,
        )

        self.model_ = model
        self.marker_mean_ = marker_mean
        self.marker_sd_ = marker_sd
        self.residual_sd_ = residual_sd
        self.validation_indices_ = validation_indices
        self.fit_summary_ = FitSummary(
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            n_train=len(training_indices),
            n_validation=len(validation_indices),
            residual_sd=tuple(float(value) for value in residual_sd),
        )
        self._fit_values = values.copy()
        self._fit_mask = mask.copy()
        self._fit_times = times.copy()
        self._fit_static = static.copy()
        self._fit_context = context.copy()
        self.device_ = device
        return self

    def _calibrate_residual_sd(
        self,
        model: BRITSMI,
        standardized: np.ndarray,
        mask: np.ndarray,
        times: np.ndarray,
        static: np.ndarray,
        context: np.ndarray,
        validation_indices: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.config.seed + 12391)
        residual_sd = np.ones(standardized.shape[2], dtype=float)
        for marker in range(standardized.shape[2]):
            candidates = np.argwhere(mask[validation_indices, :, marker] > 0.5)
            if len(candidates) < 2:
                continue
            n_hide = min(
                len(candidates),
                max(10, round(self.config.residual_holdout_fraction * len(candidates))),
            )
            chosen = candidates[rng.choice(len(candidates), size=n_hide, replace=False)]
            hidden_mask = mask.copy()
            hidden_values = standardized.copy()
            for local_subject, visit in chosen:
                subject = validation_indices[local_subject]
                hidden_mask[subject, visit, marker] = 0.0
                hidden_values[subject, visit, marker] = 0.0
            with torch.no_grad():
                predicted = model(
                    torch.from_numpy(hidden_values).to(device),
                    torch.from_numpy(hidden_mask).to(device),
                    torch.from_numpy(marker_gaps(hidden_mask > 0.5, times)).to(device),
                    torch.from_numpy(times).to(device),
                    torch.from_numpy(static).to(device),
                    torch.from_numpy(context).to(device),
                )["mean"].cpu().numpy()
            residuals = [
                standardized[validation_indices[local_subject], visit, marker]
                - predicted[validation_indices[local_subject], visit, marker]
                for local_subject, visit in chosen
            ]
            estimate = float(np.std(residuals, ddof=1))
            if np.isfinite(estimate) and estimate > self.config.residual_minimum:
                residual_sd[marker] = estimate
        return residual_sd

    def _require_fitted(self) -> None:
        if self.model_ is None or self.marker_mean_ is None or self.marker_sd_ is None:
            raise RuntimeError("fit must be called before requesting completions")

    def conditional_mean(
        self,
        values: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        times: np.ndarray | None = None,
        static: np.ndarray | None = None,
        context: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one deterministic completion while preserving measured cells."""

        draws = self.sample(
            1,
            values=values,
            mask=mask,
            times=times,
            static=static,
            context=context,
            stochastic=False,
        )
        return draws[0]

    def sample(
        self,
        n_imputations: int,
        values: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        times: np.ndarray | None = None,
        static: np.ndarray | None = None,
        context: np.ndarray | None = None,
        seed: int | None = None,
        stochastic: bool = True,
    ) -> list[np.ndarray]:
        """Draw completed trajectories; measured cells are copied unchanged."""

        self._require_fitted()
        if n_imputations < 1:
            raise ValueError("n_imputations must be positive")
        values = self._fit_values if values is None else values
        mask = self._fit_mask if mask is None else mask
        times = self._fit_times if times is None else times
        static = self._fit_static if static is None else static
        context = self._fit_context if context is None else context
        if values is None or mask is None or times is None or static is None:
            raise RuntimeError("fitted arrays are unavailable")
        values, mask, times, static, context = self._validate_arrays(
            values, mask, times, static, context
        )
        if context.shape[1] != self.model_.n_context:
            raise ValueError("new context dimension does not match the fitted model")

        standardized = (
            values - self.marker_mean_.reshape(1, 1, -1)
        ) / self.marker_sd_.reshape(1, 1, -1)
        network_values = np.nan_to_num(standardized, nan=0.0).astype(np.float32)
        network_mask = mask.astype(np.float32)
        network_times = times.astype(np.float32)
        network_static = static.astype(np.float32)
        network_context = context.astype(np.float32)
        device = self.device_ or torch.device("cpu")
        with torch.no_grad():
            mean_standardized = self.model_(
                torch.from_numpy(network_values).to(device),
                torch.from_numpy(network_mask).to(device),
                torch.from_numpy(marker_gaps(mask, times)).to(device),
                torch.from_numpy(network_times).to(device),
                torch.from_numpy(network_static).to(device),
                torch.from_numpy(network_context).to(device),
            )["mean"].cpu().numpy()

        rng = np.random.default_rng(self.config.seed + 20001 if seed is None else seed)
        missing = ~mask
        draws: list[np.ndarray] = []
        for _ in range(n_imputations):
            draw_standardized = mean_standardized.copy()
            if stochastic:
                for marker in range(draw_standardized.shape[2]):
                    marker_missing = missing[:, :, marker]
                    draw_standardized[:, :, marker][marker_missing] += rng.normal(
                        scale=self.config.variance_factor * self.residual_sd_[marker],
                        size=int(marker_missing.sum()),
                    )
            draw = (
                draw_standardized * self.marker_sd_.reshape(1, 1, -1)
                + self.marker_mean_.reshape(1, 1, -1)
            )
            draw[mask] = values[mask]
            draws.append(draw)
        return draws

    def sample_association_calibrated(
        self,
        n_imputations: int,
        outcome: np.ndarray,
        coefficients: np.ndarray,
        intercept: float = 0.0,
        calibration: AssociationCalibrationConfig | None = None,
        values: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        times: np.ndarray | None = None,
        static: np.ndarray | None = None,
        context: np.ndarray | None = None,
        seed: int | None = None,
    ) -> tuple[list[np.ndarray], list[CalibrationDiagnostics]]:
        """Draw and calibrate completions for an explanatory association model.

        This method deliberately requires outcome information and therefore is
        not the label-free deployment interface. Use :meth:`sample` for new
        patients whose outcomes are unavailable.
        """

        draws = self.sample(
            n_imputations,
            values=values,
            mask=mask,
            times=times,
            static=static,
            context=context,
            seed=seed,
            stochastic=True,
        )
        values = self._fit_values if values is None else values
        mask = self._fit_mask if mask is None else mask
        times = self._fit_times if times is None else times
        static = self._fit_static if static is None else static
        if values is None or mask is None or times is None or static is None:
            raise RuntimeError("fitted arrays are unavailable")
        calibrated: list[np.ndarray] = []
        diagnostics: list[CalibrationDiagnostics] = []
        for draw in draws:
            updated, diagnostic = calibrate_completed_trajectory(
                draw,
                values,
                mask,
                times,
                static,
                outcome,
                coefficients,
                intercept,
                calibration,
            )
            calibrated.append(updated)
            diagnostics.append(diagnostic)
        return calibrated, diagnostics
