"""BRITS-MI neural components for irregular longitudinal trajectories."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def marker_gaps(mask: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return elapsed time since the previous measured value for each marker."""

    mask = np.asarray(mask, dtype=bool)
    times = np.asarray(times, dtype=float)
    if mask.ndim != 3:
        raise ValueError("mask must have shape (subjects, visits, markers)")
    if times.ndim == 1:
        times = np.broadcast_to(times, mask.shape[:2])
    if times.shape != mask.shape[:2]:
        raise ValueError("times must have shape (visits,) or (subjects, visits)")
    gaps = np.zeros(mask.shape, dtype=np.float32)
    for subject in range(mask.shape[0]):
        for marker in range(mask.shape[2]):
            previous = float(times[subject, 0])
            seen = False
            for visit in range(mask.shape[1]):
                now = float(times[subject, visit])
                gaps[subject, visit, marker] = now - previous if seen else now - float(times[subject, 0])
                if mask[subject, visit, marker]:
                    previous = now
                    seen = True
    return gaps


def torch_marker_gaps(mask: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Differentiation-free torch implementation used after artificial masking."""

    batch, length, markers = mask.shape
    last_time = times[:, :1].expand(-1, markers).clone()
    seen = torch.zeros((batch, markers), dtype=torch.bool, device=mask.device)
    rows: list[torch.Tensor] = []
    for visit in range(length):
        now = times[:, visit].unsqueeze(1).expand(-1, markers)
        gap = torch.where(seen, now - last_time, now - times[:, :1])
        rows.append(gap)
        measured = mask[:, visit, :] > 0.5
        last_time = torch.where(measured, now, last_time)
        seen = seen | measured
    return torch.stack(rows, dim=1)


def causal_history(values: torch.Tensor, mask: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Summaries available strictly before each recurrent update."""

    batch, length, markers = values.shape
    count = torch.zeros((batch, markers), dtype=values.dtype, device=values.device)
    total = torch.zeros_like(count)
    total_sq = torch.zeros_like(count)
    first = torch.zeros_like(count)
    last = torch.zeros_like(count)
    first_time = torch.zeros_like(count)
    last_time = torch.zeros_like(count)
    rows: list[torch.Tensor] = []
    for visit in range(length):
        mean = total / count.clamp_min(1.0)
        variance = total_sq / count.clamp_min(1.0) - mean.square()
        sd = torch.sqrt(torch.clamp(variance, min=0.0))
        slope = (last - first) / (last_time - first_time).clamp_min(1e-4)
        observed_fraction = count / max(visit, 1)
        rows.append(
            torch.cat(
                [mean, sd, last, slope, observed_fraction, 1.0 - observed_fraction],
                dim=1,
            )
        )

        measured = mask[:, visit, :] > 0.5
        value = values[:, visit, :]
        now = times[:, visit].unsqueeze(1).expand(-1, markers)
        first_measurement = measured & (count <= 0)
        first = torch.where(first_measurement, value, first)
        first_time = torch.where(first_measurement, now, first_time)
        last = torch.where(measured, value, last)
        last_time = torch.where(measured, now, last_time)
        total = total + torch.where(measured, value, torch.zeros_like(value))
        total_sq = total_sq + torch.where(measured, value.square(), torch.zeros_like(value))
        count = count + measured.to(values.dtype)
    return torch.stack(rows, dim=1)


class RITSCell(nn.Module):
    """One directional recurrent imputation pass with time decay and feature mixing."""

    def __init__(
        self,
        n_markers: int,
        n_static: int,
        n_context: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.n_markers = n_markers
        self.n_context = n_context
        self.hidden_size = hidden_size
        self.decay = nn.Linear(n_markers, hidden_size)
        self.history_head = nn.Linear(hidden_size, n_markers)
        self.feature_weight = nn.Parameter(torch.empty(n_markers, n_markers))
        self.feature_bias = nn.Parameter(torch.zeros(n_markers))
        nn.init.xavier_uniform_(self.feature_weight)
        self.mix_gate = nn.Linear(2 * n_markers, n_markers)
        recurrent_width = 3 * n_markers + 6 * n_markers + n_static + n_context + 1
        self.gru = nn.GRUCell(recurrent_width, hidden_size)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        gaps: torch.Tensor,
        times: torch.Tensor,
        static: torch.Tensor,
        context: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            work_values = torch.flip(values, dims=[1])
            work_mask = torch.flip(mask, dims=[1])
            work_times = times[:, -1:] - torch.flip(times, dims=[1])
            work_gaps = torch_marker_gaps(work_mask, work_times)
        else:
            work_values = values
            work_mask = mask
            work_times = times
            work_gaps = gaps

        hidden = torch.zeros(
            (values.shape[0], self.hidden_size),
            dtype=values.dtype,
            device=values.device,
        )
        history = causal_history(work_values, work_mask, work_times)
        predictions: list[torch.Tensor] = []
        completions: list[torch.Tensor] = []
        off_diagonal = 1.0 - torch.eye(
            self.n_markers,
            dtype=values.dtype,
            device=values.device,
        )
        for visit in range(values.shape[1]):
            gamma = torch.exp(-torch.relu(self.decay(work_gaps[:, visit, :])))
            hidden = gamma * hidden
            history_mean = self.history_head(hidden)
            provisional = torch.where(
                work_mask[:, visit, :] > 0.5,
                work_values[:, visit, :],
                history_mean,
            )
            feature_mean = torch.nn.functional.linear(
                provisional,
                self.feature_weight * off_diagonal,
                self.feature_bias,
            )
            alpha = torch.sigmoid(
                self.mix_gate(
                    torch.cat([work_mask[:, visit, :], work_gaps[:, visit, :]], dim=1)
                )
            )
            prediction = alpha * feature_mean + (1.0 - alpha) * history_mean
            completed = torch.where(
                work_mask[:, visit, :] > 0.5,
                work_values[:, visit, :],
                prediction,
            )
            recurrent_input = torch.cat(
                [
                    completed,
                    work_mask[:, visit, :],
                    work_gaps[:, visit, :],
                    history[:, visit, :],
                    static,
                    context,
                    work_times[:, visit].unsqueeze(1),
                ],
                dim=1,
            )
            hidden = self.gru(recurrent_input, hidden)
            predictions.append(prediction)
            completions.append(completed)

        prediction_tensor = torch.stack(predictions, dim=1)
        completion_tensor = torch.stack(completions, dim=1)
        if reverse:
            prediction_tensor = torch.flip(prediction_tensor, dims=[1])
            completion_tensor = torch.flip(completion_tensor, dims=[1])
        return prediction_tensor, completion_tensor


class BRITSMI(nn.Module):
    """Bidirectional recurrent imputer with a downstream training head."""

    def __init__(
        self,
        n_markers: int,
        n_static: int,
        hidden_size: int = 48,
        n_context: int = 0,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        if n_markers < 3 or n_static < 3:
            raise ValueError("the default downstream design requires 3 markers and 3 static covariates")
        self.n_context = n_context
        self.n_classes = n_classes
        self.forward_rits = RITSCell(n_markers, n_static, n_context, hidden_size)
        self.backward_rits = RITSCell(n_markers, n_static, n_context, hidden_size)
        self.combine_gate = nn.Linear(2 * n_markers, n_markers)
        output_width = 1 if n_classes == 2 else n_classes
        self.downstream_head = nn.Linear(13 + n_context, output_width)

    @staticmethod
    def clinical_design(
        completed: torch.Tensor,
        times: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        """Prespecified late-level, slope, main-effect, and interaction terms."""

        late_mean = completed[:, -3:, :3].mean(dim=1)
        denominator = (times[:, -1] - times[:, -3]).clamp_min(1e-4).unsqueeze(1)
        slope = (completed[:, -1, :3] - completed[:, -3, :3]) / denominator
        diabetes, hypertension, male = static[:, 0], static[:, 1], static[:, 2]
        return torch.stack(
            [
                late_mean[:, 0],
                late_mean[:, 1],
                late_mean[:, 2],
                slope[:, 0],
                slope[:, 1],
                slope[:, 2],
                diabetes,
                hypertension,
                male,
                late_mean[:, 0] * diabetes,
                slope[:, 0] * diabetes,
                slope[:, 2] * hypertension,
                hypertension * male,
            ],
            dim=1,
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        gaps: torch.Tensor,
        times: torch.Tensor,
        static: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if context is None:
            context = values.new_zeros((values.shape[0], self.n_context))
        if context.shape != (values.shape[0], self.n_context):
            raise ValueError("context has the wrong subject or feature dimension")

        forward_mean, _ = self.forward_rits(
            values, mask, gaps, times, static, context, reverse=False
        )
        backward_mean, _ = self.backward_rits(
            values, mask, gaps, times, static, context, reverse=True
        )
        alpha = torch.sigmoid(self.combine_gate(torch.cat([mask, gaps], dim=2)))
        mean = alpha * forward_mean + (1.0 - alpha) * backward_mean
        completed = torch.where(mask > 0.5, values, mean)
        design = self.clinical_design(completed, times, static)
        downstream_input = torch.cat([design, context], dim=1)
        logits = self.downstream_head(downstream_input)
        if self.n_classes == 2:
            logits = logits.squeeze(1)
        return {
            "forward_mean": forward_mean,
            "backward_mean": backward_mean,
            "mean": mean,
            "completed": completed,
            "design": design,
            "logits": logits,
        }
