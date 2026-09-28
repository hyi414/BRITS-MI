#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time as time_module
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_effect_recovery_harder_irregularity_test as harder  # noqa: E402


run_large = harder.run_large
sim = run_large.sim
dgp1 = run_large.dgp1
TRUE_BETA = run_large.TRUE_BETA
COEFFICIENT_LABELS = run_large.COEFFICIENT_LABELS
SCENARIOS = harder.HARDER_SCENARIOS

METHOD_LABELS = {
    "brits_mi_gradient": "BRITS-MI",
    "brits_mi_no_downstream": "BRITS-MI without downstream loss",
}


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def torch_marker_gaps(mask: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    batch, length, markers = mask.shape
    last_time = times[:, :1].expand(-1, markers).clone()
    seen = torch.zeros((batch, markers), dtype=torch.bool, device=mask.device)
    rows = []
    for visit in range(length):
        current_time = times[:, visit].unsqueeze(1).expand(-1, markers)
        rows.append(torch.where(seen, current_time - last_time, current_time - times[:, :1]))
        observed = mask[:, visit, :] > 0.5
        last_time = torch.where(observed, current_time, last_time)
        seen = seen | observed
    return torch.stack(rows, dim=1)


def causal_history(y: torch.Tensor, mask: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    batch, length, markers = y.shape
    count = torch.zeros((batch, markers), dtype=y.dtype, device=y.device)
    total = torch.zeros_like(count)
    total_sq = torch.zeros_like(count)
    first = torch.zeros_like(count)
    last = torch.zeros_like(count)
    first_time = torch.zeros_like(count)
    last_time = torch.zeros_like(count)
    rows = []
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
        observed = mask[:, visit, :] > 0.5
        value = y[:, visit, :]
        current_time = times[:, visit].unsqueeze(1).expand(-1, markers)
        first_observation = observed & (count <= 0)
        first = torch.where(first_observation, value, first)
        first_time = torch.where(first_observation, current_time, first_time)
        last = torch.where(observed, value, last)
        last_time = torch.where(observed, current_time, last_time)
        total = total + torch.where(observed, value, torch.zeros_like(value))
        total_sq = total_sq + torch.where(observed, value.square(), torch.zeros_like(value))
        count = count + observed.to(y.dtype)
    return torch.stack(rows, dim=1)


def global_trajectory_summary(
    y: torch.Tensor,
    mask: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Compact mask-normalized history features for the sequence-wide branch."""
    count = mask.sum(dim=1)
    safe_count = count.clamp_min(1.0)
    mean = (y * mask).sum(dim=1) / safe_count
    variance = (((y - mean.unsqueeze(1)) * mask).square()).sum(dim=1) / safe_count
    sd = torch.sqrt(variance.clamp_min(0.0))

    first_index = torch.argmax(mask, dim=1)
    last_index = y.shape[1] - 1 - torch.argmax(torch.flip(mask, dims=[1]), dim=1)
    first = torch.gather(y, 1, first_index.unsqueeze(1)).squeeze(1)
    last = torch.gather(y, 1, last_index.unsqueeze(1)).squeeze(1)
    time_grid = times.unsqueeze(2).expand(-1, -1, y.shape[2])
    first_time = torch.gather(time_grid, 1, first_index.unsqueeze(1)).squeeze(1)
    last_time = torch.gather(time_grid, 1, last_index.unsqueeze(1)).squeeze(1)
    has_observation = count > 0
    first = torch.where(has_observation, first, torch.zeros_like(first))
    last = torch.where(has_observation, last, torch.zeros_like(last))
    slope = torch.where(
        count > 1,
        (last - first) / (last_time - first_time).clamp_min(1e-4),
        torch.zeros_like(last),
    )

    split = max(1, y.shape[1] // 2)
    early_count = mask[:, :split, :].sum(dim=1).clamp_min(1.0)
    late_count = mask[:, split:, :].sum(dim=1).clamp_min(1.0)
    early_mean = (y[:, :split, :] * mask[:, :split, :]).sum(dim=1) / early_count
    late_mean = (y[:, split:, :] * mask[:, split:, :]).sum(dim=1) / late_count
    observed_fraction = count / float(y.shape[1])
    last_gap = torch.where(
        has_observation,
        times[:, -1:].expand_as(last_time) - last_time,
        times[:, -1:].expand_as(last_time) - times[:, :1].expand_as(last_time),
    )
    return torch.cat(
        [mean, sd, first, last, slope, early_mean, late_mean, observed_fraction, last_gap],
        dim=1,
    )


class RITSDistribution(nn.Module):
    def __init__(self, n_markers: int, n_static: int, hidden_size: int) -> None:
        super().__init__()
        self.n_markers = n_markers
        self.hidden_size = hidden_size
        self.decay = nn.Linear(n_markers, hidden_size)
        self.history_mean = nn.Linear(hidden_size, n_markers)
        self.feature_weight = nn.Parameter(torch.empty(n_markers, n_markers))
        self.feature_bias = nn.Parameter(torch.zeros(n_markers))
        nn.init.xavier_uniform_(self.feature_weight)
        self.mean_gate = nn.Linear(2 * n_markers, n_markers)
        scale_input = hidden_size + 2 * n_markers + n_static + 1
        self.scale_head = nn.Sequential(
            nn.Linear(scale_input, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, n_markers),
        )
        history_size = 6 * n_markers
        self.gru = nn.GRUCell(
            n_markers + n_markers + n_markers + history_size + n_static + 1,
            hidden_size,
        )

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        static: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            y_work = torch.flip(y, dims=[1])
            mask_work = torch.flip(mask, dims=[1])
            times_work = times[:, -1:] - torch.flip(times, dims=[1])
        else:
            y_work = y
            mask_work = mask
            times_work = times
        gaps = torch_marker_gaps(mask_work, times_work)
        history = causal_history(y_work, mask_work, times_work)
        hidden = torch.zeros(
            (y.shape[0], self.hidden_size), dtype=y.dtype, device=y.device
        )
        means = []
        scales = []
        off_diagonal = 1.0 - torch.eye(
            self.n_markers, dtype=y.dtype, device=y.device
        )
        for visit in range(y.shape[1]):
            gamma = torch.exp(-torch.relu(self.decay(gaps[:, visit, :])))
            hidden = gamma * hidden
            history_mean = self.history_mean(hidden)
            provisional = torch.where(
                mask_work[:, visit, :] > 0.5,
                y_work[:, visit, :],
                history_mean,
            )
            feature_mean = torch.nn.functional.linear(
                provisional,
                self.feature_weight * off_diagonal,
                self.feature_bias,
            )
            alpha = torch.sigmoid(
                self.mean_gate(
                    torch.cat([mask_work[:, visit, :], gaps[:, visit, :]], dim=1)
                )
            )
            mean = alpha * feature_mean + (1.0 - alpha) * history_mean
            scale_context = torch.cat(
                [
                    hidden,
                    mask_work[:, visit, :],
                    gaps[:, visit, :],
                    static,
                    times_work[:, visit].unsqueeze(1),
                ],
                dim=1,
            )
            scale = torch.nn.functional.softplus(self.scale_head(scale_context)) + 0.03
            completed = torch.where(
                mask_work[:, visit, :] > 0.5,
                y_work[:, visit, :],
                mean,
            )
            cell_input = torch.cat(
                [
                    completed,
                    mask_work[:, visit, :],
                    gaps[:, visit, :],
                    history[:, visit, :],
                    static,
                    times_work[:, visit].unsqueeze(1),
                ],
                dim=1,
            )
            hidden = self.gru(cell_input, hidden)
            means.append(mean)
            scales.append(scale)
        mean_tensor = torch.stack(means, dim=1)
        scale_tensor = torch.stack(scales, dim=1)
        if reverse:
            mean_tensor = torch.flip(mean_tensor, dims=[1])
            scale_tensor = torch.flip(scale_tensor, dims=[1])
        return mean_tensor, scale_tensor


class NeuralBRITSMI(nn.Module):
    def __init__(
        self,
        n_time: int,
        n_markers: int,
        n_static: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.n_time = n_time
        self.n_markers = n_markers
        self.forward_rits = RITSDistribution(n_markers, n_static, hidden_size)
        self.backward_rits = RITSDistribution(n_markers, n_static, hidden_size)
        self.combine_gate = nn.Linear(2 * n_markers, n_markers)
        global_input = 2 * n_time * n_markers + n_time + n_static + 9 * n_markers
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input, 2 * hidden_size),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
        )
        # Preserve a regularized linear sequence-wide path for DGPs whose
        # cross-time conditional mean is simpler than the nonlinear residual.
        self.global_linear_mean = nn.Linear(global_input, n_time * n_markers)
        self.global_mean = nn.Linear(hidden_size, n_time * n_markers)
        self.global_scale = nn.Linear(hidden_size, n_time * n_markers)
        self.context_gate = nn.Linear(3 * n_markers, n_markers)
        nn.init.constant_(self.context_gate.bias, -1.5)
        self.log_scale_adjustment = nn.Parameter(torch.zeros(n_markers))
        self.association_head = nn.Linear(len(TRUE_BETA), 1)

    @staticmethod
    def downstream_design(
        completed: torch.Tensor,
        times: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        late = completed[:, -3:, :]
        late_mean = late.mean(dim=1)
        denominator = (times[:, -1] - times[:, -3]).clamp_min(1e-4).unsqueeze(1)
        slope = (completed[:, -1, :] - completed[:, -3, :]) / denominator
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
        y: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        static: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        forward_mean, forward_scale = self.forward_rits(
            y, mask, times, static, reverse=False
        )
        backward_mean, backward_scale = self.backward_rits(
            y, mask, times, static, reverse=True
        )
        gaps = torch_marker_gaps(mask, times)
        alpha = torch.sigmoid(self.combine_gate(torch.cat([mask, gaps], dim=2)))
        recurrent_mean = alpha * forward_mean + (1.0 - alpha) * backward_mean
        recurrent_variance = (
            alpha
            * (forward_scale.square() + (forward_mean - recurrent_mean).square())
            + (1.0 - alpha)
            * (backward_scale.square() + (backward_mean - recurrent_mean).square())
        )
        global_input = torch.cat(
            [
                y.reshape(y.shape[0], -1),
                mask.reshape(mask.shape[0], -1),
                times,
                static,
                global_trajectory_summary(y, mask, times),
            ],
            dim=1,
        )
        global_context = self.global_encoder(global_input)
        global_mean = (
            self.global_linear_mean(global_input) + self.global_mean(global_context)
        ).reshape_as(y)
        global_scale = (
            torch.nn.functional.softplus(self.global_scale(global_context)).reshape_as(y)
            + 0.03
        )
        context_alpha = torch.sigmoid(
            self.context_gate(
                torch.cat(
                    [mask, gaps, torch.abs(forward_mean - backward_mean)], dim=2
                )
            )
        )
        mean = context_alpha * recurrent_mean + (1.0 - context_alpha) * global_mean
        variance = (
            context_alpha
            * (recurrent_variance + (recurrent_mean - mean).square())
            + (1.0 - context_alpha)
            * (global_scale.square() + (global_mean - mean).square())
        )
        scale_adjustment = torch.exp(
            torch.clamp(self.log_scale_adjustment, min=-1.5, max=1.5)
        ).view(1, 1, -1)
        scale = torch.sqrt(variance.clamp_min(1e-5)) * scale_adjustment
        scale = scale.clamp(min=0.03, max=3.0)
        if epsilon is None:
            imputed = mean
        else:
            imputed = mean + scale * epsilon
        completed = torch.where(mask > 0.5, y, imputed)
        completed_mean = torch.where(mask > 0.5, y, mean)
        design = self.downstream_design(completed, times, static)
        mean_design = self.downstream_design(completed_mean, times, static)
        logits = self.association_head(design).squeeze(1)
        mean_logits = self.association_head(mean_design).squeeze(1)
        return {
            "forward_mean": forward_mean,
            "backward_mean": backward_mean,
            "forward_scale": forward_scale,
            "backward_scale": backward_scale,
            "recurrent_mean": recurrent_mean,
            "global_mean": global_mean,
            "mean": mean,
            "scale": scale,
            "completed": completed,
            "completed_mean": completed_mean,
            "design": design,
            "mean_design": mean_design,
            "logits": logits,
            "mean_logits": mean_logits,
        }


def artificial_mask_numpy(
    mask: np.ndarray, rate: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hidden = np.zeros_like(mask, dtype=np.float32)
    for visit in range(mask.shape[1]):
        for marker in range(mask.shape[2]):
            candidates = np.flatnonzero(mask[:, visit, marker] > 0.5)
            if len(candidates) == 0:
                continue
            n_hide = max(1, int(round(rate * len(candidates))))
            selected = rng.choice(candidates, size=min(n_hide, len(candidates)), replace=False)
            hidden[selected, visit, marker] = 1.0
    return hidden


def gaussian_nll(
    truth: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    scale = scale.clamp(min=0.03, max=3.0)
    value = 0.5 * ((truth - mean) / scale).square() + torch.log(scale)
    return (value * target).sum() / target.sum().clamp_min(1.0)


def association_score_loss(
    completed_design: torch.Tensor,
    reference_design: torch.Tensor,
    labels: torch.Tensor,
    completed_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """Preserve the standardized estimating equation for each outcome term."""
    reference_design = reference_design.detach()
    reference_residual = (
        labels - torch.sigmoid(reference_logits)
    ).detach()
    completed_residual = labels - torch.sigmoid(completed_logits)
    reference_contribution = reference_design * reference_residual.unsqueeze(1)
    completed_contribution = completed_design * completed_residual.unsqueeze(1)
    score_gap = (completed_contribution - reference_contribution).mean(dim=0)
    # Standardize by the Monte Carlo scale of a batch score. This prevents the
    # association term from vanishing relative to cell-level reconstruction.
    batch_scale = (
        reference_design.std(dim=0, unbiased=False)
        * reference_residual.std(unbiased=False)
        / math.sqrt(max(reference_design.shape[0], 1))
    ).detach().clamp_min(0.02)
    standardized_score_gap = score_gap / batch_scale
    subject_gap = completed_contribution - reference_contribution
    subject_scale = reference_design.std(dim=0, unbiased=False).detach().clamp_min(0.10)
    return standardized_score_gap.square().mean() + 0.02 * (
        subject_gap / subject_scale
    ).square().mean()


def validation_metrics(
    model: NeuralBRITSMI,
    tensors: tuple[torch.Tensor, ...],
    val_idx: np.ndarray,
    val_artificial: np.ndarray,
    lambda_down: float,
    lambda_sum: float,
    lambda_cons: float,
    downstream_objective: str,
    lambda_head: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        y_all, mask_all, times_all, static_all, label_all = tensors
        index = torch.as_tensor(val_idx, dtype=torch.long)
        y = y_all[index].to(device)
        mask = mask_all[index].to(device)
        times = times_all[index].to(device)
        static = static_all[index].to(device)
        labels = label_all[index].to(device)
        artificial = torch.as_tensor(val_artificial, dtype=y.dtype, device=device)
        input_mask = mask * (1.0 - artificial)
        input_y = torch.where(input_mask > 0.5, y, torch.zeros_like(y))
        output = model(input_y, input_mask, times, static, epsilon=None)
        reference = model(y, mask, times, static, epsilon=None)
        squared = (output["mean"] - y).square()
        reconstruction_rmse = torch.sqrt(
            (squared * artificial).sum() / artificial.sum().clamp_min(1.0)
        )
        summary_rmse = torch.sqrt(
            (output["mean_design"][:, :6] - reference["mean_design"][:, :6])
            .square()
            .mean()
        )
        missing_target = 1.0 - input_mask
        consistency_rmse = torch.sqrt(
            (
                (output["forward_mean"] - output["backward_mean"]).square()
                * missing_target
            ).sum()
            / missing_target.sum().clamp_min(1.0)
        )
        reference_logits = model.association_head(
            reference["mean_design"].detach()
        ).squeeze(1)
        head_loss = nn.functional.binary_cross_entropy_with_logits(
            reference_logits,
            labels,
        )
        if downstream_objective == "association_score":
            downstream = association_score_loss(
                output["mean_design"],
                reference["mean_design"],
                labels,
                output["mean_logits"],
                reference_logits,
            )
        else:
            downstream = nn.functional.binary_cross_entropy_with_logits(
                output["mean_logits"], labels
            )
        probabilities = torch.sigmoid(output["mean_logits"]).cpu().numpy()
        observed_labels = labels.cpu().numpy()
        try:
            auroc = float(roc_auc_score(observed_labels, probabilities))
        except ValueError:
            auroc = float("nan")
        score = (
            reconstruction_rmse
            + lambda_sum * summary_rmse
            + lambda_cons * consistency_rmse
            + (0.35 * lambda_down) * downstream
            + lambda_head * head_loss
        )
        coverage_mask = artificial > 0.5
        lower = output["mean"] - 1.96 * output["scale"]
        upper = output["mean"] + 1.96 * output["scale"]
        covered = ((y >= lower) & (y <= upper) & coverage_mask).sum()
        coverage = covered / coverage_mask.sum().clamp_min(1)
    return {
        "score": float(score.item()),
        "reconstruction_rmse": float(reconstruction_rmse.item()),
        "summary_rmse": float(summary_rmse.item()),
        "consistency_rmse": float(consistency_rmse.item()),
        "downstream_log_loss": float(downstream.item()),
        "reference_head_log_loss": float(head_loss.item()),
        "downstream_auroc": auroc,
        "cell_coverage": float(coverage.item()),
    }


def fit_fold_model(
    y_standardized: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    static: np.ndarray,
    outcome: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    lambda_down: float,
    device: torch.device,
) -> tuple[NeuralBRITSMI, np.ndarray, list[dict], dict]:
    set_torch_seed(seed)
    y_tensor = torch.from_numpy(y_standardized.astype(np.float32))
    mask_tensor = torch.from_numpy(mask.astype(np.float32))
    time_tensor = torch.from_numpy(times.astype(np.float32))
    static_tensor = torch.from_numpy(static.astype(np.float32))
    label_tensor = torch.from_numpy(outcome.astype(np.float32))
    tensors = (y_tensor, mask_tensor, time_tensor, static_tensor, label_tensor)
    dataset = TensorDataset(*tensors)
    generator = torch.Generator().manual_seed(seed + 11)
    loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_artificial = artificial_mask_numpy(
        mask[val_idx], args.validation_mask_rate, seed + 613
    )

    model = NeuralBRITSMI(
        mask.shape[1], mask.shape[2], static.shape[1], args.hidden_size
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    bce = nn.BCEWithLogitsLoss()
    downstream_objective = getattr(args, "downstream_objective", "classification")
    lambda_head = float(getattr(args, "lambda_head", 0.50))
    best_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []

    imputation_parameters = list(model.forward_rits.parameters()) + list(
        model.backward_rits.parameters()
    ) + list(model.combine_gate.parameters()) + list(model.global_encoder.parameters()) + list(
        model.global_linear_mean.parameters()
    ) + list(
        model.global_mean.parameters()
    ) + list(model.global_scale.parameters()) + list(model.context_gate.parameters()) + [
        model.log_scale_adjustment
    ]

    for epoch in range(args.max_epochs):
        model.train()
        train_totals = {
            "loss": 0.0,
            "rec": 0.0,
            "cons": 0.0,
            "summary": 0.0,
            "downstream": 0.0,
        }
        n_seen = 0
        isolated_task_gradient = float("nan")
        finetune_fraction = max(
            0.0,
            min(1.0, (epoch - args.pretrain_epochs + 1) / max(args.downstream_warmup, 1)),
        )
        active_lambda_down = lambda_down * finetune_fraction

        for batch_index, (y, observed_mask, visit_times, subject_static, labels) in enumerate(loader):
            y = y.to(device)
            observed_mask = observed_mask.to(device)
            visit_times = visit_times.to(device)
            subject_static = subject_static.to(device)
            labels = labels.to(device)
            artificial = (
                (torch.rand_like(observed_mask) < args.training_mask_rate)
                & (observed_mask > 0.5)
            ).float()
            input_mask = observed_mask * (1.0 - artificial)
            input_y = torch.where(input_mask > 0.5, y, torch.zeros_like(y))
            epsilon = torch.randn_like(y)
            output = model(input_y, input_mask, visit_times, subject_static, epsilon)
            with torch.no_grad():
                reference = model(y, observed_mask, visit_times, subject_static, None)

            reconstruction = gaussian_nll(
                y, output["mean"], output["scale"], artificial
            )
            reconstruction_mse = (
                (output["mean"] - y).square() * artificial
            ).sum() / artificial.sum().clamp_min(1.0)
            directional = 0.5 * (
                gaussian_nll(y, output["forward_mean"], output["forward_scale"], artificial)
                + gaussian_nll(y, output["backward_mean"], output["backward_scale"], artificial)
            )
            missing_target = 1.0 - input_mask
            consistency = (
                (output["forward_mean"] - output["backward_mean"]).square()
                * missing_target
            ).sum() / missing_target.sum().clamp_min(1.0)
            summary = (
                output["design"][:, :6] - reference["mean_design"][:, :6]
            ).square().mean()
            reference_design = reference["mean_design"].detach()
            reference_logits = model.association_head(reference_design).squeeze(1)
            head_loss = bce(reference_logits, labels)
            if downstream_objective == "association_score":
                score_weight = model.association_head.weight.detach()
                score_bias = model.association_head.bias.detach()
                completed_score_logits = nn.functional.linear(
                    output["design"], score_weight, score_bias
                ).squeeze(1)
                reference_score_logits = nn.functional.linear(
                    reference_design, score_weight, score_bias
                ).squeeze(1)
                downstream = association_score_loss(
                    output["design"],
                    reference_design,
                    labels,
                    completed_score_logits,
                    reference_score_logits,
                )
            else:
                downstream = bce(output["logits"], labels)
            loss = (
                reconstruction
                + args.lambda_reconstruction_mse * reconstruction_mse
                + args.lambda_directional * directional
                + args.lambda_cons * consistency
                + args.lambda_sum * summary
                + active_lambda_down * downstream
                + lambda_head * head_loss
            )

            if batch_index == 0 and active_lambda_down > 0:
                task_gradients = torch.autograd.grad(
                    downstream,
                    imputation_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                isolated_task_gradient = float(
                    math.sqrt(
                        sum(
                            float(gradient.detach().square().sum().item())
                            for gradient in task_gradients
                            if gradient is not None
                        )
                    )
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            batch_n = len(labels)
            n_seen += batch_n
            train_totals["loss"] += float(loss.item()) * batch_n
            train_totals["rec"] += float(reconstruction.item()) * batch_n
            train_totals["cons"] += float(consistency.item()) * batch_n
            train_totals["summary"] += float(summary.item()) * batch_n
            train_totals["downstream"] += float(downstream.item()) * batch_n

        val_metrics = validation_metrics(
            model,
            tensors,
            val_idx,
            val_artificial,
            lambda_down,
            args.lambda_sum,
            args.lambda_cons,
            downstream_objective,
            lambda_head,
            device,
        )
        row = {
            "epoch": epoch + 1,
            "active_lambda_down": active_lambda_down,
            "task_to_imputer_gradient_norm": isolated_task_gradient,
            **{f"train_{key}": value / max(n_seen, 1) for key, value in train_totals.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        eligible = epoch + 1 >= args.min_epochs
        if eligible and val_metrics["score"] < best_score - args.min_delta:
            best_score = val_metrics["score"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        elif eligible:
            epochs_without_improvement += 1
        if eligible and epochs_without_improvement >= args.patience:
            break

    if best_epoch == 0:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history)
        best_score = history[-1]["val_score"]
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        index = torch.as_tensor(val_idx, dtype=torch.long)
        y_val = y_tensor[index].to(device)
        mask_val = mask_tensor[index].to(device)
        times_val = time_tensor[index].to(device)
        static_val = static_tensor[index].to(device)
        artificial = torch.as_tensor(val_artificial, dtype=y_val.dtype, device=device)
        input_mask = mask_val * (1.0 - artificial)
        input_y = torch.where(input_mask > 0.5, y_val, torch.zeros_like(y_val))
        output = model(input_y, input_mask, times_val, static_val, None)
        calibration = np.ones(mask.shape[2], dtype=np.float32)
        for marker in range(mask.shape[2]):
            target = artificial[:, :, marker] > 0.5
            ratios = (
                torch.abs(y_val[:, :, marker] - output["mean"][:, :, marker])
                / output["scale"][:, :, marker].clamp_min(0.03)
            )[target]
            if ratios.numel() > 5:
                value = float(torch.quantile(ratios, 0.95).item() / 1.959963984540054)
                calibration[marker] = np.float32(np.clip(value, 0.50, 3.00))

    audit = {
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "epochs_run": len(history),
        "calibration_ast": float(calibration[0]),
        "calibration_alt": float(calibration[1]),
        "calibration_platelet": float(calibration[2]),
        "final_task_to_imputer_gradient_norm": float(
            next(
                (
                    row["task_to_imputer_gradient_norm"]
                    for row in reversed(history)
                    if np.isfinite(row["task_to_imputer_gradient_norm"])
                ),
                0.0,
            )
        ),
    }
    return model, calibration, history, audit


def generate_locked_dataset(
    seed: int, scenario: str, n: int, target_missing: float
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    diabetes = rng.binomial(1, 0.36, size=n).astype(float)
    hypertension = rng.binomial(
        1, sim._sigmoid(-0.20 + 1.05 * diabetes), size=n
    ).astype(float)
    male = rng.binomial(1, 0.46, size=n).astype(float)
    trajectory, time, latent = sim._simulate_irregular_trajectory(
        rng, diabetes, hypertension, male, scenario
    )
    true_features = dgp1._trajectory_features(trajectory, time)
    true_design = dgp1._design(true_features, diabetes, hypertension, male)
    eta = -0.84 + sum(
        TRUE_BETA[name] * true_design[name].to_numpy() for name in TRUE_BETA
    )
    outcome = rng.binomial(1, sim._sigmoid(eta)).astype(int)
    observed = sim._make_missingness_targeted(
        trajectory,
        outcome,
        diabetes,
        hypertension,
        latent,
        time,
        rng,
        target_missing,
        scenario,
    )
    trajectory_obs = np.where(observed, trajectory, np.nan)
    return {
        "trajectory": trajectory.astype(np.float32),
        "trajectory_obs": trajectory_obs.astype(np.float32),
        "observed": observed,
        "time": time.astype(np.float32),
        "times": np.broadcast_to(time.astype(np.float32), (n, len(time))).copy(),
        "static": np.column_stack([diabetes, hypertension, male]).astype(np.float32),
        "diabetes": diabetes,
        "hypertension": hypertension,
        "male": male,
        "outcome": outcome,
        "true_features": true_features,
        "target_missing": float(target_missing),
    }


def cross_fitted_completions(
    data: dict[str, np.ndarray],
    seed: int,
    args: argparse.Namespace,
    lambda_down: float,
    method: str,
    device: torch.device,
) -> tuple[list[np.ndarray], np.ndarray, list[dict], list[dict], dict]:
    trajectory_obs = data["trajectory_obs"]
    observed = data["observed"]
    outcome = data["outcome"]
    n, length, markers = trajectory_obs.shape
    completed_draws = [np.zeros_like(trajectory_obs, dtype=np.float32) for _ in range(args.m)]
    crossfit_probabilities = np.full(n, np.nan, dtype=float)
    crossfit_mean = np.zeros_like(trajectory_obs, dtype=np.float32)
    crossfit_scale = np.zeros_like(trajectory_obs, dtype=np.float32)
    history_rows: list[dict] = []
    audit_rows: list[dict] = []
    splitter = StratifiedKFold(
        n_splits=args.crossfit_folds,
        shuffle=True,
        random_state=seed + 3181,
    )
    all_indices = np.arange(n)
    start_time = time_module.time()

    for fold, (outer_train, held_out) in enumerate(
        splitter.split(all_indices, outcome), start=1
    ):
        inner_train, inner_val = train_test_split(
            outer_train,
            test_size=args.validation_fraction,
            random_state=seed + 7001 + fold,
            stratify=outcome[outer_train],
        )
        marker_mean = np.nanmean(trajectory_obs[outer_train], axis=(0, 1))
        marker_sd = np.nanstd(trajectory_obs[outer_train], axis=(0, 1))
        marker_mean = np.nan_to_num(marker_mean, nan=0.0).astype(np.float32)
        marker_sd = np.where(
            np.isfinite(marker_sd) & (marker_sd > 1e-5), marker_sd, 1.0
        ).astype(np.float32)
        standardized = (
            (trajectory_obs - marker_mean.reshape(1, 1, -1))
            / marker_sd.reshape(1, 1, -1)
        )
        standardized = np.nan_to_num(standardized, nan=0.0).astype(np.float32)

        fold_seed = seed + 100_003 * fold + (0 if lambda_down > 0 else 50_000_003)
        model, calibration, history, audit = fit_fold_model(
            standardized,
            observed.astype(np.float32),
            data["times"],
            data["static"],
            outcome,
            inner_train,
            inner_val,
            fold_seed,
            args,
            lambda_down,
            device,
        )
        for row in history:
            history_rows.append(
                {"seed": seed, "method": method, "fold": fold, **row}
            )
        audit_rows.append(
            {
                "seed": seed,
                "method": method,
                "fold": fold,
                "n_outer_train": len(outer_train),
                "n_inner_train": len(inner_train),
                "n_inner_validation": len(inner_val),
                "n_held_out": len(held_out),
                **audit,
            }
        )

        with torch.no_grad():
            held_index = torch.as_tensor(held_out, dtype=torch.long)
            y_test = torch.from_numpy(standardized[held_out]).to(device)
            mask_test = torch.from_numpy(observed[held_out].astype(np.float32)).to(device)
            times_test = torch.from_numpy(data["times"][held_out]).to(device)
            static_test = torch.from_numpy(data["static"][held_out]).to(device)
            output = model(y_test, mask_test, times_test, static_test, None)
            mean_std = output["mean"].cpu().numpy()
            scale_std = output["scale"].cpu().numpy() * calibration.reshape(1, 1, -1)
            crossfit_probabilities[held_out] = torch.sigmoid(
                output["mean_logits"]
            ).cpu().numpy()
        mean_raw = mean_std * marker_sd.reshape(1, 1, -1) + marker_mean.reshape(1, 1, -1)
        scale_raw = scale_std * marker_sd.reshape(1, 1, -1)
        observed_test = observed[held_out]
        mean_raw[observed_test] = trajectory_obs[held_out][observed_test]
        crossfit_mean[held_out] = mean_raw
        crossfit_scale[held_out] = scale_raw

        for draw_index in range(args.m):
            rng = np.random.default_rng(
                seed + 1_000_003 * (draw_index + 1) + 10_007 * fold + int(lambda_down * 1000)
            )
            sampled = mean_raw + scale_raw * rng.normal(size=mean_raw.shape)
            sampled[observed_test] = trajectory_obs[held_out][observed_test]
            completed_draws[draw_index][held_out] = sampled.astype(np.float32)

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    missing_late = ~observed[:, -3:, :]
    truth_late = data["trajectory"][:, -3:, :]
    lower = crossfit_mean[:, -3:, :] - 1.959963984540054 * crossfit_scale[:, -3:, :]
    upper = crossfit_mean[:, -3:, :] + 1.959963984540054 * crossfit_scale[:, -3:, :]
    interval_coverage = float(
        np.mean((truth_late[missing_late] >= lower[missing_late]) & (truth_late[missing_late] <= upper[missing_late]))
    )
    interval_width = float(np.mean((upper - lower)[missing_late]))
    predictive = {
        "task_auroc": float(roc_auc_score(outcome, crossfit_probabilities)),
        "task_auprc": float(average_precision_score(outcome, crossfit_probabilities)),
        "task_log_loss": float(log_loss(outcome, np.clip(crossfit_probabilities, 1e-7, 1 - 1e-7))),
        "cell_interval_coverage": interval_coverage,
        "cell_interval_width": interval_width,
        "runtime_seconds": float(time_module.time() - start_time),
    }
    return completed_draws, crossfit_probabilities, history_rows, audit_rows, predictive


def evaluate_completions(
    data: dict[str, np.ndarray],
    completed: list[np.ndarray],
    seed: int,
    scenario: str,
    method: str,
    predictive: dict,
) -> tuple[list[dict], dict]:
    feature_draws = [
        dgp1._trajectory_features(draw, data["time"]) for draw in completed
    ]
    pooled = dgp1._pool_logit_fits(
        feature_draws,
        data["outcome"],
        data["diabetes"],
        data["hypertension"],
        data["male"],
    )
    if pooled is None:
        raise RuntimeError(f"Downstream fit failed for seed={seed}, method={method}")
    params, bse, within_se, between_sd, ci, n_used, n_fit_draws = pooled
    coefficient_rows = []
    for name, truth in TRUE_BETA.items():
        lo, hi = ci.loc[name]
        coefficient_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "scenario_label": SCENARIOS[scenario]["label"],
                "target_missing": float(data.get("target_missing", 0.20)),
                "n_imputations": len(completed),
                "method": method,
                "method_label": METHOD_LABELS[method],
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
                "selected_variance_factor": np.nan,
            }
        )
    missing_late = ~data["observed"][:, -3:, :]
    completed_mean = np.mean(
        np.stack([draw[:, -3:, :] for draw in completed], axis=0), axis=0
    )
    imputation_rmse = float(
        np.sqrt(
            np.mean(
                (
                    completed_mean[missing_late]
                    - data["trajectory"][:, -3:, :][missing_late]
                )
                ** 2
            )
        )
    )
    summary_mean = np.mean(
        np.stack([frame.to_numpy() for frame in feature_draws], axis=0), axis=0
    )
    trajectory_summary_rmse = float(
        np.sqrt(
            np.mean(
                (summary_mean - data["true_features"].to_numpy()) ** 2
            )
        )
    )
    metric = {
        "seed": seed,
        "scenario": scenario,
        "scenario_label": SCENARIOS[scenario]["label"],
        "target_missing": float(data.get("target_missing", 0.20)),
        "n_imputations": len(completed),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "imputation_rmse": imputation_rmse,
        "trajectory_summary_rmse": trajectory_summary_rmse,
        "coefficient_fit_success": 1.0,
        "coefficient_n_used": int(n_used),
        "coefficient_n_fit_draws": int(n_fit_draws),
        "selected_variance_factor": np.nan,
        **predictive,
    }
    return coefficient_rows, metric


def summarize_and_plot(
    outdir: Path,
    scenario: str,
    seed_start: int,
    nsim: int,
    comparator_dir: Path,
    package_comparator_dir: Path | None,
) -> None:
    neural_coef = pd.read_csv(outdir / "coefficient_by_run.csv")
    neural_metrics = pd.read_csv(outdir / "metrics_by_run.csv")
    baseline_coef = pd.read_csv(comparator_dir / "coefficient_by_run.csv")
    baseline_metrics = pd.read_csv(comparator_dir / "metrics_by_run.csv")
    selected_seeds = set(range(seed_start, seed_start + nsim))
    baseline_coef = baseline_coef[
        baseline_coef["scenario"].eq(scenario)
        & baseline_coef["seed"].isin(selected_seeds)
    ].copy()
    baseline_metrics = baseline_metrics[
        baseline_metrics["scenario"].eq(scenario)
        & baseline_metrics["seed"].isin(selected_seeds)
    ].copy()
    # The archived MICE and Missforest rows are custom approximations. Final
    # comparisons use package-native defaults loaded below.
    excluded_archived_methods = [
        "mice",
        "missforest",
        "adaptive_posterior_vc_smc_brits",
    ]
    baseline_coef = baseline_coef[
        ~baseline_coef["method"].isin(excluded_archived_methods)
    ]
    baseline_metrics = baseline_metrics[
        ~baseline_metrics["method"].isin(excluded_archived_methods)
    ]
    if package_comparator_dir is not None and (
        package_comparator_dir / "coefficient_by_run.csv"
    ).exists():
        package_coef = pd.read_csv(package_comparator_dir / "coefficient_by_run.csv")
        package_metrics = pd.read_csv(package_comparator_dir / "metrics_by_run.csv")
        package_coef = package_coef[
            package_coef["scenario"].eq(scenario)
            & package_coef["seed"].isin(selected_seeds)
        ].copy()
        package_metrics = package_metrics[
            package_metrics["scenario"].eq(scenario)
            & package_metrics["seed"].isin(selected_seeds)
        ].copy()
        package_coef["method_label"] = package_coef["method"].map(
            {"mice": "MICE (package default)", "missforest": "Missforest (package default)"}
        ).fillna(package_coef["method_label"])
        package_metrics["method_label"] = package_metrics["method"].map(
            {"mice": "MICE (package default)", "missforest": "Missforest (package default)"}
        ).fillna(package_metrics["method_label"])
        baseline_coef = pd.concat([baseline_coef, package_coef], ignore_index=True)
        baseline_metrics = pd.concat([baseline_metrics, package_metrics], ignore_index=True)
    baseline_coef.loc[baseline_coef["method"].eq("full_data"), "method_label"] = "Full data"
    baseline_metrics.loc[baseline_metrics["method"].eq("full_data"), "method_label"] = "Full data"

    combined_coef = pd.concat([baseline_coef, neural_coef], ignore_index=True)
    combined_metrics = pd.concat([baseline_metrics, neural_metrics], ignore_index=True)
    combined_coef.to_csv(outdir / "comparison_coefficient_by_run.csv", index=False)
    combined_metrics.to_csv(outdir / "comparison_metrics_by_run.csv", index=False)
    by_term, overall = run_large.summarize_effects(combined_coef, combined_metrics)
    by_term.to_csv(outdir / "comparison_by_predictor.csv", index=False)
    overall.to_csv(outdir / "comparison_overall.csv", index=False)

    per_seed = combined_coef.groupby(
        ["seed", "method", "method_label"], as_index=False
    ).agg(
        mean_abs_run_bias=("abs_bias", "mean"),
        coverage=("covered", "mean"),
        coefficient_mse=("squared_error", "mean"),
        average_se=("std_error", "mean"),
    )
    per_seed = per_seed.merge(
        combined_metrics[
            ["seed", "method", "imputation_rmse", "trajectory_summary_rmse"]
        ],
        on=["seed", "method"],
        how="left",
    )
    paired_rows = []
    proposed = "brits_mi_gradient"
    for comparator in sorted(set(per_seed["method"]) - {proposed}):
        subset = per_seed[per_seed["method"].isin([proposed, comparator])]
        for metric in [
            "mean_abs_run_bias",
            "coverage",
            "coefficient_mse",
            "imputation_rmse",
            "trajectory_summary_rmse",
        ]:
            wide = subset.pivot(index="seed", columns="method", values=metric).dropna()
            if proposed not in wide or comparator not in wide or len(wide) < 2:
                continue
            difference = wide[proposed] - wide[comparator]
            test = stats.ttest_rel(wide[proposed], wide[comparator])
            paired_rows.append(
                {
                    "comparator_method": comparator,
                    "comparator_label": subset.loc[
                        subset["method"].eq(comparator), "method_label"
                    ].iloc[0],
                    "metric": metric,
                    "n_pairs": len(wide),
                    "brits_mi_mean": float(wide[proposed].mean()),
                    "comparator_mean": float(wide[comparator].mean()),
                    "mean_difference": float(difference.mean()),
                    "paired_t_p_value": float(test.pvalue),
                }
            )
    pd.DataFrame(paired_rows).to_csv(outdir / "paired_method_tests.csv", index=False)

    display_order = [
        "full_data",
        "brits_mi_gradient",
        "brits_mi_no_downstream",
        "adaptive_posterior_vc_smc_brits",
        "mice",
        "missforest",
        "mean_impute",
        "complete_case",
    ]
    colors = {
        "full_data": "#111827",
        "brits_mi_gradient": "#007C83",
        "brits_mi_no_downstream": "#70B7BA",
        "adaptive_posterior_vc_smc_brits": "#D97706",
        "mice": "#7651B5",
        "missforest": "#2A9D8F",
        "mean_impute": "#7A8793",
        "complete_case": "#B33A3A",
    }
    plot_data = overall.copy()
    plot_data["order"] = plot_data["method"].map(
        {method: position for position, method in enumerate(display_order)}
    )
    plot_data = plot_data.sort_values("order")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=200)
    panels = [
        ("mean_abs_signed_bias", "Mean absolute signed bias", "lower"),
        ("coverage", "Overall 95% coefficient coverage", "target"),
        ("imputation_rmse_mean", "Missing-cell RMSE", "lower"),
        ("trajectory_summary_rmse_mean", "Trajectory-summary RMSE", "lower"),
    ]
    for axis, (column, title, direction) in zip(axes.flat, panels):
        subset = plot_data[np.isfinite(plot_data[column])]
        x = np.arange(len(subset))
        values = subset[column].to_numpy(float)
        axis.scatter(
            x,
            values,
            s=90,
            color=[colors.get(method, "#555555") for method in subset["method"]],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        if column == "coverage":
            axis.axhline(0.95, color="#333333", linestyle="--", linewidth=1.2)
        axis.set_xticks(x)
        axis.set_xticklabels(subset["method_label"], rotation=28, ha="right")
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", color="#D7DEE2", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.99,
            0.96,
            "lower is better" if direction == "lower" else "target = 0.95",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="#52656D",
            fontsize=9,
        )
    fig.suptitle(
        "Matched clinical-association simulation: downstream-gradient BRITS-MI",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "matched_20_seed_comparison.png", bbox_inches="tight")
    fig.savefig(outdir / "matched_20_seed_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--comparator-dir", type=Path, default=ROOT / "outputs/effect_recovery_harder_irregularity_n500_nsim100_m20_miss20_fresh")
    parser.add_argument("--package-comparator-dir", type=Path, default=None)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="baseline_harder")
    parser.add_argument("--seed-start", type=int, default=1093000)
    parser.add_argument("--nsim", type=int, default=20)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--target-missing", type=float, default=0.20)
    parser.add_argument("--crossfit-folds", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--training-mask-rate", type=float, default=0.20)
    parser.add_argument("--validation-mask-rate", type=float, default=0.20)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-epochs", type=int, default=55)
    parser.add_argument("--min-epochs", type=int, default=18)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--downstream-warmup", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--lambda-directional", type=float, default=0.20)
    parser.add_argument("--lambda-reconstruction-mse", type=float, default=1.00)
    parser.add_argument("--lambda-cons", type=float, default=0.08)
    parser.add_argument("--lambda-sum", type=float, default=0.20)
    parser.add_argument("--lambda-down", type=float, default=0.45)
    parser.add_argument(
        "--downstream-objective",
        choices=["classification", "association_score"],
        default="classification",
    )
    parser.add_argument("--lambda-head", type=float, default=0.50)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_LABELS),
        default=list(METHOD_LABELS),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    coefficient_path = args.outdir / "coefficient_by_run.csv"
    metric_path = args.outdir / "metrics_by_run.csv"
    history_path = args.outdir / "training_history.csv"
    audit_path = args.outdir / "fold_audit.csv"
    coefficient_rows = pd.read_csv(coefficient_path).to_dict("records") if args.resume and coefficient_path.exists() else []
    metric_rows = pd.read_csv(metric_path).to_dict("records") if args.resume and metric_path.exists() else []
    history_rows = pd.read_csv(history_path).to_dict("records") if args.resume and history_path.exists() else []
    audit_rows = pd.read_csv(audit_path).to_dict("records") if args.resume and audit_path.exists() else []
    completed_keys = {(int(row["seed"]), row["method"]) for row in metric_rows}

    for offset in range(args.nsim):
        seed = args.seed_start + offset
        data = generate_locked_dataset(seed, args.scenario, args.n, args.target_missing)
        for method in args.methods:
            if (seed, method) in completed_keys:
                print(f"skip completed seed={seed} method={method}", flush=True)
                continue
            lambda_down = args.lambda_down if method == "brits_mi_gradient" else 0.0
            start = time_module.time()
            draws, _, seed_history, seed_audits, predictive = cross_fitted_completions(
                data, seed, args, lambda_down, method, device
            )
            rows, metric = evaluate_completions(
                data, draws, seed, args.scenario, method, predictive
            )
            coefficient_rows.extend(rows)
            metric_rows.append(metric)
            history_rows.extend(seed_history)
            audit_rows.extend(seed_audits)
            pd.DataFrame(coefficient_rows).to_csv(coefficient_path, index=False)
            pd.DataFrame(metric_rows).to_csv(metric_path, index=False)
            pd.DataFrame(history_rows).to_csv(history_path, index=False)
            pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
            print(
                f"seed={seed} method={METHOD_LABELS[method]} "
                f"RMSE={metric['imputation_rmse']:.3f} "
                f"trajectory={metric['trajectory_summary_rmse']:.3f} "
                f"task_AUROC={metric['task_auroc']:.3f} "
                f"runtime={time_module.time() - start:.1f}s",
                flush=True,
            )

    config = vars(args).copy()
    for key in ["outdir", "comparator_dir", "package_comparator_dir"]:
        config[key] = str(config[key])
    config.update(
        {
            "architecture": "Bidirectional RITS-GRU with learned conditional mean/scale and a regularized sequence-wide linear residual path",
            "training": (
                "subject-level cross-fitting; fixed validation masks; early stopping; "
                f"downstream objective={args.downstream_objective}"
            ),
            "natural_missing_truth_used_for_training": False,
            "held_out_subject_outcome_used_for_imputation_training": False,
            "observed_cells_fixed_in_final_draws": True,
        }
    )
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summarize_and_plot(
        args.outdir,
        args.scenario,
        args.seed_start,
        args.nsim,
        args.comparator_dir,
        args.package_comparator_dir,
    )
    overall = pd.read_csv(args.outdir / "comparison_overall.csv")
    columns = [
        "method_label",
        "mean_abs_signed_bias",
        "coverage",
        "se_ratio",
        "imputation_rmse_mean",
        "trajectory_summary_rmse_mean",
        "n_runs",
    ]
    print(overall[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
