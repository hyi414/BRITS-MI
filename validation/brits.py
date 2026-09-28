"""Bootstrap BRITS-GRU engine and association-aware masking objective.

The source-cohort outcome model supplies a fixed training target. Evaluation
uses repeated masks in the same cohort, not independent-patient validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from . import evaluation


class RealBRITSHistory(nn.Module):
    """BRITS-style GRU with causal history summaries and marker-specific gaps."""

    def __init__(
        self, n_markers: int, n_static: int, n_classes: int, hidden_size: int = 32
    ) -> None:
        super().__init__()
        self.n_markers = n_markers
        self.n_static = n_static
        self.hidden_size = hidden_size

        history_dim = 7 * n_markers
        in_dim = 2 * n_markers + history_dim + 1 + n_static
        self.forward_rnn = nn.GRU(in_dim, hidden_size, batch_first=True)
        self.backward_rnn = nn.GRU(in_dim, hidden_size, batch_first=True)
        self.forward_head = nn.Linear(hidden_size, n_markers)
        self.backward_head = nn.Linear(hidden_size, n_markers)
        self.combine_head = nn.Sequential(
            nn.Linear(2 * hidden_size + history_dim + n_static, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_markers),
        )
        self.attn = nn.Linear(2 * hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size + n_static + 5 * n_markers, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, n_classes),
        )

    @staticmethod
    def history_state(y_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, n_markers = y_obs.shape
        device = y_obs.device
        dtype = y_obs.dtype
        count = torch.zeros((batch, n_markers), device=device, dtype=dtype)
        total = torch.zeros_like(count)
        total_sq = torch.zeros_like(count)
        first = torch.zeros_like(count)
        last = torch.zeros_like(count)
        first_time = torch.zeros_like(count)
        last_time = torch.full_like(count, -1.0)
        histories = []

        for t in range(length):
            mean = total / count.clamp_min(1.0)
            var = total_sq / count.clamp_min(1.0) - mean**2
            sd = torch.sqrt(torch.clamp(var, min=0.0))
            slope = (last - first) / (last_time - first_time).clamp_min(1.0)
            miss_frac = 1.0 - count / max(t, 1)
            gap = torch.where(count > 0, t - last_time, torch.full_like(count, float(length + 1)))
            hist = torch.cat(
                [
                    mean,
                    sd,
                    last,
                    slope,
                    count / max(t, 1),
                    miss_frac,
                    gap / float(length + 1),
                ],
                dim=1,
            )
            histories.append(hist)

            obs = mask[:, t, :] > 0.5
            value = y_obs[:, t, :]
            is_first = obs & (count <= 0)
            first = torch.where(is_first, value, first)
            first_time = torch.where(is_first, torch.full_like(first_time, float(t)), first_time)
            last = torch.where(obs, value, last)
            last_time = torch.where(obs, torch.full_like(last_time, float(t)), last_time)
            total = total + torch.where(obs, value, torch.zeros_like(value))
            total_sq = total_sq + torch.where(obs, value**2, torch.zeros_like(value))
            count = count + obs.to(dtype)

        time = (
            torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
            .view(1, length, 1)
            .expand(batch, -1, -1)
        )
        hist_stack = torch.stack(histories, dim=1)
        return torch.cat([hist_stack, time], dim=2)

    def forward(
        self, y_obs: torch.Tensor, mask: torch.Tensor, static: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch, length, _ = y_obs.shape
        static_rep = static.unsqueeze(1).expand(-1, length, -1)
        hist_time = self.history_state(y_obs, mask)
        inp = torch.cat([y_obs, mask, hist_time, static_rep], dim=-1)
        f_h, _ = self.forward_rnn(inp)
        b_inp = torch.flip(inp, dims=[1])
        b_h_rev, _ = self.backward_rnn(b_inp)
        b_h = torch.flip(b_h_rev, dims=[1])
        y_f = self.forward_head(f_h)
        y_b = self.backward_head(b_h)
        hist_only = hist_time[:, :, :-1]
        y_hat = self.combine_head(torch.cat([f_h, b_h, hist_only, static_rep], dim=-1))
        completed = torch.where(mask > 0.5, y_obs, y_hat)

        attn_logits = self.attn(torch.cat([f_h, b_h], dim=-1)).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * torch.cat([f_h, b_h], dim=-1), dim=1)

        seq_mean = completed.mean(dim=1)
        seq_last = completed[:, -1, :]
        seq_slope = completed[:, -1, :] - completed[:, 0, :]
        seq_missing = 1.0 - mask.mean(dim=1)
        ever_observed = (mask.sum(dim=1) > 0).to(completed.dtype)
        summary = torch.cat([seq_mean, seq_last, seq_slope, seq_missing, ever_observed], dim=-1)
        logits = self.classifier(torch.cat([pooled, static, summary], dim=-1))
        return {
            "y_f": y_f,
            "y_b": y_b,
            "y_hat": y_hat,
            "completed": completed,
            "logits": logits,
        }


BASE_TERMS = [
    "fib4_log",
    "age",
    "bmi",
    "diabetes",
    "hypertension",
    "hyperlipidemia",
    "male",
    "study_mre",
    "fib4_log_x_diabetes",
    "hypertension_x_male",
]


TRAJECTORY_TERMS = [
    "late_ast",
    "late_alt",
    "late_platelet",
    "slope_ast",
    "slope_alt",
    "slope_platelet",
    "missing_fraction_ast",
    "missing_fraction_alt",
    "missing_fraction_platelet",
]


def reference_target(
    data_frame: pd.DataFrame,
    truth_raw: np.ndarray,
    source_mask: np.ndarray,
) -> dict[str, object]:
    outcome, design = evaluation.effect_design_observed(data_frame, truth_raw, source_mask)
    columns = BASE_TERMS + TRAJECTORY_TERMS
    design = design[columns].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(design)
    fit = sm.GLM(
        outcome.to_numpy(float),
        sm.add_constant(scaled, has_constant="add"),
        family=sm.families.Binomial(),
    ).fit(maxiter=200, disp=0)
    beta = np.asarray(fit.params[1:], dtype=np.float32)
    intercept = float(fit.params[0])
    score = intercept + scaled @ beta
    return {
        "outcome": outcome.to_numpy(np.float32),
        "design": design.to_numpy(np.float32),
        "design_columns": columns,
        "design_mean": scaler.mean_.astype(np.float32),
        "design_scale": scaler.scale_.astype(np.float32),
        "beta": beta,
        "intercept": intercept,
        "score": score.astype(np.float32),
    }


def differentiable_design(
    hybrid_standardized: torch.Tensor,
    source_mask: torch.Tensor,
    base_raw: torch.Tensor,
    marker_mean: torch.Tensor,
    marker_sd: torch.Tensor,
) -> torch.Tensor:
    raw = hybrid_standardized * marker_sd.view(1, 1, -1) + marker_mean.view(1, 1, -1)
    length = raw.shape[1]
    late_values: list[torch.Tensor] = []
    slope_values: list[torch.Tensor] = []
    missing_values: list[torch.Tensor] = []
    time_index = torch.arange(length, device=raw.device).view(1, length)
    for marker_index in (3, 4, 5):
        values = raw[:, :, marker_index]
        observed = source_mask[:, :, marker_index]
        overall_count = observed.sum(dim=1)
        overall = (values * observed).sum(dim=1) / overall_count.clamp_min(1.0)
        late_mask = observed[:, -2:]
        late_count = late_mask.sum(dim=1)
        late = (values[:, -2:] * late_mask).sum(dim=1) / late_count.clamp_min(1.0)
        late = torch.where(late_count > 0, late, overall)

        first_index = (
            torch.where(
                observed > 0.5,
                time_index,
                torch.full_like(time_index, length),
            )
            .min(dim=1)
            .values
        )
        last_index = (
            torch.where(
                observed > 0.5,
                time_index,
                torch.full_like(time_index, -1),
            )
            .max(dim=1)
            .values
        )
        valid = overall_count >= 2
        safe_first = first_index.clamp(0, length - 1)
        safe_last = last_index.clamp(0, length - 1)
        first = values.gather(1, safe_first[:, None]).squeeze(1)
        last = values.gather(1, safe_last[:, None]).squeeze(1)
        slope = torch.where(valid, last - first, torch.zeros_like(last))

        late_values.append(late)
        slope_values.append(slope)
        missing_values.append(1.0 - observed.mean(dim=1))
    trajectory = torch.stack(
        late_values + slope_values + missing_values,
        dim=1,
    )
    return torch.cat([base_raw, trajectory], dim=1)


def draw_training_hide(
    source_mask: torch.Tensor,
    target_channels: list[int],
    rate: float,
) -> torch.Tensor:
    selector = torch.zeros(source_mask.shape[-1], device=source_mask.device, dtype=torch.bool)
    selector[target_channels] = True
    return (
        (torch.rand_like(source_mask) < rate) & (source_mask > 0.5) & selector.view(1, 1, -1)
    ).to(source_mask.dtype)


def train_model(
    observed_standardized: np.ndarray,
    source_mask: np.ndarray,
    static: np.ndarray,
    target: dict[str, object],
    stage_labels: np.ndarray,
    marker_mean: np.ndarray,
    marker_sd: np.ndarray,
    target_channels: list[int],
    seed: int,
    epochs: int,
    hidden_size: int,
    batch_size: int,
    lambda_summary: float,
    lambda_downstream: float,
    lambda_outcome: float,
    lambda_covariance: float,
    lambda_stage: float,
    late_weight: float,
    use_best_training_checkpoint: bool,
    training_population: np.ndarray | None = None,
) -> tuple[RealBRITSHistory, pd.DataFrame]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = RealBRITSHistory(
        observed_standardized.shape[2],
        static.shape[1],
        len(np.unique(stage_labels)) if lambda_stage > 0 else 2,
        hidden_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    y_all = torch.as_tensor(observed_standardized, dtype=torch.float32)
    m_all = torch.as_tensor(source_mask, dtype=torch.float32)
    static_all = torch.as_tensor(static, dtype=torch.float32)
    base_all = torch.tensor(
        np.asarray(target["design"], dtype=np.float32)[:, : len(BASE_TERMS)],
        dtype=torch.float32,
    )
    design_target = torch.tensor(target["design"], dtype=torch.float32)
    design_mean = torch.tensor(target["design_mean"], dtype=torch.float32)
    design_scale = torch.tensor(target["design_scale"], dtype=torch.float32)
    beta = torch.tensor(target["beta"], dtype=torch.float32)
    intercept = torch.tensor(float(target["intercept"]), dtype=torch.float32)
    score_target = torch.tensor(target["score"], dtype=torch.float32)
    outcome = torch.tensor(target["outcome"], dtype=torch.float32)
    stage = torch.tensor(stage_labels, dtype=torch.long)
    marker_mean_tensor = torch.as_tensor(marker_mean, dtype=torch.float32)
    marker_sd_tensor = torch.as_tensor(marker_sd, dtype=torch.float32)
    rows: list[dict] = []
    rates = (0.20, 0.40, 0.60)
    n = len(y_all)
    if training_population is None:
        training_population = np.arange(n)
    training_population = np.asarray(training_population, dtype=int)
    best_loss = np.inf
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for epoch in range(epochs):
        model.train()
        order = training_population[rng.permutation(len(training_population))]
        epoch_rows: list[dict] = []
        for batch_number, start in enumerate(range(0, n, batch_size)):
            index = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            y = y_all[index]
            m = m_all[index]
            s = static_all[index]
            rate = rates[(epoch + batch_number) % len(rates)]
            hide = draw_training_hide(m, target_channels, rate)
            y_input = torch.where(hide > 0.5, torch.zeros_like(y), y)
            m_input = torch.where(hide > 0.5, torch.zeros_like(m), m)
            output = model(y_input, m_input, s)
            rec = ((output["y_hat"] - y) ** 2 * hide).sum() / hide.sum().clamp_min(1.0)
            consistency = (
                (output["y_f"] - output["y_b"]) ** 2 * m_input
            ).sum() / m_input.sum().clamp_min(1.0)
            observed_loss = ((output["y_hat"] - y) ** 2 * m_input).sum() / m_input.sum().clamp_min(
                1.0
            )
            hybrid = torch.where(hide > 0.5, output["y_hat"], y)
            design = differentiable_design(
                hybrid,
                m,
                base_all[index],
                marker_mean_tensor,
                marker_sd_tensor,
            )
            design_scaled = (design - design_mean) / design_scale
            target_scaled = (design_target[index] - design_mean) / design_scale
            summary_difference = (
                design_scaled[:, len(BASE_TERMS) : len(BASE_TERMS) + 6]
                - target_scaled[:, len(BASE_TERMS) : len(BASE_TERMS) + 6]
            )
            summary_weights = torch.tensor(
                [late_weight, late_weight, late_weight, 1.0, 1.0, 1.0],
                dtype=summary_difference.dtype,
                device=summary_difference.device,
            )
            summary_loss = (summary_difference.square() * summary_weights.view(1, -1)).sum() / (
                summary_difference.shape[0] * summary_weights.sum()
            )
            score = intercept + design_scaled @ beta
            downstream_loss = mse(score, score_target[index])
            outcome_loss = bce(score, outcome[index])
            stage_loss = (
                ce(output["logits"], stage[index])
                if lambda_stage > 0
                else torch.zeros((), dtype=score.dtype, device=score.device)
            )
            trajectory_imputed = design_scaled[:, len(BASE_TERMS) : len(BASE_TERMS) + 6]
            trajectory_target = target_scaled[:, len(BASE_TERMS) : len(BASE_TERMS) + 6]
            context = torch.cat([design_scaled[:, : len(BASE_TERMS)], outcome[index, None]], dim=1)
            centered_imputed = trajectory_imputed - trajectory_imputed.mean(dim=0, keepdim=True)
            centered_target = trajectory_target - trajectory_target.mean(dim=0, keepdim=True)
            centered_context = context - context.mean(dim=0, keepdim=True)
            denominator = max(trajectory_imputed.shape[0] - 1, 1)
            cross_covariance_imputed = centered_imputed.T @ centered_context / denominator
            cross_covariance_target = centered_target.T @ centered_context / denominator
            trajectory_covariance_imputed = centered_imputed.T @ centered_imputed / denominator
            trajectory_covariance_target = centered_target.T @ centered_target / denominator
            covariance_loss = mse(cross_covariance_imputed, cross_covariance_target) + mse(
                trajectory_covariance_imputed, trajectory_covariance_target
            )
            loss = (
                rec
                + 0.10 * consistency
                + 0.02 * observed_loss
                + lambda_summary * summary_loss
                + lambda_downstream * downstream_loss
                + lambda_outcome * outcome_loss
                + lambda_covariance * covariance_loss
                + lambda_stage * stage_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_rows.append(
                {
                    "loss": float(loss.item()),
                    "reconstruction": float(rec.item()),
                    "consistency": float(consistency.item()),
                    "summary": float(summary_loss.item()),
                    "downstream": float(downstream_loss.item()),
                    "outcome": float(outcome_loss.item()),
                    "stage": float(stage_loss.item()),
                    "covariance": float(covariance_loss.item()),
                }
            )
        row = pd.DataFrame(epoch_rows).mean().to_dict()
        row.update({"epoch": epoch + 1, "seed": seed})
        rows.append(row)
        if row["loss"] < best_loss:
            best_loss = float(row["loss"])
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(
            f"association-target BRITS seed={seed} epoch={epoch + 1}/{epochs} "
            f"loss={row['loss']:.4f} rec={row['reconstruction']:.4f} "
            f"down={row['downstream']:.4f}",
            flush=True,
        )
    if use_best_training_checkpoint:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict_completed(
    model: RealBRITSHistory,
    values: np.ndarray,
    mask: np.ndarray,
    static: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        output = model(
            torch.as_tensor(values, dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.float32),
            torch.as_tensor(static, dtype=torch.float32),
        )
    return output["completed"].cpu().numpy().astype(np.float32)


def residual_bank(
    model: RealBRITSHistory,
    observed_standardized: np.ndarray,
    source_mask: np.ndarray,
    static: np.ndarray,
    target_channels: list[int],
    seed: int,
) -> dict[int, np.ndarray]:
    banks: dict[int, list[np.ndarray]] = {channel: [] for channel in target_channels}
    for offset, rate in enumerate((0.20, 0.40, 0.60)):
        torch.manual_seed(seed + offset)
        hide = (
            draw_training_hide(
                torch.as_tensor(source_mask, dtype=torch.float32),
                target_channels,
                rate,
            ).numpy()
            > 0.5
        )
        input_values = observed_standardized.copy()
        input_mask = source_mask.copy()
        input_values[hide] = 0.0
        input_mask[hide] = 0.0
        completed = predict_completed(model, input_values, input_mask, static)
        residual = observed_standardized - completed
        for channel in target_channels:
            selected = residual[:, :, channel][hide[:, :, channel]]
            selected = selected[np.isfinite(selected)]
            if len(selected):
                banks[channel].append(selected)
    centered: dict[int, np.ndarray] = {}
    for channel, parts in banks.items():
        values = np.concatenate(parts)
        centered[channel] = values - np.mean(values)
    return centered


def stochastic_draws(
    models: list[RealBRITSHistory],
    residuals: list[dict[int, np.ndarray]],
    input_values: np.ndarray,
    input_mask: np.ndarray,
    static: np.ndarray,
    target_channels: list[int],
    n_imputations: int,
    noise_multiplier: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    conditional_means = [
        predict_completed(model, input_values, input_mask, static) for model in models
    ]
    missing = input_mask < 0.5
    draws: list[np.ndarray] = []
    for draw_index in range(n_imputations):
        model_index = draw_index % len(models)
        draw = conditional_means[model_index].copy()
        if noise_multiplier != 0:
            for channel in target_channels:
                target = missing[:, :, channel]
                bank = residuals[model_index][channel]
                sampled = rng.choice(bank, size=int(target.sum()), replace=True)
                draw[:, :, channel][target] += noise_multiplier * sampled
        draw[input_mask > 0.5] = input_values[input_mask > 0.5]
        draws.append(draw.astype(np.float32))
    return draws
