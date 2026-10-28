"""Shared recurrent BRITS engine for association MI and image-conditioned prediction.

Outcome conditioning is explicit: association imputations condition on the observed
outcome; a prediction fit never accepts query outcomes. No forest anchor is used.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .imputer import BRITSMultipleImputer
from .model import BRITSMI, RITSCell, torch_marker_gaps


@dataclass(frozen=True)
class UnifiedConfig:
    epochs: int = 120
    patience: int = 25
    hidden_size: int = 48
    batch_size: int = 128
    learning_rate: float = 0.002
    holdout_fraction: float = 0.25
    validation_fraction: float = 0.20
    consistency_weight: float = 0.10
    summary_weight: float = 0.20
    downstream_weight: float = 0.20
    residual_multiplier: float = 1.0
    residual_mode: str = "iid"
    bootstrap_members: int = 1
    seed: int = 1

    def __post_init__(self):
        if min(self.epochs, self.patience, self.hidden_size, self.batch_size,
               self.bootstrap_members) < 1:
            raise ValueError("training counts must be positive")
        if self.residual_mode not in {"iid", "correlated"}:
            raise ValueError("residual_mode must be iid or correlated")
        if not 0 < self.holdout_fraction < 1 or not 0 < self.validation_fraction < 0.5:
            raise ValueError("invalid artificial-mask or validation percentage")
        if self.residual_multiplier < 0 or self.learning_rate <= 0:
            raise ValueError("invalid residual multiplier or learning rate")
        if min(self.consistency_weight, self.summary_weight, self.downstream_weight) < 0:
            raise ValueError("loss weights must be nonnegative")

    def as_dict(self):
        return asdict(self)


def clinical_features(raw, times, static, lengths=None):
    return BRITSMI.clinical_design(raw, times, static)


def fusion_features(raw, times, static, lengths=None):
    """Differentiable trajectory summaries; image context is concatenated separately."""
    if lengths is not None and not torch.all(lengths == raw.shape[1]):
        return torch.cat([fusion_features(raw[i:i+1, :int(lengths[i])],
                          times[i:i+1, :int(lengths[i])], static[i:i+1])
                          for i in range(len(raw))], 0)
    differences = raw[:, 1:] - raw[:, :-1]
    curvature = differences[:, 1:] - differences[:, :-1]
    return torch.cat([
        raw.mean(1), raw.std(1, unbiased=False), raw[:, -3:].mean(1),
        raw[:, -3:].mean(1) - raw[:, :3].mean(1), differences.abs().mean(1),
        curvature.abs().mean(1), raw[:, -1] - raw[:, 0],
        (raw[:, -1] - raw[:, -3]) / (times[:, -1:] - times[:, -3:-2]), static,
    ], dim=1)


class RecurrentNetwork(nn.Module):
    def __init__(self, markers, static_width, context_width, classes, hidden,
                 feature_count, condition_outcome, nonlinear_head):
        super().__init__()
        self.condition_outcome = condition_outcome
        self.classes = classes
        recurrent_context = context_width + (classes if condition_outcome else 0)
        self.forward_rits = RITSCell(markers, static_width, recurrent_context, hidden)
        self.backward_rits = RITSCell(markers, static_width, recurrent_context, hidden)
        # Context acts directly on current-cell estimates, not only after a GRU update.
        self.context_mean = nn.Sequential(
            nn.Linear(static_width + recurrent_context, hidden), nn.Tanh(),
            nn.Linear(hidden, markers),
        )
        self.combine_gate = nn.Linear(2 * markers, markers)
        output = 1 if classes == 2 else classes
        if nonlinear_head:
            self.head = nn.Sequential(nn.Linear(feature_count + context_width, 48), nn.ReLU(),
                                      nn.Linear(48, output))
        else:
            self.head = nn.Linear(feature_count + context_width, output)

    def forward(self, values, mask, times, static, context, outcome_context, lengths=None):
        recurrent_context = torch.cat([context, outcome_context], 1)
        gaps = torch_marker_gaps(mask, times)
        forward, _ = self.forward_rits(values, mask, gaps, times, static, recurrent_context, lengths=lengths)
        backward, _ = self.backward_rits(values, mask, gaps, times, static, recurrent_context, True, lengths)
        shift = self.context_mean(torch.cat([static, recurrent_context], 1))[:, None, :]
        forward, backward = forward + shift, backward + shift
        gate = torch.sigmoid(self.combine_gate(torch.cat([mask, gaps], 2)))
        mean = gate * forward + (1 - gate) * backward
        completed = torch.where(mask.bool(), values, mean)
        return mean, completed, forward, backward


class UnifiedBRITSMI:
    """Canonical recurrent imputer; residual and bootstrap choices share its architecture.

``feature_function`` receives raw-scale trajectories, times, and raw static
covariates. Its output is the exact prespecified downstream design (no intercept).
Bootstrap members resample training subjects only; preprocessing and validation
use the same training/validation partition across members.
    """

    def __init__(self, config=None, *, task="association", feature_function=clinical_features):
        if task not in {"association", "prediction"}:
            raise ValueError("task must be association or prediction")
        self.config = config or UnifiedConfig()
        self.task = task
        self.feature_function = feature_function

    @staticmethod
    def _tensor(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def fit(self, values, mask, times, static, outcome, context=None, groups=None, lengths=None):
        times, lengths = self._padded_times(times, values.shape, lengths)
        values, mask, times, static, context = BRITSMultipleImputer._validate_arrays(
            values, mask, times, static, context)
        valid_cells = np.arange(values.shape[1])[None, :] < lengths[:, None]
        if np.any(mask & ~valid_cells[:, :, None]):
            raise ValueError("structural padding cannot be marked observed")
        raw_outcome = np.asarray(outcome)
        if not np.isfinite(raw_outcome).all() or np.any(raw_outcome != raw_outcome.astype(int)):
            raise ValueError("outcome labels must be finite integers")
        outcome = raw_outcome.astype(int)
        classes = np.unique(outcome[outcome >= 0])
        if self.task == "association" and np.any(outcome < 0):
            raise ValueError("association outcomes must be observed")
        if outcome.shape != (len(values),) or not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError("outcome must contain consecutive classes starting at zero")
        if len(classes) < 2 or len(values) < 40:
            raise ValueError("need at least 40 subjects and two outcome classes")
        self.classes_ = len(classes)
        self.context_width_ = context.shape[1]
        self.fit_inputs_ = (values.copy(), mask.copy(), times.copy(), static.copy(), context.copy())
        self.fit_outcome_ = outcome.copy()
        self.fit_lengths_ = lengths.copy()
        rng = np.random.default_rng(self.config.seed + 701)
        groups = np.arange(len(values)) if groups is None else np.asarray(groups)
        if groups.shape != (len(values),):
            raise ValueError("groups must align with input rows")
        group_order = rng.permutation(np.unique(groups))
        if len(group_order) < 3:
            raise ValueError("at least three distinct subject groups are required")
        nv = max(2, int(len(group_order) * self.config.validation_fraction))
        valid = np.flatnonzero(np.isin(groups, group_order[:nv]))
        train = np.flatnonzero(~np.isin(groups, group_order[:nv]))
        self.training_indices_, self.validation_indices_ = train, valid
        observed_train = np.where(mask[train], values[train], np.nan)
        self.mean_ = np.nanmean(observed_train, axis=(0, 1))
        self.sd_ = np.nanstd(observed_train, axis=(0, 1), ddof=1)
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.sd_).all():
            raise ValueError("every marker requires observed training values")
        self.sd_ = np.maximum(self.sd_, 1e-5)
        self.static_mean_ = static[train].mean(0)
        self.static_sd_ = np.maximum(static[train].std(0), 1e-5)
        self.context_mean_ = context[train].mean(0)
        self.context_sd_ = np.maximum(context[train].std(0), 1e-5)
        self.time_scale_ = float(np.median(times[train, lengths[train] - 1] - times[train, 0]))
        arrays = self._prepare(*self.fit_inputs_, outcome if self.task == "association" else None, lengths)
        obs = arrays[1]
        raw_static = self._tensor(static)
        raw_times = self._tensor(times)
        label = torch.as_tensor(outcome, dtype=torch.long)
        mean_filled = np.where(mask, values, self.mean_)
        with torch.no_grad():
            initial = self.feature_function(self._tensor(mean_filled), raw_times, raw_static, arrays[-1])
        self.feature_mean_ = initial[train].mean(0)
        self.feature_sd_ = initial[train].std(0).clamp_min(1e-4)
        fixed_hidden = self._tensor((rng.random(mask.shape) < self.config.holdout_fraction) & mask)
        self.models_, self.residual_covariances_, self.history_ = [], [], []
        self.member_diagnostics_ = []
        for member in range(self.config.bootstrap_members):
            torch.manual_seed(self.config.seed + 1009 * member)
            member_rng = np.random.default_rng(self.config.seed + 101 + member)
            population = train.copy()
            if self.config.bootstrap_members > 1:
                train_groups = np.unique(groups[train])
                chosen = member_rng.choice(train_groups, len(train_groups), replace=True)
                population = np.concatenate([np.flatnonzero(groups == g) for g in chosen])
            net = RecurrentNetwork(values.shape[2], static.shape[1], context.shape[1],
                                   self.classes_, self.config.hidden_size, initial.shape[1],
                                   self.task == "association", self.task == "prediction")
            optimizer = torch.optim.AdamW(net.parameters(), lr=self.config.learning_rate,
                                          weight_decay=1e-4)
            best_loss, best_epoch, best_state = np.inf, -1, None
            for epoch in range(self.config.epochs):
                net.train()
                order = member_rng.permutation(population)
                losses = []
                for start in range(0, len(order), self.config.batch_size):
                    idx = order[start:start + self.config.batch_size]
                    hidden = (torch.rand_like(obs[idx]) < self.config.holdout_fraction).float() * obs[idx]
                    loss, parts = self._loss(net, idx, hidden, arrays, raw_times, raw_static, label)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite recurrent training loss")
                    optimizer.zero_grad()
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("nonfinite recurrent gradient")
                    optimizer.step()
                    losses.append(float(loss.detach()))
                net.eval()
                with torch.no_grad():
                    loss, val_parts = self._loss(net, valid, fixed_hidden[valid], arrays,
                                                  raw_times, raw_static, label)
                validation_loss = float(loss)
                self.history_.append(dict(member=member, epoch=epoch, training_loss=float(np.mean(losses)),
                                          validation_loss=validation_loss, **val_parts))
                if np.isfinite(validation_loss) and validation_loss < best_loss - 1e-5:
                    best_loss, best_epoch = validation_loss, epoch
                    best_state = copy.deepcopy(net.state_dict())
                if epoch - best_epoch >= self.config.patience:
                    break
            if best_state is None:
                raise RuntimeError("no finite validation checkpoint")
            net.load_state_dict(best_state)
            net.eval()
            covariance = self._residual_covariance(net, arrays, valid, member_rng)
            self.models_.append(net)
            self.residual_covariances_.append(covariance)
            self.member_diagnostics_.append(dict(member=member, best_epoch=best_epoch,
                best_validation_loss=best_loss, residual_sd=np.sqrt(np.diag(covariance)).tolist()))
        return self

    @staticmethod
    def _padded_times(times, shape, lengths):
        times = np.asarray(times, dtype=float).copy()
        if times.ndim == 1:
            times = np.broadcast_to(times, shape[:2]).copy()
        lengths = np.full(shape[0], shape[1]) if lengths is None else np.asarray(lengths, dtype=int)
        if times.shape != shape[:2] or lengths.shape != (shape[0],):
            raise ValueError("invalid time or length dimensions")
        if np.any(lengths < 3) or np.any(lengths > shape[1]):
            raise ValueError("at least three valid visits are needed")
        for i, length in enumerate(lengths):
            if length < shape[1]:
                times[i, length:] = times[i, length - 1] + np.arange(1, shape[1] - length + 1)
        return times, lengths

    def _prepare(self, values, mask, times, static, context, outcome, lengths):
        z = np.where(mask, (values - self.mean_) / self.sd_, 0.0)
        t = (times - times[:, :1]) / self.time_scale_
        s = (static - self.static_mean_) / self.static_sd_
        c = (context - self.context_mean_) / self.context_sd_
        if self.task == "association":
            if outcome is None:
                raise ValueError("association imputations require observed outcomes")
            if not np.isin(outcome, np.arange(self.classes_)).all():
                raise ValueError("unrecognized outcome class")
            o = np.eye(self.classes_)[np.asarray(outcome, dtype=int)]
        else:
            if outcome is not None:
                raise ValueError("prediction-time outcomes must not be supplied")
            o = np.zeros((len(values), 0))
        return tuple(self._tensor(x) for x in (z, mask, t, s, c, o)) + (torch.as_tensor(lengths),)

    def _features(self, standardized, times, static, lengths):
        raw = standardized * self._tensor(self.sd_) + self._tensor(self.mean_)
        return (self.feature_function(raw, times, static, lengths) - self.feature_mean_) / self.feature_sd_

    def _loss(self, net, idx, hidden, arrays, times, static, labels):
        z, obs, t, s, c, oc, lengths = [a[idx] for a in arrays]
        fit_mask = obs * (1 - hidden)
        mean, completed, forward, backward = net(z * fit_mask, fit_mask, t, s, c, oc, lengths)
        reconstruction = ((mean - z).square() * hidden).sum() / hidden.sum().clamp_min(1)
        eligible = (torch.arange(z.shape[1])[None, :] < lengths[:, None])[:, :, None]
        missing = (1 - fit_mask) * eligible
        consistency = ((forward - backward).square() * missing).sum() / missing.sum().clamp_min(1)
        features = self._features(completed, times[idx], static[idx], lengths)
        # Only genuinely observed cells are restored in this teacher; no hidden truth is read.
        teacher = torch.where(obs.bool(), z, completed.detach())
        reference = self._features(teacher, times[idx], static[idx], lengths).detach()
        summary = (features - reference).square().mean()
        logits = net.head(torch.cat([features, c], 1))
        known = labels[idx] >= 0
        if not known.any():
            downstream = logits.sum() * 0
        elif self.classes_ == 2:
            downstream = nn.functional.binary_cross_entropy_with_logits(logits[known, 0], labels[idx][known].float())
        else:
            downstream = nn.functional.cross_entropy(logits[known], labels[idx][known])
        loss = (reconstruction + self.config.consistency_weight * consistency
                + self.config.summary_weight * summary + self.config.downstream_weight * downstream)
        return loss, {"reconstruction": float(reconstruction.detach()),
                      "consistency": float(consistency.detach()), "summary": float(summary.detach()),
                      "downstream": float(downstream.detach())}

    def _residual_covariance(self, net, arrays, valid, rng):
        z, obs, t, s, c, oc, lengths = [a[valid] for a in arrays]
        residuals, indicators = [], []
        # Calibration reads validation observations only after they are hidden from the network.
        for _ in range(6):
            hidden = self._tensor(rng.random(tuple(obs.shape)) < self.config.holdout_fraction) * obs
            fit_mask = obs * (1 - hidden)
            with torch.no_grad():
                mean = net(z * fit_mask, fit_mask, t, s, c, oc, lengths)[0]
            residuals.append(((z - mean) * hidden).numpy().reshape(len(valid), -1))
            indicators.append(hidden.numpy().reshape(len(valid), -1))
        residual = np.concatenate(residuals).astype(np.float64)
        eligible = np.concatenate(indicators).astype(np.float64)
        pair_count = eligible.T @ eligible
        second = (residual.T @ residual) / np.maximum(pair_count, 1)
        diagonal = np.maximum(np.diag(second), 0.01)
        # Shrink pairwise residual covariance and project to PSD for sparse held-out pairs.
        covariance = 0.5 * second + 0.5 * np.diag(diagonal)
        eigval, eigvec = np.linalg.eigh(covariance)
        covariance = (eigvec * np.maximum(eigval, 0.01)) @ eigvec.T
        scale = np.sqrt(diagonal / np.maximum(np.diag(covariance), 1e-12))
        covariance = covariance * scale[:, None] * scale[None, :]
        return (covariance + covariance.T) / 2

    def sample(self, m, *, values=None, mask=None, times=None, static=None, context=None,
               outcome=None, seed=None, residual_mode=None, multiplier=None, lengths=None):
        if not hasattr(self, "models_") or m < 1:
            raise ValueError("fit first and request positive imputations")
        if self.task == "prediction" and outcome is not None:
            raise ValueError("prediction-time outcomes must not be supplied")
        if values is None:
            values, mask, times, static, context = self.fit_inputs_
            outcome = self.fit_outcome_ if self.task == "association" else None
            lengths = self.fit_lengths_
        else:
            if any(x is None for x in (mask, times, static)):
                raise ValueError("all query inputs are required")
            if context is None and self.context_width_:
                raise ValueError("query image/context features are required")
        times, lengths = self._padded_times(times, values.shape, lengths)
        values, mask, times, static, context = BRITSMultipleImputer._validate_arrays(
            values, mask, times, static, context)
        if values.shape[1:] != self.fit_inputs_[0].shape[1:]:
            raise ValueError("query arrays must use the training visit/marker dimensions")
        valid_cells = np.arange(values.shape[1])[None, :] < lengths[:, None]
        if np.any(mask & ~valid_cells[:, :, None]):
            raise ValueError("structural padding cannot be marked observed")
        arrays = self._prepare(values, mask, times, static, context, outcome, lengths)
        with torch.no_grad():
            predictions = [net(*arrays)[0].numpy() for net in self.models_]
        rng = np.random.default_rng(self.config.seed + 9921 if seed is None else seed)
        mode = residual_mode or self.config.residual_mode
        factor = self.config.residual_multiplier if multiplier is None else multiplier
        if not np.isfinite(factor) or factor < 0:
            raise ValueError("residual multiplier must be finite and nonnegative")
        draws = []
        self.last_draw_members_ = []
        for draw_id in range(m):
            member = int(rng.integers(len(predictions)))
            self.last_draw_members_.append(member)
            covariance = self.residual_covariances_[member]
            if mode == "iid":
                noise = rng.normal(size=(len(values), covariance.shape[0])) * np.sqrt(np.diag(covariance))
            elif mode == "correlated":
                noise = rng.multivariate_normal(np.zeros(covariance.shape[0]), covariance,
                                                 len(values), check_valid="raise")
            else:
                raise ValueError("unknown residual mode")
            draw = (predictions[member] + factor * noise.reshape(values.shape)) * self.sd_ + self.mean_
            valid = (np.arange(values.shape[1])[None, :] < lengths[:, None])[:, :, None]
            draws.append(np.where(valid, np.where(mask, values, draw), 0.0))
        return draws

    def predict_proba(self, *, values, mask, times, static, context=None, m=20, seed=None, lengths=None):
        """Average class probabilities, not Rubin-pool AUROCs or probabilities."""
        if self.task != "prediction":
            raise ValueError("use a label-free prediction fit for new-patient probabilities")
        draws = self.sample(m, values=values, mask=mask, times=times, static=static,
                            context=context, seed=seed, lengths=lengths)
        times, lengths = self._padded_times(times, values.shape, lengths)
        if context is None:
            context = np.zeros((len(values), 0))
        ctx = self._tensor((context - self.context_mean_) / self.context_sd_)
        probabilities = []
        with torch.no_grad():
            for index, draw in enumerate(draws):
                features = (self.feature_function(self._tensor(draw), self._tensor(times), self._tensor(static), torch.as_tensor(lengths))
                            - self.feature_mean_) / self.feature_sd_
                logits = self.models_[self.last_draw_members_[index]].head(torch.cat([features, ctx], 1))
                if self.classes_ == 2:
                    p = torch.sigmoid(logits)
                    p = torch.cat([1 - p, p], 1)
                else:
                    p = logits.softmax(1)
                probabilities.append(p.numpy())
        return np.mean(probabilities, axis=0)
