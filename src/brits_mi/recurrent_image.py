"""Recurrent image-conditioned MI with a matched downstream neural fusion model."""

import copy
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, roc_curve, precision_recall_curve

from longitudinal_sim.config import ExperimentConfig
from longitudinal_sim.data import generate_dataset
from ._image_reference import image_features
from .benchmark import atomic_json, array_hash, mean_impute
from .comparators import r_impute
from .unified import UnifiedBRITSMI, fusion_features


def fusion_probabilities(draws, times, static, image, lengths, labels, train, test,
                         seed, epochs=100, task_features=fusion_features):
    """Refit the same neural fusion head for each imputation; average probabilities."""
    fit, valid = train_test_split(train, test_size=.2, random_state=seed + 51, stratify=labels[train])
    probabilities = []
    for index, draw in enumerate(draws):
        with torch.no_grad():
            clinical = task_features(torch.tensor(draw, dtype=torch.float32),
                torch.tensor(times, dtype=torch.float32), torch.tensor(static, dtype=torch.float32),
                torch.tensor(lengths)).numpy()
        design = np.column_stack([clinical, image])
        # Neither scaler nor checkpoint selection uses held-out outcome labels.
        scaler = StandardScaler().fit(design[fit])
        x = torch.tensor(scaler.transform(design), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        torch.manual_seed(seed + 500 + index)
        head = nn.Sequential(nn.Linear(x.shape[1], 48), nn.ReLU(), nn.Linear(48, 3))
        optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.01)
        best, state, last = np.inf, None, 0
        for epoch in range(epochs):
            head.train()
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(head(x[fit]), y[fit])
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5)
            optimizer.step()
            head.eval()
            with torch.no_grad():
                value = float(nn.functional.cross_entropy(head(x[valid]), y[valid]))
            if value < best:
                best, state, last = value, copy.deepcopy(head.state_dict()), epoch
            if epoch - last >= 20:
                break
        head.load_state_dict(state)
        with torch.no_grad():
            probabilities.append(head(x[test]).softmax(1).numpy())
    return np.mean(probabilities, 0)


def image_job(job, config, output):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed = job["seed"]
    dgp = ExperimentConfig(n_subjects=job["n"], seed=seed,
        mask_rate=job["missing_percentage"] / 100, dgp_profile="paper_britsadv_img065_altus",
        info_profile="balanced", n_classes=3, image_size=24)
    data = generate_dataset(dgp)
    mask = data.mask.astype(bool)
    lengths = data.lengths
    valid = np.arange(mask.shape[1])[None, :] < lengths[:, None]
    values = np.where(mask, data.y_obs, np.nan)
    values = np.where(valid[:, :, None], values, 0.)
    times, _ = UnifiedBRITSMI._padded_times(data.times, values.shape, lengths)
    static = np.column_stack([data.static, data.x.reshape(job["n"], -1), data.x_mask.reshape(job["n"], -1)])
    image = image_features(data.image)
    train, test = train_test_split(np.arange(job["n"]), test_size=.25,
                                  random_state=seed + 101, stratify=data.labels)
    training_labels = np.full(job["n"], -1)
    training_labels[train] = data.labels[train]
    input_hash = array_hash(values, mask, times, static, image, training_labels)
    rows, audits, curves = [], [], []
    # Default missForest has no inductive transform. This benchmark explicitly permits
    # all methods to use observed, unlabeled test covariates (transductive imputation).
    # Test labels are never inputs to any imputer, scaler, head, or checkpoint selector.
    for method in job.get("methods", ["BRITS-MI", "MICE", "missForest", "Mean imputation", "Complete case", "Full data"]):
        started = time.monotonic()
        audit = dict(method=method, input_hash=input_hash)
        try:
            if method == "BRITS-MI":
                model = UnifiedBRITSMI(replace(config, seed=seed), task="prediction",
                                      feature_function=fusion_features)
                model.fit(values, mask, times, static, training_labels, context=image, lengths=lengths)
                draws = model.sample(job["m"], seed=seed + 310001)
                audit["training"] = model.member_diagnostics_
            elif method in {"MICE", "missForest"}:
                draws, r_audit = r_impute(values, mask, times, static, method=method.lower(),
                    m=job["m"], seed=seed + 73001, outcome=training_labels, context=image, lengths=lengths)
                audit.update(r_audit)
            elif method == "Mean imputation":
                draws = [mean_impute(values, mask | ~valid[:, :, None])]
            elif method == "Full data":
                draws = [data.y_true.copy()]
            else:
                keep = np.all(mask | ~valid[:, :, None], axis=(1, 2))
                complete_train, complete_test = train[keep[train]], test[keep[test]]
                if min(len(complete_train), len(complete_test)) < 30 or len(np.unique(data.labels[complete_train])) < 3 or len(np.unique(data.labels[complete_test])) < 3:
                    raise ValueError("too few complete-case training/test subjects for three-class evaluation")
                prob = fusion_probabilities([data.y_true], times, static, image, lengths, data.labels,
                    complete_train, complete_test, seed, job.get("head_epochs", 100))
                # A different test population is not a paired contrast against all-subject methods.
                audit["different_test_population"] = True
                audit["n_complete_test"] = len(complete_test)
                draws = None
            evaluation = test if draws is not None else complete_test
            if draws is not None:
                prob = fusion_probabilities(draws, times, static, image, lengths, data.labels,
                    train, test, seed, job.get("head_epochs", 100))
                mean = np.mean(draws, 0)
                hidden = ~mask[test] & valid[test, :, None]
                scale = np.array([np.std(data.y_true[train, :, j][valid[train]], ddof=1) for j in range(3)])
                error = (mean[test] - data.y_true[test]) / scale
                imputation_nrmse = float(np.sqrt(np.mean(error[hidden] ** 2)))
                with torch.no_grad():
                    ftrue = fusion_features(torch.tensor(data.y_true[test]), torch.tensor(times[test]),
                        torch.tensor(static[test]), torch.tensor(lengths[test])).numpy()[:, :24]
                    fest = fusion_features(torch.tensor(mean[test]), torch.tensor(times[test]),
                        torch.tensor(static[test]), torch.tensor(lengths[test])).numpy()[:, :24]
                trajectory_nrmse = float(np.sqrt(np.mean(((fest - ftrue) / np.maximum(ftrue.std(0), 1e-6)) ** 2)))
            else:
                imputation_nrmse, trajectory_nrmse = np.nan, np.nan
            y = data.labels[evaluation]
            rows.append(dict(method=method, status="valid", auroc=float(roc_auc_score(y, prob, multi_class="ovr", average="macro")),
                auprc=float(average_precision_score(np.eye(3)[y], prob, average="macro")),
                log_loss=float(log_loss(y, prob)), imputation_nrmse=imputation_nrmse,
                trajectory_nrmse=trajectory_nrmse, n_test=len(y)))
            for cls in range(3):
                fpr, tpr, _ = roc_curve(y == cls, prob[:, cls])
                precision, recall, _ = precision_recall_curve(y == cls, prob[:, cls])
                grid = np.linspace(0, 1, 101)
                curves.append(dict(method=method, cls=cls, grid=grid.tolist(),
                    tpr=np.interp(grid, fpr, tpr).tolist(),
                    precision=np.interp(grid, recall[::-1], precision[::-1]).tolist()))
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rows.append(dict(method=method, status="invalid", reason=str(exc)))
        rows[-1]["seconds"] = time.monotonic() - started
        rows[-1]["input_hash"] = input_hash
        audits.append(audit)
        atomic_json(output / "partial.json", dict(job=job, metrics=rows, coefficients=[], audit=audits, curves=curves))
    result = dict(job=job, metrics=rows, coefficients=[], audit=audits, curves=curves,
        interpretation="held-out-label prediction with transductive imputation of observed unlabeled test covariates",
        realized_missing_percentage=float(100 * (~mask & valid[:, :, None]).sum() / (valid.sum() * 3)))
    atomic_json(output / "result.json", result)
    return result
