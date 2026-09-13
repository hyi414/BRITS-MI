from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score


def summarize_imputation(y_true: np.ndarray, y_pred: np.ndarray, missing_mask: np.ndarray) -> dict[str, float]:
    idx = missing_mask.astype(bool)
    diff = y_pred[idx] - y_true[idx]
    if diff.size == 0:
        return {"rmse": 0.0, "mae": 0.0}
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
    }


def summary_mse(y_true: np.ndarray, y_pred: np.ndarray, times: np.ndarray, lengths: np.ndarray) -> float:
    rows = []
    for i in range(y_true.shape[0]):
        length = int(lengths[i])
        yt = y_true[i, :length]
        yp = y_pred[i, :length]
        tt = times[i, :length]
        denom = np.sum((tt - tt.mean()) ** 2)
        marker_mse = []
        for k in range(yt.shape[-1]):
            ytk = yt[:, k]
            ypk = yp[:, k]
            auc_t = np.trapezoid(ytk, tt) if length > 1 else 0.0
            auc_p = np.trapezoid(ypk, tt) if length > 1 else 0.0
            slope_t = 0.0 if denom <= 0 else float(np.sum((tt - tt.mean()) * (ytk - ytk.mean())) / denom)
            slope_p = 0.0 if denom <= 0 else float(np.sum((tt - tt.mean()) * (ypk - ypk.mean())) / denom)
            sum_t = np.array([ytk[0], ytk[-1], ytk.mean(), auc_t, slope_t, ytk.std()])
            sum_p = np.array([ypk[0], ypk[-1], ypk.mean(), auc_p, slope_p, ypk.std()])
            marker_mse.append(np.mean((sum_t - sum_p) ** 2))
        rows.append(float(np.mean(marker_mse)))
    return float(np.mean(rows))


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        pred = (probs >= 0.5).astype(int)
        auroc = float(roc_auc_score(y_true, probs))
        auprc = float(average_precision_score(y_true, probs))
        ll = float(log_loss(y_true, probs, labels=[0, 1]))
    elif probs.shape[1] == 2:
        probs = np.clip(probs, 1e-7, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        pos = probs[:, 1]
        pred = np.argmax(probs, axis=1)
        auroc = float(roc_auc_score(y_true, pos))
        auprc = float(average_precision_score(y_true, pos))
        ll = float(log_loss(y_true, probs, labels=[0, 1]))
    else:
        probs = np.clip(probs, 1e-7, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        pred = np.argmax(probs, axis=1)
        auroc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        y_onehot = np.eye(probs.shape[1])[y_true]
        auprc = float(average_precision_score(y_onehot, probs, average="macro"))
        ll = float(log_loss(y_true, probs, labels=list(range(probs.shape[1]))))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "log_loss": ll,
    }


def safe_classification_metrics(y_true: np.ndarray, probs: np.ndarray, prefix: str) -> dict[str, float]:
    if len(y_true) < 8 or len(np.unique(y_true)) < 2:
        return {
            f"{prefix}_auroc": np.nan,
            f"{prefix}_auprc": np.nan,
            f"{prefix}_accuracy": np.nan,
            f"{prefix}_balanced_accuracy": np.nan,
            f"{prefix}_macro_f1": np.nan,
            f"{prefix}_log_loss": np.nan,
            f"{prefix}_n": float(len(y_true)),
        }
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if probs.ndim > 1:
        classes = np.unique(y_true)
        probs = probs[:, classes]
        probs = np.clip(probs, 1e-7, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        remap = {klass: idx for idx, klass in enumerate(classes)}
        y_true = np.asarray([remap[v] for v in y_true], dtype=int)
    out = classification_metrics(y_true, probs)
    return {
        f"{prefix}_auroc": out["auroc"],
        f"{prefix}_auprc": out["auprc"],
        f"{prefix}_accuracy": out["accuracy"],
        f"{prefix}_balanced_accuracy": out["balanced_accuracy"],
        f"{prefix}_macro_f1": out["macro_f1"],
        f"{prefix}_log_loss": out["log_loss"],
        f"{prefix}_n": float(len(y_true)),
    }
