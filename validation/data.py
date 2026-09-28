"""Validate an analysis-ready input without importing institutional source files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CLINICAL_COLUMNS = [
    "row_index",
    "outcome_3class",
    "FIB4_log",
    "FIB4_Age",
    "bmi_consolidated",
    "Diabetes_RF",
    "HTN_RF",
    "HLD_RF",
    "Sex",
    "Study_Type",
]
MARKERS = ["a1c", "sbp", "dbp", "ast", "alt", "platelet"]
STAGES = ["F0-F1", "F2", "F3-F4"]


@dataclass
class Cohort:
    clinical: pd.DataFrame
    values: np.ndarray
    mask: np.ndarray
    static: np.ndarray
    labels: np.ndarray
    standardized: np.ndarray
    marker_mean: np.ndarray
    marker_sd: np.ndarray


def external_path(path: Path, repository: Path) -> Path:
    """Keep both private inputs and derived outputs outside the checkout."""
    resolved = path.expanduser().resolve()
    if resolved == repository or repository in resolved.parents:
        raise ValueError("Validation inputs and outputs must be outside the repository")
    return resolved


def load_cohort(directory: Path) -> Cohort:
    frame = pd.read_csv(directory / "clinical.csv")
    if set(frame.columns) != set(CLINICAL_COLUMNS):
        raise ValueError(f"clinical.csv must contain exactly these columns: {CLINICAL_COLUMNS}")
    with np.load(directory / "longitudinal.npz", allow_pickle=False) as data:
        if set(data.files) != {"values", "mask", "static", "row_index", "marker_names"}:
            raise ValueError("Unexpected NPZ schema; see validation/README.md")
        values = np.asarray(data["values"], dtype=np.float32)
        mask = np.asarray(data["mask"])
        static = np.asarray(data["static"], dtype=np.float32)
        row_index = data["row_index"]
        marker_names = data["marker_names"].tolist()
    n = len(frame)
    if values.shape != (n, 6, 6) or mask.shape != values.shape or marker_names != MARKERS:
        raise ValueError("Use six ordered bins and channels a1c, sbp, dbp, ast, alt, platelet")
    if not np.array_equal(row_index, np.arange(n)) or not np.array_equal(
        frame["row_index"].to_numpy(), row_index
    ):
        raise ValueError("CSV and NPZ row_index must both equal 0, ..., N-1 in the same order")
    if not np.isin(mask, [0, 1]).all() or not np.isfinite(values[mask == 1]).all():
        raise ValueError("The mask must be binary and all measured values must be finite")
    if static.ndim != 2 or static.shape[0] != n or not np.isfinite(static).all():
        raise ValueError("static must be a finite N x P matrix of analysis covariates")
    if set(frame["outcome_3class"]) != set(STAGES):
        raise ValueError(f"All three outcome categories are required: {STAGES}")
    if np.any(mask.sum(axis=(0, 1)) == 0):
        raise ValueError("Every channel needs at least one source observation")
    for name in ["FIB4_log", "FIB4_Age", "bmi_consolidated"]:
        numeric = pd.to_numeric(frame[name], errors="coerce")
        if not np.isfinite(numeric).any() or np.isinf(numeric).any():
            raise ValueError(f"{name} must contain finite source measurements")
    for name in ["Diabetes_RF", "HTN_RF", "HLD_RF"]:
        if not frame[name].astype(str).str.lower().isin(["yes", "no", "1", "0"]).all():
            raise ValueError(f"{name} must be coded Yes/No or 1/0 without missing values")
    if not frame["Sex"].astype(str).str.lower().isin(["male", "female"]).all():
        raise ValueError("Sex must be coded Male/Female for the specified outcome model")
    observed = np.where(mask == 1, values, np.nan)
    mean = np.nanmean(observed, axis=(0, 1)).astype(np.float32)
    sd = np.nanstd(observed, axis=(0, 1))
    sd = np.where(sd > 1e-8, sd, 1.0).astype(np.float32)
    standardized = np.where(mask == 1, (values - mean) / sd, 0).astype(np.float32)
    return Cohort(
        clinical=frame.drop(columns="row_index"),
        values=np.where(mask == 1, values, 0).astype(np.float32),
        mask=mask.astype(np.float32),
        static=static,
        labels=frame["outcome_3class"].map(dict(zip(STAGES, range(3)))).to_numpy(),
        standardized=standardized,
        marker_mean=mean,
        marker_sd=sd,
    )
