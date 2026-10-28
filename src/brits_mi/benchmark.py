"""Auditable benchmark primitives, fixed-scale inference, and paired summaries."""

from __future__ import annotations

import hashlib
import json
import sys
import os
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import t
import torch

from .analysis import (COEFFICIENT_ORDER, barnard_rubin_df, design_matrix,
                       trajectory_features, validated_logistic_fit)
from .comparators import r_impute
from .unified import UnifiedBRITSMI


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(str((a.shape, str(a.dtype))).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def source_hash(root):
    digest = hashlib.sha256()
    paths = sorted(list((root / "src").rglob("*.py")) + list((root / "src").rglob("*.R"))
                   + list((root / "simulation" / "unified").rglob("*.py"))
                   + list((root / "simulation" / "clinical" / "scripts").glob("*.py"))
                   + list((root / "validation").glob("*unified*.py")))
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(content, indent=2, default=lambda x: x.item()) + "\n")
    temp.replace(path)


def mean_impute(values, mask):
    available = np.where(mask, values, np.nan)
    means = np.nanmean(available, axis=0)
    fallback = np.nanmean(available, axis=(0, 1))
    means = np.where(np.isfinite(means), means, fallback)
    return np.where(mask, values, means[None, :, :])


def pool_designs(designs, outcome, scale, truth):
    """Fit in raw units, then transform every estimate/covariance by one common SD."""
    fits = [validated_logistic_fit(outcome, design) for design in designs]
    names = list(designs[0].columns)
    scale = np.asarray(scale, dtype=float)
    estimates = np.array([np.asarray(fit.params)[1:] * scale for fit in fits])
    within = np.array([np.asarray(fit.bse)[1:] ** 2 * scale ** 2 for fit in fits]).mean(0)
    m = len(fits)
    between = estimates.var(0, ddof=1) if m > 1 else np.zeros_like(within)
    total = within + (1 + 1 / m) * between
    se = np.sqrt(total)
    df = barnard_rubin_df(within, between, m, len(outcome) - len(names) - 1)
    halfwidth = t.ppf(.975, df) * se
    estimate = estimates.mean(0)
    truth = np.asarray(truth, dtype=float)
    error = estimate - truth
    return pd.DataFrame(dict(term=names, truth=truth, estimate=estimate,
        signed_error=error, squared_error=error ** 2, standard_error=se,
        within_variance=within, between_variance=between, degrees_of_freedom=df,
        lower=estimate - halfwidth, upper=estimate + halfwidth,
        covered=(np.abs(error) <= halfwidth).astype(float), n_imputations=m,
        n_subjects=len(outcome)))


def score_imputations(draws, truth, evaluation_mask, times, reference_sd=None):
    mean = np.mean(draws, axis=0)
    scale = np.std(truth, axis=(0, 1), ddof=1) if reference_sd is None else reference_sd
    error = (mean - truth) / np.maximum(scale, 1e-8)
    true_summary = trajectory_features(truth, times).to_numpy()
    estimates = np.mean([trajectory_features(draw, times).to_numpy() for draw in draws], 0)
    summary_sd = np.maximum(true_summary.std(0, ddof=1), 1e-8)
    stacked = np.stack(draws)
    lower, upper = np.quantile(stacked, [.025, .975], axis=0)
    return dict(imputation_rmse=float(np.sqrt(np.mean((mean - truth)[evaluation_mask] ** 2))),
                imputation_nrmse=float(np.sqrt(np.mean(error[evaluation_mask] ** 2))),
                trajectory_nrmse=float(np.sqrt(np.mean(((estimates - true_summary) / summary_sd) ** 2))),
                cell_interval_inclusion=float(np.mean(((lower <= truth) & (truth <= upper))[evaluation_mask])))


def locked_clinical_data(seed, scenario, n, missing, mechanism="original"):
    root = Path(__file__).resolve().parents[2]
    path = str(root / "simulation" / "clinical" / "scripts")
    if path not in sys.path:
        sys.path.insert(0, path)
    from run_default_mice_clinical_association_sensitivity import generate_locked_dataset, TRUE_BETA
    data = generate_locked_dataset(seed, scenario, n, missing)
    values = np.asarray(data["trajectory"], float)
    static = np.column_stack([data[x] for x in ["diabetes", "hypertension", "male"]])
    times = np.broadcast_to(data["time"], values.shape[:2]).copy()
    mask = np.asarray(data["observed"], bool)
    if mechanism == "mar":
        # All predictors of masking are observed before any late-cell masking.
        rng = np.random.default_rng(seed + 39001)
        early = values[:, :3].mean(1)
        risk = (0.8 * data["outcome"] + .3 * static[:, 0] + .2 * static[:, 1]
                + .25 * early[:, 0] - .2 * early[:, 2])[:, None, None]
        risk = risk + np.array([-.3, 0, .3])[None, :, None] + np.array([.1, 0, -.1])[None, None, :]
        from scipy.special import expit
        lo, hi = -15., 15.
        for _ in range(50):
            mid = (lo + hi) / 2
            if expit(mid + risk).mean() < missing:
                lo = mid
            else:
                hi = mid
        mask = np.ones_like(values, bool)
        mask[:, -3:] = rng.random((n, 3, 3)) >= expit((lo + hi) / 2 + risk)
    elif mechanism != "original":
        raise ValueError("mechanism must be original or mar")
    beta = {name.replace("htn", "hypertension"): value for name, value in TRUE_BETA.items()}
    return values, mask, times, static, np.asarray(data["outcome"]), beta


def clinical_job(job, config, output):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed, missing = job["seed"], job["missing_percentage"] / 100
    truth, mask, times, static, outcome, beta = locked_clinical_data(
        seed, job["scenario"], job["n"], missing, job.get("mechanism", "original"))
    values = np.where(mask, truth, np.nan)
    data_hash = array_hash(values, mask, times, static, outcome)
    scale_path = output.parent / ("reference_scale_" + job["scenario"] + ".json")
    if not scale_path.exists():
        population = locked_clinical_data(870001, job["scenario"], 20000, .2)
        pop_design = design_matrix(trajectory_features(population[0], population[2]), *population[3].T)
        scale = pop_design.std(0, ddof=1).to_numpy()
        atomic_json(scale_path, dict(columns=list(pop_design), scale=scale.tolist(),
                                    source="independent 20000-subject reference; evaluation only", seed=870001))
    scale = np.array(json.loads(scale_path.read_text())["scale"])
    target = np.array([beta[k] for k in COEFFICIENT_ORDER]) * scale
    rows, terms, audits = [], [], []
    methods = job.get("methods", ["BRITS-MI", "MICE", "missForest", "Mean imputation", "Complete case", "Full data"])
    fitted = {}
    for method in methods:
        started = time.monotonic()
        audit = {"method": method, "input_hash": data_hash}
        try:
            if method.startswith("BRITS-MI"):
                members = 5 if method == "BRITS-MI-bootstrap" else config.bootstrap_members
                mode = "correlated" if method in {"BRITS-MI-correlated", "BRITS-MI-bootstrap"} else config.residual_mode
                if members not in fitted:
                    fitted[members] = UnifiedBRITSMI(replace(config, seed=seed, bootstrap_members=members))
                    fitted[members].fit(values, mask, times, static, outcome)
                imputer = fitted[members]
                draws = imputer.sample(job["m"], seed=seed + 91000, residual_mode=mode)
                audit["training"] = imputer.member_diagnostics_
                pd.DataFrame(imputer.history_).to_csv(output / f"training_{method}.csv", index=False)
            elif method in {"MICE", "missForest"}:
                draws, r_audit = r_impute(values, mask, times, static, method=method.lower(),
                    m=job["m"], seed=seed + 73001, outcome=outcome)
                audit.update(r_audit)
            elif method == "Mean imputation":
                draws = [mean_impute(values, mask)]
            elif method == "Full data":
                draws = [truth.copy()]
            else:
                keep = mask[:, -3:, :].all(axis=(1, 2))
                designs = [design_matrix(trajectory_features(truth[keep], times[keep]), *static[keep].T)]
                effects = pool_designs(designs, outcome[keep], scale, target)
                draws = None
            if draws is not None:
                designs = [design_matrix(trajectory_features(draw, times), *static.T) for draw in draws]
                # MI draws must all be valid. Partial-fit pooling is not used.
                effects = pool_designs(designs, outcome, scale, target)
                metrics = score_imputations(draws, truth, ~mask, times)
            else:
                metrics = {}
            effects["method"] = method
            terms.extend(effects.to_dict("records"))
            rows.append(dict(method=method, status="valid", coverage=float(effects.covered.mean()),
                coefficient_mse=float(effects.squared_error.mean()),
                signed_error_mean=float(effects.signed_error.mean()),
                mean_absolute_error=float(effects.signed_error.abs().mean()),
                mean_model_se=float(effects.standard_error.mean()), **metrics))
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rows.append(dict(method=method, status="invalid", reason=str(exc)))
        rows[-1]["seconds"] = time.monotonic() - started
        rows[-1]["input_hash"] = data_hash
        audits.append(audit)
        atomic_json(output / "partial.json", dict(job=job, metrics=rows, coefficients=terms, audit=audits))
    result = dict(job=job, metrics=rows, coefficients=terms, audit=audits,
                  realized_missing_percentage=float(100 * (~mask[:, -3:]).mean()))
    atomic_json(output / "result.json", result)
    return result
