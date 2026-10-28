"""TE observed-cell masking: mask first, refit all imputers, use a fixed reference SD."""

from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from brits_mi.analysis import validated_logistic_fit
from brits_mi.benchmark import atomic_json, array_hash, mean_impute, pool_designs
from brits_mi.comparators import r_impute
from brits_mi.unified import UnifiedBRITSMI


BASE_NAMES = ["fib4_log", "age", "bmi", "diabetes", "hypertension", "hyperlipidemia", "male",
              "fib4_log_x_diabetes", "hypertension_x_male"]
TRAJECTORY_NAMES = ["late_ast", "late_alt", "late_platelet", "change_ast", "change_alt", "change_platelet",
                    "missing_fraction_ast", "missing_fraction_alt", "missing_fraction_platelet"]


def real_features(log_trajectory, times, static, lengths=None):
    """Use measured-source history for a reference-comparable association target.

The last 36 static fields encode source availability, not measurement values.
Times are six equally spaced pre-examination bin centers; terminal change is
not labeled an irregular-time slope in this application.
    """
    raw = torch.expm1(log_trajectory)
    source_mask = static[:, len(BASE_NAMES):].reshape(-1, 6, 6)
    late, changes, fractions = [], [], []
    index = torch.arange(6)[None, :]
    for j in (3, 4, 5):
        observed = source_mask[:, :, j]
        overall = (raw[:, :, j] * observed).sum(1) / observed.sum(1).clamp_min(1)
        late_count = observed[:, -2:].sum(1)
        end = (raw[:, -2:, j] * observed[:, -2:]).sum(1) / late_count.clamp_min(1)
        late.append(torch.where(late_count > 0, end, overall))
        first = torch.where(observed.bool(), index, 6).min(1).values.clamp(max=5)
        last = torch.where(observed.bool(), index, -1).max(1).values.clamp(min=0)
        change = raw[:, :, j].gather(1, last[:, None])[:, 0] - raw[:, :, j].gather(1, first[:, None])[:, 0]
        changes.append(torch.where(observed.sum(1) >= 2, change, 0))
        fractions.append(1 - observed.mean(1))
    return torch.cat([static[:, :len(BASE_NAMES)], torch.stack(late + changes + fractions, 1)], 1)


def load_inputs(directory):
    directory = Path(directory)
    clinical = pd.read_csv(directory / "clinical.csv")
    with np.load(directory / "longitudinal.npz", allow_pickle=False) as archive:
        values, mask = archive["values"].copy(), archive["mask"].astype(bool)
        groups = archive["groups"].copy() if "groups" in archive else np.arange(len(values))
    if not clinical.Study_Type.astype(str).str.upper().str.contains("TE|FIBROSCAN", regex=True).all():
        raise ValueError("this application is restricted to TE")
    labels = clinical.outcome_3class.map({"F0-F1": 0, "F2": 1, "F3-F4": 2}).to_numpy()
    # Source-summary effects require at least two measured values per liver marker.
    # This is an evaluation subset, not a claim to know naturally missing values.
    keep = (mask[:, :, 3:6].sum(1) >= 2).all(1)
    numeric = clinical[["FIB4_log", "FIB4_Age", "bmi_consolidated"]].apply(pd.to_numeric, errors="coerce")
    keep &= np.isfinite(numeric).all(1).to_numpy()
    count = dict(n_te_source=len(values), n_reference_eligible=int(keep.sum()),
                 n_excluded_unavailable_source_summaries_or_covariates=int((~keep).sum()))
    clinical = clinical.loc[keep].reset_index(drop=True)
    values, mask, groups, labels = values[keep], mask[keep], groups[keep], labels[keep]
    if np.any(values[mask] < 0):
        raise ValueError("nonnegative biomarkers are required for log1p preprocessing")
    def yes(name):
        return clinical[name].astype(str).str.lower().isin(["yes", "1", "1.0", "true"]).to_numpy(float)
    numeric = numeric.loc[keep].to_numpy()
    diabetes, htn, hld = [yes(name) for name in ["Diabetes_RF", "HTN_RF", "HLD_RF"]]
    male = clinical.Sex.astype(str).str.lower().eq("male").to_numpy(float)
    base = np.column_stack([numeric, diabetes, htn, hld, male,
                            numeric[:, 0] * diabetes, htn * male])
    static = np.column_stack([base, mask.reshape(len(values), -1)])
    log_values = np.where(mask, np.log1p(np.where(mask, values, 0)), np.nan)
    times = np.broadcast_to(np.linspace(0, 1, 6), (len(values), 6)).copy()
    return log_values, mask, times, static, labels, groups, count


def real_job(job, config, output, input_dir):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    truth, source, times, static, stage, groups, counts = load_inputs(input_dir)
    y = (stage == 2).astype(int)
    columns = BASE_NAMES + TRAJECTORY_NAMES
    def to_design(a):
        return pd.DataFrame(real_features(torch.tensor(np.nan_to_num(a), dtype=torch.float64),
                           torch.tensor(times), torch.tensor(static)).numpy(), columns=columns)
    reference_design = to_design(truth)
    # Common SD is applied only to fitted coefficients/covariances, not imputer training.
    scale = reference_design.std(0, ddof=1).to_numpy()
    reference_fit = validated_logistic_fit(y, reference_design)
    target = np.asarray(reference_fit.params)[1:] * scale
    rng = np.random.default_rng(job["seed"])
    eligible = np.zeros_like(source)
    eligible[:, :, 3:6] = source[:, :, 3:6]
    hidden = eligible & (rng.random(source.shape) < job["missing_percentage"] / 100)
    mask = source & ~hidden
    values = np.where(mask, truth, np.nan)
    # Source masks contain availability only; no pre-mask values/coefficients are training inputs.
    input_hash = array_hash(values, mask, times, static, y)
    rows, terms, audits = [], [], []
    for method in job.get("methods", ["BRITS-MI", "MICE", "missForest", "Mean imputation"]):
        started = time.monotonic()
        audit = dict(method=method, input_hash=input_hash)
        try:
            if method == "BRITS-MI":
                model = UnifiedBRITSMI(replace(config, seed=job["seed"]), feature_function=real_features)
                model.fit(values, mask, times, static, y, groups=groups)
                draws = model.sample(job["m"], seed=job["seed"] + 9001)
                audit["training"] = model.member_diagnostics_
                pd.DataFrame(model.history_).to_csv(output / "training.csv", index=False)
            elif method == "Mean imputation":
                draws = [mean_impute(values, mask)]
            else:
                draws, r_audit = r_impute(values, mask, times, static, method=method.lower(),
                    m=job["m"], seed=job["seed"] + 73001, outcome=y)
                audit.update(r_audit)
            designs = [to_design(draw) for draw in draws]
            effects = pool_designs(designs, y, scale, target)
            effects["method"] = method
            raw_truth = np.expm1(np.nan_to_num(truth))
            raw_draws = [np.expm1(draw) for draw in draws]
            if not all(np.isfinite(draw).all() for draw in raw_draws):
                raise ValueError("nonfinite inverse-transformed imputation")
            predicted = np.mean(raw_draws, 0)
            marker_sd = np.array([np.std(raw_truth[:, :, j][source[:, :, j]], ddof=1) for j in range(6)])
            normalized_error = (predicted - raw_truth) / marker_sd
            summaries = np.mean([d.to_numpy() for d in designs], 0)[:, 9:15]
            summary_error = (summaries - reference_design.to_numpy()[:, 9:15]) / scale[9:15]
            probabilities = []
            for design in designs:
                classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                classifier.fit(design, stage)
                probabilities.append(classifier.predict_proba(design))
            prob = np.mean(probabilities, 0)
            onehot = np.eye(3)[stage]
            rows.append(dict(method=method, status="valid", reference_interval_inclusion=float(effects.covered.mean()),
                coefficient_mse=float(effects.squared_error.mean()),
                mean_absolute_error=float(effects.signed_error.abs().mean()),
                imputation_rmse=float(np.sqrt(np.mean((predicted - raw_truth)[hidden] ** 2))),
                imputation_nrmse=float(np.sqrt(np.mean(normalized_error[hidden] ** 2))),
                trajectory_nrmse=float(np.sqrt(np.mean(summary_error ** 2))),
                apparent_auroc=float(roc_auc_score(stage, prob, multi_class="ovr", average="macro")),
                apparent_auprc=float(average_precision_score(onehot, prob, average="macro")),
                apparent_log_loss=float(log_loss(stage, prob))))
            terms.extend(effects.to_dict("records"))
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rows.append(dict(method=method, status="invalid", reason=str(exc)))
        rows[-1]["seconds"] = time.monotonic() - started
        rows[-1]["input_hash"] = input_hash
        audits.append(audit)
        atomic_json(output / "partial.json", dict(job=job, metrics=rows, coefficients=terms, audit=audits))
    result = dict(job=job, metrics=rows, coefficients=terms, audit=audits, cohort_counts=counts,
        interpretation="Repeated-mask reference-estimate inclusion and apparent classification; not population coverage",
        reference_terms=dict(zip(columns, target.tolist())),
        realized_missing_percentage=float(100 * hidden.sum() / eligible.sum()))
    atomic_json(output / "result.json", result)
    return result
