"""Run same-cohort observed-cell masking with source-relative coefficient recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation import brits  # noqa: E402
from validation import evaluation as ev  # noqa: E402
from validation.data import external_path, load_cohort  # noqa: E402

LABELS = {
    "brits": "BRITS-MI",
    "mice": "MICE",
    "missforest": "missForest",
    "mean": "Mean imputation",
}
CHANNELS = [3, 4, 5]


def package_draws(values, mask, static, method, m, seed):
    # Share the same package-native R wrapper as the clinical simulations.
    source = ROOT / "simulation/clinical/scripts"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from package_default_imputation import run_package_imputer

    frame, targets = ev.build_frame(values, mask, static)
    frames, audit = run_package_imputer(frame, targets, method, m, seed)
    return [ev.completed_array(values, mask, frame) for frame in frames], audit


def mean_draw(values, mask):
    counts = mask.sum(axis=0)
    means = np.divide(
        (values * mask).sum(axis=0), counts, out=np.zeros_like(counts), where=counts > 0
    )
    return np.where(mask > 0.5, values, means[None, :, :])


def atomic_csv(frame, path):
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_summaries(metrics, coefficients, output):
    group = ["masking_percent", "method"]
    names = [
        "effect_abs_bias",
        "effect_mse",
        "effect_coverage",
        "masked_cell_rmse",
        "masked_cell_standardized_rmse",
        "trajectory_summary_rmse",
        "trajectory_summary_nrmse",
        "auroc",
        "auprc",
        "log_loss",
    ]
    rows = []
    for keys, part in metrics.groupby(group):
        for name in names:
            x = part[name].dropna()
            rows.append(
                dict(zip(group, keys))
                | {
                    "metric": name,
                    "mean": x.mean(),
                    "sd": x.std(ddof=1),
                    "mcse": x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                    "n_valid_repeats": len(x),
                    "n_attempted_repeats": len(part),
                }
            )
    atomic_csv(pd.DataFrame(rows), output / "summary.csv")
    if not coefficients.empty:
        term = coefficients.groupby(group + ["term"], as_index=False).agg(
            signed_deviation=("bias", "mean"),
            mse=("squared_error", "mean"),
            coverage=("covered", "mean"),
            mean_model_se=("std_error", "mean"),
            repeated_mask_sd=("estimate", "std"),
            n_valid_repeats=("repeat", "nunique"),
        )
        atomic_csv(term, output / "term_summary.csv")


def run(args):
    source = external_path(args.input_dir, ROOT)
    output = external_path(args.output, ROOT)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Keep validation output separate from the input directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory; existing analyses are not overwritten")
    if {"mice", "missforest"}.intersection(args.methods) and shutil.which("Rscript") is None:
        raise RuntimeError("MICE and missForest require Rscript and the corresponding R packages")
    cohort = load_cohort(source)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    reference = ev.reference_coefficients(cohort.clinical, cohort.values, cohort.mask)
    settings = {k: v for k, v in vars(args).items() if k not in {"input_dir", "output"}}
    settings.update(
        analysis="same-cohort observed-cell masking",
        coefficient_reference="pre-mask fit",
        natural_missing_cells_scored=False,
        bootstrap_models=True,
        residual_noise=0.0,
        input_sha256={
            name: file_hash(source / name) for name in ["clinical.csv", "longitudinal.npz"]
        },
    )
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    atomic_csv(
        pd.DataFrame({"term": list(reference), "reference": list(reference.values())}),
        output / "reference_coefficients.csv",
    )
    models, histories = [], []
    if "brits" in args.methods:
        target = brits.reference_target(cohort.clinical, cohort.values, cohort.mask)
        for index in range(args.ensemble_size):
            seed = args.training_seed + 1009 * index
            population = np.random.default_rng(seed + 41003).integers(
                0, len(cohort.values), size=len(cohort.values)
            )
            model, history = brits.train_model(
                cohort.standardized,
                cohort.mask,
                cohort.static,
                target,
                cohort.labels,
                cohort.marker_mean,
                cohort.marker_sd,
                CHANNELS,
                seed,
                args.epochs,
                args.hidden_size,
                args.batch_size,
                args.lambda_summary,
                args.lambda_downstream,
                args.lambda_outcome,
                0.0,
                0.0,
                1.0,
                False,
                population,
            )
            model.eval()
            models.append(model)
            histories.append(history.assign(model_index=index))
            atomic_csv(pd.concat(histories, ignore_index=True), output / "training_history.csv")
    metric_rows, coefficient_rows, audits = [], [], []
    for percent in args.missing_percentages:
        fraction = percent / 100
        for repeat in range(args.repeats):
            seed = args.mask_seed + int(round(1000 * fraction)) + repeat
            hidden = ev._hide_observed_cells(
                cohort.mask, fraction, np.random.default_rng(seed), CHANNELS
            )
            mask = np.where(hidden, 0, cohort.mask)
            values = np.where(hidden, 0, cohort.standardized)
            for method in args.methods:
                if method == "brits":
                    draws = brits.stochastic_draws(
                        models,
                        [{} for _ in models],
                        values,
                        mask,
                        cohort.static,
                        CHANNELS,
                        len(models),
                        0,
                        seed,
                    )
                elif method == "mean":
                    draws = [mean_draw(values, mask)]
                else:
                    # Preserve the original method-specific seeds even when running one arm.
                    offset = 1009 if method == "missforest" else 0
                    draws, audit = package_draws(
                        values,
                        mask,
                        cohort.static,
                        method,
                        args.mice_imputations,
                        seed + 58103 + offset,
                    )
                    audits.append(
                        {
                            "seed": seed,
                            "masking_percent": percent,
                            "method": LABELS[method],
                            **audit,
                        }
                    )
                raw = [draw * cohort.marker_sd + cohort.marker_mean for draw in draws]
                if any(not np.isfinite(draw).all() for draw in raw):
                    raise RuntimeError(f"Nonfinite imputed values: {method}, repeat {repeat}")
                mean_raw = np.mean(raw, axis=0)
                terms, effects = ev.pooled_effect_rows(
                    cohort.clinical,
                    raw,
                    cohort.values,
                    cohort.mask,
                    hidden,
                    reference,
                    seed,
                    repeat,
                    fraction,
                    LABELS[method],
                    LABELS[method],
                )
                coefficient_rows.extend(terms)
                metric_rows.append(
                    {
                        "seed": seed,
                        "repeat": repeat,
                        "masking_percent": percent,
                        "method": LABELS[method],
                        "n_imputations": len(draws),
                        **effects,
                        **ev._masked_metrics(mean_raw, cohort.values, hidden),
                        **ev.trajectory_summary_metrics(
                            mean_raw, cohort.values, cohort.mask, hidden, CHANNELS
                        ),
                        **ev.pooled_apparent_prediction(raw, mask, cohort.static, cohort.labels),
                    }
                )
            metrics = pd.DataFrame(metric_rows)
            coefficients = pd.DataFrame(coefficient_rows)
            atomic_csv(metrics, output / "metrics_by_repeat.csv")
            atomic_csv(coefficients, output / "coefficients_by_repeat.csv")
            if audits:
                atomic_csv(pd.DataFrame(audits), output / "package_audit.csv")
            print(f"Masking {percent}%: repeat {repeat + 1}/{args.repeats}", flush=True)
    write_summaries(metrics, coefficients, output)
    return metrics, coefficients


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--missing-percentages", type=int, nargs="+", default=[20, 40, 60])
    parser.add_argument("--methods", nargs="+", choices=list(LABELS), default=list(LABELS))
    parser.add_argument("--ensemble-size", type=int, default=10)
    parser.add_argument("--mice-imputations", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-seed", type=int, default=210000)
    parser.add_argument("--mask-seed", type=int, default=110000)
    parser.add_argument("--lambda-summary", type=float, default=0.6666666667)
    parser.add_argument("--lambda-downstream", type=float, default=0.5)
    parser.add_argument("--lambda-outcome", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if (
        min(
            args.repeats,
            args.ensemble_size,
            args.mice_imputations,
            args.epochs,
            args.hidden_size,
            args.batch_size,
            args.threads,
        )
        < 1
    ):
        parser.error("Counts must be positive")
    if any(not 0 < p < 100 for p in args.missing_percentages):
        parser.error("Missing percentages must be between 0 and 100")
    if len(set(args.missing_percentages)) != len(args.missing_percentages):
        parser.error("Missing percentages must be unique")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Methods must be unique")
    if (
        min(
            args.lambda_summary,
            args.lambda_downstream,
            args.lambda_outcome,
            args.mask_seed,
            args.training_seed,
        )
        < 0
    ):
        parser.error("Seeds and loss weights must be nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
