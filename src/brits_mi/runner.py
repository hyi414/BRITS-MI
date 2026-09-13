"""Reproducible simulation runners built on the package API."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .analysis import (
    coefficient_recovery,
    imputation_recovery,
    pool_logistic_regressions,
)
from .config import TrainingConfig
from .imputer import BRITSMultipleImputer
from .simulation import TRUE_BETA, SimulationConfig, simulate_clinical_association


def run_replicate(
    simulation: SimulationConfig,
    training: TrainingConfig,
    n_imputations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one known-truth BRITS-MI replicate and return tidy metrics."""

    data = simulate_clinical_association(simulation)
    imputer = BRITSMultipleImputer(replace(training, seed=simulation.seed))
    imputer.fit(
        data.observed_values,
        data.observed_mask,
        data.times,
        data.static_covariates,
        data.outcome,
    )
    completions = imputer.sample(n_imputations, seed=simulation.seed + 120001)
    pooled = pool_logistic_regressions(
        completions,
        data.times,
        data.static_covariates,
        data.outcome,
    )
    coefficients = coefficient_recovery(pooled, TRUE_BETA)
    coefficients.insert(0, "seed", simulation.seed)
    coefficients.insert(1, "scenario", simulation.scenario)
    coefficients.insert(2, "target_missing", simulation.target_missing)
    coefficients.insert(3, "method", "BRITS-MI")

    draw_metrics = imputation_recovery(
        completions,
        data.true_values,
        data.observed_mask,
        data.times,
    )
    metrics = pd.DataFrame(
        [
            {
                "seed": simulation.seed,
                "scenario": simulation.scenario,
                "target_missing": simulation.target_missing,
                "method": "BRITS-MI",
                "realized_missing_all": float((~data.observed_mask).mean()),
                "realized_missing_late": float((~data.observed_mask[:, -3:, :]).mean()),
                "missing_cell_nrmse": float(draw_metrics["missing_cell_nrmse"].mean()),
                "trajectory_summary_nrmse": float(draw_metrics["trajectory_summary_nrmse"].mean()),
                "mean_absolute_coefficient_error": float(coefficients["absolute_error"].mean()),
                "coefficient_mse": float(coefficients["squared_error"].mean()),
                "coverage": float(coefficients["covered"].mean()),
                "best_epoch": int(imputer.fit_summary_.best_epoch),
                "best_validation_loss": float(imputer.fit_summary_.best_validation_loss),
                "n_imputations": n_imputations,
            }
        ]
    )
    return coefficients, metrics


def run_simulation_grid(
    output: str | Path,
    n_runs: int,
    n_subjects: int,
    missing_rates: Iterable[float],
    scenario: str,
    n_imputations: int,
    training: TrainingConfig,
    first_seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run and save a deterministic grid over missingness rates and seeds."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    coefficient_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    rates = [float(rate) for rate in missing_rates]
    for rate_index, rate in enumerate(rates):
        for replicate in range(n_runs):
            seed = first_seed + rate_index * 100000 + replicate
            coefficients, metrics = run_replicate(
                SimulationConfig(
                    n_subjects=n_subjects,
                    target_missing=rate,
                    scenario=scenario,
                    seed=seed,
                ),
                training,
                n_imputations,
            )
            coefficient_frames.append(coefficients)
            metric_frames.append(metrics)
    coefficient_results = pd.concat(coefficient_frames, ignore_index=True)
    metric_results = pd.concat(metric_frames, ignore_index=True)
    coefficient_results.to_csv(output / "coefficient_recovery.csv", index=False)
    metric_results.to_csv(output / "replicate_metrics.csv", index=False)
    summary = metric_results.groupby(["scenario", "target_missing", "method"], as_index=False).agg(
        mean_absolute_coefficient_error=("mean_absolute_coefficient_error", "mean"),
        coefficient_mse=("coefficient_mse", "mean"),
        coverage=("coverage", "mean"),
        missing_cell_nrmse=("missing_cell_nrmse", "mean"),
        trajectory_summary_nrmse=("trajectory_summary_nrmse", "mean"),
        n_runs=("seed", "nunique"),
    )
    term_summary = summarize_coefficients(coefficient_results)
    systematic = term_summary.groupby(["scenario", "target_missing", "method"], as_index=False).agg(
        mean_absolute_bias=("absolute_bias", "mean")
    )
    summary = summary.merge(
        systematic, on=["scenario", "target_missing", "method"], validate="one_to_one"
    )
    term_summary.to_csv(output / "term_summary.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    manifest = {
        "simulation": {
            "n_runs": n_runs,
            "n_subjects": n_subjects,
            "missing_rates": rates,
            "scenario": scenario,
            "n_imputations": n_imputations,
            "first_seed": first_seed,
        },
        "training": asdict(training),
        "outputs": [
            "coefficient_recovery.csv",
            "replicate_metrics.csv",
            "term_summary.csv",
            "summary.csv",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return coefficient_results, metric_results


def summarize_coefficients(coefficient_results):
    """Aggregate runs before taking absolute bias; retain each predictor term."""
    groups = ["scenario", "target_missing", "method", "term"]
    table = coefficient_results.groupby(groups, as_index=False).agg(
        signed_bias=("signed_error", "mean"),
        coefficient_mse=("squared_error", "mean"),
        coverage=("covered", "mean"),
        empirical_se=("estimate", "std"),
        mean_model_se=("standard_error", "mean"),
        n_runs=("seed", "nunique"),
    )
    table["absolute_bias"] = table["signed_bias"].abs()
    return table
