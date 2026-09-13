#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_package_default_association_comparators as association  # noqa: E402


SCENARIO = "baseline_harder"
METHODS = ("full_data", "mean_impute", "complete_case")
association.METHOD_LABELS.update(
    {
        "full_data": "Full data",
        "mean_impute": "Mean imputation",
        "complete_case": "Complete case",
    }
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_one(task: tuple[int, float, int]) -> tuple[list[dict], list[dict]]:
    seed, missing_rate, n_subjects = task
    data = association.generate_locked_dataset(
        seed, SCENARIO, n_subjects, missing_rate
    )
    full = association.hist_sim._cell_frame(data["trajectory"])
    observed = data["late_observed"].copy()
    mean_completed = observed.copy()
    for column in association.TARGET_COLUMNS:
        mean_completed[column] = mean_completed[column].fillna(
            mean_completed[column].mean()
        )
    draws = {
        "full_data": [full],
        "mean_impute": [mean_completed],
        "complete_case": [observed],
    }

    coefficient_rows: list[dict] = []
    metric_rows: list[dict] = []
    for method in METHODS:
        try:
            rows, metric = association.score_method(
                data,
                draws[method],
                seed,
                SCENARIO,
                method,
                missing_rate,
            )
            coefficient_rows.extend(rows)
            metric_rows.append(metric)
        except Exception as exc:
            metric_rows.append(
                {
                    "seed": seed,
                    "scenario": SCENARIO,
                    "scenario_label": association.SCENARIOS[SCENARIO]["label"],
                    "target_missing": missing_rate,
                    "n_imputations": 1,
                    "method": method,
                    "method_label": association.METHOD_LABELS[method],
                    "imputation_rmse": np.nan,
                    "trajectory_summary_rmse": np.nan,
                    "coefficient_fit_success": 0.0,
                    "coefficient_n_used": 0,
                    "coefficient_n_fit_draws": 0,
                    "failure": repr(exc),
                }
            )
    return coefficient_rows, metric_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--nsim", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1_200_000)
    parser.add_argument(
        "--missing-rates", nargs="+", type=float, default=[0.20, 0.40, 0.60]
    )
    parser.add_argument("--n-workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    coefficient_path = args.outdir / "coefficient_by_run.csv"
    metric_path = args.outdir / "metrics_by_run.csv"
    coefficient_rows = (
        pd.read_csv(coefficient_path).to_dict("records")
        if args.resume and coefficient_path.exists()
        else []
    )
    metric_rows = (
        pd.read_csv(metric_path).to_dict("records")
        if args.resume and metric_path.exists()
        else []
    )
    completed = {
        (int(row["seed"]), float(row["target_missing"]))
        for row in metric_rows
        if row.get("method") == "mean_impute"
    }
    tasks = [
        (seed, rate, args.n)
        for rate in args.missing_rates
        for seed in range(args.seed_start, args.seed_start + args.nsim)
        if (seed, rate) not in completed
    ]

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(run_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            coefficients, metrics = future.result()
            coefficient_rows.extend(coefficients)
            metric_rows.extend(metrics)
            if index == 1 or index % 10 == 0 or index == len(tasks):
                atomic_csv(pd.DataFrame(coefficient_rows), coefficient_path)
                atomic_csv(pd.DataFrame(metric_rows), metric_path)
                print(f"simple baselines {index}/{len(tasks)}", flush=True)

    coefficients = pd.DataFrame(coefficient_rows).drop_duplicates(
        ["seed", "target_missing", "method", "coefficient"], keep="last"
    )
    metrics = pd.DataFrame(metric_rows).drop_duplicates(
        ["seed", "target_missing", "method"], keep="last"
    )
    atomic_csv(coefficients, coefficient_path)
    atomic_csv(metrics, metric_path)
    config = {
        "experiment": "clinical association simulation simple references",
        "scenario": SCENARIO,
        "n_subjects": args.n,
        "n_runs_per_missingness_level": args.nsim,
        "missingness_levels": args.missing_rates,
        "seed_start": args.seed_start,
        "methods": list(METHODS),
        "mean_imputation": "column means of observed late-cell values",
    }
    (args.outdir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
