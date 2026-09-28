#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_package_default_association_comparators as association  # noqa: E402
from package_default_imputation import run_package_imputer  # noqa: E402


SCENARIO = "baseline_harder"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_one(task: tuple[int, float, int, int]) -> tuple[list[dict], dict, list[dict]]:
    seed, missing_rate, n_subjects, n_imputations = task
    data = association.generate_locked_dataset(
        seed, SCENARIO, n_subjects, missing_rate
    )
    frame = pd.concat(
        [
            data["late_observed"].reset_index(drop=True),
            data["base"].reset_index(drop=True),
        ],
        axis=1,
    )
    draws: list[pd.DataFrame] = []
    audits: list[dict] = []
    for draw_index in range(n_imputations):
        draw_seed = seed + 720_260 + 104_729 * draw_index
        completed, metadata = run_package_imputer(
            frame,
            association.TARGET_COLUMNS,
            "missforest",
            1,
            draw_seed,
        )
        draws.extend(completed)
        audits.append(
            {
                "seed": seed,
                "missing_rate": missing_rate,
                "scenario": SCENARIO,
                "draw": draw_index + 1,
                "draw_seed": draw_seed,
                **metadata,
            }
        )

    coefficients, metrics = association.score_method(
        data,
        draws,
        seed,
        SCENARIO,
        "missforest",
        missing_rate,
    )
    return coefficients, metrics, audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--nsim", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=1_200_000)
    parser.add_argument("--missing-rates", nargs="+", type=float, default=[0.20, 0.40, 0.60])
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    coefficient_path = args.outdir / "coefficient_by_run.csv"
    metric_path = args.outdir / "metrics_by_run.csv"
    audit_path = args.outdir / "package_audit.csv"
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
    audit_rows = (
        pd.read_csv(audit_path).to_dict("records")
        if args.resume and audit_path.exists()
        else []
    )
    completed = {
        (int(row["seed"]), float(row["target_missing"])) for row in metric_rows
    }
    tasks = [
        (seed, missing_rate, args.n, args.m)
        for missing_rate in args.missing_rates
        for seed in range(args.seed_start, args.seed_start + args.nsim)
        if (seed, missing_rate) not in completed
    ]

    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(run_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                coefficients, metrics, audits = future.result()
            except Exception as exc:
                failures.append(
                    {"seed": task[0], "missing_rate": task[1], "error": repr(exc)}
                )
                atomic_csv(pd.DataFrame(failures), args.outdir / "failed_tasks.csv")
                print(
                    f"Missforest-MI failed seed={task[0]} rate={task[1]:.0%}: {exc}",
                    flush=True,
                )
                continue
            coefficient_rows.extend(coefficients)
            metric_rows.append(metrics)
            audit_rows.extend(audits)
            if index == 1 or index % 5 == 0 or index == len(tasks):
                atomic_csv(pd.DataFrame(coefficient_rows), coefficient_path)
                atomic_csv(pd.DataFrame(metric_rows), metric_path)
                atomic_csv(pd.DataFrame(audit_rows), audit_path)
                print(
                    f"Missforest five-draw pooling {index}/{len(tasks)} completed tasks",
                    flush=True,
                )

    coefficients = pd.DataFrame(coefficient_rows).drop_duplicates(
        ["seed", "target_missing", "coefficient"], keep="last"
    )
    metrics = pd.DataFrame(metric_rows).drop_duplicates(
        ["seed", "target_missing", "method"], keep="last"
    )
    audits = pd.DataFrame(audit_rows).drop_duplicates(
        ["seed", "missing_rate", "draw"], keep="last"
    )
    atomic_csv(coefficients, coefficient_path)
    atomic_csv(metrics, metric_path)
    atomic_csv(audits, audit_path)
    by_term, overall = association.run_large.summarize_effects(coefficients, metrics)
    atomic_csv(by_term, args.outdir / "effect_recovery_by_predictor.csv")
    atomic_csv(overall, args.outdir / "effect_recovery_overall.csv")

    config = {
        "experiment": "clinical association simulation",
        "scenario": SCENARIO,
        "n_subjects": args.n,
        "n_runs_per_missingness_level": args.nsim,
        "missingness_levels": args.missing_rates,
        "seed_start": args.seed_start,
        "method_label": "Missforest",
        "n_imputations": args.m,
        "completion_rule": "five independently seeded calls to package-default missForest",
        "pooling": "downstream model refitted in every completion and combined with Rubin rules",
        "missforest_defaults": "maxiter=10, ntree=100, mtry=floor(sqrt(p)), no predictor preselection",
    }
    (args.outdir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
