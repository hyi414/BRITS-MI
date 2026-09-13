#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import finalize_clinical_association_nsim200 as finalizer  # noqa: E402


N_SUBJECTS = 2000
N_RUNS = 100
RATES = (20, 40, 60)
MISSFOREST_DIR = (
    ROOT / "outputs" / "clinical_association_missforest5_n2000_nsim100_20260802"
)


def source_paths(rate: int) -> dict[str, list[Path]]:
    return {
        "proposed": [
            ROOT
            / "outputs"
            / f"refined_brits_mi_assoc_n2000_nsim100_m50_miss{rate}_20260802"
        ],
        "package": [
            ROOT
            / "outputs"
            / f"package_default_mice_assoc_n2000_nsim100_m50_miss{rate}_20260802"
        ],
        "simple": [
            ROOT
            / "outputs"
            / f"simple_assoc_n2000_nsim100_miss{rate}_20260802"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=(
            ROOT / "outputs" / "clinical_association_n2000_nsim100_final_20260802"
        ),
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    finalizer.source_paths = source_paths
    finalizer.MISSFOREST5_DIR = MISSFOREST_DIR
    coefficient, metrics = finalizer.load_data()
    coefficient, metrics, complete_case_audit = (
        finalizer.exclude_unstable_complete_case(coefficient, metrics)
    )
    coefficient.to_csv(args.outdir / "coefficient_by_run.csv", index=False)
    metrics.to_csv(args.outdir / "metrics_by_run.csv", index=False)
    complete_case_audit.to_csv(
        args.outdir / "complete_case_estimability_audit.csv", index=False
    )
    summary, run_level = finalizer.summarize(
        coefficient, metrics, args.n_bootstrap
    )
    summary.to_csv(
        args.outdir / "summary_with_monte_carlo_uncertainty.csv", index=False
    )
    run_level.to_csv(args.outdir / "effect_metrics_by_run.csv", index=False)
    finalizer.paired_tests(run_level).to_csv(
        args.outdir / "paired_effect_tests.csv", index=False
    )
    manifest = {
        "experiment": "clinical association simulation sample-size sensitivity",
        "n_subjects": N_SUBJECTS,
        "n_runs_per_missingness_level": N_RUNS,
        "missingness_levels": [0.20, 0.40, 0.60],
        "scenario": "baseline_harder",
        "seed_start": 1_200_000,
        "brits_mi_imputations": 50,
        "mice": "R mice package defaults with m=50 and maxit=5",
        "missforest": "Five independently seeded R missForest package-default completions with downstream refitting and Rubin pooling",
        "predictor_preselection": "none for package-default MICE or Missforest",
        "methods": finalizer.METHOD_ORDER,
    }
    (args.outdir / "config.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        summary.sort_values(["metric", "missing_rate", "method_label"]).to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
