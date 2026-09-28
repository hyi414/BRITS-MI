#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_effect_recovery_harder_irregularity_test as harder  # noqa: E402
from package_default_imputation import run_package_imputer  # noqa: E402
from run_default_mice_clinical_association_sensitivity import (  # noqa: E402
    generate_locked_dataset,
)


run_large = harder.run_large
dgp1 = run_large.dgp1
hist_sim = run_large.hist_sim
TRUE_BETA = run_large.TRUE_BETA
COEFFICIENT_LABELS = run_large.COEFFICIENT_LABELS
SCENARIOS = harder.HARDER_SCENARIOS
TARGET_COLUMNS = [
    "ast_v4", "alt_v4", "platelet_v4",
    "ast_v5", "alt_v5", "platelet_v5",
    "ast_v6", "alt_v6", "platelet_v6",
]
METHOD_LABELS = {
    "mice": "MICE",
    "missforest": "Missforest",
}


def score_method(
    data: dict,
    cell_draws: list[pd.DataFrame],
    seed: int,
    scenario: str,
    method: str,
    target_missing: float,
) -> tuple[list[dict], dict]:
    trajectories = [
        hist_sim._frame_to_late_trajectory(draw, data["trajectory_obs"])
        for draw in cell_draws
    ]
    feature_draws = [
        dgp1._trajectory_features(trajectory, data["time"])
        for trajectory in trajectories
    ]
    pooled = dgp1._pool_logit_fits(
        feature_draws,
        data["outcome"],
        data["diabetes"],
        data["hypertension"],
        data["male"],
    )
    if pooled is None:
        raise RuntimeError(f"Downstream fit failed for seed {seed}, {scenario}, {method}")
    params, bse, within_se, between_sd, ci, n_used, n_fit_draws = pooled
    coefficient_rows = []
    for name, truth in TRUE_BETA.items():
        lo, hi = ci.loc[name]
        coefficient_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "scenario_label": SCENARIOS[scenario]["label"],
                "target_missing": target_missing,
                "n_imputations": len(cell_draws),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "coefficient": name,
                "coefficient_label": COEFFICIENT_LABELS[name],
                "true_value": truth,
                "estimate": float(params[name]),
                "std_error": float(bse[name]),
                "within_std_error": float(within_se[name]),
                "between_imputation_sd": float(between_sd[name]),
                "bias": float(params[name] - truth),
                "abs_bias": float(abs(params[name] - truth)),
                "squared_error": float((params[name] - truth) ** 2),
                "covered": float(lo <= truth <= hi),
                "n_used": int(n_used),
                "n_fit_draws": int(n_fit_draws),
            }
        )

    missing_late = ~data["observed"][:, -3:, :]
    true_late = data["trajectory"][:, -3:, :]
    completed_mean = np.mean(
        np.stack([trajectory[:, -3:, :] for trajectory in trajectories], axis=0),
        axis=0,
    )
    imputation_rmse = float(
        np.sqrt(np.mean((completed_mean[missing_late] - true_late[missing_late]) ** 2))
    )
    summary_mean = np.mean(
        np.stack([features.to_numpy() for features in feature_draws], axis=0),
        axis=0,
    )
    trajectory_summary_rmse = float(
        np.sqrt(np.mean((summary_mean - data["true_features"].to_numpy()) ** 2))
    )
    metric_row = {
        "seed": seed,
        "scenario": scenario,
        "scenario_label": SCENARIOS[scenario]["label"],
        "target_missing": target_missing,
        "n_imputations": len(cell_draws),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "imputation_rmse": imputation_rmse,
        "trajectory_summary_rmse": trajectory_summary_rmse,
        "coefficient_fit_success": 1.0,
        "coefficient_n_used": int(n_used),
        "coefficient_n_fit_draws": int(n_fit_draws),
    }
    return coefficient_rows, metric_row


def run_one(task: tuple) -> tuple[list[dict], list[dict], list[dict]]:
    seed, scenario, n, m_values, target_missing, methods = task
    data = generate_locked_dataset(seed, scenario, n, target_missing)
    frame = pd.concat(
        [data["late_observed"].reset_index(drop=True), data["base"].reset_index(drop=True)],
        axis=1,
    )
    coefficient_rows: list[dict] = []
    metric_rows: list[dict] = []
    audit_rows: list[dict] = []
    for method_index, method in enumerate(methods):
        requested_m = max(m_values) if method == "mice" else 1
        draws, metadata = run_package_imputer(
            frame,
            TARGET_COLUMNS,
            method,
            requested_m,
            seed + 720260 + 1009 * method_index,
        )
        evaluation_sizes = m_values if method == "mice" else [1]
        for m_value in evaluation_sizes:
            rows, metric = score_method(
                data,
                draws[:m_value],
                seed,
                scenario,
                method,
                target_missing,
            )
            coefficient_rows.extend(rows)
            metric_rows.append(metric)
        audit_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "method": method,
                "requested_m": requested_m,
                **metadata,
            }
        )
    return coefficient_rows, metric_rows, audit_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--nsim", type=int, default=100)
    parser.add_argument("--mice-m", type=int, default=20)
    parser.add_argument("--m-values", type=int, nargs="+", default=None)
    parser.add_argument("--target-missing", type=float, default=0.20)
    parser.add_argument("--seed-start", type=int, default=993000)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIOS),
        default=list(SCENARIOS),
    )
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_LABELS), default=list(METHOD_LABELS))
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    m_values = sorted(set(args.m_values or [args.mice_m]))
    starts = {
        "mild_harder": args.seed_start,
        "baseline_harder": args.seed_start + 100000,
        "strong_harder": args.seed_start + 200000,
    }
    methods = tuple(args.methods)
    tasks = [
        (start + offset, scenario, args.n, m_values, args.target_missing, methods)
        for scenario, start in starts.items()
        if scenario in args.scenarios
        for offset in range(args.nsim)
    ]
    coefficient_rows: list[dict] = []
    metric_rows: list[dict] = []
    audit_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        for index, (coef, metrics, audit) in enumerate(executor.map(run_one, tasks), 1):
            coefficient_rows.extend(coef)
            metric_rows.extend(metrics)
            audit_rows.extend(audit)
            if index == 1 or index % 10 == 0:
                print(f"association comparators {index}/{len(tasks)}", flush=True)

    coef_df = pd.DataFrame(coefficient_rows)
    metric_df = pd.DataFrame(metric_rows)
    audit_df = pd.DataFrame(audit_rows)
    by_term, overall = run_large.summarize_effects(coef_df, metric_df)
    coef_df.to_csv(args.outdir / "coefficient_by_run.csv", index=False)
    metric_df.to_csv(args.outdir / "metrics_by_run.csv", index=False)
    audit_df.to_csv(args.outdir / "package_audit.csv", index=False)
    by_term.to_csv(args.outdir / "effect_recovery_by_predictor.csv", index=False)
    overall.to_csv(args.outdir / "effect_recovery_overall.csv", index=False)
    config = {
        "experiment": "clinical association simulation",
        "n": args.n,
        "nsim_per_scenario": args.nsim,
        "mice_m": max(m_values),
        "m_values": m_values,
        "target_missing": args.target_missing,
        "seed_starts": starts,
        "methods": list(methods),
        "scenarios": list(args.scenarios),
        "mice": "R mice package defaults except m fixed to the manuscript value",
        "missforest": "R missForest package defaults; one completed dataset",
        "outcome_in_imputation_model": False,
        "predictor_preselection": "none beyond package automatic constant/collinearity handling",
    }
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
