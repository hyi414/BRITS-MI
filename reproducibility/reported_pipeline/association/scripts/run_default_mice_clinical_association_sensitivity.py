#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_effect_recovery_harder_irregularity_test as harder  # noqa: E402


run_large = harder.run_large
sim = run_large.sim
dgp1 = run_large.dgp1
hist_sim = run_large.hist_sim
TRUE_BETA = run_large.TRUE_BETA
COEFFICIENT_LABELS = run_large.COEFFICIENT_LABELS
SCENARIOS = harder.HARDER_SCENARIOS

TARGET_COLUMNS = [
    "ast_v4",
    "alt_v4",
    "platelet_v4",
    "ast_v5",
    "alt_v5",
    "platelet_v5",
    "ast_v6",
    "alt_v6",
    "platelet_v6",
]


def generate_locked_dataset(seed: int, scenario: str, n: int, target_missing: float) -> dict:
    rng = np.random.default_rng(seed)
    diabetes = rng.binomial(1, 0.36, size=n).astype(float)
    hypertension = rng.binomial(
        1,
        sim._sigmoid(-0.20 + 1.05 * diabetes),
        size=n,
    ).astype(float)
    male = rng.binomial(1, 0.46, size=n).astype(float)
    trajectory, time, latent = sim._simulate_irregular_trajectory(
        rng,
        diabetes,
        hypertension,
        male,
        scenario,
    )
    true_features = dgp1._trajectory_features(trajectory, time)
    true_design = dgp1._design(true_features, diabetes, hypertension, male)
    eta = -0.84 + sum(
        TRUE_BETA[name] * true_design[name].to_numpy()
        for name in TRUE_BETA
    )
    outcome = rng.binomial(1, sim._sigmoid(eta)).astype(int)
    observed = sim._make_missingness_targeted(
        trajectory,
        outcome,
        diabetes,
        hypertension,
        latent,
        time,
        rng,
        target_missing,
        scenario,
    )
    trajectory_obs = np.where(observed, trajectory, np.nan)
    base = dgp1._build_base_frame(
        trajectory_obs,
        observed,
        diabetes,
        hypertension,
        male,
        time,
    )
    late_observed = hist_sim._cell_frame(trajectory_obs)
    return {
        "diabetes": diabetes,
        "hypertension": hypertension,
        "male": male,
        "trajectory": trajectory,
        "time": time,
        "outcome": outcome,
        "observed": observed,
        "trajectory_obs": trajectory_obs,
        "true_features": true_features,
        "base": base,
        "late_observed": late_observed,
    }


def run_r_mice(
    frame: pd.DataFrame,
    m: int,
    seed: int,
    r_script: Path,
) -> tuple[list[pd.DataFrame], dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="brits_mi_default_mice_") as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / "input.csv"
        output_path = tmpdir / "completed.csv"
        metadata_path = tmpdir / "metadata.txt"
        frame.to_csv(input_path, index=False)
        completed = subprocess.run(
            [
                "Rscript",
                str(r_script),
                str(input_path),
                str(output_path),
                str(metadata_path),
                str(m),
                str(seed),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"R mice failed for seed {seed}: {completed.stderr.strip()}"
            )
        long = pd.read_csv(output_path)
        metadata = {}
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            key, value = line.split("=", 1)
            metadata[key] = value
        draws = []
        for draw_id in sorted(long[".imp"].unique()):
            draw = long.loc[long[".imp"].eq(draw_id), TARGET_COLUMNS].reset_index(drop=True)
            draws.append(draw)
        return draws, metadata


def run_one(task: tuple) -> tuple[list[dict], dict, dict]:
    seed, scenario, n, m, target_missing, r_script = task
    data = generate_locked_dataset(seed, scenario, n, target_missing)
    mice_frame = pd.concat(
        [
            data["late_observed"].reset_index(drop=True),
            data["base"].reset_index(drop=True),
        ],
        axis=1,
    )
    cell_draws, metadata = run_r_mice(
        mice_frame,
        m,
        seed + 720260,
        r_script,
    )
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
        raise RuntimeError(f"Downstream fit failed for seed {seed}, {scenario}")
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
                "n_imputations": m,
                "method": "mice_default_pmm",
                "method_label": "Default MICE (PMM)",
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
        "n_imputations": m,
        "method": "mice_default_pmm",
        "method_label": "Default MICE (PMM)",
        "imputation_rmse": imputation_rmse,
        "trajectory_summary_rmse": trajectory_summary_rmse,
        "coefficient_fit_success": 1.0,
        "coefficient_n_used": int(n_used),
        "coefficient_n_fit_draws": int(n_fit_draws),
    }
    audit_row = {
        "seed": seed,
        "scenario": scenario,
        "mice_version": metadata.get("mice_version", ""),
        "methods": metadata.get("methods", ""),
        "logged_events": int(metadata.get("logged_events", "0")),
    }
    return coefficient_rows, metric_row, audit_row


def paired_comparison(coef: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    per_seed_coef = coef.groupby(
        ["scenario", "scenario_label", "method", "method_label", "seed"],
        as_index=False,
    ).agg(
        mean_abs_run_bias=("abs_bias", "mean"),
        coverage=("covered", "mean"),
        average_se=("std_error", "mean"),
    )
    per_seed = per_seed_coef.merge(
        metrics[
            [
                "scenario",
                "method",
                "seed",
                "imputation_rmse",
                "trajectory_summary_rmse",
            ]
        ],
        on=["scenario", "method", "seed"],
        how="left",
    )
    rows = []
    measures = [
        "mean_abs_run_bias",
        "coverage",
        "average_se",
        "imputation_rmse",
        "trajectory_summary_rmse",
    ]
    for scenario, group in per_seed.groupby("scenario", sort=False):
        wide = group.pivot(index="seed", columns="method", values=measures)
        for measure in measures:
            default = wide[(measure, "mice_default_pmm")]
            custom = wide[(measure, "mice_custom_gaussian")]
            paired = pd.concat([default, custom], axis=1).dropna()
            paired.columns = ["default", "custom"]
            difference = paired["default"] - paired["custom"]
            n_pairs = len(difference)
            se = float(difference.std(ddof=1) / np.sqrt(n_pairs))
            critical = float(stats.t.ppf(0.975, df=n_pairs - 1))
            test = stats.ttest_rel(paired["default"], paired["custom"])
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": group["scenario_label"].iloc[0],
                    "measure": measure,
                    "n_pairs": n_pairs,
                    "default_mean": float(paired["default"].mean()),
                    "custom_mean": float(paired["custom"].mean()),
                    "mean_difference_default_minus_custom": float(difference.mean()),
                    "difference_ci_low": float(difference.mean() - critical * se),
                    "difference_ci_high": float(difference.mean() + critical * se),
                    "paired_t_p_value": float(test.pvalue),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--poster-output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--nsim", type=int, default=50)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--target-missing", type=float, default=0.20)
    parser.add_argument("--seed-start", type=int, default=993000)
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    starts = {
        "mild_harder": args.seed_start,
        "baseline_harder": args.seed_start + 100000,
        "strong_harder": args.seed_start + 200000,
    }
    r_script = ROOT / "scripts" / "run_default_mice_association_one.R"
    tasks = [
        (start + offset, scenario, args.n, args.m, args.target_missing, r_script)
        for scenario, start in starts.items()
        for offset in range(args.nsim)
    ]

    coefficient_rows: list[dict] = []
    metric_rows: list[dict] = []
    audit_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        for index, (rows, metric, audit) in enumerate(executor.map(run_one, tasks), 1):
            coefficient_rows.extend(rows)
            metric_rows.append(metric)
            audit_rows.append(audit)
            print(
                f"[{index}/{len(tasks)}] {metric['scenario']} seed={metric['seed']} "
                f"cell_RMSE={metric['imputation_rmse']:.3f} "
                f"summary_RMSE={metric['trajectory_summary_rmse']:.3f}",
                flush=True,
            )

    default_coef = pd.DataFrame(coefficient_rows)
    default_metrics = pd.DataFrame(metric_rows)
    audit = pd.DataFrame(audit_rows)
    poster_coef = pd.read_csv(args.poster_output / "coefficient_by_run.csv")
    poster_metrics = pd.read_csv(args.poster_output / "metrics_by_run.csv")
    selected_seeds = {seed for task in tasks for seed in [task[0]]}
    custom_coef = poster_coef.loc[
        poster_coef["method"].eq("mice") & poster_coef["seed"].isin(selected_seeds)
    ].copy()
    custom_metrics = poster_metrics.loc[
        poster_metrics["method"].eq("mice") & poster_metrics["seed"].isin(selected_seeds)
    ].copy()
    custom_coef["method"] = "mice_custom_gaussian"
    custom_coef["method_label"] = "Poster MICE (custom Gaussian)"
    custom_metrics["method"] = "mice_custom_gaussian"
    custom_metrics["method_label"] = "Poster MICE (custom Gaussian)"

    combined_coef = pd.concat([custom_coef, default_coef], ignore_index=True)
    combined_metrics = pd.concat([custom_metrics, default_metrics], ignore_index=True)
    by_term, summary = run_large.summarize_effects(combined_coef, combined_metrics)
    paired = paired_comparison(combined_coef, combined_metrics)

    combined_coef.to_csv(args.outdir / "coefficient_by_run.csv", index=False)
    combined_metrics.to_csv(args.outdir / "metrics_by_run.csv", index=False)
    audit.to_csv(args.outdir / "default_mice_audit.csv", index=False)
    by_term.to_csv(args.outdir / "mice_comparison_by_predictor.csv", index=False)
    summary.to_csv(args.outdir / "mice_comparison_summary.csv", index=False)
    paired.to_csv(args.outdir / "paired_method_differences.csv", index=False)
    config = {
        "n": args.n,
        "nsim_per_scenario": args.nsim,
        "m": args.m,
        "target_missing": args.target_missing,
        "seed_starts": starts,
        "scenarios": list(starts),
        "poster_output": str(args.poster_output),
        "default_mice": {
            "package": "mice",
            "method": "package defaults (predictive mean matching for incomplete continuous variables)",
            "maxit": 5,
            "m": args.m,
            "predictor_preselection": "none; default predictor matrix",
            "outcome_in_imputation_model": False,
        },
        "poster_mice": {
            "method": "custom two-iteration chained Gaussian linear regression",
            "m": args.m,
            "predictor_preselection": "none",
            "outcome_in_imputation_model": False,
        },
    }
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nMatched 50-seed comparison\n")
    display_columns = [
        "scenario_label",
        "method_label",
        "mean_abs_signed_bias",
        "coverage",
        "empirical_se",
        "average_se",
        "se_ratio",
        "imputation_rmse_mean",
        "trajectory_summary_rmse_mean",
        "n_runs",
    ]
    print(summary[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPaired differences: default MICE minus poster MICE\n")
    print(paired.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
