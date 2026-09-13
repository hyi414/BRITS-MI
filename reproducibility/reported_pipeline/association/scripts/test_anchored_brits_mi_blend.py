#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_neural_brits_mi_downstream_gradient as neural  # noqa: E402
from package_default_imputation import run_package_imputer  # noqa: E402


run_large = neural.run_large
dgp1 = neural.dgp1
hist_sim = run_large.hist_sim


def missforest_anchor(data: dict[str, np.ndarray], seed: int) -> np.ndarray:
    base = dgp1._build_base_frame(
        data["trajectory_obs"],
        data["observed"],
        data["diabetes"],
        data["hypertension"],
        data["male"],
        data["time"],
    )
    late = hist_sim._cell_frame(data["trajectory_obs"])
    frame = pd.concat([late.reset_index(drop=True), base.reset_index(drop=True)], axis=1)
    target_columns = list(late.columns)
    draws, _ = run_package_imputer(
        frame, target_columns, "missforest", 1, seed + 8_142_119
    )
    return hist_sim._frame_to_late_trajectory(draws[0], data["trajectory_obs"])


def blend_draws(
    neural_draws: list[np.ndarray],
    anchor: np.ndarray,
    observed: np.ndarray,
    mean_weight: float,
    noise_weight: float,
) -> list[np.ndarray]:
    stack = np.stack(neural_draws, axis=0)
    neural_mean = stack.mean(axis=0)
    mean = anchor + mean_weight * (neural_mean - anchor)
    output = []
    for draw in stack:
        completed = mean + noise_weight * (draw - neural_mean)
        completed[observed] = anchor[observed]
        output.append(completed.astype(np.float32))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1092900)
    parser.add_argument("--nsim", type=int, default=5)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--target-missing", type=float, default=0.20)
    parser.add_argument("--scenario", default="baseline_harder")
    parser.add_argument("--mean-weights", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--noise-weights", nargs="+", type=float, default=[0.0, 0.35, 0.7, 1.0])
    parser.add_argument("--smc-step-sizes", nargs="+", type=float, default=[0.01, 0.025, 0.05, 0.10])
    parser.add_argument("--smc-steps", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--include-smc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--final-label", default="")
    parser.add_argument(
        "--downstream-objective",
        choices=["classification", "association_score"],
        default="association_score",
    )
    parser.add_argument("--lambda-head", type=float, default=0.50)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    train_args = argparse.Namespace(
        m=args.m,
        crossfit_folds=5,
        validation_fraction=0.15,
        validation_mask_rate=0.20,
        training_mask_rate=0.20,
        hidden_size=64,
        batch_size=96,
        max_epochs=80,
        min_epochs=24,
        pretrain_epochs=15,
        downstream_warmup=10,
        patience=15,
        min_delta=1e-4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        gradient_clip=5.0,
        lambda_directional=0.20,
        lambda_reconstruction_mse=1.50,
        lambda_cons=0.08,
        lambda_sum=0.30,
        lambda_down=0.45,
        downstream_objective=args.downstream_objective,
        lambda_head=args.lambda_head,
    )
    device = torch.device(args.device)
    coefficient_rows: list[dict] = []
    metric_rows: list[dict] = []

    for offset in range(args.nsim):
        seed = args.seed_start + offset
        data = neural.generate_locked_dataset(
            seed, args.scenario, args.n, args.target_missing
        )
        cache_path = args.outdir / f"raw_seed_{seed}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            raw_draws = [draw for draw in cached["raw_draws"]]
            anchor = cached["anchor"]
            predictive = {
                "task_auroc": np.nan,
                "task_auprc": np.nan,
                "task_log_loss": np.nan,
                "cell_interval_coverage": np.nan,
                "cell_interval_width": np.nan,
                "runtime_seconds": np.nan,
            }
        else:
            raw_draws, _, _, _, predictive = neural.cross_fitted_completions(
                data,
                seed,
                train_args,
                train_args.lambda_down,
                "brits_mi_gradient",
                device,
            )
            anchor = missforest_anchor(data, seed)
            np.savez_compressed(
                cache_path,
                raw_draws=np.stack(raw_draws, axis=0),
                anchor=anchor,
            )
        for mean_weight in args.mean_weights:
            for noise_weight in args.noise_weights:
                method = f"anchored_mean{mean_weight:.2f}_noise{noise_weight:.2f}"
                neural.METHOD_LABELS[method] = args.final_label or (
                    f"Anchored BRITS-MI (mean={mean_weight:.2f}, noise={noise_weight:.2f})"
                )
                completed = blend_draws(
                    raw_draws,
                    anchor,
                    data["observed"],
                    mean_weight,
                    noise_weight,
                )
                rows, metric = neural.evaluate_completions(
                    data, completed, seed, args.scenario, method, predictive
                )
                coefficient_rows.extend(rows)
                metric_rows.append(metric)

        if args.include_smc:
            anchored = blend_draws(raw_draws, anchor, data["observed"], 0.0, 1.0)
            anchored_cells = [hist_sim._cell_frame(draw) for draw in anchored]
            for step_size in args.smc_step_sizes:
                for n_steps in args.smc_steps:
                    rng = np.random.default_rng(
                        seed + 91_003 + int(round(step_size * 100_000)) + 1009 * n_steps
                    )
                    targeted_cells = hist_sim._crossfit_targeted_association_fluctuation(
                        anchored_cells,
                        data["trajectory_obs"],
                        data["observed"],
                        data["time"],
                        data["outcome"],
                        data["diabetes"],
                        data["hypertension"],
                        data["male"],
                        dgp1._trajectory_features,
                        dgp1._design,
                        rng,
                        n_folds=5,
                        n_steps=n_steps,
                        step_size=step_size,
                    )
                    targeted = [
                        hist_sim._frame_to_late_trajectory(frame, data["trajectory_obs"])
                        for frame in targeted_cells
                    ]
                    method = f"anchored_smc_step{step_size:.3f}_n{n_steps}"
                    neural.METHOD_LABELS[method] = (
                        f"Anchored BRITS-MI + SMC (step={step_size:.3f}, n={n_steps})"
                    )
                    rows, metric = neural.evaluate_completions(
                        data, targeted, seed, args.scenario, method, predictive
                    )
                    coefficient_rows.extend(rows)
                    metric_rows.append(metric)
        pd.DataFrame(coefficient_rows).to_csv(
            args.outdir / "coefficient_by_run.csv", index=False
        )
        pd.DataFrame(metric_rows).to_csv(args.outdir / "metrics_by_run.csv", index=False)
        print(f"completed development seed {seed}", flush=True)

    coefficient = pd.DataFrame(coefficient_rows)
    metrics = pd.DataFrame(metric_rows)
    by_term, overall = run_large.summarize_effects(coefficient, metrics)
    by_term.to_csv(args.outdir / "by_predictor.csv", index=False)
    overall.to_csv(args.outdir / "overall.csv", index=False)
    ranking = overall.copy()
    ranking["coverage_error"] = (ranking["coverage"] - 0.95).abs()
    ranking["score"] = (
        ranking["imputation_rmse_mean"]
        + 0.50 * ranking["trajectory_summary_rmse_mean"]
        + 1.50 * ranking["mean_abs_signed_bias"]
        + 0.75 * ranking["coverage_error"]
    )
    ranking = ranking.sort_values("score")
    ranking.to_csv(args.outdir / "development_ranking.csv", index=False)
    config = vars(args).copy()
    config["outdir"] = str(config["outdir"])
    config["locked_interpretation"] = (
        "Outcome-blind package-default Missforest conditional-mean anchor plus "
        "association-score-trained BRITS-MI residual mean and stochastic residual draws"
    )
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        ranking[
            [
                "method_label",
                "mean_abs_signed_bias",
                "coverage",
                "imputation_rmse_mean",
                "trajectory_summary_rmse_mean",
                "score",
            ]
        ].head(12).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
