#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_neural_brits_mi_downstream_gradient as neural  # noqa: E402
from test_anchored_brits_mi_blend import blend_draws, missforest_anchor  # noqa: E402


run_large = neural.run_large


def training_args(max_m: int, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        m=max_m,
        crossfit_folds=args.crossfit_folds,
        validation_fraction=0.20,
        validation_mask_rate=0.20,
        training_mask_rate=0.20,
        hidden_size=args.hidden_size,
        batch_size=96,
        max_epochs=args.max_epochs,
        min_epochs=16,
        pretrain_epochs=8,
        downstream_warmup=6,
        patience=8,
        min_delta=1e-4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        gradient_clip=5.0,
        lambda_directional=0.20,
        lambda_reconstruction_mse=1.00,
        lambda_cons=0.08,
        lambda_sum=0.20,
        lambda_down=0.45,
        downstream_objective="classification",
        lambda_head=0.10,
    )


def predictive_from_cache(cache: np.lib.npyio.NpzFile) -> dict[str, float]:
    names = [
        "task_auroc",
        "task_auprc",
        "task_log_loss",
        "cell_interval_coverage",
        "cell_interval_width",
        "runtime_seconds",
    ]
    return {
        name: float(cache[name]) if name in cache.files else np.nan
        for name in names
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1_200_000)
    parser.add_argument("--nsim", type=int, default=100)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--m-values", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--target-missing", type=float, required=True)
    parser.add_argument("--scenario", default="baseline_harder")
    parser.add_argument("--mean-weight", type=float, required=True)
    parser.add_argument("--noise-weight", type=float, default=0.70)
    parser.add_argument("--crossfit-folds", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--max-epochs", type=int, default=45)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.m_values = sorted(set(args.m_values))
    args.outdir.mkdir(parents=True, exist_ok=True)
    max_m = max(args.m_values)
    train_args = training_args(max_m, args)
    device = torch.device(args.device)
    method = "brits_mi_refined"
    neural.METHOD_LABELS[method] = "BRITS-MI"

    coefficient_path = args.outdir / "coefficient_by_run.csv"
    metric_path = args.outdir / "metrics_by_run.csv"
    audit_path = args.outdir / "fold_audit.csv"
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
    completed_keys = {
        (int(row["seed"]), int(row["n_imputations"])) for row in metric_rows
    }

    for offset in range(args.nsim):
        seed = args.seed_start + offset
        required = {(seed, m) for m in args.m_values}
        if required.issubset(completed_keys):
            print(f"skip completed seed={seed}", flush=True)
            continue
        started = time.time()
        data = neural.generate_locked_dataset(
            seed,
            args.scenario,
            args.n,
            args.target_missing,
        )
        cache_path = args.outdir / f"raw_seed_{seed}.npz"
        if args.resume and cache_path.exists():
            cache = np.load(cache_path)
            raw_draws = [draw for draw in cache["raw_draws"]]
            anchor = cache["anchor"]
            predictive = predictive_from_cache(cache)
        else:
            raw_draws, _, _, seed_audits, predictive = neural.cross_fitted_completions(
                data,
                seed,
                train_args,
                train_args.lambda_down,
                "brits_mi_gradient",
                device,
            )
            anchor = missforest_anchor(data, seed)
            audit_rows.extend(seed_audits)
            np.savez_compressed(
                cache_path,
                raw_draws=np.stack(raw_draws, axis=0),
                anchor=anchor,
                **predictive,
            )
        completed_all = blend_draws(
            raw_draws,
            anchor,
            data["observed"],
            args.mean_weight,
            args.noise_weight,
        )
        for m_value in args.m_values:
            if (seed, m_value) in completed_keys:
                continue
            rows, metric = neural.evaluate_completions(
                data,
                completed_all[:m_value],
                seed,
                args.scenario,
                method,
                predictive,
            )
            coefficient_rows.extend(rows)
            metric_rows.append(metric)
            completed_keys.add((seed, m_value))
        pd.DataFrame(coefficient_rows).to_csv(coefficient_path, index=False)
        pd.DataFrame(metric_rows).to_csv(metric_path, index=False)
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        print(
            f"completed seed={seed} rate={args.target_missing:.0%} "
            f"M={args.m_values} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    coefficient = pd.DataFrame(coefficient_rows)
    metrics = pd.DataFrame(metric_rows)
    by_term, overall = run_large.summarize_effects(coefficient, metrics)
    by_term.to_csv(args.outdir / "effect_recovery_by_predictor.csv", index=False)
    overall.to_csv(args.outdir / "effect_recovery_overall.csv", index=False)
    config = vars(args).copy()
    config["outdir"] = str(config["outdir"])
    config.update(
        {
            "proposal_engine": "cross-fitted bidirectional RITS-GRU BRITS",
            "conditional_mean_anchor": "outcome-blind package-default Missforest",
            "gate": (
                "development-selected convex BRITS residual correction, "
                f"mean_weight={args.mean_weight:.2f}"
            ),
            "variance_calibration": (
                f"BRITS residual spread multiplied by {args.noise_weight:.2f}"
            ),
            "m_design": (
                "one fitted proposal distribution and 50 draws; M=10,20,50 "
                "are nested draw subsets without retraining"
            ),
            "model_selection_priority": (
                "coefficient bias, SE calibration, coverage, missing-cell RMSE, "
                "trajectory-summary RMSE; AUROC is secondary"
            ),
        }
    )
    (args.outdir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(
        overall[
            [
                "n_imputations",
                "mean_abs_signed_bias",
                "coverage",
                "se_ratio",
                "imputation_rmse_mean",
                "trajectory_summary_rmse_mean",
                "n_runs",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        flush=True,
    )


if __name__ == "__main__":
    main()
