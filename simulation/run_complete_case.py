"""Rerun unpenalized complete-case logistic regression with existence diagnostics."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import statsmodels.api as sm

ARCHIVE = Path(__file__).resolve().parent / "clinical" / "scripts"
sys.path.insert(0, str(ARCHIVE))
import run_package_default_association_comparators as association  # noqa: E402


def separated(x, y):
    """Find a nonzero, nonnegative signed-margin direction (including quasi-separation)."""
    scale = np.maximum(np.max(np.abs(x), axis=0), 1.0)
    signed = (2 * y[:, None] - 1) * x / scale
    result = linprog(
        -signed.sum(axis=0), A_ub=-signed, b_ub=np.zeros(len(y)),
        bounds=[(-1, 1)] * x.shape[1], method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Separation diagnostic failed: {result.message}")
    return bool(-result.fun > 1e-7), float(-result.fun)


def run_one(task):
    seed, missing = task
    data = association.generate_locked_dataset(seed, "baseline_harder", 500, missing)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        features = association.dgp1._trajectory_features(data["trajectory_obs"], data["time"])
    keep = features.notna().all(axis=1).to_numpy()
    frame = association.dgp1._design(
        features.loc[keep].reset_index(drop=True), data["diabetes"][keep],
        data["hypertension"][keep], data["male"][keep],
    )
    x = sm.add_constant(frame, has_constant="add")
    y = data["outcome"][keep]
    rank = int(np.linalg.matrix_rank(x.to_numpy())) if len(x) else 0
    audit = {
        "seed": seed, "missing_percentage": int(round(100 * missing)),
        "n_subjects": 500, "n_complete": len(x), "events": int(y.sum()),
        "parameters_including_intercept": x.shape[1], "design_rank": rank,
        "original_minimum_n_met": len(x) >= 25,
        "lp_separation": None, "lp_margin_objective": None,
        "optimizer_converged": False, "finite_estimates_and_se": False,
        "legacy_stability_pass": False, "valid_finite_mle": False,
        "reason": "", "warnings": "",
    }
    if rank < x.shape[1] or len(np.unique(y)) < 2:
        audit["reason"] = "rank_deficient_or_single_outcome_class"
        return audit, []
    is_separated, objective = separated(x.to_numpy(), y)
    audit.update(lp_separation=is_separated, lp_margin_objective=objective)
    if len(x) < 25:
        audit["reason"] = "original_minimum_sample_not_met"
        return audit, []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=100, disp=False)
        audit["warnings"] = " | ".join(sorted({str(w.message) for w in caught}))
        beta = fit.params.drop("const")
        se = fit.bse.drop("const")
        finite = bool(np.isfinite(beta).all() and np.isfinite(se).all() and (se > 0).all())
        audit.update(
            optimizer_converged=bool(fit.converged), finite_estimates_and_se=finite,
            max_abs_estimate=float(np.max(np.abs(beta))), max_std_error=float(se.max()),
            legacy_stability_pass=bool(len(x) >= 10 * len(beta) and finite and
                                       np.max(np.abs(beta)) <= 10 and se.max() <= 10),
            valid_finite_mle=bool(fit.converged and finite and not is_separated),
        )
        audit["reason"] = ("separation" if is_separated else
                           "valid" if audit["valid_finite_mle"] else "numerical_fit_failure")
        rows = []
        for term, truth in association.TRUE_BETA.items():
            estimate, error = float(beta[term]), float(se[term])
            rows.append({
                "seed": seed, "missing_percentage": int(round(100 * missing)),
                "term": term, "truth": truth, "estimate": estimate, "std_error": error,
                "bias": estimate - truth, "squared_error": (estimate - truth) ** 2,
                "covered": float(estimate - 1.96 * error <= truth <= estimate + 1.96 * error),
                "valid_finite_mle": audit["valid_finite_mle"],
            })
        return audit, rows
    except Exception as exc:
        audit["reason"] = "optimizer_exception"
        audit["warnings"] = repr(exc)
        return audit, []


def summarize(audit, coefficients):
    rows = []
    for percentage, group in audit.groupby("missing_percentage"):
        row = {
            "missing_percentage": int(percentage), "attempted_runs": len(group),
            "valid_runs": int(group.valid_finite_mle.sum()),
            "valid_fraction": float(group.valid_finite_mle.mean()),
            "legacy_retained_runs": int(group.legacy_stability_pass.sum()),
            "mean_complete_cases": float(group.n_complete.mean()),
            "min_complete_cases": int(group.n_complete.min()),
            "max_complete_cases": int(group.n_complete.max()),
            "mean_absolute_coefficient_bias": None, "coefficient_mse": None,
            "coverage": None, "bias_mcse": None, "mse_mcse": None, "coverage_mcse": None,
        }
        valid = coefficients.loc[
            coefficients.missing_percentage.eq(percentage) & coefficients.valid_finite_mle
        ] if len(coefficients) else pd.DataFrame()
        if len(valid):
            bias = valid.pivot(index="seed", columns="term", values="bias").to_numpy()
            coverage = valid.pivot(index="seed", columns="term", values="covered").to_numpy()
            row.update(
                mean_absolute_coefficient_bias=float(np.abs(bias.mean(axis=0)).mean()),
                coefficient_mse=float((bias ** 2).mean()), coverage=float(coverage.mean()),
            )
            if len(bias) > 1:
                rng = np.random.default_rng(20260920 + int(percentage))
                idx = rng.integers(0, len(bias), size=(10000, len(bias)))
                row.update(
                    bias_mcse=float(np.abs(bias[idx].mean(axis=1)).mean(axis=1).std(ddof=1)),
                    mse_mcse=float((bias ** 2).mean(axis=1).std(ddof=1) / np.sqrt(len(bias))),
                    coverage_mcse=float(coverage.mean(axis=1).std(ddof=1) / np.sqrt(len(bias))),
                )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Sanity checks for overlap and complete separation.
    assert not separated(np.array([[1, -1], [1, -1], [1, 1], [1, 1.]]),
                         np.array([0, 1, 0, 1]))[0]
    assert separated(np.array([[1, -1], [1, -.5], [1, .5], [1, 1.]]),
                     np.array([0, 0, 1, 1]))[0]
    tasks = [(seed, rate) for rate in [.4, .6]
             for seed in range(1200000, 1200000 + args.runs)]
    audits, coefficients = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (audit, rows) in enumerate(pool.map(run_one, tasks), 1):
            audits.append(audit)
            coefficients.extend(rows)
            if index % 50 == 0 or index == len(tasks):
                pd.DataFrame(audits).to_csv(args.outdir / "fit_audit.csv", index=False)
                pd.DataFrame(coefficients).to_csv(args.outdir / "all_optimizer_outputs.csv", index=False)
                print(f"Complete-case fits {index}/{len(tasks)}", flush=True)
    audits = pd.DataFrame(audits)
    coefficients = pd.DataFrame(coefficients)
    table = summarize(audits, coefficients)
    table.to_csv(args.outdir / "valid_fit_summary.csv", index=False)
    if len(coefficients):
        coefficients.loc[coefficients.valid_finite_mle].to_csv(
            args.outdir / "valid_coefficient_by_run.csv", index=False)
    (args.outdir / "config.json").write_text(json.dumps({
        "n": 500, "runs_per_missing_percentage": args.runs,
        "missing_percentages": [40, 60], "scenario": "baseline_harder",
        "seeds": [1200000, 1200000 + args.runs - 1],
        "model": "Unpenalized binomial-logit GLM; same 13 terms plus intercept",
        "fitting": "Original maxiter=100 and minimum sample=25; no penalization",
        "validation": "Full rank, no complete/quasi separation, convergence, finite SE",
        "summary": "Conditional on valid fits; report success denominator alongside metrics",
        "legacy_stability_filter": "Recorded separately, not used to declare MLE existence",
    }, indent=2))
    print(table.to_string(index=False), flush=True)
    print(audits.groupby(["missing_percentage", "reason"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()
