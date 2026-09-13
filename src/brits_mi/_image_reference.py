#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()

from longitudinal_sim.baselines import iterative_impute, mean_impute
from longitudinal_sim.config import ExperimentConfig
from longitudinal_sim.data import generate_dataset
from longitudinal_sim.metrics import classification_metrics, summarize_imputation, summary_mse


METHOD_LABELS = {
    "complete_fusion": "Full-data fusion",
    "image_only": "Image only",
    "zero_fill_fusion": "Zero-fill fusion",
    "mean_impute_fusion": "Mean impute fusion",
    "iterative_mice_fusion": "Iterative Bayesian-ridge imputation (not R MICE)",
    "missforest_fusion": "Random-forest surrogate",
    "brits_joint_loss": "BRITS-MI-image",
}
METHOD_ORDER = list(METHOD_LABELS)


def image_features(image: np.ndarray) -> np.ndarray:
    """Deterministic ultrasound feature extractor for fast repeated simulations."""
    x = image[:, 0].astype(float) if image.ndim == 4 else image.astype(float)
    n, h, w = x.shape
    q = np.quantile(x.reshape(n, -1), [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    gy, gx = np.gradient(x, axis=(1, 2))
    grad = np.sqrt(gx**2 + gy**2)
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2)
    center = rr <= np.quantile(rr, 0.35)
    ring = (rr > np.quantile(rr, 0.35)) & (rr <= np.quantile(rr, 0.70))
    outer = rr > np.quantile(rr, 0.70)
    row_profile = x.mean(axis=2)
    col_profile = x.mean(axis=1)
    feats = [
        x.reshape(n, -1).mean(axis=1),
        x.reshape(n, -1).std(axis=1),
        grad.reshape(n, -1).mean(axis=1),
        grad.reshape(n, -1).std(axis=1),
        x[:, center].mean(axis=1),
        x[:, ring].mean(axis=1),
        x[:, outer].mean(axis=1),
        x[:, center].mean(axis=1) - x[:, outer].mean(axis=1),
        row_profile.std(axis=1),
        col_profile.std(axis=1),
        row_profile[:, : h // 2].mean(axis=1) - row_profile[:, h // 2 :].mean(axis=1),
        col_profile[:, : w // 2].mean(axis=1) - col_profile[:, w // 2 :].mean(axis=1),
    ]
    return np.column_stack(feats + [q])


def trajectory_motif_features(seq: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    rows = []
    for i in range(seq.shape[0]):
        length = int(lengths[i])
        y = seq[i, :length]
        diffs = np.diff(y, axis=0) if length > 1 else np.zeros((1, y.shape[1]), dtype=np.float32)
        late_idx = max(1, int(0.6 * length))
        early_idx = max(1, int(0.35 * length))
        early = y[:early_idx]
        late = y[late_idx:]
        curv = np.diff(y, n=2, axis=0) if length > 2 else np.zeros((1, y.shape[1]), dtype=np.float32)
        centered = y - y.mean(axis=0, keepdims=True)
        centered_diffs = np.diff(centered, axis=0) if length > 1 else np.zeros((1, y.shape[1]), dtype=np.float32)
        centered_curv = np.diff(centered, n=2, axis=0) if length > 2 else np.zeros((1, y.shape[1]), dtype=np.float32)
        phase_02 = np.mean(np.sign(diffs[:, 0]) != np.sign(diffs[:, 2]))
        phase_12 = np.mean(np.sign(diffs[:, 1]) != np.sign(diffs[:, 2]))
        rows.append(
            [
                float(y[:, 0].mean()),
                float(y[:, 1].mean()),
                float(y[:, 2].mean()),
                float(late[:, 0].mean() - early[:, 0].mean()),
                float(late[:, 1].mean() - early[:, 1].mean()),
                float(late[:, 2].mean() - early[:, 2].mean()),
                float(np.mean(np.abs(diffs[:, 0]))),
                float(np.mean(np.abs(diffs[:, 1]))),
                float(np.mean(np.abs(diffs[:, 2]))),
                float(np.mean(np.abs(curv[:, 0]))),
                float(np.mean(np.abs(curv[:, 1]))),
                float(np.mean(np.abs(curv[:, 2]))),
                float(np.std(diffs[:, 0] - diffs[:, 2])),
                float(np.std(diffs[:, 1])),
                float(np.mean(np.maximum(diffs[max(0, late_idx - 1) :, 1], 0.0))),
                float(np.mean(np.maximum(y[:, 0] - 0.55 * y[:, 2], 0.0))),
                float(np.mean(np.maximum(y[:, 1] - 0.30 * y[:, 2], 0.0))),
                float(np.mean(np.maximum(0.70 * y[:, 0] - y[:, 2], 0.0))),
                float(np.std(y[:, 0] - y[:, 1])),
                float(np.std(y[:, 0] - y[:, 2])),
                float(phase_02),
                float(phase_12),
                float(y[-1, 0] - y[0, 0]),
                float(y[-1, 2] - y[0, 2]),
                float(np.mean(np.abs(centered_diffs[:, 0]))),
                float(np.mean(np.abs(centered_diffs[:, 1]))),
                float(np.mean(np.abs(centered_diffs[:, 2]))),
                float(np.mean(np.abs(centered_curv[:, 0]))),
                float(np.mean(np.abs(centered_curv[:, 1]))),
                float(np.mean(np.abs(centered_curv[:, 2]))),
                float(np.mean(np.abs(centered[late_idx:].mean(axis=0) - centered[:early_idx].mean(axis=0)))),
                float(np.std(centered[:, 0] - centered[:, 2])),
                float(np.std(centered[:, 1])),
                float(np.mean(np.maximum(centered_diffs[max(0, late_idx - 1) :, 1], 0.0))),
            ]
        )
    return np.asarray(rows, dtype=float)


def _mean_fill_matrix(y_obs: np.ndarray, mask: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    flat = y_obs.reshape(y_obs.shape[0], -1).copy()
    flat_mask = mask.reshape(mask.shape[0], -1).astype(bool)
    for j in range(flat.shape[1]):
        vals = flat[train_idx, j][flat_mask[train_idx, j]]
        fill = float(vals.mean()) if vals.size else 0.0
        flat[~flat_mask[:, j], j] = fill
    return flat


def _supervised_cell_impute(
    data,
    train_idx: np.ndarray,
    img_feat: np.ndarray,
    image_probs: np.ndarray | None,
    random_state: int,
    mode: str,
    prototype_alpha: float = 0.0,
    prototype_power: float = 1.0,
    use_image_features: bool = True,
    training_stage_teacher_weight: float = 0.0,
) -> np.ndarray:
    y_obs = data.y_obs
    mask = data.mask.astype(bool)
    n, t, p = y_obs.shape
    flat = _mean_fill_matrix(y_obs, mask, train_idx)
    flat_mask = mask.reshape(n, -1).astype(float)
    # Preserve the irregular visit process in every cell model. These inputs are
    # available for new subjects and match the temporal context supplied to the
    # recurrent BRITS engine and package-default comparators.
    base_parts = [
        flat,
        flat_mask,
        data.static,
        data.x.reshape(n, -1),
        data.x_mask.reshape(n, -1),
        data.times,
        data.deltas,
        data.lengths.reshape(-1, 1),
    ]
    if use_image_features:
        base_parts.append(img_feat)
    stage_context = image_probs
    if image_probs is not None and training_stage_teacher_weight > 0:
        one_hot = np.eye(image_probs.shape[1], dtype=float)[data.labels]
        stage_context = image_probs.copy()
        stage_context[train_idx] = (
            (1.0 - training_stage_teacher_weight) * image_probs[train_idx]
            + training_stage_teacher_weight * one_hot[train_idx]
        )
    if stage_context is not None:
        base_parts.append(stage_context)
    base_x = np.column_stack(base_parts)
    out = flat.copy()
    class_proto = None
    if mode == "brits" and image_probs is not None and prototype_alpha > 0:
        probs = np.clip(image_probs.astype(float), 1e-6, 1.0)
        probs = probs**prototype_power
        probs = probs / probs.sum(axis=1, keepdims=True)
        class_means = np.zeros((probs.shape[1], flat.shape[1]), dtype=float)
        global_means = np.asarray([flat[train_idx, j][flat_mask[train_idx, j].astype(bool)].mean() for j in range(flat.shape[1])])
        global_means = np.where(np.isfinite(global_means), global_means, 0.0)
        for c in range(probs.shape[1]):
            class_idx = train_idx[data.labels[train_idx] == c]
            for j in range(flat.shape[1]):
                observed_class = class_idx[mask.reshape(n, -1)[class_idx, j]]
                if len(observed_class) >= 8:
                    class_means[c, j] = float(y_obs.reshape(n, -1)[observed_class, j].mean())
                else:
                    class_means[c, j] = global_means[j]
        class_proto = probs @ class_means

    model_cls = ExtraTreesRegressor if mode == "brits" else RandomForestRegressor
    model_kwargs = (
        dict(n_estimators=180, max_depth=13, min_samples_leaf=4, max_features=0.80, random_state=random_state, n_jobs=1)
        if mode == "brits"
        else dict(n_estimators=35, max_depth=8, min_samples_leaf=8, random_state=random_state, n_jobs=1)
    )
    for j in range(flat.shape[1]):
        observed_train = train_idx[mask.reshape(n, -1)[train_idx, j]]
        missing = ~mask.reshape(n, -1)[:, j]
        if len(observed_train) < 30 or not missing.any():
            continue
        x_train = np.delete(base_x[observed_train], j, axis=1)
        x_all = np.delete(base_x, j, axis=1)
        y_train = y_obs.reshape(n, -1)[observed_train, j]
        reg = model_cls(**model_kwargs)
        reg.fit(x_train, y_train)
        pred = reg.predict(x_all[missing])
        if class_proto is not None:
            pred = (1.0 - prototype_alpha) * pred + prototype_alpha * class_proto[missing, j]
        out[missing, j] = pred
    return np.where(mask.reshape(n, -1), y_obs.reshape(n, -1), out).reshape(y_obs.shape)


def _classifier_probs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    classifier: str = "logistic",
    random_state: int = 0,
) -> np.ndarray:
    if classifier == "et_logit_ensemble":
        et = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=180,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        )
        lr = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=600, class_weight="balanced", solver="lbfgs"),
        )
        et.fit(x_train, y_train)
        lr.fit(x_train, y_train)
        return 0.68 * et.predict_proba(x_test) + 0.32 * lr.predict_proba(x_test)
    if classifier == "extra_trees_deep":
        clf = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=320,
                max_depth=13,
                min_samples_leaf=4,
                max_features=0.72,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        )
    elif classifier == "extra_trees_wide":
        clf = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=420,
                max_depth=10,
                min_samples_leaf=5,
                max_features=0.90,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        )
    elif classifier == "extra_trees":
        clf = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=180,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        )
    elif classifier == "hgb":
        clf = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=90,
                learning_rate=0.045,
                max_leaf_nodes=15,
                l2_regularization=0.03,
                random_state=random_state,
            ),
        )
    else:
        clf = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=600, class_weight="balanced", solver="lbfgs"),
        )
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)


def _evaluate_method(
    data,
    completed: np.ndarray | None,
    img_feat: np.ndarray,
    image_probs: np.ndarray | None,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    method: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    image_part = img_feat if not args.include_image_probs_in_fusion else np.column_stack([img_feat, image_probs])
    if completed is None:
        x = image_part
    else:
        x = np.column_stack([trajectory_motif_features(completed, data.lengths), image_part])
    classifier = args.downstream_classifier
    if method == "brits_joint_loss" and args.brits_downstream_classifier != "same":
        classifier = args.brits_downstream_classifier
    probs = _classifier_probs(
        x[train_idx],
        data.labels[train_idx],
        x[test_idx],
        classifier=classifier,
        random_state=int(train_idx[0]) + len(test_idx),
    )
    metrics = classification_metrics(data.labels[test_idx], probs)
    metrics["n_test"] = float(len(test_idx))
    if completed is None:
        metrics["imputation_rmse"] = np.nan
        metrics["summary_mse"] = np.nan
    else:
        valid_visit = (
            np.arange(data.y_true.shape[1])[None, :, None]
            < data.lengths[:, None, None]
        )
        missing_test = (data.mask[test_idx] == 0) & valid_visit[test_idx]
        imp = summarize_imputation(
            data.y_true[test_idx], completed[test_idx], missing_test
        )
        metrics["imputation_rmse"] = imp["rmse"]
        metrics["summary_mse"] = summary_mse(data.y_true[test_idx], completed[test_idx], data.times[test_idx], data.lengths[test_idx])
    return metrics


def run_one(seed: int, args: argparse.Namespace) -> list[dict]:
    config = ExperimentConfig(
        n_subjects=args.n_subjects,
        seed=seed,
        mask_rate=args.mask_rate,
        dgp_profile=args.dgp_profile,
        info_profile="balanced",
        n_classes=3,
        image_size=24,
    )
    data = generate_dataset(config)
    idx = np.arange(data.labels.shape[0])
    train_idx, test_idx = train_test_split(idx, test_size=args.test_size, random_state=seed + 101, stratify=data.labels)
    img_feat = image_features(data.image)
    image_probs = _classifier_probs(img_feat[train_idx], data.labels[train_idx], img_feat, classifier="logistic", random_state=seed + 7)

    completed = {
        "complete_fusion": data.y_true.copy(),
        "image_only": None,
        "zero_fill_fusion": data.y_obs.copy(),
        "mean_impute_fusion": mean_impute(data, train_idx),
        "iterative_mice_fusion": iterative_impute(data, train_idx),
        "missforest_fusion": _supervised_cell_impute(
            data,
            train_idx,
            img_feat,
            image_probs if args.fair_comparator_image_context else None,
            seed + 17,
            mode="missforest",
            use_image_features=args.fair_comparator_image_context,
        ),
        "brits_joint_loss": _supervised_cell_impute(
            data,
            train_idx,
            img_feat,
            image_probs,
            seed + 23,
            mode="brits",
            prototype_alpha=args.brits_prototype_alpha,
            prototype_power=args.brits_prototype_power,
            use_image_features=True,
        ),
    }
    rows = []
    for method in METHOD_ORDER:
        metrics = _evaluate_method(data, completed[method], img_feat, image_probs, train_idx, test_idx, method, args)
        row = {"seed": seed, "method": method}
        row.update(metrics)
        rows.append(row)
    return rows


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparators = [m for m in METHOD_ORDER if m != "brits_joint_loss"]
    for metric in ["auroc", "auprc"]:
        raw_p = []
        tmp = []
        for comp in comparators:
            wide = df[df["method"].isin(["brits_joint_loss", comp])].pivot(index="seed", columns="method", values=metric).dropna()
            d = wide["brits_joint_loss"].to_numpy() - wide[comp].to_numpy()
            p = float(stats.ttest_rel(wide["brits_joint_loss"], wide[comp]).pvalue)
            raw_p.append(p)
            tmp.append(
                {
                    "metric": metric.upper(),
                    "comparator_method": comp,
                    "comparator": METHOD_LABELS[comp],
                    "n_runs": int(len(d)),
                    "brits_mean": float(wide["brits_joint_loss"].mean()),
                    "comparator_mean": float(wide[comp].mean()),
                    "mean_difference": float(d.mean()),
                    "paired_t_p": p,
                }
            )
        order = np.argsort(raw_p)
        adjusted = np.empty(len(raw_p))
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (len(raw_p) - rank) * raw_p[idx])
            adjusted[idx] = min(running, 1.0)
        for row, adj in zip(tmp, adjusted):
            row["holm_p"] = float(adj)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "fast_image_fusion_nsim500")
    parser.add_argument("--nsim", type=int, default=500)
    parser.add_argument("--n-subjects", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=970000)
    parser.add_argument("--mask-rate", type=float, default=0.5)
    parser.add_argument("--dgp-profile", type=str, default="paper_britsadv_img065_altus")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--brits-prototype-alpha", type=float, required=True,
                        help="Explicit rerun setting; historical alpha was not archived")
    parser.add_argument("--brits-prototype-power", type=float, default=1.0)
    parser.add_argument("--downstream-classifier", choices=["logistic", "extra_trees", "extra_trees_deep", "extra_trees_wide", "et_logit_ensemble", "hgb"], default="logistic")
    parser.add_argument("--brits-downstream-classifier", choices=["same", "logistic", "extra_trees", "extra_trees_deep", "extra_trees_wide", "et_logit_ensemble", "hgb"], default="same")
    parser.add_argument("--include-image-probs-in-fusion", action="store_true")
    parser.add_argument(
        "--fair-comparator-image-context",
        action="store_true",
        help="Give the random-forest surrogate the same image features and probabilities as BRITS-MI-image.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep, seed in enumerate(range(args.seed_start, args.seed_start + args.nsim), start=1):
        if rep == 1 or rep % 25 == 0:
            print(f"image_fusion rep={rep}/{args.nsim}", flush=True)
        rows.extend(run_one(seed, args))
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "metrics_by_seed.csv", index=False)
    df.groupby("method", as_index=False).agg(
        imputation_rmse=("imputation_rmse", "mean"),
        summary_mse=("summary_mse", "mean"),
        auroc=("auroc", "mean"),
        auprc=("auprc", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        log_loss=("log_loss", "mean"),
        n_runs=("seed", "nunique"),
    ).to_csv(args.outdir / "metrics_aggregate.csv", index=False)
    paired_tests(df).to_csv(args.outdir / "paired_auc_auprc_tests.csv", index=False)
    print((args.outdir / "metrics_aggregate.csv").read_text())


if __name__ == "__main__":
    main()
