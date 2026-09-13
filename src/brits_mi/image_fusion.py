"""BRITS-MI-image tree extension and feature-level fusion.

This explicit fit/transform API packages the image-conditioning concept. It is
not a recurrent BRITS engine and does not implement stochastic MI. It is a new,
testable API, not a claim to recover an unrecorded historical tuning invocation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from longitudinal_sim.config import ExperimentConfig
from longitudinal_sim.data import generate_dataset
from longitudinal_sim.metrics import classification_metrics

from ._image_reference import image_features, trajectory_motif_features


@dataclass(frozen=True)
class ImageFusionConfig:
    seed: int = 970000
    cell_trees: int = 180
    fusion_trees: int = 180
    prototype_alpha: float = 0.0
    prototype_power: float = 1.0
    min_cell_training: int = 30
    engine: str = "extra_trees"

    def __post_init__(self):
        if not 0 <= self.prototype_alpha <= 1 or self.prototype_power <= 0:
            raise ValueError("Prototype alpha must be in [0,1]; power must be positive")
        if self.cell_trees < 1 or self.fusion_trees < 1 or self.min_cell_training < 1:
            raise ValueError("Tree and training counts must be positive")
        if self.engine not in {"extra_trees", "random_forest"}:
            raise ValueError("Unknown cell engine")


class BRITSMIImage:
    """Fit image context, cell regressions, then the completed-data classifier.

    fit receives training labels. complete and predict_proba have no outcome
    argument. All supplied measured cells remain fixed. Time padding is zero
    and is excluded from cell training and trajectory features.
    """

    def __init__(self, config: ImageFusionConfig | None = None):
        self.config = config or ImageFusionConfig()

    def _inputs(self, values, mask, times, lengths, covariates, images):
        y = np.asarray(values, dtype=float)
        raw_mask = np.asarray(mask)
        if not np.isin(raw_mask, [0, 1]).all():
            raise ValueError("Mask must be binary")
        m = raw_mask.astype(bool)
        t = np.asarray(times, dtype=float)
        ll = np.asarray(lengths, dtype=int)
        s = np.asarray(covariates, dtype=float)
        if y.ndim != 3 or y.shape[-1] != 3 or m.shape != y.shape:
            raise ValueError("Expected values and mask shaped (subjects, visits, 3)")
        n, visits, _ = y.shape
        if t.shape != (n, visits) or ll.shape != (n,) or s.ndim != 2 or len(s) != n:
            raise ValueError("Inconsistent input shapes")
        if np.any(ll < 3) or np.any(ll > visits):
            raise ValueError("Each subject must have 3 to max_visits visits")
        valid = np.arange(visits)[None, :] < ll[:, None]
        if np.any(m & ~valid[:, :, None]):
            raise ValueError("Padding cannot be marked observed")
        if not np.isfinite(y[m]).all() or not np.isfinite(s).all() or not np.isfinite(t).all():
            raise ValueError("Observed values, times, and covariates must be finite")
        if any(np.any(np.diff(t[i, : ll[i]]) < 0) for i in range(n)):
            raise ValueError("Visits must be ordered")
        v = image_features(np.asarray(images))
        if len(v) != n or not np.isfinite(v).all():
            raise ValueError("Images must be finite and subject aligned")
        y = np.where(m, y, 0.0)
        return y, m, t, ll, s, v, valid

    def _design(self, inp):
        y, m, t, ll, s, v, valid = inp
        n = len(y)
        f = y.reshape(n, -1)
        fm = m.reshape(n, -1)
        completed = np.where(fm, f, self.cell_means_)
        gaps = np.concatenate([np.zeros((n, 1)), np.diff(t, axis=1)], axis=1)
        gaps = np.where(valid, gaps, 0.0)
        probabilities = self.image_model_.predict_proba(v)
        return np.column_stack([completed, fm, s, t, gaps, ll, v, probabilities]), probabilities

    def fit(self, values, mask, times, lengths, covariates, images, labels):
        inp = self._inputs(values, mask, times, lengths, covariates, images)
        y, m, _, _, s, v, _ = inp
        labels = np.asarray(labels)
        if labels.shape != (len(y),) or not np.array_equal(np.unique(labels), [0, 1, 2]):
            raise ValueError("Training must contain all three labels 0, 1, 2")
        self.shape_ = y.shape[1:]
        self.n_covariates_ = s.shape[1]
        f = y.reshape(len(y), -1)
        fm = m.reshape(len(y), -1)
        self.cell_means_ = np.divide(
            f.sum(0), fm.sum(0), out=np.zeros(f.shape[1]), where=fm.sum(0) > 0
        )
        self.image_model_ = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=600, class_weight="balanced")
        )
        self.image_model_.fit(v, labels)
        x, _ = self._design(inp)
        self.prototypes_ = np.tile(self.cell_means_, (3, 1))
        self.cell_models_ = []
        for j in range(f.shape[1]):
            for c in range(3):
                idx = fm[:, j] & (labels == c)
                if idx.sum() >= 8:
                    self.prototypes_[c, j] = f[idx, j].mean()
            idx = fm[:, j]
            if idx.sum() < self.config.min_cell_training:
                self.cell_models_.append(None)
                continue
            if self.config.engine == "extra_trees":
                model = ExtraTreesRegressor(
                    n_estimators=self.config.cell_trees,
                    max_depth=13,
                    min_samples_leaf=4,
                    max_features=0.8,
                    random_state=self.config.seed,
                    n_jobs=1,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=self.config.cell_trees,
                    max_depth=8,
                    min_samples_leaf=8,
                    random_state=self.config.seed,
                    n_jobs=1,
                )
            model.fit(np.delete(x[idx], j, axis=1), f[idx, j])
            self.cell_models_.append(model)
        completed = self.complete(values, mask, times, lengths, covariates, images)
        design = np.column_stack([trajectory_motif_features(completed, lengths), v])
        self.fusion_model_ = ExtraTreesClassifier(
            n_estimators=self.config.fusion_trees,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=self.config.seed,
            n_jobs=1,
        )
        self.fusion_model_.fit(design, labels)
        return self

    def complete(self, values, mask, times, lengths, covariates, images):
        if not hasattr(self, "cell_models_"):
            raise RuntimeError("Call fit first")
        inp = self._inputs(values, mask, times, lengths, covariates, images)
        y, m, _, _, s, _, valid = inp
        if y.shape[1:] != self.shape_ or s.shape[1] != self.n_covariates_:
            raise ValueError("Feature dimensions differ from training")
        x, probs = self._design(inp)
        weights = np.clip(probs, 1e-6, 1) ** self.config.prototype_power
        weights /= weights.sum(1, keepdims=True)
        proto = weights @ self.prototypes_
        out = np.tile(self.cell_means_, (len(y), 1))
        for j, model in enumerate(self.cell_models_):
            if model is not None:
                pred = model.predict(np.delete(x, j, axis=1))
                out[:, j] = (
                    1 - self.config.prototype_alpha
                ) * pred + self.config.prototype_alpha * proto[:, j]
        out = np.where(m, y, out.reshape(y.shape))
        return np.where(valid[:, :, None], out, 0.0)

    def predict_proba(self, values, mask, times, lengths, covariates, images):
        completed = self.complete(values, mask, times, lengths, covariates, images)
        design = np.column_stack(
            [trajectory_motif_features(completed, lengths), image_features(images)]
        )
        return self.fusion_model_.predict_proba(design)


def image_inputs(data, idx):
    """Exclude generator-only metadata and outcome from deployable inputs."""
    n = len(idx)
    cov = np.column_stack(
        [data.static[idx], data.x[idx].reshape(n, -1), data.x_mask[idx].reshape(n, -1)]
    )
    return (
        data.y_obs[idx],
        data.mask[idx],
        data.times[idx],
        data.lengths[idx],
        cov,
        data.image[idx],
    )


def run_image_replicate(n_subjects=500, seed=970000, missing_rate=0.5, config=None):
    """New deterministic image run with train-only fitting; not archived scores."""
    cfg = config or ImageFusionConfig(seed=seed)
    dgp = ExperimentConfig(
        n_subjects=n_subjects,
        seed=seed,
        mask_rate=missing_rate,
        dgp_profile="paper_britsadv_img065_altus",
        info_profile="balanced",
        n_classes=3,
        image_size=24,
    )
    data = generate_dataset(dgp)
    train, test = train_test_split(
        np.arange(n_subjects), test_size=0.25, random_state=seed + 101, stratify=data.labels
    )
    model = BRITSMIImage(cfg).fit(*image_inputs(data, train), data.labels[train])
    probs = model.predict_proba(*image_inputs(data, test))
    completed = model.complete(*image_inputs(data, test))
    valid = np.arange(data.y_obs.shape[1])[None, :, None] < data.lengths[test, None, None]
    hidden = valid & (data.mask[test] == 0)
    result = classification_metrics(data.labels[test], probs)
    result.update(
        seed=seed,
        method="BRITS-MI-image",
        n_subjects=n_subjects,
        missing_cell_rmse=float(
            np.sqrt(np.mean((completed[hidden] - data.y_true[test][hidden]) ** 2))
        ),
        realized_missing=float(hidden.sum() / np.broadcast_to(valid, completed.shape).sum()),
        status="new API verification run; not manuscript benchmark",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-subjects", type=int, default=500)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument(
        "--prototype-alpha",
        type=float,
        required=True,
        help="Must be explicit: historical alpha was not archived",
    )
    parser.add_argument("--trees", type=int, default=180)
    args = parser.parse_args()
    if args.n_runs < 1 or args.n_subjects < 40 or not 0 < args.missing_rate < 1:
        parser.error("Require n_runs>=1, n_subjects>=40, and 0<missing_rate<1")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in range(args.n_runs):
        cfg = ImageFusionConfig(
            seed=args.seed + run,
            cell_trees=args.trees,
            fusion_trees=args.trees,
            prototype_alpha=args.prototype_alpha,
        )
        rows.append(run_image_replicate(args.n_subjects, args.seed + run, args.missing_rate, cfg))
        pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "output": str(args.output),
                "model": asdict(cfg),
                "status": "New API, not exact historical retraining",
            },
            indent=2,
        )
        + "\n"
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
