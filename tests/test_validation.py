"""Synthetic-only tests of the portable masking analysis and privacy boundary."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation import brits  # noqa: E402
from validation import evaluation as ev  # noqa: E402
from validation.data import MARKERS, external_path, load_cohort  # noqa: E402
from validation.run_validation import parse_args, run  # noqa: E402


@pytest.fixture
def synthetic_cohort(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    rng = np.random.default_rng(873)
    n = 180
    values = rng.normal(size=(n, 6, 6)).astype(np.float32)
    mask = (rng.random(values.shape) > 0.2).astype(np.float32)
    frame = pd.DataFrame(
        {
            "row_index": np.arange(n),
            "outcome_3class": rng.choice(["F0-F1", "F2", "F3-F4"], n),
            "FIB4_log": rng.normal(size=n),
            "FIB4_Age": rng.normal(55, 10, n),
            "bmi_consolidated": rng.normal(29, 3, n),
            "Diabetes_RF": rng.choice(["Yes", "No"], n),
            "HTN_RF": rng.choice(["Yes", "No"], n),
            "HLD_RF": rng.choice(["Yes", "No"], n),
            "Sex": rng.choice(["Male", "Female"], n),
            "Study_Type": rng.choice(["MRE", "VCTE"], n),
        }
    )
    frame.to_csv(folder / "clinical.csv", index=False)
    np.savez(
        folder / "longitudinal.npz",
        values=values,
        mask=mask,
        static=rng.normal(size=(n, 4)).astype(np.float32),
        row_index=np.arange(n),
        marker_names=np.array(MARKERS),
    )
    return folder


def test_input_contract_and_mask_targets(synthetic_cohort):
    data = load_cohort(synthetic_cohort)
    hidden = ev._hide_observed_cells(data.mask, 0.4, np.random.default_rng(1), [3, 4, 5])
    assert not hidden[:, :, :3].any()
    assert not hidden[data.mask == 0].any()
    assert np.isfinite(data.standardized).all()
    assert np.all(data.standardized[data.mask == 0] == 0)


def test_row_alignment_and_identifier_columns_rejected(synthetic_cohort):
    path = synthetic_cohort / "clinical.csv"
    frame = pd.read_csv(path)
    frame.iloc[::-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="row_index"):
        load_cohort(synthetic_cohort)
    frame.assign(patient_id="not-a-real-identifier").to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly"):
        load_cohort(synthetic_cohort)


def test_output_and_input_cannot_be_inside_checkout(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        external_path(ROOT / "validation/results", ROOT)
    link = tmp_path / "repository_link"
    link.symlink_to(ROOT, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        external_path(link / "validation/input", ROOT)


def test_differentiable_design_matches_refit_design(synthetic_cohort):
    data = load_cohort(synthetic_cohort)
    target = brits.reference_target(data.clinical, data.values, data.mask)
    actual = brits.differentiable_design(
        torch.tensor(data.standardized),
        torch.tensor(data.mask),
        torch.tensor(target["design"][:, :10]),
        torch.tensor(data.marker_mean),
        torch.tensor(data.marker_sd),
    ).numpy()
    np.testing.assert_allclose(actual, target["design"], atol=2e-6, rtol=2e-6)


def test_identity_imputation_recovers_reference(synthetic_cohort):
    data = load_cohort(synthetic_cohort)
    hidden = ev._hide_observed_cells(data.mask, 0.2, np.random.default_rng(9), [3, 4, 5])
    reference = ev.reference_coefficients(data.clinical, data.values, data.mask)
    rows, metrics = ev.pooled_effect_rows(
        data.clinical,
        [data.values, data.values],
        data.values,
        data.mask,
        hidden,
        reference,
        9,
        0,
        0.2,
        "BRITS-MI",
        "BRITS-MI",
    )
    assert len(rows) == 19
    assert metrics["effect_abs_bias"] == pytest.approx(0)
    assert metrics["effect_coverage"] == 1
    assert metrics["effect_terms"] == 12


def test_bootstrap_draws_keep_observed_cells(synthetic_cohort):
    data = load_cohort(synthetic_cohort)
    model = brits.RealBRITSHistory(6, 4, 2, 8).eval()
    draws = brits.stochastic_draws(
        [model], [{}], data.standardized, data.mask, data.static, [3, 4, 5], 1, 0, 99
    )
    np.testing.assert_array_equal(draws[0][data.mask == 1], data.standardized[data.mask == 1])


def test_validation_runner_synthetic_smoke(synthetic_cohort, tmp_path):
    output = tmp_path / "results"
    args = parse_args(
        [
            "--input-dir",
            str(synthetic_cohort),
            "--output",
            str(output),
            "--repeats",
            "1",
            "--missing-percentages",
            "20",
            "--methods",
            "brits",
            "mean",
            "--epochs",
            "1",
            "--ensemble-size",
            "2",
            "--hidden-size",
            "8",
        ]
    )
    metrics, coefficients = run(args)
    assert set(metrics.method) == {"BRITS-MI", "Mean imputation"}
    assert len(coefficients) == 38
    assert np.isfinite(metrics.masked_cell_standardized_rmse).all()
    assert metrics.effect_coverage.between(0, 1).all()
    assert pd.read_csv(output / "term_summary.csv").shape[0] == 38
    assert {p.suffix for p in output.iterdir()} <= {".csv", ".json"}
    for path in output.glob("*.csv"):
        assert "patient_id" not in pd.read_csv(path).columns
    with pytest.raises(ValueError, match="empty output"):
        run(args)


def test_duplicate_settings_rejected(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input-dir",
                str(tmp_path),
                "--output",
                str(tmp_path),
                "--missing-percentages",
                "20",
                "20",
            ]
        )
