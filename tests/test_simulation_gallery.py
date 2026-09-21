"""Check public experiment routing and result provenance without running benchmarks."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "simulation" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clinical_launcher_matches_seeds(tmp_path):
    runner = load_script("run_simulation")
    args = SimpleNamespace(experiment="clinical", subjects=500, runs=2, first_seed=1200000,
                           workers=1, missing_percentages=[20, 40, 60], output=tmp_path,
                           methods=["brits", "mice", "missforest", "baselines"])
    commands = runner.commands(args)
    assert len(commands) == 8
    for command in commands:
        script = command[1]
        assert Path(script).is_file()
        seed = int(command[command.index("--seed-start")+1])
        actual = seed + 100000 if "run_package_default_association" in script else seed
        assert actual == args.first_seed


def test_gallery_excludes_invalid_image_arm_and_matches_pairs():
    plotting = load_script("plot_results")
    data = plotting.image_metrics()
    assert not data.method.str.contains("mice|zero_fill").any()
    assert data.groupby("method").seed.nunique().eq(500).all()
    tests = plotting.image_tests(data)
    assert tests.matched_runs.eq(500).all()
    assert (tests.p_holm >= tests.p_two_sided).all()
    assert tests.p_holm.between(0, 1).all()
    assert np.isfinite(tests.mean_difference).all()


def test_actual_clinical_counts_and_metrics():
    plotting = load_script("plot_results")
    for n, expected in [(500, 200), (2000, 100)]:
        frame = plotting.clinical_summary(n)
        practical = frame[frame.method_label.isin(plotting.PRACTICAL)]
        assert practical.n_runs.eq(expected).all()
        assert set(practical.metric) == {
            "bias", "mse", "coverage", "imputation_rmse", "trajectory_rmse"
        }
        assert (practical.ci_low <= practical.estimate).all()
        assert (practical.ci_high >= practical.estimate).all()


def test_complete_case_counts_and_zero_success_interval():
    plotting = load_script("plot_results")
    data = pd.read_csv(ROOT / "simulation/results/complete_case/fit_audit.csv")
    counts = data.groupby("missing_percentage").valid_finite_mle.agg(["count", "sum"])
    assert counts["count"].tolist() == [500, 500]
    assert counts["sum"].tolist() == [4, 0]
    lo, hi = plotting.wilson_interval(0, 500)
    assert lo == 0 and 0 < hi < .01
