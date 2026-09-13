#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_large_adaptive_effect_recovery as run_large  # noqa: E402


HARDER_SCENARIOS = {
    "mild_harder": {
        "label": "Mild harder irregularity",
        "time": np.array([0.00, 0.10, 0.27, 0.50, 0.77, 1.00]),
        "late_accel_mult": 1.05,
        "pulse_mult": 1.05,
        "noise_mult": 1.05,
        "missing_abnormality_mult": 1.10,
        "missing_slope_mult": 1.15,
    },
    "baseline_harder": {
        "label": "Baseline harder irregularity",
        "time": np.array([0.00, 0.07, 0.22, 0.46, 0.73, 1.00]),
        "late_accel_mult": 1.30,
        "pulse_mult": 1.25,
        "noise_mult": 1.15,
        "missing_abnormality_mult": 1.40,
        "missing_slope_mult": 1.50,
    },
    "strong_harder": {
        "label": "Strong harder irregularity",
        "time": np.array([0.00, 0.03, 0.14, 0.37, 0.67, 1.00]),
        "late_accel_mult": 1.65,
        "pulse_mult": 1.45,
        "noise_mult": 1.25,
        "missing_abnormality_mult": 1.80,
        "missing_slope_mult": 2.00,
    },
}

run_large.LOCKED_SCENARIOS.clear()
run_large.LOCKED_SCENARIOS.update(HARDER_SCENARIOS)
run_large.sim.SCENARIOS.update(HARDER_SCENARIOS)


if __name__ == "__main__":
    run_large.main()
