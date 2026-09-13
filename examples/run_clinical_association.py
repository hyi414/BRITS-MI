"""Minimal end-to-end BRITS-MI clinical-association example."""

from brits_mi.config import TrainingConfig
from brits_mi.runner import run_simulation_grid

if __name__ == "__main__":
    run_simulation_grid(
        output="example_output",
        n_runs=2,
        n_subjects=200,
        missing_rates=[0.20, 0.40],
        scenario="baseline_irregular",
        n_imputations=5,
        training=TrainingConfig(epochs=10, hidden_size=32, batch_size=64),
    )
