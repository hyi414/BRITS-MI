"""Run a simulation grid from a versioned JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brits_mi.config import LossWeights, TrainingConfig
from brits_mi.runner import run_simulation_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    loss_settings = settings.get("losses", {})
    training = TrainingConfig(
        epochs=settings["epochs"],
        hidden_size=settings["hidden_size"],
        batch_size=settings["batch_size"],
        seed=args.first_seed,
        device=args.device,
        losses=LossWeights(**loss_settings),
    )
    run_simulation_grid(
        output=args.output,
        n_runs=settings["n_runs"],
        n_subjects=settings["n_subjects"],
        missing_rates=settings["missing_rates"],
        scenario=settings["scenario"],
        n_imputations=settings["n_imputations"],
        training=training,
        first_seed=args.first_seed,
    )


if __name__ == "__main__":
    main()
