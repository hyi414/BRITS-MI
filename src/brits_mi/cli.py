"""Command-line interface for the packaged clinical-association simulation."""

from __future__ import annotations

import argparse

from .config import TrainingConfig
from .runner import run_simulation_grid
from .simulation import SCENARIOS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="directory for tidy CSV results")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--subjects", type=int, default=500)
    parser.add_argument("--missing-rates", type=float, nargs="+", default=[0.20])
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="baseline_harder",
    )
    parser.add_argument("--imputations", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--variance-factor", type=float, default=0.85)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_simulation_grid(
        output=args.output,
        n_runs=args.runs,
        n_subjects=args.subjects,
        missing_rates=args.missing_rates,
        scenario=args.scenario,
        n_imputations=args.imputations,
        first_seed=args.first_seed,
        training=TrainingConfig(
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            variance_factor=args.variance_factor,
            seed=args.first_seed,
            device=args.device,
        ),
    )


if __name__ == "__main__":
    main()
