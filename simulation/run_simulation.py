"""Run the public clinical, MNAR, or image experiments from one entry point."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "simulation/clinical/scripts"


def commands(args):
    """Keep the archived implementation canonical and match comparator seeds."""
    common = ["--n", str(args.subjects), "--nsim", str(args.runs)]
    seed = ["--seed-start", str(args.first_seed)]
    workers = ["--n-workers", str(args.workers)]
    rates = [str(p / 100) for p in args.missing_percentages]
    out = args.output.resolve()
    if args.experiment == "image":
        return [[sys.executable, "-m", "brits_mi.image_fusion",
                 "--output", str(out), "--n-subjects", str(args.subjects),
                 "--n-runs", str(args.runs), "--seed", str(args.first_seed),
                 "--missing-rate", rates[0], "--prototype-alpha", str(args.prototype_alpha),
                 "--trees", str(args.trees)]]
    if args.experiment == "mnar":
        return [[sys.executable, str(ARCHIVE / "run_clinical_association_mnar_sensitivity.py"),
                 *common, *seed, *workers, "--m", "20", "--missforest-draws", "5",
                 "--target-missing", rates[0], "--mnar-strengths", "0", "0.5", "1",
                 "--outdir", str(out)]]
    result = []
    for method in args.methods:
        if method in {"brits", "mice"}:
            for p, rate in zip(args.missing_percentages, rates):
                dest = str(out / f"{method}_missing{p}")
                if method == "brits":
                    result.append([
                        sys.executable, str(ROOT / "simulation/run_clinical.py"),
                        "--n", str(args.subjects), "--runs", str(args.runs), *seed,
                        "--missing-rate", rate, "--mean-weight", "0.20" if p == 60 else "0.10",
                        "--output", dest, "--execute",
                    ])
                else:
                    # The archival runner adds 100000 for baseline_harder.
                    result.append([
                        sys.executable, str(ARCHIVE / "run_package_default_association_comparators.py"),
                        *common, *workers, "--seed-start", str(args.first_seed - 100000),
                        "--scenarios", "baseline_harder", "--methods", "mice",
                        "--m-values", "50", "--target-missing", rate, "--outdir", dest,
                    ])
        else:
            name = ("run_missforest5_clinical_association.py" if method == "missforest"
                    else "run_simple_clinical_association_baselines.py")
            result.append([sys.executable, str(ARCHIVE / name), *common, *seed, *workers,
                           "--missing-rates", *rates, "--outdir", str(out / method)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["clinical", "mnar", "image"], required=True)
    parser.add_argument("--subjects", type=int, default=500)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--missing-percentages", type=int, nargs="+", default=[20, 40, 60])
    parser.add_argument("--methods", nargs="+", choices=["brits", "mice", "missforest", "baselines"],
                        default=["brits", "mice", "missforest", "baselines"])
    parser.add_argument("--first-seed", type=int, default=1200000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prototype-alpha", type=float)
    parser.add_argument("--trees", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Default: print commands only")
    args = parser.parse_args()
    if min(args.subjects, args.runs, args.workers) < 1 or args.first_seed < 100000:
        parser.error("Positive sample size, runs, workers and first-seed >=100000 required")
    if any(p <= 0 or p >= 100 for p in args.missing_percentages):
        parser.error("Missing percentages must be between 0 and 100")
    if args.experiment == "clinical":
        if args.subjects not in {500, 2000} or any(
            p not in {20, 40, 60} for p in args.missing_percentages
        ):
            parser.error("Locked clinical settings: subjects 500/2000 and missing 20/40/60")
    elif len(args.missing_percentages) != 1:
        parser.error("Specify one missing percentage for image or MNAR experiments")
    if args.experiment == "image" and args.prototype_alpha is None:
        parser.error("Image runs require explicit --prototype-alpha")
    jobs = commands(args)
    for job in jobs:
        print(shlex.join(job), flush=True)
    if not args.execute:
        print("Dry run only. Add --execute to launch these commands.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for index, job in enumerate(jobs, 1):
        subprocess.run(job, cwd=ROOT, check=True)
        (args.output / "launcher_progress.json").write_text(
            json.dumps({"completed_jobs": index, "total_jobs": len(jobs),
                        "requested_replicates": args.runs, "commands": jobs}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
