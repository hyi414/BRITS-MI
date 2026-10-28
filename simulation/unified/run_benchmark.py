"""Version-locked clinical/image/real-data jobs with explicit validity status."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import subprocess
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/brits_mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from brits_mi.benchmark import atomic_json, clinical_job, source_hash  # noqa: E402
from brits_mi.unified import UnifiedConfig  # noqa: E402


@contextmanager
def exclusive_run(directory):
    """A POSIX advisory lock prevents duplicate runners, including stale-PID reuse."""
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "RUNNING.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("a benchmark runner already holds this output lock") from exc
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "state": "running"}, handle)
        handle.flush()
        try:
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            json.dump({"pid": None, "state": "released"}, handle)
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def execute(task):
    job, config, directory, input_dir = task
    try:
        if job["kind"] == "clinical":
            result = clinical_job(job, UnifiedConfig(**config), directory)
        elif job["kind"] == "image":
            from brits_mi.recurrent_image import image_job
            result = image_job(job, UnifiedConfig(**config), directory)
        else:
            from validation.unified_validation import real_job
            if input_dir is None:
                raise ValueError("provide the confirmed cohort before real-data jobs")
            result = real_job(job, UnifiedConfig(**config), directory, input_dir)
        return {"directory": str(directory), "status": "completed", "job": job,
                "valid_methods": [r["method"] for r in result["metrics"] if r["status"] == "valid"],
                "invalid_methods": [r["method"] for r in result["metrics"] if r["status"] != "valid"]}
    except Exception:
        error = traceback.format_exc()
        atomic_json(Path(directory) / "error.json", {"job": job, "traceback": error})
        return {"directory": str(directory), "status": "failed", "job": job, "error": error}


def run(args):
    if args.kind == "real":
        if args.input_dir is None:
            raise ValueError("provide the confirmed cohort before real-data jobs")
        import pandas as pd
        source_rows = len(pd.read_csv(args.input_dir / "clinical.csv"))
        if source_rows != args.expected_cohort_size:
            raise ValueError(f"cohort has {source_rows} rows; expected {args.expected_cohort_size}")
        args.n = source_rows
    args.output.mkdir(parents=True, exist_ok=True)
    config = asdict(UnifiedConfig(epochs=args.epochs, bootstrap_members=args.bootstrap_members,
                                 residual_mode=args.residual_mode, hidden_size=args.hidden_size))
    settings = dict(kind=args.kind, runs=args.runs, n=args.n, m=args.m,
        missing_percentages=args.missing_percentages, scenarios=args.scenarios,
        mechanism=args.mechanism, seed_start=args.seed_start, config=config,
        source_sha256=source_hash(ROOT), pilot=args.pilot,
        input_sha256=None, runtime={name: importlib.metadata.version(name) for name in
            ["numpy", "scipy", "pandas", "statsmodels", "torch", "scikit-learn"]})
    settings["runtime"]["python"] = sys.version
    if args.kind == "image":
        settings["image_comparator_outcomes"] = "none"
        settings["classifier_fits_per_method"] = args.m
    settings["runtime"]["R"] = subprocess.check_output(
        ["Rscript", "-e", 'cat(R.version.string, "\\nmice=", as.character(packageVersion("mice")), "\\nmissForest=", as.character(packageVersion("missForest")), sep="")'],
        text=True).strip()
    if args.input_dir:
        settings["input_sha256"] = {name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
                                     for name in ["clinical.csv", "longitudinal.npz"]}
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise RuntimeError("incompatible resume: source code or scientific settings differ")
    atomic_json(manifest, settings)
    tasks, completed = [], []
    for scenario in (args.scenarios if args.kind == "clinical" else [args.kind]):
        for percent in args.missing_percentages:
            for repeat in range(args.runs):
                seed = args.seed_start + repeat
                job = dict(kind=args.kind, scenario=scenario, n=args.n, m=args.m,
                           missing_percentage=percent, seed=seed, mechanism=args.mechanism)
                if args.pilot and args.kind == "clinical":
                    job["methods"] = ["BRITS-MI", "BRITS-MI-correlated", "BRITS-MI-bootstrap",
                                      "MICE", "missForest", "Mean imputation", "Complete case", "Full data"]
                folder = args.output / "jobs" / f"{scenario}_{percent}_{seed}"
                if (folder / "result.json").exists():
                    completed.append(dict(directory=str(folder), status="completed", job=job, resumed=True))
                else:
                    tasks.append((job, config, folder, args.input_dir))
    total = len(tasks) + len(completed)
    started = time.monotonic()
    def status(state):
        atomic_json(args.output / "status.json", dict(state=state, total_jobs=total,
            completed_jobs=sum(x["status"] == "completed" for x in completed),
            failed_jobs=sum(x["status"] == "failed" for x in completed),
            remaining_jobs=total - len(completed),
            elapsed_seconds=time.monotonic() - started, last_jobs=completed[-5:], pid=os.getpid()))
    status("running")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending, index, stopped = set(), 0, False
        while index < len(tasks) or pending:
            while index < len(tasks) and len(pending) < args.workers and not stopped:
                if source_hash(ROOT) != settings["source_sha256"]:
                    raise RuntimeError("source changed while benchmark was running")
                if shutil.disk_usage(args.output).free < 8 * 1024**3:
                    status("stopped_low_disk")
                    return
                pending.add(executor.submit(execute, tasks[index]))
                index += 1
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                record = future.result()
                completed.append(record)
                stopped |= record["status"] == "failed"
                status("running")
                print(json.dumps(record), flush=True)
            if stopped and not pending:
                break
    status("complete_with_failures" if any(x["status"] == "failed" for x in completed) else "complete")
    from summarize import summarize
    summarize(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["clinical", "image", "real"], default="clinical")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--expected-cohort-size", type=int, default=3995)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--missing-percentages", type=int, nargs="+", default=[20, 40, 60])
    parser.add_argument("--scenarios", nargs="+", default=["mild_harder", "baseline_harder", "strong_harder"])
    parser.add_argument("--mechanism", choices=["original", "mar"], default="original")
    parser.add_argument("--seed-start", type=int, default=910000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--bootstrap-members", type=int, default=1)
    parser.add_argument("--residual-mode", choices=["iid", "correlated"], default="iid")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pilot", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.runs, arguments.n, arguments.m, arguments.workers) < 1:
        parser.error("positive counts required")
    if any(not 0 < p < 100 for p in arguments.missing_percentages):
        parser.error("missing percentages must lie between 0 and 100")
    with exclusive_run(arguments.output):
        run(arguments)
