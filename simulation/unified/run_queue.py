"""Sequential benchmark stages from a frozen source tree; no cohort data are copied."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def save(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan["source_root"])
    destination = Path(plan["output_root"])
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / (plan["name"] + "_queue_status.json")
    state = dict(state="running", pid=os.getpid(), stages=[],
                 started_at=datetime.now(timezone.utc).isoformat())
    save(status_path, state)
    environment = os.environ.copy()
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        environment[name] = "1"
    environment["MPLCONFIGDIR"] = "/private/tmp/brits_mpl"
    for stage in plan["stages"]:
        command = [sys.executable, str(root / "simulation/unified/run_benchmark.py"),
                   "--output", str(destination / stage["name"]), *stage["arguments"]]
        log_path = destination / (stage["name"] + ".log")
        state["current_stage"] = stage["name"]
        state["command"] = command
        save(status_path, state)
        with log_path.open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=root, env=environment)
            state["child_pid"] = process.pid
            save(status_path, state)
            code = process.wait()
        report_path = destination / stage["name"] / "status.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        record = dict(name=stage["name"], exit_code=code,
                      benchmark_state=report.get("state", "no status"), log=str(log_path))
        state["stages"].append(record)
        if code or report.get("state") != "complete":
            state["state"] = "stopped_after_failure"
            save(status_path, state)
            return 1
        save(status_path, state)
    state["state"] = "complete"
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    state.pop("current_stage", None)
    state.pop("child_pid", None)
    save(status_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
