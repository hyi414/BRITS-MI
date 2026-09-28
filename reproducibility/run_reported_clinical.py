"""Compatibility entry point; use simulation/run_clinical.py for new runs."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "simulation/run_clinical.py"),
        run_name="__main__",
    )
