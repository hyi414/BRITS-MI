from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
R_SCRIPT = ROOT / "scripts" / "run_package_default_imputer.R"


def run_package_imputer(
    frame: pd.DataFrame,
    target_columns: list[str],
    method: str,
    m: int,
    seed: int,
) -> tuple[list[pd.DataFrame], dict[str, str]]:
    """Run package-native R mice or missForest and return target-cell draws."""
    if method not in {"mice", "mice_selected", "missforest"}:
        raise ValueError(f"Unsupported method: {method}")
    with tempfile.TemporaryDirectory(prefix=f"package_{method}_") as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / "input.csv"
        target_path = tmpdir / "targets.txt"
        output_path = tmpdir / "completed.csv"
        metadata_path = tmpdir / "metadata.txt"
        frame.to_csv(input_path, index=False)
        target_path.write_text("\n".join(target_columns) + "\n", encoding="utf-8")
        result = subprocess.run(
            [
                "Rscript",
                str(R_SCRIPT),
                str(input_path),
                str(target_path),
                method,
                str(output_path),
                str(metadata_path),
                str(m),
                str(seed),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Package {method} failed for seed {seed}: {result.stderr.strip()}"
            )
        completed = pd.read_csv(output_path)
        metadata: dict[str, str] = {}
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            key, value = line.split("=", 1)
            metadata[key] = value
        draws = []
        for draw_id in sorted(completed[".imp"].unique()):
            draw = completed.loc[
                completed[".imp"].eq(draw_id), target_columns
            ].reset_index(drop=True)
            draws.append(draw)
        return draws, metadata
