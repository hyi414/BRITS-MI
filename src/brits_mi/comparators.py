"""Package-native R comparators with an explicit, shared predictor-information set."""

from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd


def comparator_frame(values, mask, times, static, outcome=None, context=None, lengths=None):
    n, visits, markers = values.shape
    values = np.where(mask, values, np.nan)
    if lengths is not None:
        valid = np.arange(visits)[None, :] < np.asarray(lengths)[:, None]
        # Structural padding is not an imputation target.
        values = np.where(valid[:, :, None], values, 0.0)
    targets = [f"marker_{j}_visit_{t}" for t in range(visits) for j in range(markers)]
    flat = values.reshape(n, -1)
    columns = {name: flat[:, j] for j, name in enumerate(targets)}
    if np.asarray(times).ndim == 1:
        times = np.broadcast_to(times, (n, visits))
    for t in range(visits):
        columns[f"time_{t}"] = times[:, t]
    for j in range(static.shape[1]):
        columns[f"static_{j}"] = static[:, j]
    if context is not None:
        for j in range(context.shape[1]):
            columns[f"image_{j}"] = context[:, j]
    if lengths is not None:
        columns["visits"] = lengths
    if outcome is not None:
        columns["outcome"] = np.where(np.asarray(outcome) >= 0, outcome, np.nan)
    return pd.DataFrame(columns), targets


def r_impute(values, mask, times, static, *, method, m, seed, outcome=None,
             context=None, lengths=None, maxit=5):
    if method not in {"mice", "missforest"}:
        raise ValueError("method must be mice or missforest")
    frame, targets = comparator_frame(values, mask, times, static, outcome, context, lengths)
    with tempfile.TemporaryDirectory(prefix="brits_private_") as directory:
        root = Path(directory)
        frame.to_csv(root / "input.csv", index=False)
        script = Path(__file__).parent / "r" / "impute.R"
        run = subprocess.run(["Rscript", str(script), str(root), method, str(m),
                              str(seed), str(maxit), str(len(targets))],
                             capture_output=True, text=True, check=False)
        if run.returncode:
            raise RuntimeError(f"{method} failed: {run.stderr[-3000:]}")
        long = pd.read_csv(root / "draws.csv")
        audit = pd.read_csv(root / "audit.csv").iloc[0].to_dict()
        events = root / "events.csv"
        audit["events"] = pd.read_csv(events).fillna("").to_dict("records") if events.exists() else []
        models = root / "target_models.csv"
        audit["target_models"] = pd.read_csv(models).fillna("").to_dict("records") if models.exists() else []
        audit["warnings"] = run.stderr
        draws = []
        for _, part in long.groupby("imputation", sort=True):
            draw = part[targets].to_numpy().reshape(values.shape)
            if not np.isfinite(draw).all():
                raise ValueError("comparator returned nonfinite imputed values")
            if not np.allclose(draw[mask], values[mask], rtol=1e-7, atol=1e-7):
                raise ValueError("comparator changed observed cells")
            draw[mask] = values[mask]
            draws.append(draw)
    return draws, audit
