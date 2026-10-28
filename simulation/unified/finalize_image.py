"""Render a completed, corrected synthetic image comparison after verifying it."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from release_results import (verify_and_read, summarise, contrasts, panel_figure,
                             curve_figure, display_table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, metrics, co, curves, hashes, events = verify_and_read(args.run, args.frozen_source)
    if manifest["kind"] != "image" or manifest.get("image_comparator_outcomes") != "none":
        raise ValueError("not the corrected outcome-blind image comparison")
    for path in (args.run / "jobs").glob("*/result.json"):
        result = json.loads(path.read_text())
        valid = {r["method"].lower() for r in result["metrics"] if r["status"] == "valid"}
        for audit in result["audit"]:
            if audit["method"].lower() not in valid:
                continue
            if audit.get("classifier_fits_completed") != manifest["m"]:
                raise ValueError("unequal completed classifier fitting budget")
            if audit["method"].lower() in {"mice", "missforest"}:
                if audit.get("imputer_outcome_policy") != "none":
                    raise ValueError("comparator outcome policy differs")
                if any(t.get("outcome_predictor", False) for t in audit.get("target_models", [])):
                    raise ValueError("outcome entered a comparator target model")
    args.output.mkdir(parents=True, exist_ok=True)
    table, _ = summarise(metrics, co)
    table.to_csv(args.output / "table.csv", index=False)
    contrasts(metrics).to_csv(args.output / "paired_contrasts.csv", index=False)
    events.to_csv(args.output / "native_comparator_diagnostics.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Serif", "pdf.fonttype": 42})
    with PdfPages(args.output / "image_figures.pdf") as book:
        for fig in [panel_figure(table, "image", "Clinical + image: matched comparison",
                                args.output / "image_metrics", image=True),
                    curve_figure(curves, table, args.output / "image_curves",
                                 title="Clinical + image: matched comparison")]:
            book.savefig(fig)
            plt.close(fig)
    verification = {"complete": True, "jobs": len(hashes), "runs_per_percentage": manifest["runs"],
                    "source_hash_verified": True, "equal_classifier_fits": True,
                    "comparator_outcomes_absent": True, "manifest": manifest}
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    (args.output / "result_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    page = "<!doctype html><meta charset='utf-8'><title>Corrected image benchmark</title><style>body{font-family:Georgia;max-width:1400px;margin:auto;padding:30px}img{max-width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd}</style><h1>Recurrent clinical + image benchmark</h1>"
    page += f"<p>{manifest['runs']} independent cohorts per missing percentage; n = {manifest['n']}; M = {manifest['m']}. Image MICE and missForest are outcome-blind. All final methods use {manifest['m']} classifier fits and probability averaging.</p>"
    page += "<p>Points show means and bars show 1.96 Monte Carlo SE across cohorts. ROC and precision-recall lines are class-averaged within cohort, then averaged across cohorts; shaded bands are pointwise Monte Carlo intervals. AUPRC is macro average precision. All attempted/valid counts are retained; paired tests are two-sided and unadjusted.</p>"
    page += "<img src='image_metrics.png'><img src='image_curves.png'>" + display_table(table)
    (args.output / "report.html").write_text(page)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
