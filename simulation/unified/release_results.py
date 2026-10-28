"""Verify saved synthetic benchmarks and build public aggregate tables/figures.

No clinical input is accepted. Original job records and source trees are read-only.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

METHODS = ["BRITS-MI", "MICE", "missForest", "Mean imputation", "Complete case", "Full data"]
COLORS = dict(zip(METHODS, ["#007C83", "#7752AE", "#D97923", "#7D8890", "#9A6144", "#222C34"]))
PRIMARY = METHODS[:3]
NAMES = {"baseline_harder": "Baseline", "mild_harder": "Mild", "strong_harder": "Strong", "image": "Clinical + image"}
KEYS = ["scenario", "missing_percentage", "method"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_and_read(directory, source):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["kind"] not in {"clinical", "image"} or manifest.get("input_sha256"):
        raise ValueError("Only synthetic clinical/image benchmarks may be published")
    from brits_mi.benchmark import source_hash
    if source_hash(source) != manifest["source_sha256"]:
        raise ValueError("frozen source hash differs from benchmark manifest")
    scenarios = manifest["scenarios"] if manifest["kind"] == "clinical" else ["image"]
    expected = {(s, p, seed) for s in scenarios for p in manifest["missing_percentages"]
                for seed in range(manifest["seed_start"], manifest["seed_start"] + manifest["runs"])}
    seen, metrics, coefficients, curves, hashes, events = set(), [], [], [], {}, []
    for path in sorted((directory / "jobs").glob("*/result.json")):
        result = json.loads(path.read_text())
        job = result["job"]
        key = (job["scenario"], job["missing_percentage"], job["seed"])
        if key not in expected or key in seen:
            raise ValueError(f"unexpected or duplicate synthetic job: {key}")
        seen.add(key)
        if {m["method"] for m in result["metrics"]} != set(METHODS):
            raise ValueError(f"incomplete comparator set: {key}")
        if len({m["input_hash"] for m in result["metrics"]}) != 1:
            raise ValueError(f"unmatched method inputs: {key}")
        if len(result["metrics"]) != len(METHODS):
            raise ValueError("duplicate method rows")
        meta = {k: job[k] for k in ("kind", "scenario", "n", "m", "missing_percentage", "seed", "mechanism")}
        metrics.extend([{**meta, **m} for m in result["metrics"]])
        coefficients.extend([{**meta, **r} for r in result["coefficients"]])
        valid_methods = {m["method"] for m in result["metrics"] if m["status"] == "valid"}
        if manifest["kind"] == "clinical":
            for method in valid_methods:
                terms = [r for r in result["coefficients"] if r["method"] == method]
                if len(terms) != 13 or len({r["term"] for r in terms}) != 13:
                    raise ValueError("each valid clinical fit must retain all 13 coefficients")
                for r in terms:
                    np.testing.assert_allclose(r["signed_error"], r["estimate"] - r["truth"])
                    np.testing.assert_allclose(r["squared_error"], r["signed_error"] ** 2)
                    total = r["within_variance"] + (1 + 1 / r["n_imputations"]) * r["between_variance"]
                    np.testing.assert_allclose(r["standard_error"] ** 2, total)
                    half = stats.t.ppf(.975, r["degrees_of_freedom"]) * r["standard_error"]
                    np.testing.assert_allclose([r["lower"], r["upper"]], [r["estimate"] - half, r["estimate"] + half])
                    if bool(r["covered"]) != (r["lower"] <= r["truth"] <= r["upper"]):
                        raise ValueError("coefficient interval inclusion is inconsistent")
        for curve in result.get("curves", []):
            for x, y, z in zip(curve["grid"], curve["tpr"], curve["precision"]):
                if not all(np.isfinite([x, y, z])) or not all(0 <= v <= 1 for v in [x, y, z]):
                    raise ValueError("invalid ROC/precision-recall coordinates")
                curves.append({**meta, "method": curve["method"], "class": curve["cls"],
                               "grid": x, "tpr": y, "precision": z})
        for audit in result["audit"]:
            if audit["method"] in {"mice", "missForest"}:
                events.append({**meta, "method": audit["method"], "version": audit["version"],
                    "logged_events": audit.get("logged_events", 0),
                    "outcome_targets": sum(bool(x.get("outcome_predictor")) for x in audit.get("target_models", []))})
        hashes[str(path.relative_to(directory))] = sha(path)
    if seen != expected:
        raise ValueError(f"benchmark incomplete: {len(seen)} of {len(expected)} jobs")
    if list((directory / "jobs").glob("*/error.json")):
        raise ValueError("unexpected job error artifacts remain")
    return manifest, pd.DataFrame(metrics), pd.DataFrame(coefficients), pd.DataFrame(curves), hashes, pd.DataFrame(events)


def summarise(metrics, coefficients):
    rows, terms = [], []
    rng = np.random.default_rng(67011)
    for key, part in metrics.groupby(KEYS, sort=True):
        valid = part[part.status.eq("valid")]
        row = dict(zip(KEYS, key)) | {"attempted": len(part), "valid": len(valid), "invalid": len(part) - len(valid)}
        for metric in ("coverage", "coefficient_mse", "mean_model_se", "imputation_nrmse", "trajectory_nrmse", "auroc", "auprc", "log_loss", "seconds"):
            if metric not in valid:
                continue
            x = valid[metric].dropna()
            row[metric] = x.mean()
            row[metric + "_mcse"] = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        if len(coefficients):
            co = coefficients[(coefficients.scenario == key[0]) & (coefficients.missing_percentage == key[1]) & (coefficients.method == key[2])]
            wide = co.pivot(index="seed", columns="term", values="signed_error")
            if len(wide):
                if wide.isna().any().any():
                    raise ValueError("missing term inside an otherwise valid coefficient fit")
                row["mean_absolute_bias"] = wide.mean().abs().mean()
                a = wide.to_numpy()
                boot = rng.integers(0, len(a), size=(1000, len(a)))
                row["mean_absolute_bias_mcse"] = np.abs(a[boot].mean(1)).mean(1).std(ddof=1)
                for term, t in co.groupby("term"):
                    terms.append(dict(zip(KEYS, key)) | dict(term=term, n_valid=len(t),
                        bias=t.signed_error.mean(), bias_mcse=t.signed_error.std(ddof=1)/np.sqrt(len(t)),
                        mse=t.squared_error.mean(), mse_mcse=t.squared_error.std(ddof=1)/np.sqrt(len(t)),
                        coverage=t.covered.mean(), coverage_mcse=t.covered.std(ddof=1)/np.sqrt(len(t)),
                        model_se=t.standard_error.mean(), empirical_se=t.estimate.std(ddof=1),
                        within_variance=t.within_variance.mean(), between_variance=t.between_variance.mean()))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(terms)


def contrasts(metrics):
    rows = []
    for key, part in metrics.groupby(KEYS[:2]):
        for metric in ("coefficient_mse", "coverage", "imputation_nrmse", "auroc", "auprc"):
            if metric not in part:
                continue
            wide = part[part.status.eq("valid")].pivot(index="seed", columns="method", values=metric)
            for comparator in ("MICE", "missForest", "Mean imputation"):
                paired = wide[["BRITS-MI", comparator]].dropna()
                if len(paired) < 2:
                    continue
                diff = paired["BRITS-MI"] - paired[comparator]
                mcse = diff.std(ddof=1) / np.sqrt(len(diff))
                rows.append(dict(zip(KEYS[:2], key)) | dict(metric=metric, comparator=comparator,
                    pairs=len(diff), difference=diff.mean(), mcse=mcse,
                    lower=diff.mean() - stats.t.ppf(.975, len(diff)-1)*mcse,
                    upper=diff.mean() + stats.t.ppf(.975, len(diff)-1)*mcse,
                    p_value=stats.ttest_1samp(diff, 0).pvalue if mcse > 0 else np.nan))
    return pd.DataFrame(rows)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DEE3E6", linewidth=.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=12)


def legend(fig, methods=PRIMARY):
    fig.legend(handles=[Line2D([0], [0], marker="o", lw=2, color=COLORS[m], label=m) for m in methods],
               loc="upper center", bbox_to_anchor=(.5, .935), ncol=len(methods), frameon=False, fontsize=13)


def panel_figure(table, scenario, title, path, image=False):
    specs = [("auroc", "AUROC", "Higher is better"), ("auprc", "AUPRC", "Higher is better"),
             ("imputation_nrmse", "Imputation NRMSE", "Lower is better"),
             ("trajectory_nrmse", "Trajectory-summary NRMSE", "Lower is better")] if image else [
             ("mean_absolute_bias", "Mean absolute coefficient bias", "Lower is better"),
             ("coefficient_mse", "Coefficient MSE", "Lower is better"),
             ("coverage", "95% interval coverage", "Target: 0.95"),
             ("imputation_nrmse", "Imputation NRMSE", "Lower is better")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=.1, right=.97, bottom=.11, top=.81, hspace=.52, wspace=.28)
    fig.suptitle(title, fontsize=20, y=.985, fontweight="bold")
    legend(fig)
    for letter, ax, (metric, label, direction) in zip("ABCD", axes.flat, specs):
        for offset, method in zip([-.9, 0, .9], PRIMARY):
            data = table[table.scenario.eq(scenario) & table.method.eq(method)].sort_values("missing_percentage")
            ax.errorbar(data.missing_percentage + offset, data[metric], yerr=1.96 * data[metric + "_mcse"],
                fmt="o-", color=COLORS[method], ms=7, lw=1.8, elinewidth=1.8, capsize=4)
        if metric == "coverage":
            ax.axhline(.95, color="#24353C", ls="--", lw=1.2)
            ax.set_ylim(max(0, ax.get_ylim()[0] - .01), min(1.005, max(.975, ax.get_ylim()[1])))
        ax.set_title(f"{letter}  {label}\n{direction}", loc="left", fontsize=14, pad=14)
        ax.set_ylabel(label, fontsize=13)
        ax.set_xlabel("Target missing percentage", fontsize=13)
        ax.set_xticks([20, 40, 60], ["20%", "40%", "60%"])
        style(ax)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=240)
    return fig


def curve_figure(curves, table, path, title="Clinical + image: archived 200-run comparison"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(left=.07, right=.98, bottom=.09, top=.87, hspace=.35, wspace=.27)
    fig.suptitle(title, fontsize=20, fontweight="bold")
    for c, percent in enumerate([20, 40, 60]):
        for r, (metric, xlabel, ylabel, area) in enumerate([
                ("tpr", "False-positive rate", "True-positive rate", "auroc"),
                ("precision", "Recall", "Precision", "auprc")]):
            ax = axes[r, c]
            for method in METHODS[:4] + ["Full data"]:
                part = curves[curves.missing_percentage.eq(percent) & curves.method.eq(method)]
                # Average within cohort across classes, then across independent cohorts.
                lines = part.groupby(["seed", "grid"])[metric].mean().unstack("grid")
                avg = lines.mean()
                band = 1.96 * lines.std(ddof=1) / np.sqrt(len(lines))
                auc = table.loc[table.missing_percentage.eq(percent) & table.method.eq(method), area].iloc[0]
                ax.plot(avg.index, avg, color=COLORS[method], lw=2, label=f"{method}: {auc:.3f}")
                ax.fill_between(avg.index, np.maximum(0, avg-band), np.minimum(1, avg+band), color=COLORS[method], alpha=.08)
            ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel=xlabel, ylabel=ylabel)
            ax.set_title(f"{percent}% missing | {'ROC' if r == 0 else 'Precision-recall'}", fontsize=15)
            ax.legend(frameon=False, fontsize=10, loc="lower right" if r == 0 else "upper right")
            style(ax)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=240)
    return fig


def display_table(table):
    records = []
    columns = [x for x in ["mean_absolute_bias", "coefficient_mse", "coverage", "imputation_nrmse", "trajectory_nrmse", "auroc", "auprc", "log_loss"] if x in table]
    for _, row in table.iterrows():
        item = {"Scenario": NAMES.get(row.scenario, row.scenario), "Missing percentage": f"{row.missing_percentage}%",
                "Method": row.method, "Valid / attempted": f"{int(row.valid)} / {int(row.attempted)}"}
        for col in columns:
            value = row.get(col)
            item[col.replace("_", " ")] = f"{value:.3f} ({row[col + '_mcse']:.3f})" if pd.notna(value) else "Not estimable"
        records.append(item)
    return pd.DataFrame(records).to_html(index=False, escape=True, classes="results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    sections, verification = [], {}
    with PdfPages(args.output / "updated_figures.pdf") as book:
        for name in ("clinical_n500_200", "clinical_n2000_200", "clinical_mar_200", "image_200"):
            manifest, metrics, co, curves, hashes, events = verify_and_read(args.evaluation / name, args.frozen_source)
            destination = args.output / name
            destination.mkdir(exist_ok=True)
            table, terms = summarise(metrics, co)
            table.to_csv(destination / "table.csv", index=False)
            if len(terms):
                terms.to_csv(destination / "individual_coefficients.csv", index=False)
            contrasts(metrics).to_csv(destination / "paired_contrasts.csv", index=False)
            events.to_csv(destination / "native_comparator_diagnostics.csv", index=False)
            (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (destination / "result_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
            verification[name] = {"complete": True, "jobs": len(hashes), "runs_per_setting": manifest["runs"],
                "source_hash_verified": True, "matched_method_inputs": True,
                "all_13_terms_and_pooling_replayed": bool(len(co)),
                "observed_cell_invariance": "covered by implementation tests; completed runs did not retain draws"}
            figs = []
            for scenario in sorted(table.scenario.unique()):
                title = ("Clinical association: " + NAMES[scenario] + (" | n = 2,000" if manifest["n"] == 2000 else "")
                         + (" | observed-history masking" if manifest["mechanism"] == "mar" else "")) if name != "image_200" else "Clinical + image: archived comparison"
                path = destination / scenario
                fig = panel_figure(table, scenario, title, path, image=name == "image_200")
                book.savefig(fig)
                plt.close(fig)
                figs.append(path.relative_to(args.output).with_suffix(".png").as_posix())
            if name == "image_200":
                fig = curve_figure(curves, table, destination / "roc_pr_curves")
                book.savefig(fig)
                plt.close(fig)
                figs.append((destination.relative_to(args.output) / "roc_pr_curves.png").as_posix())
            warning = "<p class='warning'>Archived image results: training labels entered MICE/missForest; BRITS-MI/MICE received 20 classifier fits, single-imputation methods received one. The corrected comparison is a separate run, not these numbers.</p>" if name == "image_200" else ""
            sections.append(f"<section id='{name}'><h2>{html.escape(name.replace('_', ' '))}</h2>{warning}"
                + "".join(f"<img src='{f}' alt='Simulation metrics with Monte Carlo uncertainty'>" for f in figs)
                + display_table(table) + f"<p><a href='{name}/table.csv'>Numerical table</a> | <a href='{name}/paired_contrasts.csv'>Matched contrasts</a>"
                + (f" | <a href='{name}/individual_coefficients.csv'>All 13 coefficient terms</a>" if len(co) else "") + "</p></section>")
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    intro = """<h1>Unified recurrent BRITS-MI: updated results</h1>
    <p>Completed evaluation: 200 independent cohorts per scenario and missing percentage; M = 20 for multiple imputation. The current conditional-mean, bootstrap and residual-noise engine is distinct from the newer small joint-likelihood/HMC pilot.</p>
    <h2>Interpretation</h2><p>At n = 500, BRITS-MI has lower coefficient MSE than MICE and missForest in the original clinical scenarios. Its reconstruction error is higher. At n = 2,000, MICE has lower coefficient MSE and coverage closer to 0.95. These differences are retained rather than selecting only favorable settings.</p>
    <p>Points show estimates across independent cohorts; bars are 1.96 times Monte Carlo SE, not individual coefficient confidence intervals. Overall coverage averages the 13 term indicators within a cohort before averaging across cohorts. Mean absolute bias averages the absolute term-specific biases; its Monte Carlo SE uses 1,000 cohort bootstrap resamples. Other Monte Carlo SEs use between-cohort variation. All methods and invalid-fit denominators are in the tables; figures emphasize the three primary competing imputers. AUPRC is macro average precision, not the area under the interpolated displayed precision-recall line.</p>
    <p>No real-data records or results are included in this public bundle. The 3,995-person TE source and the older 2,209-person eligible analysis are different populations. The newer full-cohort, 18-term specification still needs its own masking analysis.</p>"""
    page = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Unified BRITS-MI results</title><style>body{font-family:Georgia,serif;max-width:1400px;margin:38px auto;padding:0 26px;color:#22343e;background:#f8fafb}h1,h2{color:#075d67}p{font-size:18px;line-height:1.55;max-width:1100px}section{background:white;padding:22px;margin:28px 0;overflow:auto}img{max-width:100%;height:auto}.results{border-collapse:collapse;font-family:system-ui;font-size:13px;white-space:nowrap}.results td,.results th{padding:9px;border-bottom:1px solid #cdd8dd;text-align:right}.results th{background:#e7eff1}.warning{border-left:5px solid #bc751e;padding-left:16px}a{color:#006977}</style>" + intro + "".join(sections) + "</html>"
    (args.output / "report.html").write_text(page)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
