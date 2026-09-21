"""Rebuild the public results gallery without fitting models or accessing EHR data."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "simulation/results"
COLORS = {"BRITS-MI": "#007F82", "MICE": "#7653B7", "missForest": "#CE791C",
          "Mean imputation": "#84919A", "Full data": "#244A70", "Image only": "#B65D72",
          "BRITS-MI-image": "#007F82", "Forest surrogate": "#CE791C"}
PRACTICAL = ["BRITS-MI", "MICE", "missForest", "Mean imputation"]
IMAGE_NAMES = {"brits_joint_loss": "BRITS-MI-image", "missforest_fusion": "Forest surrogate",
               "mean_impute_fusion": "Mean imputation", "complete_fusion": "Full data",
               "image_only": "Image only"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titlesize": 16, "axes.labelsize": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#93A3AC", "axes.labelcolor": "#203B46",
                         "text.color": "#203B46", "svg.fonttype": "none",
                         "pdf.fonttype": 42})


def save(fig, output, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output / f"{name}.{ext}", dpi=220, bbox_inches="tight",
                    metadata={"Creator": "BRITS-MI simulation/plot_results.py"}
                    if ext != "png" else None)
    plt.close(fig)


def clinical_summary(n):
    folder = DATA / f"clinical_n{n}"
    frame = pd.read_csv(folder / "summary_with_monte_carlo_uncertainty.csv")
    runs = pd.read_csv(folder / "metrics_by_run.csv")
    rows = []
    for (rate, method), part in runs.groupby(["missing_rate", "method_label"]):
        values = part.trajectory_summary_rmse.dropna().to_numpy(float)
        if len(values) < 2:
            continue
        mean = values.mean()
        se = values.std(ddof=1) / np.sqrt(len(values))
        rows.append({"missing_rate": rate, "method_label": method,
                     "metric": "trajectory_rmse", "estimate": mean, "mcse": se,
                     "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
                     "n_runs": len(values)})
    frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    frame["method_label"] = frame.method_label.replace({"Missforest": "missForest"})
    return frame


def plot_clinical(n, output):
    frame = clinical_summary(n)
    frame.to_csv(output / f"clinical_n{n}_plot_data.csv", index=False)
    fig = plt.figure(figsize=(17, 10), layout="constrained")
    grid = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(grid[0, i:i+2]) for i in [0, 2, 4]]
    axes += [fig.add_subplot(grid[1, :3]), fig.add_subplot(grid[1, 3:])]
    metrics = [("bias", "Mean absolute coefficient bias"), ("mse", "Coefficient MSE"),
               ("coverage", "95% interval coverage"),
               ("imputation_rmse", "Missing-cell RMSE"),
               ("trajectory_rmse", "Trajectory-summary RMSE")]
    for index, (ax, (metric, title)) in enumerate(zip(axes, metrics)):
        for k, method in enumerate(PRACTICAL):
            part = frame[(frame.metric == metric) & (frame.method_label == method)]
            part = part.sort_values("missing_rate")
            y = part.estimate.to_numpy()
            errors = np.vstack([y - part.ci_low, part.ci_high - y])
            if (errors < -1e-10).any():
                raise ValueError("An archived interval does not enclose its estimate")
            ax.errorbar(100 * part.missing_rate + (k-1.5)*1.5, y,
                        yerr=np.maximum(errors, 0), fmt="o", ms=7, capsize=4,
                        elinewidth=1.8, color=COLORS[method], label=method)
        direction = "Target: 0.95" if metric == "coverage" else "Lower is better"
        ax.set_title(f"{chr(65+index)}  {title}\n{direction}", loc="left", pad=14)
        ax.set_xticks([20, 40, 60], ["20%", "40%", "60%"])
        ax.set_xlim(14, 66)
        ax.set_xlabel("Target missing percentage")
        ax.set_ylabel("Coverage" if metric == "coverage" else "Error")
        ax.grid(axis="y", color="#DAE1E4", alpha=.8)
        ax.set_axisbelow(True)
        if metric == "coverage":
            ax.axhline(.95, color="#49626D", ls="--", lw=1.3)
        else:
            ax.set_ylim(bottom=0)
    fig.suptitle(f"Clinical association and biomarker recovery | n = {n:,}",
                 fontsize=21, fontweight="bold")
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS[m], label=m)
               for m in PRACTICAL]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    save(fig, output, f"clinical_n{n}")


def image_metrics():
    frame = pd.read_csv(ROOT / "results/image_extension/metrics_by_seed.csv")
    frame = frame[frame.method.isin(IMAGE_NAMES)].copy()
    if frame.duplicated(["seed", "method"]).any():
        raise ValueError("Duplicate image replicate/method records")
    frame["method_label"] = frame.method.map(IMAGE_NAMES)
    frame["trajectory_rmse"] = np.sqrt(frame.summary_mse)
    return frame


def image_tests(frame):
    rows = []
    for metric in ["auroc", "auprc"]:
        wide = frame.pivot(index="seed", columns="method_label", values=metric)
        for method in ["Forest surrogate", "Mean imputation"]:
            pair = wide[["BRITS-MI-image", method]].dropna()
            delta = pair.iloc[:, 0] - pair.iloc[:, 1]
            se = stats.sem(delta)
            critical = stats.t.ppf(.975, len(delta)-1)
            test = stats.ttest_rel(pair.iloc[:, 0], pair.iloc[:, 1])
            rows.append({"metric": metric, "comparator": method, "matched_runs": len(pair),
                         "mean_difference": delta.mean(), "ci_low": delta.mean()-critical*se,
                         "ci_high": delta.mean()+critical*se, "p_two_sided": test.pvalue})
    result = pd.DataFrame(rows)
    order = np.argsort(result.p_two_sided.to_numpy())
    adjusted = np.minimum(1, np.maximum.accumulate(
        result.p_two_sided.to_numpy()[order] * np.arange(len(result), 0, -1)))
    result.loc[result.index[order], "p_holm"] = adjusted
    return result


def plot_image(output):
    frame = image_metrics()
    tests = image_tests(frame)
    tests.to_csv(output / "image_paired_tests.csv", index=False)
    frame.to_csv(output / "image_plot_data.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")
    metrics = [("auroc", "AUROC", "Higher is better"),
               ("auprc", "AUPRC", "Higher is better"),
               ("imputation_rmse", "Missing-cell RMSE", "Lower is better"),
               ("trajectory_rmse", "Trajectory-summary RMSE", "Lower is better")]
    for i, (ax, (metric, title, direction)) in enumerate(zip(axes.flat, metrics)):
        methods = list(IMAGE_NAMES.values()) if i < 2 else list(IMAGE_NAMES.values())[:3]
        data = [frame.loc[frame.method_label == m, metric].dropna() for m in methods]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=.58,
                        medianprops={"color": "#17333E", "linewidth": 2},
                        whiskerprops={"color": "#607681"}, capprops={"color": "#607681"})
        for box, method in zip(bp["boxes"], methods):
            box.set(facecolor=COLORS[method], edgecolor=COLORS[method], alpha=.8)
        ax.set_xticks(np.arange(1, len(methods)+1),
                      [m.replace(" ", "\n") for m in methods], fontsize=11)
        ax.set_title(f"{chr(65+i)}  {title}\n{direction}", loc="left", pad=14)
        ax.set_ylabel(title)
        ax.grid(axis="y", color="#DAE1E4")
        ax.set_axisbelow(True)
        if i < 2:
            test = tests[(tests.metric == metric) & (tests.comparator == "Forest surrogate")].iloc[0]
            ax.set_xlabel(f"BRITS-MI-image vs forest: mean difference {test.mean_difference:.3f}; "
                          f"Holm p = {test.p_holm:.2g}", labelpad=16, fontsize=11)
    fig.suptitle("Clinical + ultrasound fusion | archived image prototype", fontsize=21,
                 fontweight="bold")
    save(fig, output, "image_fusion")


def wilson_interval(k, n):
    z = stats.norm.ppf(.975)
    p = k / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return (0.0 if k == 0 else max(0, center-half),
            1.0 if k == n else min(1, center+half))


def plot_complete_case(output):
    audit = pd.read_csv(DATA / "complete_case/fit_audit.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    groups = [part for _, part in audit.groupby("missing_percentage")]
    percentages = sorted(audit.missing_percentage.unique())
    axes[0].boxplot([part.n_complete for part in groups], showfliers=False,
                    medianprops={"color": COLORS["BRITS-MI"], "linewidth": 2})
    axes[0].set_xticks(range(1, len(groups)+1), [f"{p}%" for p in percentages])
    axes[0].set_ylabel("Eligible subjects")
    axes[0].set_title("A  Complete-case sample size", loc="left")
    for i, part in enumerate(groups, 1):
        count = int(part.valid_finite_mle.sum())
        n = len(part)
        p = count / n
        lo, hi = wilson_interval(count, n)
        axes[1].errorbar(i, p, yerr=[[max(0, p-lo)], [max(0, hi-p)]], fmt="o", ms=9, capsize=5,
                         color=COLORS["BRITS-MI"], lw=2)
        axes[1].annotate(f"{count}/{n}", (i, hi), xytext=(0, 9), textcoords="offset points",
                         ha="center", fontweight="bold")
    axes[1].set_xticks(range(1, len(groups)+1), [f"{p}%" for p in percentages])
    axes[1].set_ylim(-.001, .029)
    axes[1].set_xlim(.5, 2.5)
    axes[1].set_ylabel("Proportion of identifiable finite fits")
    axes[1].set_title("B  Fit availability (95% Wilson intervals)", loc="left")
    for ax in axes:
        ax.set_xlabel("Target missing percentage")
        ax.grid(axis="y", color="#DAE1E4")
        ax.set_axisbelow(True)
    fig.suptitle("Complete-case analysis | 13 coefficients plus intercept", fontsize=20,
                 fontweight="bold")
    save(fig, output, "complete_case")


def plot_mnar(output):
    frame = pd.read_csv(DATA / "mnar/summary_with_monte_carlo_uncertainty.csv")
    frame["method_label"] = frame.method_label.replace({"Missforest": "missForest"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    metrics = [("absolute_bias", "Mean absolute coefficient bias"),
               ("coefficient_mse", "Coefficient MSE"), ("coverage", "95% interval coverage"),
               ("imputation_rmse", "Missing-cell RMSE")]
    for i, (ax, (metric, title)) in enumerate(zip(axes.flat, metrics)):
        for k, method in enumerate(PRACTICAL[:3]):
            part = frame[(frame.method_label == method) & (frame.metric == metric)]
            part = part.sort_values("mnar_strength")
            if part.empty:
                raise ValueError(f"Missing MNAR results: {method}, {metric}")
            y = part.estimate.to_numpy()
            ax.errorbar(part.mnar_strength + (k-1)*.045, y,
                        yerr=[y-part.ci_low, part.ci_high-y], fmt="o", ms=8,
                        capsize=4, lw=1.8, color=COLORS[method], label=method)
        direction = "Target: 0.95" if metric == "coverage" else "Lower is better"
        ax.set_title(f"{chr(65+i)}  {title}\n{direction}", loc="left", pad=14)
        ax.set_xticks([0, .5, 1], ["0", "0.5", "1.0"])
        ax.set_xlabel("MNAR coefficient per SD of unobserved abnormality")
        ax.set_ylabel("Coverage" if metric == "coverage" else "Error")
        ax.grid(axis="y", color="#DAE1E4")
        ax.set_axisbelow(True)
        if metric == "coverage":
            ax.axhline(.95, color="#49626D", ls="--", lw=1.3)
        else:
            ax.set_ylim(bottom=0)
    fig.suptitle("MNAR sensitivity | target missing percentage held at 40%",
                 fontsize=20, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save(fig, output, "mnar_sensitivity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "simulation/visualization")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    for n in [500, 2000]:
        plot_clinical(n, args.output)
    plot_image(args.output)
    plot_complete_case(args.output)
    plot_mnar(args.output)
    sources = sorted(DATA.rglob("*.csv")) + [ROOT / "results/image_extension/metrics_by_seed.csv"]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources}
    (args.output / "source_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rebuilt five figures, plot data, paired tests and source hashes in {args.output}")


if __name__ == "__main__":
    main()
