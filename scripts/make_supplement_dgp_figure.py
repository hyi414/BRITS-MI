"""Create the simulation-design diagnostic used in the JBI supplement."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from brits_mi.simulation import BIOMARKERS, SimulationConfig, simulate_clinical_association


COLORS = {"ast": "#007C83", "alt": "#7651B5", "platelet": "#D97706"}


def build_figure(output: Path, seed: int = 20260904) -> None:
    data = simulate_clinical_association(
        SimulationConfig(
            n_subjects=500,
            target_missing=0.40,
            scenario="baseline_harder",
            seed=seed,
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)

    rng = np.random.default_rng(seed + 1)
    selected = rng.choice(len(data.outcome), size=18, replace=False)
    ax = axes[0, 0]
    for marker, name in enumerate(BIOMARKERS):
        for subject in selected:
            ax.plot(
                data.times,
                data.true_values[subject, :, marker],
                color=COLORS[name],
                alpha=0.18,
                linewidth=1.0,
            )
        ax.plot([], [], color=COLORS[name], linewidth=3, label=name.upper())
    ax.set_title("A  Nonlinear biomarker trajectories on a shared irregular grid", loc="left")
    ax.set_xlabel("Scaled follow-up time")
    ax.set_ylabel("Standardized biomarker value")
    ax.legend(frameon=False, ncol=3, loc="upper left")

    ax = axes[0, 1]
    display = data.observed_mask[:50].transpose(0, 2, 1).reshape(50, -1)
    ax.imshow(display, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    boundaries = [5.5, 11.5]
    for boundary in boundaries:
        ax.axvline(boundary, color="white", linewidth=2.5)
    ax.set_title("B  Nonmonotone cell-level observation pattern", loc="left")
    ax.set_xlabel("Biomarker and visit")
    ax.set_ylabel("Illustrative subjects")
    ax.set_xticks([2.5, 8.5, 14.5], ["AST", "ALT", "Platelet"])
    ax.set_yticks([])

    ax = axes[1, 0]
    for marker, name in enumerate(BIOMARKERS):
        missing = (~data.observed_mask[:, :, marker]).mean(axis=0)
        ax.plot(
            data.times,
            missing,
            marker="o",
            markersize=6,
            linewidth=2.2,
            color=COLORS[name],
            label=name.upper(),
        )
    ax.axhline(0.40, color="#4B5563", linestyle="--", linewidth=1.4, label="Late-cell target")
    ax.set_title("C  Missingness varies by visit and biomarker", loc="left")
    ax.set_xlabel("Scaled follow-up time")
    ax.set_ylabel("Realized missing-cell proportion")
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    ax = axes[1, 1]
    abnormality = np.stack(
        [data.true_values[:, :, 0], data.true_values[:, :, 1], -data.true_values[:, :, 2]],
        axis=2,
    )
    labels = ["Least abnormal", "Q2", "Q3", "Q4", "Most abnormal"]
    x = np.arange(5)
    width = 0.24
    for marker, name in enumerate(BIOMARKERS):
        values = abnormality[:, :, marker].ravel()
        missing = (~data.observed_mask[:, :, marker]).ravel()
        bins = np.quantile(values, np.linspace(0, 1, 6))
        group = np.clip(np.digitize(values, bins[1:-1], right=True), 0, 4)
        rate = [missing[group == level].mean() for level in range(5)]
        ax.plot(
            x,
            rate,
            marker="o",
            markersize=6,
            linewidth=2.2,
            color=COLORS[name],
            label=name.upper(),
        )
    ax.set_title("D  Current unobserved abnormality informs deletion", loc="left")
    ax.set_xlabel("True current-value abnormality quintile")
    ax.set_ylabel("Realized missing-cell proportion")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=False, ncol=3, loc="upper left")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D9E2E8", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
    fig.suptitle(
        "Known-truth clinical-association data-generating process",
        fontsize=17,
        fontweight="bold",
        color="#17324D",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    build_figure(args.output, args.seed)


if __name__ == "__main__":
    main()
