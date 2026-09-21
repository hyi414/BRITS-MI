# Simulation code and visualized results

This folder is the entry point for running the clinical-association and
clinical-image experiments and inspecting their saved results. It contains
executable launchers, the complete-case diagnostic, public numerical outputs,
and a reproducible figure gallery. The model and DGP implementations remain
in their canonical source files linked below, so there are no competing copies
of the training code.

## Results gallery

![Clinical association benchmark](visualization/clinical_n500.png)

**Clinical association, n=500.** Points show across-replicate estimates;
bars show the archived 95% Monte Carlo intervals. Trajectory-summary RMSE
intervals are rebuilt as mean +/- 1.96 times the Monte Carlo SE. Methods are
offset horizontally at each target missing percentage. Coverage targets 0.95;
all error measures favor smaller values. Full-data and complete-case references
are available in the CSV tables but omitted from these practical-method plots.

![Clinical-image fusion](visualization/image_fusion.png)

**Clinical-image fusion.** Boxes summarize 500 independent synthetic replicates:
median, interquartile range, and whiskers extending to the most extreme points
within 1.5 interquartile ranges. Outliers are omitted only from display.
BRITS-MI-image here is the archived image-conditioned tree extension; the forest
comparator is a random-forest surrogate, not R package missForest. The invalid
historical image arm labeled MICE and zero-fill are excluded. Trajectory RMSE is
the square root of each replicate's trajectory-summary MSE, not the square
root of an across-replicate average. Pairwise tests use matched seeds, two-sided
paired t-tests, and Holm adjustment across two metrics and two practical
comparators. ROC/PR curves cannot be reconstructed from AUC summaries alone and
are not fabricated here.

[Larger-sample benchmark](visualization/clinical_n2000.png) |
[MNAR sensitivity](visualization/mnar_sensitivity.png) |
[Complete-case fit availability](visualization/complete_case.png) |
[All vector PDFs, SVGs, and plot data](visualization/)

## Saved experiment inventory

| Saved experiment | Subjects per replicate | Completed replicates | Other settings |
| --- | ---: | ---: | --- |
| Clinical association | 500 | 200 per missing percentage | 20%, 40%, 60%; BRITS-MI/MICE 50 completions; missForest 5 seeded fits |
| Larger-sample association | 2,000 | 100 per missing percentage | Same percentages and completion counts |
| MNAR strength sensitivity | 500 | 100 per strength | 40% target missingness; strengths 0, 0.5, 1; BRITS-MI/MICE 20 completions |
| Archived image prototype | 500 | 500 | 3-class outcome; 24 x 24 ultrasound-like images |
| Complete-case diagnostic | 500 | 500 attempts per missing percentage | 40% and 60%; only 4 and 0 valid finite fits, respectively |

Counts above describe the saved outputs. The command-line default of 500 runs
controls a **new** experiment; it does not change the number behind an existing
result. This folder contains synthetic-data results only. It does not include
the real cohort, clinical weights, masks, or patient-level records.

## Installation

From the repository root, in a Python environment:

```bash
python -m pip install -e '.[dev]'
```

Clinical BRITS-MI anchoring and the MICE/missForest comparators require R and
the `mice` and `missForest` packages. The archived versions were mice 3.18.0
and missForest 1.5; record package versions when using other releases.
Plotting the saved results requires Python only and does not train models.

## Run the experiments

### Clinical association

```bash
python simulation/run_simulation.py --experiment clinical \
  --subjects 500 --runs 500 --missing-percentages 20 40 60 \
  --output results_local/clinical
```

This prints all commands. Add `--execute` to run them. Select individual arms
with `--methods brits mice missforest baselines`. The `baselines` arm produces
mean-imputation, full-data, and complete-case results. The launcher matches
the underlying subject seeds across methods, including correction for the
historical MICE runner's scenario-specific seed offset. Existing seed-level
outputs should not be counted twice when combining batches.

The reported BRITS-MI clinical procedure combines cross-fitted recurrent
proposals with a missForest conditional-mean anchor. Neural mean weights are
0.10 at 20%/40% and 0.20 at 60%; residual weight is 0.70. The canonical launcher
preserves its 45-epoch ceiling, 48 hidden units, and 50 completions. Five
missForest fits quantify algorithmic variation, not automatically proper
posterior imputation. Archived MICE and missForest inputs exclude the outcome.

### Larger sample and MNAR sensitivity

Use `--subjects 2000` for the larger-sample clinical benchmark. For the separate
MNAR sensitivity experiment:

```bash
python simulation/run_simulation.py --experiment mnar \
  --subjects 500 --runs 500 --missing-percentages 40 \
  --output results_local/mnar --execute
```

This changes the unobserved-abnormality coefficient over 0, 0.5, and 1, while
recalibrating the missingness intercept to preserve the target percentage.
It uses 20 BRITS-MI/MICE completions and five missForest fits. It is a distinct
sensitivity configuration, not a relabeling of the primary benchmark.

### Clinical-image generation and fusion

```bash
python simulation/run_simulation.py --experiment image \
  --subjects 500 --runs 2 --missing-percentages 50 \
  --prototype-alpha 0.0 --trees 8 \
  --output results_local/image_smoke --execute
```

This short example runs the public image API. For a larger new experiment,
increase runs and trees explicitly. Image context comprises 17 pixel features
and training-fitted class probabilities; completion precedes a fusion classifier
using image features and 34 trajectory summaries. The synthetic images include
speckle, depth attenuation, capsule, vessel shadows, and stage-linked texture.
The percentage argument is a generator control in this image DGP, not a
guarantee of the realized percentage after probability clipping.

The historical prototype-correction invocation was not preserved, so this
command is not claimed to reproduce the archived 500-run image results exactly.
The saved results can still be visualized exactly from their released metrics.
See [image implementation details](../docs/image_fusion.md).

### Complete-case diagnostic

```bash
python simulation/run_complete_case.py --runs 500 --workers 4 \
  --outdir results_local/complete_case
```

Unlike the main launcher, this command runs immediately. It checks predictor
rank, outcome classes, complete/quasi-complete separation, convergence, and
finite positive standard errors. The original model has 13 coefficients plus
an intercept. At 40% missingness only 4/500 fits are valid; at 60% none are valid.
The gallery therefore shows fit availability, not misleading coefficient
coverage conditional on a few successful fits. The conditional numbers remain
in `results/complete_case/valid_fit_summary.csv` for auditability.

## Rebuild the visualizations

```bash
python simulation/plot_results.py
# Or write an independent gallery without overwriting the checked-in version:
python simulation/plot_results.py --output results_local/gallery
```

The script writes five PNG/PDF/SVG figures, clinical/image plot data, paired
image tests, and hashes identifying the source CSVs. It neither refits models
nor accesses a private dataset. The main gallery uses consistent method colors:
teal BRITS-MI, purple MICE, orange missForest/forest surrogate, and gray mean
imputation. Clinical figures show RMSE on the generator's scale, not NRMSE.

## Code map

| Component | Canonical implementation |
| --- | --- |
| Experiment launcher | [run_simulation.py](run_simulation.py) |
| Figure regeneration and paired tests | [plot_results.py](plot_results.py) |
| Complete-case fit checks | [run_complete_case.py](run_complete_case.py) |
| Clinical data and outcome construction | [generate_locked_dataset](../reproducibility/reported_pipeline/association/scripts/run_default_mice_clinical_association_sensitivity.py) |
| Trajectories and missingness | [trajectory generator](../reproducibility/reported_pipeline/association/scripts/run_trajectory_history_imputation_sim.py), [irregularity scenarios](../reproducibility/reported_pipeline/association/scripts/run_effect_recovery_harder_irregularity_test.py) |
| Recurrent training and downstream gradients | [neural training](../reproducibility/reported_pipeline/association/scripts/run_neural_brits_mi_downstream_gradient.py) |
| Reported clinical BRITS-MI composition | [anchored recurrent simulation](../reproducibility/reported_pipeline/association/scripts/run_refined_brits_mi_association.py) |
| MICE and missForest | [Python wrapper](../reproducibility/reported_pipeline/association/scripts/package_default_imputation.py), [R implementation](../reproducibility/reported_pipeline/association/scripts/run_package_default_imputer.R) |
| Outcome features, fitting, and pooling | [association analysis](../reproducibility/reported_pipeline/association/scripts/run_irregular_predictor_trajectory_sim.py) |
| Image/clinical DGP and renderer | [data.py](../src/longitudinal_sim/data.py) |
| Public image completion and fusion | [image_fusion.py](../src/brits_mi/image_fusion.py) |
| Historical image runner | [archived runner](../reproducibility/reported_pipeline/image_prototype/scripts/run_fast_image_fusion_nsim500.py) |
| MNAR experiment | [sensitivity runner](../reproducibility/reported_pipeline/association/scripts/run_clinical_association_mnar_sensitivity.py) |

## Reading the statistics

For coefficient error `e[r,j] = estimate[r,j] - truth[j]`, the summaries are:

- Mean absolute bias: average over terms of `abs(mean over replicates of e[r,j])`.
- Coefficient MSE: average of `e[r,j]^2` over replicates and terms.
- Coverage: average of the indicator that each nominal 95% interval contains its true coefficient; this is not simultaneous coverage.
- Missing-cell RMSE: error evaluated on artificially missing synthetic biomarker cells.
- Trajectory-summary RMSE: error in the derived late levels and slopes, calculated after trajectory completion on the original generated scale.

Clinical coefficient intervals in the gallery use the archived Monte Carlo
intervals rather than reinterpreting model standard errors as Monte Carlo
uncertainty. The complete-case availability plot uses Wilson intervals for
the proportion of valid fits, not coefficient confidence intervals.
