# Simulation

Experiment scripts for the manuscript's clinical-association and clinical-image
analyses. Run commands from the repository root after installing the package.
No clinical records are used by these simulations.

## Organization

| Location | Purpose |
| --- | --- |
| `run_simulation.py` | Main launcher: clinical comparisons, missingness sensitivity, or image API |
| `run_clinical.py` | Manuscript BRITS-MI clinical training, anchoring, and pooling |
| `clinical/scripts/` | Clinical DGP, recurrent training, competing methods, and sensitivity analyses |
| `image/scripts/` | Archived clinical-image comparison runner |
| `R/` | Standalone R comparator interface |
| `run_complete_case.py` | Complete-case estimability and separation checks |
| `diagnostics/` | Archived numerical and figure-verification scripts |
| `plot_results.py` | Optional visualization of the retained synthetic-data outputs |

Shared models and generators stay in `src/brits_mi/` and
`src/longitudinal_sim/`. They are imported, not duplicated, by the runners.

## Clinical-Association Comparison

```bash
python simulation/run_simulation.py --experiment clinical \
  --subjects 500 --runs 500 --missing-percentages 20 40 60 \
  --methods brits mice missforest baselines \
  --output results_local/clinical --execute
```

Without `--execute`, the launcher prints the commands. Use `--subjects 2000`
for the larger-sample setting, or select individual methods to rerun one arm.
Each method receives the same underlying simulation seeds and masking setting.

| Argument to `--methods` | Analysis |
| --- | --- |
| `brits` | Recurrent BRITS-MI draws with the specified missForest conditional-mean anchor |
| `mice` | R `mice`: 50 imputations, 5 iterations, default predictor matrix and conditional models |
| `missforest` | Five independently seeded R `missForest` fits, followed by outcome-model pooling |
| `baselines` | Mean imputation, complete-case analysis, and full-data analysis |

MICE and missForest use the supplied outcome-blind predictor frame in this
benchmark. The five missForest fits quantify algorithmic variation; they are
not claimed to be draws from a posterior imputation distribution. The clinical
BRITS-MI configuration uses 50 imputations, 48 hidden units, a 45-epoch ceiling,
a neural mean weight of 0.10 at 20%/40% and 0.20 at 60%, and residual weight 0.70.
The underlying scripts expose additional configuration arguments.

The DGP has six non-equidistant visits, three longitudinal biomarkers, static
covariates, and a logistic outcome with 13 non-intercept coefficients.
The missing percentage targets eligible cells in the last three visits, not
all cells in the trajectory. See the generator and outcome-model definitions
in `clinical/scripts/run_default_mice_clinical_association_sensitivity.py` and
`clinical/scripts/run_irregular_predictor_trajectory_sim.py`.

### Output Files

Each method writes its own directory under `--output`; the clinical launcher
does not merge different methods' files automatically.

| File | Contents |
| --- | --- |
| `coefficient_by_run.csv` | Per-run, per-coefficient estimates, model SE, signed error, squared error, and interval coverage |
| `metrics_by_run.csv` | Imputation and trajectory-summary error, plus aggregate association metrics |
| `effect_recovery_by_predictor.csv` | Across-run bias, MSE, coverage, and SE diagnostics for each coefficient, where written by the method runner |
| `effect_recovery_overall.csv` | Across-coefficient summaries, where written by the method runner |
| `package_audit.csv` / `fold_audit.csv` | Comparator versions or imputation diagnostics, where applicable |
| `config.json` | The method's run settings |
| `launcher_progress.json` | Requested jobs and completed-job count |

Retain the method runners' column names when joining outputs.
`run_complete_case.py` additionally writes
fit-availability and separation diagnostics. Do not calculate unconditional
coverage from only the few runs with estimable complete-case models.

## Missingness Sensitivity

```bash
python simulation/run_simulation.py --experiment mnar \
  --subjects 500 --runs 500 --missing-percentages 40 \
  --output results_local/mnar --execute
```

This varies the unobserved-abnormality coefficient over 0, 0.5, and 1 while
recalibrating the masking intercept. It uses 20 BRITS-MI/MICE imputations and
five missForest fits. It is a separate analysis configuration.

## Clinical + Image Input

For an end-to-end check of image generation, imputation, and fusion:

```bash
python simulation/run_simulation.py --experiment image \
  --subjects 500 --runs 2 --missing-percentages 50 \
  --prototype-alpha 0.0 --trees 8 \
  --output results_local/image_smoke --execute
```

This runs the public BRITS-MI-image API and writes `metrics.csv` and
`config.json`. It is a single-method example. Use larger tree and replicate
counts explicitly for a new analysis.

The archived multi-method comparison is
[`image/scripts/run_fast_image_fusion_nsim500.py`](image/scripts/run_fast_image_fusion_nsim500.py).
Its historical tree-based image arm, forest surrogate, and invalid historical
MICE arm must not be confused with the package-default clinical comparisons.
See [the implementation record](../docs/image_fusion.md) before interpreting
or rerunning that script. The public API does not reconstruct an unrecorded
historical tuning configuration.

## Dependencies and Reproducibility

Clinical comparisons require R, `mice`, and `missForest`; the archived package
versions were 3.18.0 and 1.5, respectively. Record the installed versions when
rerunning. The Python unit tests do not require R.

Saved synthetic summaries are retained under `results/` and previously
generated assets under `visualization/`, but are not displayed in this README.
Run counts in retained files describe completed experiments. The launcher
default of 500 specifies a new run and does not alter saved results.
See [reproducibility](../docs/reproducibility.md) for the historical analysis map.
