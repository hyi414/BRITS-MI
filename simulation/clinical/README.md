# Clinical-Association Benchmark

Use `python simulation/run_simulation.py --experiment clinical --help` from the
repository root. The [main simulation guide](../README.md) lists method choices,
settings, and output files. This folder preserves the manuscript implementation
and its dependency chain; the reusable package API is in `src/brits_mi/`.

| Component | Function and file | Returns |
| --- | --- | --- |
| Data generation and masking | `generate_locked_dataset` in `scripts/run_default_mice_clinical_association_sensitivity.py` | Complete and observed trajectories, masks, covariates, and outcome |
| Recurrent model fitting | `fit_fold_model` in `scripts/run_neural_brits_mi_downstream_gradient.py` | Trained fold-specific model and fitting diagnostics |
| Recurrent imputation | `cross_fitted_completions` in the same file | Stochastic trajectories and fold diagnostics |
| Conditional-mean anchor | `missforest_anchor` in `scripts/test_anchored_brits_mi_blend.py` | R missForest conditional-mean trajectory |
| Anchored imputation distribution | `blend_draws` in the same file | Imputed datasets using the specified mean and residual weights |
| Outcome refitting and evaluation | `evaluate_completions` in `scripts/run_neural_brits_mi_downstream_gradient.py` | Coefficient-level estimates and aggregate recovery metrics |
| R competing methods | `run_package_imputer` in `scripts/package_default_imputation.py` | Imputed predictor frames and package metadata |

Historical filenames are retained for reproducibility; a filename beginning
with `test_` here denotes an experiment component, not a pytest unit test.
No data-generating or fitting equations were changed by the folder reorganization.
