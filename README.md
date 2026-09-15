# BRITS-MI

Research software for association-aware longitudinal multiple imputation, with
the **BRITS-MI-image** extension for clinical-image fusion.

**Repository:** https://github.com/hyi414/BRITS-MI

The scientific target is the analysis performed after imputation: preserving
biomarker levels, slopes, interactions, and coefficient uncertainty, as well as
recovering individual measurements. The image extension examines a separate
target, classification after completing the clinical branch of a fusion model.

This is research code, not a validated clinical decision tool. It contains no
individual-level clinical records or weights fitted to the restricted cohort.

## Two entry points

| Purpose | Public entry point | Implementation |
| --- | --- | --- |
| Recurrent stochastic completion | `BRITSMultipleImputer.fit`, `.sample` | Reference RITS-GRU, weighted loss, residual draws, optional score calibration |
| Image-conditioned completion and fusion | `BRITSMIImage.fit`, `.complete`, `.predict_proba` | Tree extension; deterministic completion, 17 pixel features and 34 trajectory motifs |

The clinical benchmark additionally anchors recurrent draws to a package-default
missForest estimate. Its archived pipeline is included; the compact neural API
alone does not reproduce the hybrid. The image extension is not a trained
recurrent or CNN benchmark.

## Install and test

Use Python 3.10 or later. The release was checked locally on Python 3.14;
GitHub Actions also tests Python 3.10 and 3.12. A CPU is sufficient for the
small examples. Full neural benchmarks require substantially more time.

```bash
git clone https://github.com/hyi414/BRITS-MI.git
cd BRITS-MI
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
brits-mi-sim --help
brits-mi-image --help
```

R comparators need R, `mice`, and `missForest`. The archived environment used
mice 3.18.0 and missForest 1.5. Unit tests do not require R.

```r
install.packages(c("mice", "missForest"))
```

That command installs currently available releases, not necessarily the
archived versions. Record `sessionInfo()` and package versions for a rerun.

## Implementation details?

| Goal | Route | Relation to the manuscript |
| --- | --- | --- |
| Understand or adapt recurrent multiple imputation | `src/brits_mi` and `examples` | Compact reference implementation of the reusable components |
| Inspect or rerun the reported clinical hybrid | `reproducibility/run_reported_clinical.py` | Recurrent proposals plus the reported missForest anchor |
| Generate ultrasound-like images and fit a fusion model | `brits-mi-image` | Explicit tree-based image API; not a recurrent or CNN benchmark |
| Inspect historical image result generation | `reproducibility/reported_pipeline/image_prototype` | Original script, with documented missing configuration and comparator issues |
| Run package-default comparator wrappers | `reproducibility/R` and archived association scripts | Requires R; distinguish deterministic forest fits from proper MI |

Do not interchange the compact reference API and the archived clinical
hybrid when claiming reproduction of a manuscript table.

## Recurrent completion

```python
from brits_mi import BRITSMultipleImputer, TrainingConfig
from brits_mi.simulation import SimulationConfig, simulate_clinical_association

data = simulate_clinical_association(SimulationConfig(n_subjects=100, seed=21))
model = BRITSMultipleImputer(TrainingConfig(epochs=3, hidden_size=16, seed=21))
model.fit(data.observed_values, data.observed_mask, data.times,
          data.static_covariates, data.outcome)
draws = model.sample(5, seed=22)
```

This is a software example, not an adequately trained manuscript-scale model.
Use `brits_mi.analysis.pool_logistic_regressions` to refit and pool each draw.

### Training and completion workflow

1. Represent measured values, masks, visit times, static covariates, and the
   training outcome separately. Zero is a valid measured value; it does not
   identify missingness.
2. Artificially hide a subset of measured training cells. Only those cells
   provide known reconstruction targets; naturally missing cells do not.
3. Run forward and backward RITS-GRU passes, reconcile their predictions, and
   optimize reconstruction, directional consistency, trajectory-summary, and
   downstream losses. The planned outcome model supplies training feedback.
4. Select a checkpoint using an artificial validation mask. Use validation
   residuals to calibrate the stochastic completion scale.
5. Freeze the model, draw completed trajectories, retain every supplied
   measurement, and recompute downstream features within each draw.
6. Refit the outcome model to every completed dataset and pool estimates and
   within/between-completion uncertainty. Numerical convergence does not by
   itself guarantee valid multiple-imputation coverage.

Training uses outcome information. The ordinary `sample` interface for a new
patient does not require that patient's outcome. The optional
`sample_association_calibrated` interface does use an outcome and a supplied
coefficient target; it is an explanatory analysis operation, not label-free
prediction. See [methodology](docs/methodology.md) and
[code walkthrough](docs/code_walkthrough.md).

### Input contract

| Input | Shape | Meaning |
| --- | --- | --- |
| `values` | `N x T x J` | Numeric biomarker values; availability is determined by the mask |
| `mask` | `N x T x J` | True for measured, supplied cells |
| `times` | `N x T` | Ordered visit times on a consistent scale |
| `static` | `N x P` | Subject-level clinical/demographic predictors |
| `outcome` | `N` | Binary target for the reference clinical association model |
| `context` | `N x K`, optional | Additional supplied context, such as image features |

The compact clinical API uses a fixed-width sequence. The image API additionally
accepts per-subject lengths and excludes padding. Harmonize biomarker scales
and use clinically justified time windows before applying the interface to
another cohort. See the [data contract](docs/data_contract.md).

### Clinical smoke run and outputs

```bash
brits-mi-sim --output results_local/clinical_smoke --runs 2 --subjects 80 \
  --missing-rates 0.2 --imputations 3 --epochs 2 --hidden-size 8 \
  --first-seed 21 --device cpu
```

| Output | Contents |
| --- | --- |
| `manifest.json` | Simulation and training settings |
| `coefficient_recovery.csv` | Run-by-term coefficient estimates, errors, and intervals |
| `replicate_metrics.csv` | Reconstruction and within-run aggregate metrics |
| `term_summary.csv` | Bias, MSE, coverage, and SE summaries across runs for each term |
| `summary.csv` | Across-term summary of the repeated-run experiment |

Systematic bias is calculated from the mean signed error across independent
runs for each coefficient. Mean absolute error within a run is not systematic
bias. Coverage is a repeated-run property and cannot be established from one
small example. The smoke command checks execution, not scientific performance.

## Image generation and fusion

```bash
brits-mi-image --output results_local/image_smoke --n-subjects 200 \
  --n-runs 2 --trees 8 --prototype-alpha 0.0
```

This generates the alternative sector-ultrasound DGP, fits only training
subjects, and evaluates held-out subjects. Larger counts do not recover the
unknown historical tuning invocation. See [workflow](docs/image_fusion.md).

### Image generation, completion, and prediction

The joint image generator simulates three biomarkers at 3-12 visits,
patient-specific gamma-distributed gaps, clinical covariates, a three-class
fibrosis outcome, and one 24 x 24 grayscale sector image per subject. The
renderer includes speckle, depth attenuation, capsule, vessel-like shadows,
and stage-linked texture. These are controlled ultrasound-like textures, not
validated clinical scans or calibrated elastography stiffness maps.

The observed image is reduced to 17 pixel features. A training-fitted logistic
image model supplies three class probabilities. Cell-specific tree regressors
use image context together with incomplete trajectories, masks, times, and
covariates. After completion, 34 trajectory motifs and the 17 image features
form the 51-column design of an ExtraTrees fusion classifier.

```python
from brits_mi.image_fusion import BRITSMIImage, ImageFusionConfig

model = BRITSMIImage(ImageFusionConfig(prototype_alpha=0.0))
model.fit(train_values, train_mask, train_times, train_lengths,
          train_covariates, train_images, train_labels)
completed = model.complete(test_values, test_mask, test_times, test_lengths,
                           test_covariates, test_images)
probabilities = model.predict_proba(
    test_values, test_mask, test_times, test_lengths,
    test_covariates, test_images)
```

Here `train_images` and `test_images` have shape `N x 1 x 24 x 24`, and labels
are coded 0, 1, and 2. Fit the entire pipeline only on training subjects. The
test calls neither accept labels nor refit models. The example requires actual
input arrays; the CLI above generates them end to end.

The image CLI writes `metrics.csv` and `config.json`. It fails explicitly if a
rare class has insufficient observations for a stratified split. Do not
silently resample until favorable class frequencies or performance appear.

## Manuscript analyses

- `reproducibility/reported_pipeline/association/scripts` contains the clinical
  DGP, training, anchoring, pooling, comparator wrappers, and dependencies.
- `reproducibility/run_reported_clinical.py` prepares or executes an explicit
  hybrid run without relying on the original workstation path.
- `src/longitudinal_sim/data.py` contains the image-linked longitudinal DGP and
  image renderers. `src/brits_mi/_image_reference.py` retains the available
  runner. Use `python -m brits_mi._image_reference --help`.
- `reproducibility/source_manifest.json` gives original hashes;
  `docs/release_audit.md` records verification and unresolved provenance.
- `results` contains aggregate summaries. Historical identifiers are retained
  for joins. The old image MICE arm is invalid and must not be shown as MICE.

### Reported clinical configuration

```bash
python reproducibility/run_reported_clinical.py \
  --n 500 --runs 200 --missing-rate 0.2 --mean-weight 0.10 \
  --output results_local/clinical_reported_20
```

This prints the command without launching the expensive run. Add `--execute`
to launch it after reviewing the settings. The archived mean weight is 0.10
at 20% and 40% late-cell deletion and 0.20 at 60%; residual weight is 0.70.
The launcher uses 50 completions. The larger-sample analysis uses `--n 2000`
and `--runs 100`. These are the saved counts, not 1000-run experiments.

Comparator, sample-size, MNAR, and aggregation scripts are in
`reproducibility/reported_pipeline/association/scripts`. Each script exposes
`--help`. Archived finalizers expect the documented historical output-directory
structure; they do not download missing benchmark files automatically.

### Evidence and reproducibility boundaries

- The reported clinical result is a recurrent-plus-forest hybrid; it does not
  isolate the recurrent engine or prove that every loss term is necessary.
- Historical image predictions can be summarized, but exact retraining cannot
  be certified because the prototype-correction invocation was not archived.
- The historical image arm labeled MICE incorrectly passed zero-filled cells
  to a NaN-based iterative imputer. It is excluded from valid comparisons. The
  encoding correction is regression-tested; a full corrected benchmark is
  still needed.
- The historical image forest comparator is a random-forest surrogate, not
  the R `missForest` algorithm. The image extension itself is deterministic,
  not a claim of stochastic recurrent multiple imputation.
- Source-cohort masking measures recovery relative to an unmasked cohort fit.
  It is not population-truth coverage or external validation; its pre-mask
  training and unequal-information boundary are documented in the manuscript.

## Repository map

```text
src/brits_mi/                  Recurrent API, pooling, calibration, image API
src/longitudinal_sim/          Joint longitudinal/image generator and utilities
examples/                     Small Python usage examples
configs/                      Reference grids and archived configurations
reproducibility/R/            R comparator interface
reproducibility/reported_pipeline/  Historical simulations and dependencies
reproducibility/editorial_audit/    Audit and figure-building source scripts
results/                      Aggregate results only
tests/                        Unit and regression tests
.github/workflows/ci.yml       Source tests, lint, and package build
```

## Tests and numerical checks

The release passes 12 tests covering measured-cell invariance, deterministic
seeding, label-free image prediction, query-cohort invariance, calibration,
pooling, correct coefficient-bias aggregation, and mask-to-NaN conversion.
The same tests pass after wheel installation. Two short clinical runs and two
short image runs also completed. They are software checks, not full benchmark
reruns. The exact check outputs and tested library versions are in
`reproducibility/verification_20260913.json`.

```bash
ruff check src tests examples
pytest
python -m build
```

## Citation and references

See [CITATION.cff](CITATION.cff) for software authorship. The associated
manuscript is in preparation; no accepted-paper citation or DOI is asserted.

- Cao et al. BRITS. NeurIPS 2018:
  https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html
- van Buuren and Groothuis-Oudshoorn. mice:
  https://doi.org/10.18637/jss.v045.i03
- Stekhoven and Buehlmann. missForest:
  https://doi.org/10.1093/bioinformatics/btr597
- Bartlett et al. Substantive-model-compatible imputation:
  https://doi.org/10.1177/0962280214521348
- Geurts et al. Extremely randomized trees:
  https://doi.org/10.1007/s10994-006-6226-1
- Hayat et al. MedFuse:
  https://proceedings.mlr.press/v182/hayat22a.html
- Yao et al. DrFuse:
  https://doi.org/10.1609/aaai.v38i15.29578

MedFuse and DrFuse motivate the fusion setting. Their neural architectures
were not implemented in the archived tree experiment.

## Public release

Review authorship, license, and institutional approvals before a manuscript
submission or tagged release. Do not upload raw EHR, person IDs,
trained clinical weights, credentials, or manuscript comments. Add an archive
DOI to `CITATION.cff` only after an archive is actually created. Full benchmark reruns and corrected
image MICE comparisons are not completed by this packaging task. This research
software is not a clinical decision tool.

Code is released under the [MIT license](LICENSE). See
[DATA_POLICY.md](DATA_POLICY.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
