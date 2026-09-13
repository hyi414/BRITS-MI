# Release audit

This is a source release for review, not a newly validated clinical product.
The local manuscript revision and the package use **BRITS-MI-image** for the
image-conditioning extension. Its tree implementation is stated explicitly.

## Included

- Recurrent BRITS-GRU reference implementation, weighted training, artificial
  masks, validation, calibrated residual draws, and analysis refitting/pooling.
- Optional bounded estimating-score calibration, separate from the active
  outcome-supervised loss in the primary clinical experiment.
- Clinical hybrid training and missForest anchoring scripts, default R MICE
  and missForest wrappers, known-truth DGP, sample-size and MNAR sensitivities,
  aggregation scripts, and their local Python/R dependency chain.
- Image-linked longitudinal DGP, both image renderers, 17 pixel features,
  34 trajectory motifs, cell trees, and fusion classifier.
- A public label-free image fit/complete/predict API and unit tests.
- Original source hashes and machine-readable verification records.

## Important limits

The compact neural API is not the archived forest-anchored clinical hybrid.
Historical image metrics are auditable, but exact retraining is not certified
because the prototype-alpha invocation was not archived. Existing aggregate
CSV files retain old machine identifiers for joins. The legacy image MICE arm
is invalid and must be excluded; corrected code does not constitute a full
corrected benchmark. The forest image surrogate is not default R missForest.

Small smoke runs check execution, not statistical performance. In the image
DGP, very small samples can lack enough observations in a rare outcome class
for stratified splitting; the runner fails rather than silently resampling or
changing the DGP. Two 200-subject runs were used for software verification.

The current clinical paper reports 200 runs per rate at n=500 and 100 at
n=2000, not 1000. The image table uses 500 runs. Source-cohort masking is not
external validation or population coefficient coverage. Public release excludes
all individual-level clinical records, models trained on those records, and
collaborator manuscript comments.

## Historical scripts

Archived scripts preserve their original relative `outputs/` contracts. Run
them from their own directory, supply explicit `--outdir`/run arguments, and
inspect `--help`. Final aggregation expects the named historical result
directories. The available source is shipped for inspection and rerunning;
not every paper-scale command has been executed during packaging.

## Publication preparation

Review LICENSE, authorship in CITATION.cff, institution approval, and the
manuscript-to-code map. Create the GitHub repository only after that review.
The public repository is https://github.com/hyi414/BRITS-MI. No manuscript
acceptance or archived-release DOI is asserted.

## Metric aggregation

The reference clinical API reports single-run absolute coefficient errors
separately from systematic bias. Bias is computed by averaging signed errors
across independent runs for each term, then taking absolute values and
averaging terms. A cancellation regression test checks this distinction.
This correction does not alter any archived manuscript number.
