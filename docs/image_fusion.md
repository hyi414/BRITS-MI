# BRITS-MI-image

The image extension conditions biomarker completion on pixel features and
image-derived fibrosis probabilities before fitting a clinical-image classifier.
The reported image implementation is tree based. The framework name does not
imply that this archived experiment trained recurrent BRITS, a CNN, or a proper
multiple-imputation distribution.

## Input and fitting order

1. Supply an N x T x 3 biomarker tensor, observation mask, subject visit times
   and lengths, covariates, and N x 1 x 24 x 24 ultrasound-like images.
2. Split subjects before fitting. Estimate provisional means from measured
   training cells only. An observed zero is never a missing marker.
3. Extract 17 pixel features. Fit a standardized, balanced, three-class logistic
   model on training images and labels. Predicted probabilities, not query
   labels, provide the completion context.
4. Fit a separate ExtraTrees regression per visit-marker target using other
   provisional values, masks, covariates, times, gaps, image features, and image
   probabilities. Exclude the target's own provisional value. Fewer than 30
   measured training targets trigger a mean fallback.
5. Optionally blend predictions with probability-weighted class-specific means
   of observed training cells. `prototype_alpha` controls this correction and
   must be explicit in command-line runs. This is not neural gradient calibration.
6. Restore measured cells. Recompute 34 trajectory motifs, concatenate with 17
   image features, and fit the fusion classifier to the 51-column design.

`fit` implements steps 2-6. `complete` and `predict_proba` take no outcome and
fit nothing. Training image probabilities are in-sample; cross-fitted context
would be a separate extension. The public API is a refactoring, not an exact
replay of the historical invocation.

## Architecture

The cell model has 180 trees, depth 13, leaf size 4, and feature fraction 0.8.
Fusion has 180 trees, depth 8, leaf size 8, and balanced class weights. No
data-driven feature preselection is performed. Seventeen pixel summaries
include intensity moments and quantiles, gradient magnitude, radial-region
means, and directional contrasts. Thirty-four trajectory motifs include levels,
late-early contrasts, changes, curvature, and cross-marker patterns. Centered
differences duplicate uncentered differences algebraically; these are retained
for fidelity, not as independent information. Generator-only metadata such as
nodule count never enters the estimator.

## Reproducibility boundary

The reported 500-run predictions are archived. Their exact prototype correction
invocation was not saved; a new run cannot be certified as exact reproduction.
The old image MICE arm supplied zeros to an iterative imputer expecting NaN and
is invalid. Corrected encoding is in `longitudinal_sim/baselines.py`. The archived
forest comparator is not R missForest. The public API's optional forest engine
uses the same image inputs; its results do not replace historical results.

## References

- Geurts et al. Extremely randomized trees: https://doi.org/10.1007/s10994-006-6226-1
- Hayat et al. MedFuse: https://proceedings.mlr.press/v182/hayat22a.html
- Yao et al. DrFuse: https://doi.org/10.1609/aaai.v38i15.29578

MedFuse and DrFuse motivate integration; their architectures are not the
algorithm used for the archived tree benchmark.
