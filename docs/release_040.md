# Recurrent implementation and comparison correction

Version 0.4.0 publishes one recurrent conditional-mean BRITS-GRU engine for
association and image-conditioned prediction. Its training, bootstrap members,
and correlated residual sampling are unchanged from the completed October 22
evaluation. Commit `ae52dfa` preserves that evaluation implementation.

## Completed clinical evidence

The saved evaluation contains 200 independent cohorts at each missing percentage:

| Experiment | Subjects | Scenarios | Missing percentages | Completed jobs |
| --- | ---: | ---: | --- | ---: |
| Clinical association | 500 | 3 | 20, 40, 60 | 1,800 |
| Larger sample | 2,000 | 1 | 20, 40, 60 | 600 |
| Observed-history masking sensitivity | 500 | 1 | 20, 40, 60 | 600 |

The release script checks frozen source hashes, the exact seed inventory,
matched method-input hashes, all 13 coefficient terms for each valid clinical
fit, pooled variance, confidence-interval limits, and coverage indicators.
It preserves invalid-method counts. Completed jobs are not synonymous with
valid complete-case analyses. Saved draws were not retained in this experiment;
observed-cell invariance is tested in the implementation, not retrospectively
verified for every saved draw.

## Corrected image comparison

The archived image evaluation completed 600 jobs (200 cohorts at each missing
percentage). Its MICE and missForest imputers received observed training labels
and missing test-label entries. Its final classifier was fitted once for
single-imputation methods and 20 times for multiple-imputation methods.

For the revised comparison:

1. MICE and missForest receive biomarker histories, visit times, static
   covariates and the same 17 image descriptors, without an outcome column.
2. BRITS-MI retains its supervised training-label classification loss. It does
   not condition query imputation on test outcomes.
3. Every method receives 20 final classifier fits, using matched classifier
   seeds and train/validation/test partitions. A single-imputation method reuses
   the same imputed array; MICE and BRITS-MI use their 20 different arrays.
4. Probabilities are averaged before evaluation. The number of classifier
   fits is not the number of independent cohorts, and averaging predictions
   does not invoke Rubin's coefficient-pooling rules.

No DGP, missingness parameter, uncertainty multiplier, recurrent training weight,
native comparator model, or evaluation seed is changed to obtain a ranking.
The frozen revised run uses n = 500, M = 20, 200 cohorts per missing percentage,
seeds 930000--930199, five bootstrap members, and correlated residual sampling.
Its results must be completed and checked before replacing the archived image
results. Tests cover absence of comparator outcomes, equal fitting budgets,
held-out-label invariance, and duplicate-runner rejection.

## Scope

The joint-likelihood/HMC development pilot is not part of this release. Its two
independent cohorts per scenario cannot replace the 200-cohort evaluation.
The private 2,209-participant reference-eligible masking analysis is not the
newer full 3,995-participant TE analysis. No participant data are distributed.
