# Locked clinical + image fusion benchmark

This folder restores the 500-run benchmark used in the poster and JAMIA figures. It is the primary clinical + image result; the later 200-run, three-missingness-level experiment remains a sensitivity analysis and does not replace it.

Across 500 matched runs, BRITS-MI achieved mean AUROC 0.790 and AUPRC 0.519, compared with 0.784 and 0.515 for Missforest. The paired mean differences were 0.00569 for AUROC (Holm-adjusted p = 6.06e-13) and 0.00438 for AUPRC (Holm-adjusted p = 0.00239). The displayed representative curves have AUROC 0.811 versus 0.792 and AUPRC 0.548 versus 0.541 for BRITS-MI and Missforest, respectively.

`figure4_image_downstream.png` contains the representative ROC and precision-recall curves. `figure5_image_imputation.png` contains the missing-cell and trajectory-summary recovery distributions. The CSV files preserve all matched-run results and tests.

## Implementation provenance

The historical generator was `scripts/run_fast_image_fusion_nsim500.py`. In that fast benchmark, the BRITS-labeled arm used an image-conditioned extremely randomized trees cell imputer plus class-prototype correction; the Missforest-labeled arm used a random-forest regression surrogate. These are not, respectively, the recurrent BRITS-MI engine and the R `missForest` package implementation. The restored results should therefore be identified as the poster-era surrogate image benchmark. They must not be used as evidence for a package-default Missforest comparison or described as a direct recurrent-BRITS run without this qualification.
