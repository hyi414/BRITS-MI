# Public synthetic-data results

- `clinical_n500/` and `clinical_n2000/`: saved configuration, coefficient/reconstruction summaries with Monte Carlo intervals, and replicate-level reconstruction metrics.
- `mnar/`: aggregate sensitivity results at MNAR coefficients 0, 0.5, and 1, holding target missing percentage at 40%.
- `complete_case/`: the 500-attempt-per-percentage complete-case rerun, fit audit, and conditional numerical diagnostic. Only four fits at 40% and none at 60% were valid; conditional performance is not an overall method estimate.
- Image replicate metrics are read from the existing canonical file [metrics_by_seed.csv](../../results/image_extension/metrics_by_seed.csv); valid display methods and labels are selected by [plot_results.py](../plot_results.py).

The clinical result files were copied without changing their numerical values
from the final aggregate outputs. Source folder names are
`clinical_association_n500_nsim200_final_20260802`,
`clinical_association_n2000_nsim100_final_20260802`,
`clinical_association_mnar_sensitivity_n500_nsim100_m20_20260803`, and
`complete_case_table2_rerun500_20260920`. All rows concern synthetic datasets.
Run identifiers are pseudorandom seeds, not patient identifiers.

New output belongs under the ignored `results_local/` directory, not in this
saved-results archive. See the [gallery](../visualization/) for figures and
derived plotting tables. Real cohort data and fitted clinical weights are not
part of this folder.
