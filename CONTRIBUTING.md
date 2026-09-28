# Contributing

Use a focused branch and include a test for every behavioral change. Before
opening a pull request, run:

```bash
ruff check src tests examples scripts validation
pytest
python -m build
```

Do not submit patient-level data, credentials, local absolute paths, or fitted
objects derived from restricted clinical data. Changes to the simulation DGP or
default loss weights must update the corresponding configuration and
reproducibility documentation.

Keep reusable model components in `src/`, simulation orchestration in
`simulation/`, and cohort masking analysis in `validation/`. Unit tests must
use generated data only. Preserve matched datasets and masking seeds in method
comparisons, and document any change to input features or pooling conventions.
