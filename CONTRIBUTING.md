# Contributing

Use a focused branch and include a test for every behavioral change. Before
opening a pull request, run:

```bash
ruff check src tests examples scripts
pytest
python -m build
```

Do not submit patient-level data, credentials, local absolute paths, or fitted
objects derived from restricted clinical data. Changes to the simulation DGP or
default loss weights must update the corresponding configuration and
reproducibility documentation.
