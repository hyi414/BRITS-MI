import pandas as pd

from brits_mi.runner import summarize_coefficients


def test_bias_is_not_mean_absolute_error():
    frame = pd.DataFrame(
        {
            "scenario": ["baseline"] * 2,
            "target_missing": [0.2] * 2,
            "method": ["BRITS-MI"] * 2,
            "term": ["AST"] * 2,
            "signed_error": [-1.0, 1.0],
            "squared_error": [1.0, 1.0],
            "covered": [1, 0],
            "estimate": [0.0, 2.0],
            "standard_error": [0.5, 0.5],
            "seed": [1, 2],
        }
    )
    row = summarize_coefficients(frame).iloc[0]
    assert row.absolute_bias == 0
    assert row.coefficient_mse == 1
    assert row.coverage == 0.5
