"""Small executable example; not a manuscript-performance benchmark."""

from brits_mi import UnifiedBRITSMI, UnifiedConfig
from brits_mi.analysis import design_matrix, trajectory_features
from brits_mi.benchmark import pool_designs
from brits_mi.simulation import SimulationConfig, simulate_clinical_association


def main():
    data = simulate_clinical_association(SimulationConfig(n_subjects=150, seed=21))
    model = UnifiedBRITSMI(UnifiedConfig(epochs=5, hidden_size=16,
        bootstrap_members=2, residual_mode="correlated", seed=22))
    model.fit(data.observed_values, data.observed_mask, data.times,
              data.static_covariates, data.outcome)
    draws = model.sample(5, seed=23)
    designs = [design_matrix(trajectory_features(draw, data.times),
                             *data.static_covariates.T) for draw in draws]
    # Unit scale returns raw-unit estimates. No true coefficients enter fitting.
    result = pool_designs(designs, data.outcome, [1.] * 13, [0.] * 13)
    print(result[["term", "estimate", "standard_error", "lower", "upper"]].to_string(index=False))


if __name__ == "__main__":
    main()
