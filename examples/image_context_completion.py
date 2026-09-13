"""Supply ultrasound-derived context to BRITS-MI without exposing an image label."""

import numpy as np

from brits_mi import BRITSMultipleImputer, TrainingConfig
from brits_mi.simulation import SimulationConfig, simulate_clinical_association
from brits_mi.ultrasound import simulate_ultrasound_context

data = simulate_clinical_association(SimulationConfig(n_subjects=200, seed=23))
stage = np.digitize(data.latent["severity"], [-0.35, 0.70])
_, image_context = simulate_ultrasound_context(
    data.latent["severity"],
    data.latent["late_activity"],
    stage,
    seed=23,
)

imputer = BRITSMultipleImputer(TrainingConfig(epochs=20, seed=23))
imputer.fit(
    data.observed_values,
    data.observed_mask,
    data.times,
    data.static_covariates,
    data.outcome,
    context=image_context,
)
completed = imputer.sample(5, context=image_context, seed=23001)
print(len(completed), completed[0].shape)
