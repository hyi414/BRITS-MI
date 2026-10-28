"""Public API for longitudinal BRITS multiple imputation."""

from .calibration import AssociationCalibrationConfig, CalibrationDiagnostics
from .config import LossWeights, TrainingConfig
from .imputer import BRITSMultipleImputer
from .model import BRITSMI
from .unified import UnifiedBRITSMI, UnifiedConfig

__all__ = [
    "BRITSMI",
    "AssociationCalibrationConfig",
    "BRITSMultipleImputer",
    "CalibrationDiagnostics",
    "LossWeights",
    "TrainingConfig",
    "UnifiedBRITSMI",
    "UnifiedConfig",
]

__version__ = "0.3.0"
