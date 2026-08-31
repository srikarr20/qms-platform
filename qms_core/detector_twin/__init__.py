"""
QMS detector-twin primitives.

Generic detector diagnostics are kept independent of
Pyxel. Pyxel-specific inversion is exposed through the
pyxel_cti module.

Current validation supports controlled computational
detector-twin behavior and does not establish calibrated
real-detector degradation inference.
"""

from .residuals import (
    rmse,
    residual_features,
)

from .stages import (
    StageDivergence,
    first_divergent_stage,
)

from .inference import (
    MechanismCandidate,
    MechanismInference,
    rank_mechanism_library,
)

from .calibration import (
    MonotonicInverseCalibration,
    build_inverse_calibration,
)

from .controller import (
    DetectorPrediction,
    DetectorInnovation,
    constant_velocity_predict,
    relative_innovation,
    assess_detector_state,
)

from .io import (
    latest_file,
    load_pixel,
    load_image,
    validate_measurement_directory,
    ingest_measurement,
    archive_measurement,
    load_twin_state,
    save_twin_state,
)

__all__ = [
    "rmse",
    "residual_features",
    "StageDivergence",
    "first_divergent_stage",
    "MechanismCandidate",
    "MechanismInference",
    "rank_mechanism_library",
    "MonotonicInverseCalibration",
    "build_inverse_calibration",
    "DetectorPrediction",
    "DetectorInnovation",
    "constant_velocity_predict",
    "relative_innovation",
    "assess_detector_state",
    "latest_file",
    "load_pixel",
    "load_image",
    "validate_measurement_directory",
    "ingest_measurement",
    "archive_measurement",
    "load_twin_state",
    "save_twin_state",
]
