"""
QMS causal measurement-state twin.

This package provides reusable observable-state tracking,
causal prediction, innovation calibration, future-only
alerting, and alert consolidation.

It does not assign physical detector-health or degradation
meaning unless an independently calibrated physical model
supports such interpretation.
"""

from .state import (
    MeasurementState,
    aggregate_increment_window,
    normalized_grid_distribution,
    cosine_similarity,
    jensen_shannon_divergence,
    build_measurement_states,
)

from .prediction import (
    MeasurementPrediction,
    constant_velocity_prediction,
    predict_sequence,
)

from .calibration import (
    FrozenInnovationCalibration,
    calibrate_frozen_threshold,
    frozen_robust_z,
)

from .alerting import (
    MeasurementAlert,
    evaluate_prediction,
    evaluate_future_predictions,
)

from .episodes import (
    AlertEpisode,
    consolidate_alerts,
)

from .causal import (
    CausalReplayResult,
    run_causal_replay,
)

__all__ = [
    "MeasurementState",
    "aggregate_increment_window",
    "normalized_grid_distribution",
    "cosine_similarity",
    "jensen_shannon_divergence",
    "build_measurement_states",
    "MeasurementPrediction",
    "constant_velocity_prediction",
    "predict_sequence",
    "FrozenInnovationCalibration",
    "calibrate_frozen_threshold",
    "frozen_robust_z",
    "MeasurementAlert",
    "evaluate_prediction",
    "evaluate_future_predictions",
    "AlertEpisode",
    "consolidate_alerts",
    "CausalReplayResult",
    "run_causal_replay",
]
