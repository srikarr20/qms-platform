"""
QMS quantum-twin primitives.

These utilities implement reusable finite-dimensional
computational dynamics, observability, reconstruction,
and causal linear-Gaussian estimation.

They do not constitute experimental quantum-field
tomography or a physical quantum-field twin.
"""

from .dynamics import (
    build_two_mode_drift,
    default_drive_matrix,
    default_diffusion,
    gaussian_rhs,
    propagate_gaussian_state,
    discretize_linear_dynamics,
)

from .observability import (
    ObservabilityDiagnostics,
    observability_matrix,
    null_space,
    analyze_observability,
    build_discrete_observation_matrix,
    null_overlap,
)

from .estimation import (
    reconstruct_state,
    KalmanState,
    KalmanUpdate,
    kalman_predict,
    kalman_update,
    kalman_step,
)

from .state import (
    symplectic_form_two_mode,
    gaussian_physicality,
)

__all__ = [
    "build_two_mode_drift",
    "default_drive_matrix",
    "default_diffusion",
    "gaussian_rhs",
    "propagate_gaussian_state",
    "discretize_linear_dynamics",
    "ObservabilityDiagnostics",
    "observability_matrix",
    "null_space",
    "analyze_observability",
    "build_discrete_observation_matrix",
    "null_overlap",
    "reconstruct_state",
    "KalmanState",
    "KalmanUpdate",
    "kalman_predict",
    "kalman_update",
    "kalman_step",
    "symplectic_form_two_mode",
    "gaussian_physicality",
]

from .divergence import (
    ResidualStatistics,
    DivergenceAssessment,
    lag1_autocorrelation,
    summarize_residuals,
    classify_residual_structure,
)

from .identification import (
    ParameterFit,
    FamilyFit,
    build_window_matrix,
    fit_state_for_model,
    fit_parameter_grid,
    candidate_parameter_sets,
    fit_model_family,
    identify_model_family,
    adequacy_threshold,
    reject_unknown_mechanism,
)

from .adaptation import (
    AdaptationResult,
    estimate_parameter_from_trajectory,
)

from .recommendation import (
    MeasurementRecommendation,
    orthonormal_trajectory_basis,
    subspace_mismatch,
    recommend_measurement_configuration,
)

__all__ += [
    "ResidualStatistics",
    "DivergenceAssessment",
    "lag1_autocorrelation",
    "summarize_residuals",
    "classify_residual_structure",
    "ParameterFit",
    "FamilyFit",
    "build_window_matrix",
    "fit_state_for_model",
    "fit_parameter_grid",
    "candidate_parameter_sets",
    "fit_model_family",
    "identify_model_family",
    "adequacy_threshold",
    "reject_unknown_mechanism",
    "AdaptationResult",
    "estimate_parameter_from_trajectory",
    "MeasurementRecommendation",
    "orthonormal_trajectory_basis",
    "subspace_mismatch",
    "recommend_measurement_configuration",
]
