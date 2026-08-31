from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Optional
import numpy as np


# ============================================================
# QMS PLATFORM — SHARED INTEGRATION CONTRACTS
#
# Measurement
#     ↓
# ReconstructedField
#     ↓
# DetectorState
#     ↓
# ObservableManifold
#     ↓
# DynamicState
#     ↓
# PlatformTwinState
# ============================================================


@dataclass
class DetectorDiagnostics:
    """
    Detector-plane quality / measurement diagnostics.

    Examples:
        visibility
        uncertainty
        detector quality
        calibration state
    """

    visibility: Optional[float] = None

    quality_status: Optional[str] = None

    metadata: Dict[str, Any] = dc_field(
        default_factory=dict
    )


@dataclass
class ReconstructedField:
    """
    Field reconstructed from downstream measurement data.

    This is modality-independent.

    Examples:
        optical complex field
        MRI reconstructed 4D field
        acoustic pressure field
        semiconductor process field
    """

    data: Any

    domain: str

    coordinates: Any = None

    timestamp: Optional[float] = None

    metadata: Dict[str, Any] = dc_field(
        default_factory=dict
    )


@dataclass
class UpstreamEstimate:
    """
    Estimated hidden / upstream source state.

    DPI currently populates this directly from a reconstructed
    virtual propagation volume.
    """

    x: float
    y: float
    z: float

    field: Optional[np.ndarray] = None

    confidence: Optional[float] = None

    metadata: Dict[str, Any] = dc_field(
        default_factory=dict
    )


@dataclass
class DetectorState:
    """
    Detector-conditioned representation derived from a
    reconstructed field.

    Examples:
        AURORA deltaV
        accumulation field
        phase-aware detector state
        optical-flow detector representation
        topology-aware detector representation
    """

    data: Any

    detector_type: str

    timestamp: Optional[float] = None

    metadata: Dict[str, Any] = dc_field(
        default_factory=dict
    )


@dataclass
class ObservableManifold:
    """
    Low-dimensional observable representation.

    Examples:
        AURORA X(t) = [C(t), K(t), E(t)]
        source-position trajectory
        phase / coherence manifold
        regime state space
    """

    state: Any

    names: Any = None

    timestamps: Any = None

    metadata: Dict[str, Any] = dc_field(
        default_factory=dict
    )


@dataclass
class DynamicState:
    """
    Temporal interpretation of observable evolution.
    """

    trajectory: Any = None

    phase: Any = None

    attractor: Any = None

    regime: Any = None

    prediction: Any = None

    instability: Any = None

    metadata: Dict[str, Any] = dc_field(
        default_factory=dict
    )


@dataclass
class ObservabilityState:
    """
    Describes what latent/source parameters are observable
    from the available measurement configuration.
    """

    depth_score: Any = None

    parameter_map: Any = None

    uncertainty: Any = None

    degeneracy: Any = None

    metadata: Dict[str, Any] = dc_field(
        default_factory=dict
    )


@dataclass
class PlatformTwinState:
    """
    Unified measurement-twin state.

    This object can hold both reconstruction and dynamical
    observability without requiring one modality-specific
    pipeline.
    """

    measurement: Any

    detector_diagnostics: Optional[
        DetectorDiagnostics
    ] = None

    reconstructed_field: Optional[
        ReconstructedField
    ] = None

    upstream: Optional[
        UpstreamEstimate
    ] = None

    detector_state: Optional[
        DetectorState
    ] = None

    manifold: Optional[
        ObservableManifold
    ] = None

    dynamics: Optional[
        DynamicState
    ] = None

    observability: Optional[
        ObservabilityState
    ] = None

    predicted_measurement: Any = None

    residual: Any = None

    version: int = 0

    metadata: Dict[str, Any] = dc_field(
        default_factory=dict
    )
