from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time


# ============================================================
# QMS PLATFORM — SHARED STATE MODEL
#
# This file defines the common language exchanged between:
#
#   QMS Runtime
#       ↓
#   DPI reconstruction
#       ↓
#   virtual upstream twin
#       ↓
#   AURORA observability
#       ↓
#   validation
#
# No reconstruction physics belongs here.
# No detector-specific code belongs here.
# ============================================================


@dataclass
class Measurement:
    """
    Generic downstream measurement.

    Examples:
        intensity frame
        quadrature measurement
        hologram
        event batch
        wavefront measurement
    """

    data: Any

    modality: str

    timestamp: float = field(
        default_factory=time.time
    )

    detector_id: str = "unknown"

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class DetectorEvent:
    """
    One sparse detector event.
    """

    x: int
    y: int

    value: float = 1.0

    timestamp: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class DetectorFrame:
    """
    Conventional detector frame.
    """

    data: Any

    timestamp: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ComplexFieldState:
    """
    Reconstructed complex field at one measurement plane.

        U(x,y,t)

    This is normally produced by DPI reconstruction.
    """

    field: Any

    wavelength: Optional[float] = None

    pixel_spacing: Optional[float] = None

    z: float = 0.0

    timestamp: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class VirtualVolumeState:
    """
    Source-facing virtual propagation volume.

        Psi(x,y,z,t)

    The planes are computational / virtual states,
    not additional physical detectors.
    """

    field: Any

    depths: Any

    timestamp: Optional[float] = None

    wavelength: Optional[float] = None

    pixel_spacing: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class SourceEstimate:
    """
    Current inferred upstream source state.
    """

    position: Optional[
        Tuple[float, float, float]
    ] = None

    velocity: Optional[
        Tuple[float, float, float]
    ] = None

    phase: Optional[float] = None

    separation: Optional[float] = None

    orientation: Optional[float] = None

    confidence: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ObservableEstimate:
    """
    One inferred source or field observable.
    """

    name: str

    value: Any

    confidence: Optional[float] = None

    best_depth: Optional[float] = None

    uncertainty: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class TwinState:
    """
    Central detector-driven virtual twin state.

    This is the object that eventually gets continuously
    updated as downstream measurements arrive.
    """

    measurement: Optional[
        Measurement
    ] = None

    detector_field: Optional[
        ComplexFieldState
    ] = None

    virtual_volume: Optional[
        VirtualVolumeState
    ] = None

    source: Optional[
        SourceEstimate
    ] = None

    observables: List[
        ObservableEstimate
    ] = field(
        default_factory=list
    )

    version: int = 0

    timestamp: float = field(
        default_factory=time.time
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ValidationRecord:
    """
    Comparison between reconstructed twin state,
    known truth, and/or conventional baseline.
    """

    experiment: str

    estimate: Any

    truth: Any = None

    baseline: Any = None

    metrics: Dict[str, float] = field(
        default_factory=dict
    )

    passed: Optional[bool] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )
