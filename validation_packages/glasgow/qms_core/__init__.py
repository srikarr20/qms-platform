"""
QMS Platform
============

Reusable measurement-system observability, reconstruction,
digital-twin, inference, prediction, and adaptation software.

Major capability layers:

    qms_core.quantum_twin
        Finite-dimensional dynamical observability,
        state estimation, model identification,
        divergence diagnostics, adaptation, and
        measurement recommendation.

    qms_core.detector_twin
        Detector-stage observability, residual analysis,
        mechanism-library inference, detector calibration,
        adaptive control, external measurement I/O,
        and optional Pyxel-specific inversion.

    qms_core.measurement_twin
        Observable-state construction, causal prediction,
        innovation calibration, future-only alerting,
        and alert-episode consolidation.

Scientific interpretation depends on the measurement model,
calibration, and validation available for each application.
"""

from .state import (
    Measurement,
    DetectorEvent,
    DetectorFrame,
    ComplexFieldState,
    VirtualVolumeState,
    SourceEstimate,
    ObservableEstimate,
    TwinState,
    ValidationRecord,
)

from .runtime import (
    TwinRuntime,
)

from .contracts import (
    DetectorDiagnostics,
    ReconstructedField,
    UpstreamEstimate,
    DetectorState,
    ObservableManifold,
    DynamicState,
    ObservabilityState,
    PlatformTwinState,
)

from .ingestion import (
    MeasurementAdapter,
    QuadratureMeasurementAdapter,
)

from .reconstruction import (
    ReconstructionAdapter,
)


__all__ = [
    "Measurement",
    "DetectorEvent",
    "DetectorFrame",
    "ComplexFieldState",
    "VirtualVolumeState",
    "SourceEstimate",
    "ObservableEstimate",
    "TwinState",
    "ValidationRecord",
    "TwinRuntime",
    "DetectorDiagnostics",
    "ReconstructedField",
    "UpstreamEstimate",
    "DetectorState",
    "ObservableManifold",
    "DynamicState",
    "ObservabilityState",
    "PlatformTwinState",
    "MeasurementAdapter",
    "QuadratureMeasurementAdapter",
    "ReconstructionAdapter",
]
