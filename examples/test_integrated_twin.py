import numpy as np

from qms_core import Measurement
from twin import QuantumMeasurementTwin
from adapters.dpi_twin_adapter import adapt_dpi_twin_state
from adapters.qms_diagnostics_adapter import enrich_with_qms_diagnostics

from dpi.reconstruction import (
    make_frequency_grid,
    propagate,
)


N = 96
dx = 8e-6
wavelength = 810e-9

coord = (
    np.arange(N)
    - N // 2
) * dx

X, Y = np.meshgrid(
    coord,
    coord,
    indexing="xy",
)

FX, FY = make_frequency_grid(
    N,
    dx,
)

reference = (
    1.5
    * np.exp(
        1j * (
            2*np.pi/160e-6 * X
            +
            2*np.pi/210e-6 * Y
        )
    )
)

true_depth = 0.073

source = np.exp(
    -(
        (X - 25e-6)**2
        +
        (Y + 15e-6)**2
    )
    /
    (
        2 * (30e-6)**2
    )
).astype(complex)

sensor = propagate(
    source,
    FX,
    FY,
    wavelength,
    true_depth,
)


def detector(E, ref):
    return np.abs(
        E + ref
    )**2


measurement = Measurement(
    data=np.stack([
        detector(sensor, reference),
        detector(sensor, 1j*reference),
        detector(sensor, -reference),
        detector(sensor, -1j*reference),
    ]),
    modality="quadrature",
    detector_id="integrated-demo",
)


depths = (
    np.arange(
        20.0,
        110.1,
        1.0,
    )
    * 1e-3
)


twin = QuantumMeasurementTwin(
    n=N,
    dx=dx,
    wavelength=wavelength,
    depths=depths,
    reference=reference,
)


state = twin.update(
    measurement
)


print()
print("=" * 72)
print("QMS PLATFORM — INTEGRATED TWIN")
print("=" * 72)

print(
    "Version:",
    state.version
)

print(
    "Detector:",
    state.measurement.detector_id
)

print(
    "Recovered source:",
    state.source.position
)

print(
    "True depth:",
    true_depth
)

print(
    "Recovered depth:",
    state.source.position[2]
)

print(
    "Depth error (mm):",
    (
        state.source.position[2]
        - true_depth
    ) * 1e3
)

print()
print("INTEGRATED TWIN OK")


platform_state = adapt_dpi_twin_state(state)

print()
print("=" * 72)
print("QMS PLATFORM — UNIFIED STATE")
print("=" * 72)

print(
    "Version:",
    platform_state.version
)

print(
    "Upstream position:",
    (
        platform_state.upstream.x,
        platform_state.upstream.y,
        platform_state.upstream.z,
    ),
)

print(
    "Direction:",
    platform_state.metadata["direction"]
)

print(
    "Depth score available:",
    platform_state.observability.depth_score is not None
)

print()
print("UNIFIED PLATFORM STATE OK")


platform_state = enrich_with_qms_diagnostics(
    platform_state
)

print()

print("=" * 72)
print("QMS PLATFORM — QMS DIAGNOSTICS")
print("=" * 72)

print(
    "Mean quadrature visibility:",
    platform_state.detector_diagnostics.visibility
)

print(
    "Quadrature visibility:",
    platform_state.detector_diagnostics.metadata[
        "quadrature_visibility"
    ]
)

print(
    "Quadrature uncertainty:",
    platform_state.detector_diagnostics.metadata[
        "quadrature_visibility_uncertainty"
    ]
)

print(
    "Visibility range:",
    (
        platform_state.detector_diagnostics.metadata[
            "visibility_min"
        ],
        platform_state.detector_diagnostics.metadata[
            "visibility_max"
        ],
    )
)

print(
    "Diagnostic source:",
    platform_state.detector_diagnostics.metadata[
        "qms_diagnostic"
    ]
)

print()
print("QMS DIAGNOSTICS INTEGRATION OK")
