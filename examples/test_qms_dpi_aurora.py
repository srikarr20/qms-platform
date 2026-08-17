import numpy as np

from qms_core import Measurement

from twin import (
    QuantumMeasurementTwin,
    build_observability_layer,
)

from dpi.reconstruction import (
    make_frequency_grid,
    propagate,
)

from adapters.dpi_twin_adapter import (
    adapt_dpi_twin_state,
)

from adapters.qms_diagnostics_adapter import (
    enrich_with_qms_diagnostics,
)

from adapters.aurora_dynamics_adapter import (
    enrich_with_aurora_dynamics,
)


N = 96
dx = 8e-6
wavelength = 810e-9
depth = 0.073

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


def detector(E, ref):
    return np.abs(E + ref)**2


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

print("QMS + DPI + AURORA SETUP OK")

reconstructed_fields = []
platform_state = None

N_STEPS = 16

for step in range(N_STEPS):

    x0 = -35e-6 + step * 4e-6
    y0 = -20e-6 + step * 2.5e-6

    sigma = (
        28e-6
        + 2e-6 * np.sin(step * 0.5)
    )

    source = np.exp(
        -(
            (X - x0)**2
            +
            (Y - y0)**2
        )
        /
        (
            2 * sigma**2
        )
    ).astype(complex)

    source *= np.exp(
        1j * (
            0.25
            + 0.08 * np.sin(step * 0.4)
        )
    )

    sensor = propagate(
        source,
        FX,
        FY,
        wavelength,
        depth,
    )

    measurement = Measurement(
        data=np.stack([
            detector(sensor, reference),
            detector(sensor, 1j * reference),
            detector(sensor, -reference),
            detector(sensor, -1j * reference),
        ]),
        modality="quadrature",
        detector_id="qms-dpi-aurora-demo",
    )

    dpi_state = twin.update(
        measurement
    )

    platform_state = adapt_dpi_twin_state(
        dpi_state
    )

    platform_state = enrich_with_qms_diagnostics(
        platform_state
    )

    reconstructed_fields.append(
        dpi_state.detector_field.field
    )

reconstructed_fields = np.asarray(
    reconstructed_fields
)

print(
    "STREAM STATES:",
    reconstructed_fields.shape
)

print(
    "LAST UPSTREAM:",
    (
        platform_state.upstream.x,
        platform_state.upstream.y,
        platform_state.upstream.z,
    )
)

print(
    "LAST QMS VISIBILITY:",
    platform_state.detector_diagnostics.visibility
)

print("STREAM + UNIFIED STATE OK")

detector_state, manifold = build_observability_layer(
    reconstructed_fields,
    field_domain="optical_complex_detector_field",
)

platform_state.detector_state = detector_state
platform_state.manifold = manifold

print(
    "DETECTOR STATE:",
    platform_state.detector_state.data.shape
)

print(
    "MANIFOLD:",
    platform_state.manifold.state.shape
)

print(
    "MANIFOLD NAMES:",
    platform_state.manifold.names
)

platform_state = enrich_with_aurora_dynamics(
    platform_state
)

print()
print("=" * 78)
print("QMS + DPI + AURORA — UNIFIED DYNAMICS")
print("=" * 78)

print(
    "AURORA drift:",
    platform_state.dynamics.trajectory[
        "drift"
    ]
)

print(
    "AURORA mean speed:",
    float(
        np.mean(
            platform_state.dynamics.trajectory[
                "speed"
            ]
        )
    )
)

print(
    "AURORA attractor radius:",
    platform_state.dynamics.attractor[
        "radius"
    ]
)

print(
    "AURORA attractor distortion:",
    platform_state.dynamics.attractor[
        "distortion_score"
    ]
)

print(
    "AURORA mean local instability:",
    float(
        np.mean(
            platform_state.dynamics.instability
        )
    )
)

print(
    "AURORA phase samples:",
    len(
        platform_state.dynamics.phase[
            "phase"
        ]
    )
)

print()
print("FULL PIPELINE:")
print(
    "measurement"
    " -> QMS diagnostics"
    " -> DPI inverse reconstruction"
    " -> reconstructed field evolution"
    " -> detector state"
    " -> observable manifold"
    " -> AURORA dynamics"
)

print()
print("QMS + DPI + AURORA INTEGRATION OK")
