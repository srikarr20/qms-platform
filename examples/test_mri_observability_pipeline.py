import numpy as np

from adapters.mri_reconstructed_adapter import (
    MRIReconstructedMeasurementAdapter,
    MRIReconstructedFieldAdapter,
)

from twin import (
    build_observability_layer,
)

from adapters.aurora_dynamics_adapter import (
    enrich_with_aurora_dynamics,
)


# ============================================================
# SYNTHETIC 4D MRI-LIKE FIELD
#
# Original convention:
#     H x W x Z x T
# ============================================================

H = 64
W = 64
Z = 8
T = 20

x = np.linspace(-1, 1, H)
y = np.linspace(-1, 1, W)
z = np.linspace(-1, 1, Z)

X, Y, ZG = np.meshgrid(
    x,
    y,
    z,
    indexing="ij",
)

V = np.zeros(
    (H, W, Z, T),
    dtype=float,
)


for t in range(T):

    # Smooth cyclic motion / deformation
    phase = (
        2 * np.pi * t / T
    )

    cx = (
        0.20
        * np.sin(phase)
    )

    cy = (
        0.12
        * np.cos(phase)
    )

    sigma_x = (
        0.28
        + 0.03
        * np.sin(phase)
    )

    sigma_y = (
        0.22
        + 0.02
        * np.cos(phase)
    )

    amplitude = (
        1.0
        + 0.12
        * np.sin(
            phase + 0.4
        )
    )

    frame = (
        amplitude
        * np.exp(
            -(
                (X-cx)**2
                /
                (
                    2*sigma_x**2
                )
                +
                (Y-cy)**2
                /
                (
                    2*sigma_y**2
                )
                +
                ZG**2
                /
                (
                    2*0.45**2
                )
            )
        )
    )

    V[..., t] = frame


# ============================================================
# MRI INGESTION
# ============================================================

measurement_adapter = (
    MRIReconstructedMeasurementAdapter(
        detector_id="synthetic-cine-mri"
    )
)

measurement = (
    measurement_adapter.to_measurement(
        V
    )
)


# ============================================================
# MRI FIELD ADAPTER
# ============================================================

reconstruction_adapter = (
    MRIReconstructedFieldAdapter()
)

platform_state, field_sequence = (
    reconstruction_adapter.reconstruct(
        measurement
    )
)


# ============================================================
# SHARED OBSERVABILITY LAYER
# ============================================================

detector_state, manifold = (
    build_observability_layer(
        field_sequence,
        field_domain=
            "mri_reconstructed_volume",
    )
)

platform_state.detector_state = (
    detector_state
)

platform_state.manifold = (
    manifold
)


# ============================================================
# AURORA DYNAMICS
# ============================================================

platform_state = (
    enrich_with_aurora_dynamics(
        platform_state
    )
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 78)
print("QMS PLATFORM — MRI OBSERVABILITY PIPELINE")
print("=" * 78)

print(
    "Original MRI:",
    V.shape
)

print(
    "Platform field:",
    field_sequence.shape
)

print(
    "Detector state:",
    platform_state.detector_state.data.shape
)

print(
    "Observable manifold:",
    platform_state.manifold.state.shape
)

print(
    "Manifold names:",
    platform_state.manifold.names
)

print()

print(
    "AURORA drift:",
    platform_state.dynamics.trajectory[
        "drift"
    ]
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
print(
    "Pipeline:"
)

print(
    "reconstructed MRI"
    " -> temporal detector state"
    " -> CKE manifold"
    " -> AURORA dynamics"
)

print()
print(
    "MRI OBSERVABILITY PIPELINE OK"
)
