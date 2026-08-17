import numpy as np

from twin import MeasurementTwinPlatform

from adapters.mri_kspace_adapter import (
    MRIKSpaceMeasurementAdapter,
    MRIKSpaceReconstructionAdapter,
)


# ============================================================
# SYNTHETIC MRI TRUTH
#
# Platform image convention:
#     (T,H,W,Z)
# ============================================================

T = 12
H = 64
W = 64
Z = 6

x = np.linspace(-1.0, 1.0, H)
y = np.linspace(-1.0, 1.0, W)
z = np.linspace(-1.0, 1.0, Z)

X, Y, ZG = np.meshgrid(
    x,
    y,
    z,
    indexing="ij",
)

truth = np.zeros(
    (T, H, W, Z),
    dtype=np.complex128,
)


for t in range(T):

    phase_t = (
        2*np.pi*t/T
    )

    cx = (
        0.20
        * np.sin(phase_t)
    )

    cy = (
        0.14
        * np.cos(phase_t)
    )

    sigma_x = (
        0.24
        + 0.02*np.sin(phase_t)
    )

    sigma_y = (
        0.20
        + 0.02*np.cos(phase_t)
    )

    magnitude = np.exp(
        -(
            (X-cx)**2
            /
            (2*sigma_x**2)
            +
            (Y-cy)**2
            /
            (2*sigma_y**2)
            +
            ZG**2
            /
            (2*0.50**2)
        )
    )

    phase = (
        0.20
        + 0.12*X
        - 0.08*Y
        + 0.04*np.sin(phase_t)
    )

    truth[t] = (
        magnitude
        * np.exp(1j*phase)
    )


# ============================================================
# FORWARD ENCODING:
#
# image-space
#     ->
# Cartesian k-space
#
# Adapter expects:
#     (T,Z,KY,KX)
# ============================================================

truth_tzhw = np.transpose(
    truth,
    (0, 3, 1, 2),
)

kspace = np.empty_like(
    truth_tzhw,
    dtype=np.complex128,
)


for t in range(T):
    for zi in range(Z):

        image_slice = (
            truth_tzhw[
                t,
                zi,
            ]
        )

        kspace[t, zi] = (
            np.fft.fftshift(
                np.fft.fft2(
                    np.fft.ifftshift(
                        image_slice
                    )
                )
            )
        )


# ============================================================
# PLATFORM
# ============================================================

measurement_adapter = (
    MRIKSpaceMeasurementAdapter(
        detector_id=
            "raw-cartesian-mri-runtime",
        metadata={
            "encoding":
                "Cartesian",

            "coil_count":
                1,
        },
    )
)

reconstruction_adapter = (
    MRIKSpaceReconstructionAdapter()
)

platform = MeasurementTwinPlatform(
    reconstruction_adapter=
        reconstruction_adapter,

    measurement_adapter=
        measurement_adapter,

    min_dynamics_states=4,
)


# ============================================================
# ONE RAW INGEST CALL
# ============================================================

state = platform.ingest(
    kspace
)


# ============================================================
# RECONSTRUCTION VALIDATION
# ============================================================

reconstructed = (
    state.reconstructed_field.data
)

complex_error = (
    np.linalg.norm(
        reconstructed
        - truth
    )
    /
    (
        np.linalg.norm(
            truth
        )
        + 1e-15
    )
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 80)
print("QMS PLATFORM — RAW MRI INGEST RUNTIME")
print("=" * 80)

print(
    "Input k-space:",
    kspace.shape
)

print(
    "Reconstructed field:",
    reconstructed.shape
)

print(
    "Normalized complex error:",
    f"{complex_error:.12e}"
)

print()

print(
    "Temporal source:",
    state.metadata[
        "temporal_source"
    ]
)

print(
    "Detector state:",
    state.detector_state.data.shape
)

print(
    "Observable manifold:",
    state.manifold.state.shape
)

print(
    "Manifold names:",
    state.manifold.names
)

print(
    "Dynamics ready:",
    state.dynamics is not None
)

print()

print(
    "AURORA drift:",
    state.dynamics.trajectory[
        "drift"
    ]
)

print(
    "AURORA attractor radius:",
    state.dynamics.attractor[
        "radius"
    ]
)

print(
    "AURORA attractor distortion:",
    state.dynamics.attractor[
        "distortion_score"
    ]
)

print(
    "AURORA mean instability:",
    float(
        np.mean(
            state.dynamics.instability
        )
    )
)

print()

print(
    "Pipeline:"
)

print(
    "raw k-space"
    " -> MeasurementAdapter"
    " -> MRI inverse reconstruction"
    " -> reconstructed 4D field"
    " -> DetectorState"
    " -> CKE manifold"
    " -> AURORA dynamics"
)

print()

print(
    "RAW MRI -> PLATFORM INGEST OK"
)
