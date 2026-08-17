import numpy as np

from adapters.mri_kspace_adapter import (
    MRIKSpaceMeasurementAdapter,
    MRIKSpaceReconstructionAdapter,
)


# ============================================================
# SYNTHETIC 4D MRI IMAGE VOLUME
#
# Platform convention:
#     (T, H, W, Z)
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
        2.0
        * np.pi
        * t
        / T
    )

    cx = (
        0.22
        * np.sin(
            phase_t
        )
    )

    cy = (
        0.16
        * np.cos(
            phase_t
        )
    )

    sigma_x = (
        0.24
        + 0.025
        * np.sin(
            phase_t
        )
    )

    sigma_y = (
        0.20
        + 0.020
        * np.cos(
            phase_t
        )
    )

    amplitude = (
        1.0
        + 0.10
        * np.sin(
            phase_t + 0.3
        )
    )

    magnitude = (
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
                    2*0.50**2
                )
            )
        )
    )

    spatial_phase = (
        0.25
        + 0.15 * X
        - 0.10 * Y
        + 0.05 * np.sin(
            phase_t
        )
    )

    truth[t] = (
        magnitude
        * np.exp(
            1j * spatial_phase
        )
    )


# ============================================================
# FORWARD MRI ENCODING
#
# Image convention:
#     (T,H,W,Z)
#
# K-space adapter expects:
#     (T,Z,KY,KX)
#
# For each z slice:
#     image -> centered FFT2 -> k-space
# ============================================================

image_tzhw = np.transpose(
    truth,
    (0, 3, 1, 2),
)

kspace = np.empty_like(
    image_tzhw,
    dtype=np.complex128,
)


for t in range(T):
    for zi in range(Z):

        image_slice = (
            image_tzhw[
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
# INGEST ONLY K-SPACE
# ============================================================

measurement_adapter = (
    MRIKSpaceMeasurementAdapter(
        detector_id=
            "synthetic-cartesian-mri",
        metadata={
            "encoding":
                "Cartesian",
            "coil_count":
                1,
        },
    )
)

measurement = (
    measurement_adapter.to_measurement(
        kspace
    )
)


# ============================================================
# INVERSE RECONSTRUCTION
# ============================================================

reconstruction_adapter = (
    MRIKSpaceReconstructionAdapter()
)

state, reconstructed = (
    reconstruction_adapter.reconstruct(
        measurement
    )
)


# ============================================================
# VALIDATION
# ============================================================

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


magnitude_error = (
    np.linalg.norm(
        np.abs(
            reconstructed
        )
        -
        np.abs(
            truth
        )
    )
    /
    (
        np.linalg.norm(
            np.abs(
                truth
            )
        )
        + 1e-15
    )
)


phase_difference = np.angle(
    reconstructed
    * np.conj(
        truth
    )
)

weights = (
    np.abs(
        truth
    )**2
)

weighted_phase_error = (
    np.sum(
        weights
        * np.abs(
            phase_difference
        )
    )
    /
    (
        np.sum(
            weights
        )
        + 1e-15
    )
)


max_abs_error = float(
    np.max(
        np.abs(
            reconstructed
            - truth
        )
    )
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 78)
print("QMS PLATFORM — MRI K-SPACE INVERSE RECONSTRUCTION")
print("=" * 78)

print(
    "Truth shape:",
    truth.shape
)

print(
    "K-space shape:",
    kspace.shape
)

print(
    "Reconstructed shape:",
    reconstructed.shape
)

print()

print(
    "Normalized complex error:",
    f"{complex_error:.12e}"
)

print(
    "Normalized magnitude error:",
    f"{magnitude_error:.12e}"
)

print(
    "Weighted phase error:",
    f"{weighted_phase_error:.12e}",
    "rad"
)

print(
    "Maximum absolute error:",
    f"{max_abs_error:.12e}"
)

print()

print(
    "Field domain:",
    state.reconstructed_field.domain
)

print(
    "Inverse reconstruction performed:",
    state.reconstructed_field.metadata[
        "raw_mri_inverse_reconstruction"
    ]
)

print(
    "Reconstruction model:",
    state.metadata[
        "reconstruction_model"
    ]
)

print()

print(
    "k-space"
    " -> inverse FFT"
    " -> reconstructed 4D MRI field"
)

print()
print(
    "MRI K-SPACE RECONSTRUCTION OK"
)
