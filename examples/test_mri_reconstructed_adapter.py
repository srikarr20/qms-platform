import numpy as np

from adapters.mri_reconstructed_adapter import (
    MRIReconstructedMeasurementAdapter,
    MRIReconstructedFieldAdapter,
)


# Synthetic reconstructed cine-MRI-like volume:
#
# H x W x Z x T

V = np.zeros(
    (
        64,
        64,
        8,
        20,
    ),
    dtype=float,
)

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

reconstruction_adapter = (
    MRIReconstructedFieldAdapter()
)

state, field = (
    reconstruction_adapter.reconstruct(
        measurement
    )
)


print("=" * 72)
print("QMS PLATFORM — MRI RECONSTRUCTED FIELD ADAPTER")
print("=" * 72)

print(
    "Measurement modality:",
    measurement.modality
)

print(
    "Original MRI shape:",
    measurement.data.shape
)

print(
    "Platform field shape:",
    field.shape
)

print(
    "Field domain:",
    state.reconstructed_field.domain
)

print(
    "Time axis:",
    state.reconstructed_field.metadata[
        "time_axis"
    ]
)

print(
    "Raw k-space reconstruction performed:",
    state.reconstructed_field.metadata[
        "raw_mri_inverse_reconstruction"
    ]
)

print()
print(
    "MRI RECONSTRUCTED FIELD ADAPTER OK"
)
