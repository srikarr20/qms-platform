import numpy as np

from qms_core import (
    QuadratureMeasurementAdapter,
)

raw = np.zeros(
    (4, 32, 32)
)

adapter = QuadratureMeasurementAdapter(
    detector_id="test-4q",
    metadata={
        "wavelength": 810e-9,
    },
)

measurement = adapter.to_measurement(
    raw
)

print("=" * 72)
print("QMS PLATFORM — MEASUREMENT INGESTION")
print("=" * 72)

print(
    "Modality:",
    measurement.modality
)

print(
    "Detector:",
    measurement.detector_id
)

print(
    "Shape:",
    measurement.data.shape
)

print(
    "Wavelength:",
    measurement.metadata[
        "wavelength"
    ]
)

print()
print(
    "MEASUREMENT INGESTION OK"
)
