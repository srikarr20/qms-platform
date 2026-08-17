from qms_core import (
    Measurement,
    TwinState,
    SourceEstimate,
    ObservableEstimate,
)


measurement = Measurement(
    data="example detector frame",
    modality="quadrature",
    detector_id="detector-01",
    metadata={
        "wavelength": 810e-9,
        "pixel_spacing": 8e-6,
    },
)


source = SourceEstimate(
    position=(
        10e-6,
        -5e-6,
        0.100,
    ),
    confidence=0.91,
)


observable = ObservableEstimate(
    name="source_depth",
    value=0.100,
    confidence=0.91,
    best_depth=0.100,
)


twin = TwinState(
    measurement=measurement,
    source=source,
    observables=[
        observable
    ],
    version=1,
)


print()
print("=" * 70)
print("QMS PLATFORM — CORE STATE TEST")
print("=" * 70)

print(
    "Measurement modality:",
    twin.measurement.modality
)

print(
    "Detector:",
    twin.measurement.detector_id
)

print(
    "Source position:",
    twin.source.position
)

print(
    "Source confidence:",
    twin.source.confidence
)

print(
    "Observable:",
    twin.observables[0].name
)

print(
    "Observable value:",
    twin.observables[0].value
)

print(
    "Twin version:",
    twin.version
)

print()
print("CORE STATE MODEL OK")
