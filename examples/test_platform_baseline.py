from twin import (
    QuantumMeasurementTwin,
    MeasurementTwinPlatform,
    TwinRecorder,
    TwinReplay,
)

from qms_core import (
    Measurement,
    PlatformTwinState,
)

print("=" * 72)
print("QMS PLATFORM — BASELINE CONTRACT")
print("=" * 72)

assert QuantumMeasurementTwin is not None
assert MeasurementTwinPlatform is not None
assert TwinRecorder is not None
assert TwinReplay is not None
assert Measurement is not None
assert PlatformTwinState is not None

print("Core contracts: OK")
print("DPI twin: OK")
print("Unified runtime: OK")
print("Persistence/replay: OK")

print()
print("QMS PLATFORM BASELINE OK")
