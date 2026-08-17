import qms_platform as qmp

print("=" * 72)
print("QMS PLATFORM — PUBLIC API")
print("=" * 72)

print("Version:", qmp.__version__)

required = [
    "MeasurementTwinPlatform",
    "DPIReconstructionAdapter",
    "MRIKSpaceReconstructionAdapter",
    "TwinRecorder",
    "TwinReplay",
]

for name in required:
    assert hasattr(qmp, name), name
    print(name + ": OK")

print()
print("QMS PLATFORM PUBLIC API OK")
