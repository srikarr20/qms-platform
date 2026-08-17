from pathlib import Path
import shutil

from twin import (
    TwinRecorder,
    TwinReplay,
)

from qms_core import (
    Measurement,
    PlatformTwinState,
    DetectorDiagnostics,
    UpstreamEstimate,
)


RUN_DIR = Path(
    "artifacts/test_twin_run"
)

if RUN_DIR.exists():
    shutil.rmtree(
        RUN_DIR
    )


recorder = TwinRecorder(
    RUN_DIR
)


for version in range(1, 4):

    state = PlatformTwinState(
        measurement=Measurement(
            data=None,
            modality="test",
            detector_id="test-detector",
        ),

        detector_diagnostics=
            DetectorDiagnostics(
                visibility=
                    0.1 * version
            ),

        upstream=
            UpstreamEstimate(
                x=version * 1e-6,
                y=version * 2e-6,
                z=0.073,
            ),

        version=version,
    )

    recorder.record(
        state
    )


manifest = recorder.finalize()


replay = TwinReplay(
    RUN_DIR
)


print()
print("=" * 72)
print("QMS PLATFORM — PERSISTENCE / REPLAY TEST")
print("=" * 72)

print(
    "Manifest:",
    manifest
)

print(
    "Recorded states:",
    len(replay)
)


last = replay.state_metadata(
    3
)


print(
    "Last version:",
    last["version"]
)

print(
    "Last visibility:",
    last[
        "detector_diagnostics"
    ][
        "visibility"
    ]
)

print(
    "Last upstream z:",
    last[
        "upstream"
    ][
        "z"
    ]
)

print()
print(
    "TWIN PERSISTENCE + REPLAY OK"
)
