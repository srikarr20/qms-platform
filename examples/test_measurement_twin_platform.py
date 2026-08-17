import numpy as np

from qms_core import Measurement

from twin import (
    QuantumMeasurementTwin,
    MeasurementTwinPlatform,
    TwinRecorder,
    TwinReplay,
)

from dpi.reconstruction import (
    make_frequency_grid,
    propagate,
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


dpi_twin = QuantumMeasurementTwin(
    n=N,
    dx=dx,
    wavelength=wavelength,
    depths=depths,
    reference=reference,
)


RUN_DIR = "artifacts/live_measurement_twin_run"

recorder = TwinRecorder(
    RUN_DIR
)

platform = MeasurementTwinPlatform(
    dpi_twin=dpi_twin,
    min_dynamics_states=4,
    recorder=recorder,
)


print("MEASUREMENT TWIN PLATFORM SETUP OK")

print()

for step in range(12):

    x0 = -30e-6 + step * 5e-6
    y0 = -15e-6 + step * 2e-6

    sigma = (
        29e-6
        + 1.5e-6 * np.sin(step * 0.45)
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
            + 0.06 * np.sin(step * 0.35)
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
        detector_id="measurement-twin-runtime-demo",
    )

    state = platform.update(
        measurement
    )

    manifold_shape = (
        None
        if state.manifold is None
        else state.manifold.state.shape
    )

    print(
        f"step={step:02d}"
        f"  version={state.version:02d}"
        f"  z={state.upstream.z*1e3:7.3f} mm"
        f"  visibility="
        f"{state.detector_diagnostics.visibility:.6f}"
        f"  manifold={manifold_shape}"
        f"  dynamics={state.dynamics is not None}"
    )

print()
print("MEASUREMENT TWIN STREAM OK")

manifest = recorder.finalize()

replay = TwinReplay(
    "artifacts/live_measurement_twin_run"
)

print()
print("=" * 78)
print("LIVE TWIN RECORDING")
print("=" * 78)

print(
    "Manifest:",
    manifest
)

print(
    "Recorded states:",
    len(replay)
)

last_record = replay.state_metadata(
    platform.current_state.version
)

print(
    "Replay last version:",
    last_record["version"]
)

print(
    "Replay upstream z:",
    last_record[
        "upstream"
    ][
        "z"
    ]
)

print(
    "Replay visibility:",
    last_record[
        "detector_diagnostics"
    ][
        "visibility"
    ]
)

print()
print(
    "LIVE QMS + DPI + AURORA REPLAY OK"
)
