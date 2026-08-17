import numpy as np

from qms_core import (
    QuadratureMeasurementAdapter,
)

from twin import (
    QuantumMeasurementTwin,
    MeasurementTwinPlatform,
)

from dpi.reconstruction import (
    make_frequency_grid,
    propagate,
)

from adapters.dpi_reconstruction_adapter import (
    DPIReconstructionAdapter,
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


adapter = QuadratureMeasurementAdapter(
    detector_id="raw-4q-runtime",
    metadata={
        "wavelength": wavelength,
        "pixel_spacing": dx,
    },
)


reconstruction_adapter = DPIReconstructionAdapter(
    dpi_twin=dpi_twin,
)

platform = MeasurementTwinPlatform(
    reconstruction_adapter=reconstruction_adapter,
    min_dynamics_states=4,
    measurement_adapter=adapter,
)


print()
print("=" * 76)
print("QMS PLATFORM — RAW INGEST TEST")
print("=" * 76)


for step in range(8):

    x0 = (
        -20e-6
        + step * 5e-6
    )

    y0 = (
        -10e-6
        + step * 2e-6
    )

    sigma = 30e-6

    source = np.exp(
        -(
            (X-x0)**2
            +
            (Y-y0)**2
        )
        /
        (
            2*sigma**2
        )
    ).astype(complex)

    source *= np.exp(
        1j * (
            0.25
            + 0.05 * step
        )
    )

    sensor = propagate(
        source,
        FX,
        FY,
        wavelength,
        depth,
    )

    raw = np.stack([
        detector(
            sensor,
            reference,
        ),

        detector(
            sensor,
            1j*reference,
        ),

        detector(
            sensor,
            -reference,
        ),

        detector(
            sensor,
            -1j*reference,
        ),
    ])

    state = platform.ingest(
        raw
    )

    print(
        f"step={step:02d}"
        f"  modality={state.measurement.modality}"
        f"  detector={state.measurement.detector_id}"
        f"  z={state.upstream.z*1e3:.3f} mm"
        f"  dynamics={state.dynamics is not None}"
    )


print()
print(
    "Measurements processed:",
    platform.measurement_count
)

print(
    "Final upstream depth:",
    platform.current_state.upstream.z
)

print(
    "Dynamics ready:",
    platform.dynamics_ready
)

print()
print(
    "RAW DETECTOR -> PLATFORM INGEST OK"
)
