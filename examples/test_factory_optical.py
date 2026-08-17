import numpy as np
import qms_platform as qmp

from dpi.reconstruction import (
    make_frequency_grid,
    propagate,
)

N = 96
dx = 8e-6
wavelength = 810e-9
depth = 0.073

coord = (np.arange(N) - N // 2) * dx
X, Y = np.meshgrid(coord, coord, indexing="xy")

FX, FY = make_frequency_grid(N, dx)

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

depths = np.arange(
    20.0,
    110.1,
    1.0,
) * 1e-3


def detector(E, ref):
    return np.abs(E + ref) ** 2


platform = qmp.create_optical_platform(
    n=N,
    dx=dx,
    wavelength=wavelength,
    depths=depths,
    reference=reference,
    detector_id="factory-optical",
)


for step in range(8):

    x0 = -20e-6 + step * 5e-6
    y0 = -10e-6 + step * 2e-6

    source = np.exp(
        -(
            (X-x0)**2
            +
            (Y-y0)**2
        )
        /
        (
            2*(30e-6)**2
        )
    ).astype(complex)

    sensor = propagate(
        source,
        FX,
        FY,
        wavelength,
        depth,
    )

    raw = np.stack([
        detector(sensor, reference),
        detector(sensor, 1j*reference),
        detector(sensor, -reference),
        detector(sensor, -1j*reference),
    ])

    state = platform.ingest(raw)


print("=" * 72)
print("QMS PLATFORM — OPTICAL FACTORY TEST")
print("=" * 72)

print(
    "Measurements:",
    platform.measurement_count
)

print(
    "Recovered depth:",
    state.upstream.z
)

print(
    "Dynamics ready:",
    state.dynamics is not None
)

print(
    "Visibility:",
    state.detector_diagnostics.visibility
)

print()
print("OPTICAL FACTORY OK")
