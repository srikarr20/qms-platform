import numpy as np

from qms_core import Measurement
from twin import QuantumMeasurementTwin, build_observability_layer
from dpi.reconstruction import make_frequency_grid, propagate


N = 96
dx = 8e-6
wavelength = 810e-9
depth = 0.073

coord = (np.arange(N) - N // 2) * dx
X, Y = np.meshgrid(coord, coord, indexing="xy")

FX, FY = make_frequency_grid(N, dx)

reference = 1.5 * np.exp(
    1j * (
        2 * np.pi / 160e-6 * X
        + 2 * np.pi / 210e-6 * Y
    )
)


def detector(E, ref):
    return np.abs(E + ref) ** 2


depths = np.arange(20.0, 110.1, 1.0) * 1e-3

twin = QuantumMeasurementTwin(
    n=N,
    dx=dx,
    wavelength=wavelength,
    depths=depths,
    reference=reference,
)

reconstructed_fields = []

for step in range(12):

    x0 = -30e-6 + step * 5e-6
    y0 = -15e-6 + step * 2e-6
    sigma = 30e-6

    source = np.exp(
        -(
            (X - x0) ** 2
            + (Y - y0) ** 2
        )
        / (2 * sigma ** 2)
    ).astype(complex)

    source *= np.exp(
        1j * (0.25 + step * 0.04)
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
        detector_id="temporal-demo",
    )

    state = twin.update(measurement)

    reconstructed_fields.append(
        state.detector_field.field
    )


reconstructed_fields = np.asarray(
    reconstructed_fields
)

detector_state, manifold = build_observability_layer(
    reconstructed_fields,
    field_domain="optical_complex_detector_field",
)


print()
print("=" * 74)
print("QMS PLATFORM — OBSERVABLE MANIFOLD TEST")
print("=" * 74)

print(
    "Field sequence shape:",
    reconstructed_fields.shape,
)

print(
    "Detector state shape:",
    detector_state.data.shape,
)

print(
    "Manifold shape:",
    manifold.state.shape,
)

print(
    "Manifold names:",
    manifold.names,
)

print()

for name in ("C", "K", "E"):
    values = manifold.metadata[name]

    print(
        f"{name} range:",
        (
            float(np.min(values)),
            float(np.max(values)),
        ),
    )

print()

print(
    "First manifold state:",
    manifold.state[0],
)

print(
    "Last manifold state:",
    manifold.state[-1],
)

print()
print("OBSERVABLE MANIFOLD OK")
