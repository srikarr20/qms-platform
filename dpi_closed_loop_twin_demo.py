import numpy as np

from qms_core import (
    Measurement,
    ComplexFieldState,
    VirtualVolumeState,
    SourceEstimate,
    ObservableEstimate,
    TwinState,
    TwinRuntime,
)

# ============================================================
# QMS PLATFORM — CLOSED-LOOP DPI TWIN
#
# Physical:
#     source -> detector measurement
#
# Inverse twin:
#     detector -> virtual upstream source
#
# Forward twin:
#     reconstructed source -> predicted detector
#
# Validation:
#     actual detector - predicted detector = residual
# ============================================================

N = 96
wavelength = 810e-9
dx = 8e-6
source_depth = 0.100

coord = (np.arange(N) - N // 2) * dx

X, Y = np.meshgrid(
    coord,
    coord,
    indexing="xy",
)

fx = np.fft.fftfreq(N, d=dx)
fy = np.fft.fftfreq(N, d=dx)

FX, FY = np.meshgrid(
    fx,
    fy,
    indexing="xy",
)


def transfer(z):
    return np.exp(
        -1j
        * np.pi
        * wavelength
        * z
        * (FX**2 + FY**2)
    )


def propagate(field, H):
    return np.fft.ifft2(
        np.fft.fft2(field)
        * H
    )


H_FORWARD = transfer(source_depth)

# ============================================================
# REFERENCE
# ============================================================

reference_amp = 1.5

kx = 2*np.pi / 160e-6
ky = 2*np.pi / 210e-6

R = reference_amp * np.exp(
    1j * (kx*X + ky*Y)
)


def detector(E, ref):
    return np.abs(E + ref)**2


# ============================================================
# SOURCE
# ============================================================

sigma = 30e-6


def source_field(x0, y0):

    field = np.exp(
        -(
            (X-x0)**2
            + (Y-y0)**2
        )
        / (2*sigma**2)
    )

    return (
        field.astype(complex)
        * np.exp(1j*0.35)
    )


# ============================================================
# PHYSICAL MEASUREMENT
# ============================================================

def make_measurement(x0, y0, step):

    source = source_field(
        x0,
        y0,
    )

    sensor = propagate(
        source,
        H_FORWARD,
    )

    data = np.stack([
        detector(sensor, R),
        detector(sensor, 1j*R),
        detector(sensor, -R),
        detector(sensor, -1j*R),
    ])

    return Measurement(
        data=data,
        modality="quadrature",
        detector_id="simulated-4Q-stream",
        metadata={
            "step": step,
            "truth_x": x0,
            "truth_y": y0,
            "truth_z": source_depth,
        },
    )


# ============================================================
# VIRTUAL DEPTHS
# ============================================================

DEPTHS = np.arange(
    0.0,
    0.1001,
    0.005,
)


# ============================================================
# DPI UPDATE
# ============================================================

def dpi_update(
    measurement,
    previous_state,
):

    I0, I90, I180, I270 = measurement.data

    Qc = I0 - I180
    Qs = I90 - I270

    sensor_field = (
        Qc + 1j*Qs
    ) / (
        4*np.conj(R)
        + 1e-15
    )

    detector_state = ComplexFieldState(
        field=sensor_field,
        wavelength=wavelength,
        pixel_spacing=dx,
        z=0.0,
        timestamp=measurement.timestamp,
    )

    # --------------------------------------------------------
    # INVERSE VIRTUAL VOLUME
    # --------------------------------------------------------

    volume = np.empty(
        (
            len(DEPTHS),
            N,
            N,
        ),
        dtype=np.complex64,
    )

    for zi, z in enumerate(DEPTHS):

        volume[zi] = propagate(
            sensor_field,
            transfer(-z),
        )

    volume_state = VirtualVolumeState(
        field=volume,
        depths=DEPTHS,
        wavelength=wavelength,
        pixel_spacing=dx,
        timestamp=measurement.timestamp,
        metadata={
            "direction":
                "downstream-to-upstream"
        },
    )

    reconstructed_source_field = (
        volume[-1]
    )

    intensity = (
        np.abs(
            reconstructed_source_field
        )**2
    )

    total = (
        intensity.sum()
        + 1e-15
    )

    cx = float(
        np.sum(
            intensity * X
        ) / total
    )

    cy = float(
        np.sum(
            intensity * Y
        ) / total
    )

    source = SourceEstimate(
        position=(
            cx,
            cy,
            source_depth,
        ),
        confidence=1.0,
    )

    # --------------------------------------------------------
    # CLOSED-LOOP FORWARD TWIN
    #
    # reconstructed upstream state -> predicted detector field
    # --------------------------------------------------------

    predicted_sensor_field = propagate(
        reconstructed_source_field,
        H_FORWARD,
    )

    predicted_measurement = np.stack([
        detector(
            predicted_sensor_field,
            R,
        ),
        detector(
            predicted_sensor_field,
            1j*R,
        ),
        detector(
            predicted_sensor_field,
            -R,
        ),
        detector(
            predicted_sensor_field,
            -1j*R,
        ),
    ])

    # --------------------------------------------------------
    # DETECTOR RESIDUAL
    # --------------------------------------------------------

    actual = measurement.data

    residual = (
        actual
        - predicted_measurement
    )

    normalized_residual = (
        np.linalg.norm(residual)
        /
        (
            np.linalg.norm(actual)
            + 1e-15
        )
    )

    # complex detector-field consistency
    field_residual = (
        np.linalg.norm(
            sensor_field
            - predicted_sensor_field
        )
        /
        (
            np.linalg.norm(
                sensor_field
            )
            + 1e-15
        )
    )

    observable = ObservableEstimate(
        name="detector_consistency",
        value=1.0-normalized_residual,
        confidence=1.0,
        metadata={
            "measurement_residual":
                float(normalized_residual),

            "field_residual":
                float(field_residual),
        },
    )

    trajectory = []

    residual_history = []

    if previous_state is not None:

        trajectory = list(
            previous_state.metadata.get(
                "trajectory",
                [],
            )
        )

        residual_history = list(
            previous_state.metadata.get(
                "residual_history",
                [],
            )
        )

    trajectory.append(
        source.position
    )

    residual_history.append(
        float(
            normalized_residual
        )
    )

    return TwinState(
        measurement=measurement,
        detector_field=detector_state,
        virtual_volume=volume_state,
        source=source,
        observables=[
            observable
        ],
        metadata={
            "trajectory":
                trajectory,

            "residual_history":
                residual_history,

            "predicted_detector_field":
                predicted_sensor_field,

            "predicted_measurement":
                predicted_measurement,

            "measurement_residual":
                float(
                    normalized_residual
                ),

            "field_residual":
                float(
                    field_residual
                ),

            "pipeline":
                (
                    "measurement -> inverse twin -> "
                    "upstream state -> forward twin -> "
                    "predicted detector -> residual"
                ),
        },
    )


# ============================================================
# STREAMING RUNTIME
# ============================================================

runtime = TwinRuntime(
    update_fn=dpi_update
)

N_STEPS = 12

measurement_residuals = []
field_residuals = []

print()
print("=" * 76)
print("QMS PLATFORM — CLOSED-LOOP DPI TWIN")
print("=" * 76)


for step in range(N_STEPS):

    x0 = (
        -45e-6
        + step*8e-6
    )

    y0 = (
        -25e-6
        + step*4e-6
    )

    measurement = make_measurement(
        x0,
        y0,
        step,
    )

    state = runtime.update(
        measurement
    )

    rx, ry, rz = (
        state.source.position
    )

    source_error_um = (
        1e6
        * np.sqrt(
            (rx-x0)**2
            + (ry-y0)**2
        )
    )

    measurement_residual = (
        state.metadata[
            "measurement_residual"
        ]
    )

    field_residual = (
        state.metadata[
            "field_residual"
        ]
    )

    measurement_residuals.append(
        measurement_residual
    )

    field_residuals.append(
        field_residual
    )

    print(
        f"step {step:02d}"
        f"  source_error={source_error_um:.9f} um"
        f"  measurement_residual={measurement_residual:.10e}"
        f"  field_residual={field_residual:.10e}"
    )


# ============================================================
# SUMMARY
# ============================================================

measurement_residuals = np.asarray(
    measurement_residuals
)

field_residuals = np.asarray(
    field_residuals
)

print()
print("=" * 76)
print("CLOSED-LOOP TWIN SUMMARY")
print("=" * 76)

print(
    "Measurements processed:",
    runtime.measurement_count
)

print(
    "Mean detector-measurement residual:",
    f"{measurement_residuals.mean():.12e}"
)

print(
    "Max detector-measurement residual:",
    f"{measurement_residuals.max():.12e}"
)

print(
    "Mean complex-field residual:",
    f"{field_residuals.mean():.12e}"
)

print(
    "Max complex-field residual:",
    f"{field_residuals.max():.12e}"
)

print()
print(
    "Closed loop:"
)

print(
    "detector -> upstream twin -> predicted detector -> residual"
)

print()
print(
    "CLOSED-LOOP DPI TWIN OK"
)
