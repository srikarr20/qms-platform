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
# QMS PLATFORM — STREAMING DPI TWIN
#
# A simulated source moves upstream.
#
# Physical direction:
#
#     source -> detector
#
# Twin direction:
#
#     detector measurement -> upstream reconstruction
#
# The TwinState is updated once for every measurement.
# ============================================================


# ============================================================
# PHYSICS
# ============================================================

N = 96

wavelength = 810e-9

dx = 8e-6

source_depth = 0.100


coord = (
    np.arange(N)
    - N // 2
) * dx


X, Y = np.meshgrid(
    coord,
    coord,
    indexing="xy",
)


fx = np.fft.fftfreq(
    N,
    d=dx,
)

fy = np.fft.fftfreq(
    N,
    d=dx,
)


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
        * (
            FX**2
            + FY**2
        )
    )


def propagate(
    field,
    H,
):

    return np.fft.ifft2(
        np.fft.fft2(field)
        * H
    )


H_FORWARD = transfer(
    source_depth
)


# ============================================================
# QUADRATURE REFERENCE
# ============================================================

reference_amp = 1.5

kx = 2*np.pi / 160e-6

ky = 2*np.pi / 210e-6


R = (
    reference_amp
    * np.exp(
        1j
        * (
            kx*X
            + ky*Y
        )
    )
)


def detector(
    E,
    ref,
):

    return np.abs(
        E + ref
    )**2


# ============================================================
# SOURCE GENERATOR
# ============================================================

sigma = 30e-6


def source_field(
    x0,
    y0,
):

    field = np.exp(
        -(
            (X-x0)**2
            +
            (Y-y0)**2
        )
        /
        (
            2*sigma**2
        )
    )

    return (
        field.astype(complex)
        * np.exp(
            1j*0.35
        )
    )


# ============================================================
# FORWARD MEASUREMENT
# ============================================================

def make_measurement(
    x0,
    y0,
    step,
):

    source = source_field(
        x0,
        y0,
    )

    sensor = propagate(
        source,
        H_FORWARD,
    )

    I0 = detector(
        sensor,
        R,
    )

    I90 = detector(
        sensor,
        1j*R,
    )

    I180 = detector(
        sensor,
        -R,
    )

    I270 = detector(
        sensor,
        -1j*R,
    )

    data = np.stack(
        [
            I0,
            I90,
            I180,
            I270,
        ]
    )

    return Measurement(
        data=data,
        modality="quadrature",
        detector_id="simulated-4Q-stream",
        metadata={
            "step": step,

            "truth_x": x0,

            "truth_y": y0,

            "truth_z": source_depth,

            "wavelength": wavelength,

            "pixel_spacing": dx,
        },
    )


# ============================================================
# DPI UPDATE FUNCTION
#
# This is the function TwinRuntime calls for every measurement.
# ============================================================

DEPTHS = np.arange(
    0.0,
    0.1001,
    0.005,
)


def dpi_update(
    measurement,
    previous_state,
):

    I0 = measurement.data[0]

    I90 = measurement.data[1]

    I180 = measurement.data[2]

    I270 = measurement.data[3]


    # --------------------------------------------------------
    # Recover downstream complex detector field
    # --------------------------------------------------------

    Qc = (
        I0
        - I180
    )

    Qs = (
        I90
        - I270
    )


    sensor_field = (
        Qc
        + 1j*Qs
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

        metadata={
            "method":
                "4-quadrature reconstruction"
        },
    )


    # --------------------------------------------------------
    # Reconstruct virtual propagation volume
    # --------------------------------------------------------

    volume = np.empty(
        (
            len(DEPTHS),
            N,
            N,
        ),
        dtype=np.complex64,
    )


    for zi, z in enumerate(
        DEPTHS
    ):

        volume[zi] = propagate(
            sensor_field,
            transfer(-z),
        )


    volume_state = VirtualVolumeState(
        field=volume,

        depths=DEPTHS,

        timestamp=measurement.timestamp,

        wavelength=wavelength,

        pixel_spacing=dx,

        metadata={
            "direction":
                "downstream-to-upstream"
        },
    )


    # --------------------------------------------------------
    # Source estimate at 100 mm virtual plane
    # --------------------------------------------------------

    source_plane = volume[-1]

    intensity = (
        np.abs(source_plane)**2
    )

    total = (
        intensity.sum()
        + 1e-15
    )


    cx = float(
        np.sum(
            intensity*X
        )
        / total
    )

    cy = float(
        np.sum(
            intensity*Y
        )
        / total
    )


    source = SourceEstimate(
        position=(
            cx,
            cy,
            source_depth,
        ),

        confidence=1.0,

        metadata={
            "method":
                "source-plane centroid"
        },
    )


    observable = ObservableEstimate(
        name="source_position",

        value=source.position,

        confidence=source.confidence,

        best_depth=source_depth,
    )


    # --------------------------------------------------------
    # Maintain simple trajectory history
    # --------------------------------------------------------

    trajectory = []

    if previous_state is not None:

        trajectory = list(
            previous_state.metadata.get(
                "trajectory",
                [],
            )
        )


    trajectory.append(
        source.position
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

            "pipeline":
                (
                    "measurement -> detector field -> "
                    "virtual volume -> source state"
                ),
        },
    )


# ============================================================
# RUNTIME
# ============================================================

runtime = TwinRuntime(
    update_fn=dpi_update
)


# ============================================================
# SIMULATED STREAM
#
# Move source diagonally through 12 measurement updates.
# ============================================================

N_STEPS = 12


truth_positions = []

recovered_positions = []


print()
print("=" * 74)
print("QMS PLATFORM — STREAMING DPI TWIN")
print("=" * 74)


for step in range(
    N_STEPS
):

    x0 = (
        -45e-6
        + step
        * 8e-6
    )

    y0 = (
        -25e-6
        + step
        * 4e-6
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


    error_um = (
        1e6
        * np.sqrt(
            (rx-x0)**2
            +
            (ry-y0)**2
        )
    )


    truth_positions.append(
        (
            x0,
            y0,
        )
    )

    recovered_positions.append(
        (
            rx,
            ry,
        )
    )


    print(
        f"step {step:02d}"
        f"  twin_version={state.version:02d}"
        f"  true=({x0*1e6:7.2f},"
        f"{y0*1e6:7.2f}) um"
        f"  recovered=({rx*1e6:7.2f},"
        f"{ry*1e6:7.2f}) um"
        f"  error={error_um:.6f} um"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

truth_positions = np.asarray(
    truth_positions
)

recovered_positions = np.asarray(
    recovered_positions
)


errors_um = (
    1e6
    * np.sqrt(
        np.sum(
            (
                truth_positions
                - recovered_positions
            )**2,
            axis=1,
        )
    )
)


print()
print("=" * 74)
print("STREAMING TWIN SUMMARY")
print("=" * 74)


print(
    "Measurements processed:",
    runtime.measurement_count
)


print(
    "Final twin version:",
    runtime.current_state.version
)


print(
    "Trajectory states:",
    len(
        runtime.current_state.metadata[
            "trajectory"
        ]
    )
)


print(
    "Mean source-position error:",
    f"{errors_um.mean():.9f}",
    "microns"
)


print(
    "Max source-position error:",
    f"{errors_um.max():.9f}",
    "microns"
)


print()
print(
    "Physical direction:"
)

print(
    "source -> propagation -> detector"
)


print()
print(
    "Twin direction:"
)

print(
    "detector -> reconstruction -> virtual upstream source"
)


print()
print(
    "STREAMING DPI TWIN OK"
)
