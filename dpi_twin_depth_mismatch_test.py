import numpy as np
import matplotlib.pyplot as plt

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
# QMS PLATFORM — DEPTH MISMATCH DETECTION
#
# Physical world:
#   source at true depth z_true
#
# Twin model:
#   always assumes z_model = 100 mm
#
# Test:
#   detector -> inverse twin -> upstream state
#   -> forward twin -> predicted detector
#
# Measure:
#   residual vs depth mismatch
# ============================================================

N = 96

wavelength = 810e-9
dx = 8e-6

MODEL_DEPTH = 0.100

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


def propagate(field, H):

    return np.fft.ifft2(
        np.fft.fft2(field)
        * H
    )


# ============================================================
# REFERENCE
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


def detector(E, ref):

    return np.abs(
        E + ref
    )**2


# ============================================================
# SOURCE
# ============================================================

sigma = 30e-6

source_x = 25e-6
source_y = -15e-6


def source_field():

    field = np.exp(
        -(
            (X-source_x)**2
            +
            (Y-source_y)**2
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


SOURCE_TRUE = source_field()


# ============================================================
# PHYSICAL MEASUREMENT
# ============================================================

def make_measurement(
    true_depth,
):

    sensor_true = propagate(
        SOURCE_TRUE,
        transfer(true_depth),
    )

    data = np.stack([
        detector(
            sensor_true,
            R,
        ),

        detector(
            sensor_true,
            1j*R,
        ),

        detector(
            sensor_true,
            -R,
        ),

        detector(
            sensor_true,
            -1j*R,
        ),
    ])

    return Measurement(
        data=data,

        modality="quadrature",

        detector_id="depth-mismatch-sim",

        metadata={
            "true_depth":
                true_depth,

            "model_depth":
                MODEL_DEPTH,
        },
    )


# ============================================================
# TWIN UPDATE
# ============================================================

DEPTHS = np.arange(
    0.0,
    MODEL_DEPTH + 0.0001,
    0.005,
)


def dpi_update(
    measurement,
    previous_state,
):

    I0, I90, I180, I270 = (
        measurement.data
    )

    sensor_field = (
        (
            I0 - I180
        )
        +
        1j * (
            I90 - I270
        )
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
    # INVERSE TWIN:
    # assumes source lies at MODEL_DEPTH
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

    reconstructed_source = (
        volume[-1]
    )

    volume_state = VirtualVolumeState(
        field=volume,

        depths=DEPTHS,

        wavelength=wavelength,

        pixel_spacing=dx,

        timestamp=measurement.timestamp,

        metadata={
            "assumed_source_depth":
                MODEL_DEPTH,
        },
    )

    # --------------------------------------------------------
    # SOURCE POSITION ESTIMATE
    # --------------------------------------------------------

    I_source = np.abs(
        reconstructed_source
    )**2

    total = (
        I_source.sum()
        + 1e-15
    )

    cx = float(
        np.sum(
            I_source * X
        )
        / total
    )

    cy = float(
        np.sum(
            I_source * Y
        )
        / total
    )

    source = SourceEstimate(
        position=(
            cx,
            cy,
            MODEL_DEPTH,
        ),
        confidence=1.0,
    )

    # --------------------------------------------------------
    # FORWARD TWIN:
    # reconstructed source propagated using SAME assumed model
    # --------------------------------------------------------

    predicted_sensor_field = propagate(
        reconstructed_source,
        transfer(MODEL_DEPTH),
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
    # RESIDUALS
    # --------------------------------------------------------

    actual = measurement.data

    measurement_residual = (
        np.linalg.norm(
            actual
            - predicted_measurement
        )
        /
        (
            np.linalg.norm(actual)
            + 1e-15
        )
    )

    field_residual = (
        np.linalg.norm(
            sensor_field
            - predicted_sensor_field
        )
        /
        (
            np.linalg.norm(sensor_field)
            + 1e-15
        )
    )

    # --------------------------------------------------------
    # UPSTREAM SOURCE ERROR
    # only available here because this is a validation test
    # --------------------------------------------------------

    aligned = reconstructed_source

    c = np.vdot(
        aligned,
        SOURCE_TRUE,
    )

    if abs(c) > 1e-15:

        aligned = (
            aligned
            * np.exp(
                1j*np.angle(c)
            )
        )

    source_field_error = (
        np.linalg.norm(
            aligned
            - SOURCE_TRUE
        )
        /
        (
            np.linalg.norm(
                SOURCE_TRUE
            )
            + 1e-15
        )
    )

    observable = ObservableEstimate(
        name="twin_consistency",

        value=(
            1.0
            - measurement_residual
        ),

        confidence=1.0,

        metadata={
            "measurement_residual":
                float(
                    measurement_residual
                ),

            "field_residual":
                float(
                    field_residual
                ),

            "source_field_error":
                float(
                    source_field_error
                ),
        },
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
            "measurement_residual":
                float(
                    measurement_residual
                ),

            "field_residual":
                float(
                    field_residual
                ),

            "source_field_error":
                float(
                    source_field_error
                ),
        },
    )


# ============================================================
# SWEEP TRUE PHYSICAL DEPTH
# ============================================================

runtime = TwinRuntime(
    update_fn=dpi_update
)

TRUE_DEPTHS_MM = np.array([
    80,
    85,
    90,
    95,
    97.5,
    100,
    102.5,
    105,
    110,
    115,
    120,
])

measurement_residuals = []

field_residuals = []

source_field_errors = []

position_errors_um = []


print()
print("=" * 78)
print("QMS PLATFORM — DEPTH MISMATCH DETECTION")
print("=" * 78)

print()
print(
    "Twin assumed depth:",
    MODEL_DEPTH*1e3,
    "mm"
)

print()


for depth_mm in TRUE_DEPTHS_MM:

    runtime.reset()

    true_depth = (
        depth_mm
        * 1e-3
    )

    measurement = make_measurement(
        true_depth
    )

    state = runtime.update(
        measurement
    )

    mr = state.metadata[
        "measurement_residual"
    ]

    fr = state.metadata[
        "field_residual"
    ]

    se = state.metadata[
        "source_field_error"
    ]

    rx, ry, _ = (
        state.source.position
    )

    pe = (
        1e6
        * np.sqrt(
            (rx-source_x)**2
            +
            (ry-source_y)**2
        )
    )

    measurement_residuals.append(
        mr
    )

    field_residuals.append(
        fr
    )

    source_field_errors.append(
        se
    )

    position_errors_um.append(
        pe
    )

    mismatch_mm = (
        depth_mm
        - MODEL_DEPTH*1e3
    )

    print(
        f"true_depth={depth_mm:6.1f} mm"
        f"  mismatch={mismatch_mm:+6.1f} mm"
        f"  detector_residual={mr:.10e}"
        f"  field_residual={fr:.10e}"
        f"  source_error={se:.6f}"
        f"  position_error={pe:.6f} um"
    )


measurement_residuals = np.asarray(
    measurement_residuals
)

field_residuals = np.asarray(
    field_residuals
)

source_field_errors = np.asarray(
    source_field_errors
)

position_errors_um = np.asarray(
    position_errors_um
)


mismatch_mm = (
    TRUE_DEPTHS_MM
    - MODEL_DEPTH*1e3
)


# ============================================================
# FIGURE 1 — CLOSED-LOOP RESIDUAL
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    mismatch_mm,
    measurement_residuals,
    marker="o",
    label="Detector measurement residual",
)

plt.plot(
    mismatch_mm,
    field_residuals,
    marker="o",
    label="Complex detector-field residual",
)

plt.xlabel(
    "True depth - twin depth (mm)"
)

plt.ylabel(
    "Normalized residual"
)

plt.title(
    "Closed-loop residual vs source-depth mismatch"
)

plt.legend()

plt.tight_layout()

FIG1 = (
    "dpi_twin_depth_mismatch_residual.png"
)

plt.savefig(
    FIG1,
    dpi=220
)

plt.close()


# ============================================================
# FIGURE 2 — UPSTREAM ERROR
# ============================================================

fig, axs = plt.subplots(
    1,
    2,
    figsize=(11, 4.5)
)

axs[0].plot(
    mismatch_mm,
    source_field_errors,
    marker="o",
)

axs[0].set_xlabel(
    "True depth - twin depth (mm)"
)

axs[0].set_ylabel(
    "Normalized source-field error"
)

axs[0].set_title(
    "Upstream field reconstruction"
)


axs[1].plot(
    mismatch_mm,
    position_errors_um,
    marker="o",
)

axs[1].set_xlabel(
    "True depth - twin depth (mm)"
)

axs[1].set_ylabel(
    "Source centroid error (microns)"
)

axs[1].set_title(
    "Source-position estimate"
)

plt.tight_layout()

FIG2 = (
    "dpi_twin_depth_mismatch_upstream.png"
)

plt.savefig(
    FIG2,
    dpi=220
)

plt.close()


# ============================================================
# SAVE
# ============================================================

NPZ = (
    "dpi_twin_depth_mismatch_results.npz"
)

np.savez_compressed(
    NPZ,

    true_depths_mm=
        TRUE_DEPTHS_MM,

    mismatch_mm=
        mismatch_mm,

    measurement_residual=
        measurement_residuals,

    field_residual=
        field_residuals,

    source_field_error=
        source_field_errors,

    position_error_um=
        position_errors_um,

    model_depth_mm=
        MODEL_DEPTH*1e3,
)


# ============================================================
# SUMMARY
# ============================================================

matched_idx = int(
    np.argmin(
        np.abs(
            mismatch_mm
        )
    )
)


print()
print("=" * 78)
print("DEPTH MISMATCH SUMMARY")
print("=" * 78)

print(
    "Matched-model detector residual:",
    f"{measurement_residuals[matched_idx]:.12e}"
)

print(
    "Matched-model source-field error:",
    f"{source_field_errors[matched_idx]:.8f}"
)

print()

print(
    "Largest tested mismatch:",
    f"{np.max(np.abs(mismatch_mm)):.1f}",
    "mm"
)

print(
    "Largest source-field error:",
    f"{source_field_errors.max():.8f}"
)

print()

print(
    "Saved:"
)

print(
    FIG1
)

print(
    FIG2
)

print(
    NPZ
)

print()

print(
    "DEPTH MISMATCH TEST COMPLETE"
)
