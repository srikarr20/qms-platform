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
# QMS PLATFORM — BLIND UPSTREAM DEPTH TWIN
#
# Physical:
#       hidden source at unknown z
#                 ↓
#              detector
#
# Twin:
#       detector complex field
#                 ↓
#       virtual propagation volume
#                 ↓
#       source-likeness score vs z
#                 ↓
#          inferred x,y,z
#
# IMPORTANT:
# Minimum spatial width is a SOURCE PRIOR / heuristic.
# It is not a universal law of depth inference.
# ============================================================

N = 96

wavelength = 810e-9
dx = 8e-6

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


# ============================================================
# QUADRATURE REFERENCE
# ============================================================

reference_amp = 1.5

kx = 2*np.pi / 160e-6
ky = 2*np.pi / 210e-6

R = (
    reference_amp
    * np.exp(
        1j * (kx*X + ky*Y)
    )
)


def detector(E, ref):

    return np.abs(E + ref)**2


# ============================================================
# SOURCE
# ============================================================

sigma = 30e-6

source_x = 28e-6
source_y = -18e-6


def source_field():

    A = np.exp(
        -(
            (X-source_x)**2
            + (Y-source_y)**2
        )
        / (2*sigma**2)
    )

    return (
        A.astype(complex)
        * np.exp(1j*0.35)
    )


SOURCE = source_field()


# ============================================================
# PHYSICAL MEASUREMENT
# ============================================================

def make_measurement(
    true_depth,
    index,
):

    sensor_field = propagate(
        SOURCE,
        transfer(true_depth),
    )

    data = np.stack([
        detector(sensor_field, R),
        detector(sensor_field, 1j*R),
        detector(sensor_field, -R),
        detector(sensor_field, -1j*R),
    ])

    return Measurement(
        data=data,
        modality="quadrature",
        detector_id="blind-depth-4Q",
        metadata={
            "index": index,

            # validation truth only:
            "true_depth": true_depth,

            "truth_x": source_x,
            "truth_y": source_y,
        },
    )


# ============================================================
# VIRTUAL SEARCH REGION
# ============================================================

DEPTHS_MM = np.arange(
    20.0,
    111.0,
    1.0,
)

DEPTHS = DEPTHS_MM * 1e-3


# ============================================================
# SOURCE-LIKENESS METRIC
#
# Here:
#
#     source-like = spatially compact
#
# Therefore:
#
#     inferred depth = minimum RMS width
#
# ============================================================

def spatial_metrics(field):

    I = np.abs(field)**2

    total = (
        I.sum()
        + 1e-15
    )

    cx = float(
        np.sum(I * X)
        / total
    )

    cy = float(
        np.sum(I * Y)
        / total
    )

    r2 = (
        (X-cx)**2
        + (Y-cy)**2
    )

    width = float(
        np.sqrt(
            np.sum(
                I * r2
            )
            / total
        )
    )

    return cx, cy, width


# ============================================================
# DPI TWIN UPDATE
# ============================================================

def dpi_update(
    measurement,
    previous_state,
):

    I0, I90, I180, I270 = (
        measurement.data
    )

    # --------------------------------------------------------
    # Downstream complex-field recovery
    # --------------------------------------------------------

    sensor_field = (
        (I0-I180)
        + 1j*(I90-I270)
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
    # Search entire upstream virtual volume
    # --------------------------------------------------------

    volume = np.empty(
        (
            len(DEPTHS),
            N,
            N,
        ),
        dtype=np.complex64,
    )

    widths = np.zeros(
        len(DEPTHS)
    )

    centroids = np.zeros(
        (
            len(DEPTHS),
            2,
        )
    )

    for i, z in enumerate(DEPTHS):

        field = propagate(
            sensor_field,
            transfer(-z),
        )

        volume[i] = field

        cx, cy, width = spatial_metrics(
            field
        )

        centroids[i] = [
            cx,
            cy,
        ]

        widths[i] = width

    # --------------------------------------------------------
    # BLIND DEPTH INFERENCE
    # --------------------------------------------------------

    best_index = int(
        np.argmin(widths)
    )

    inferred_depth = float(
        DEPTHS[best_index]
    )

    inferred_x = float(
        centroids[
            best_index,
            0
        ]
    )

    inferred_y = float(
        centroids[
            best_index,
            1
        ]
    )

    # --------------------------------------------------------
    # Confidence from score contrast
    #
    # Not a calibrated probability.
    # Just an internal quality indicator.
    # --------------------------------------------------------

    sorted_widths = np.sort(
        widths
    )

    if len(sorted_widths) > 1:

        confidence = float(
            max(
                0.0,
                (
                    sorted_widths[1]
                    - sorted_widths[0]
                )
                /
                (
                    sorted_widths[0]
                    + 1e-15
                )
            )
        )

    else:
        confidence = 0.0

    volume_state = VirtualVolumeState(
        field=volume,
        depths=DEPTHS,
        wavelength=wavelength,
        pixel_spacing=dx,
        timestamp=measurement.timestamp,
        metadata={
            "search":
                "blind upstream depth",

            "source_metric":
                "minimum RMS spatial width",

            "widths_m":
                widths,
        },
    )

    source = SourceEstimate(
        position=(
            inferred_x,
            inferred_y,
            inferred_depth,
        ),
        confidence=confidence,
        metadata={
            "depth_method":
                "minimum RMS width",

            "best_index":
                best_index,

            "best_width_m":
                float(
                    widths[
                        best_index
                    ]
                ),
        },
    )

    depth_observable = ObservableEstimate(
        name="source_depth",
        value=inferred_depth,
        confidence=confidence,
        best_depth=inferred_depth,
        metadata={
            "source_likeness":
                "spatial compactness"
        },
    )

    return TwinState(
        measurement=measurement,
        detector_field=detector_state,
        virtual_volume=volume_state,
        source=source,
        observables=[
            depth_observable
        ],
        metadata={
            "depth_score_mm":
                DEPTHS_MM.copy(),

            "depth_width_m":
                widths.copy(),

            "inferred_depth":
                inferred_depth,
        },
    )


# ============================================================
# RUNTIME
# ============================================================

runtime = TwinRuntime(
    update_fn=dpi_update
)


# ============================================================
# BLIND TEST SET
# ============================================================

TRUE_DEPTHS_MM = np.array([
    32.0,
    38.0,
    45.0,
    53.0,
    61.0,
    68.0,
    76.0,
    84.0,
    91.0,
    99.0,
    105.0,
])

predicted_depths_mm = []

xy_errors_um = []

depth_errors_mm = []

example_curves = []


print()
print("=" * 78)
print("QMS PLATFORM — BLIND UPSTREAM DEPTH TWIN")
print("=" * 78)
print()
print(
    "Virtual search:",
    DEPTHS_MM[0],
    "to",
    DEPTHS_MM[-1],
    "mm"
)
print()


for i, true_depth_mm in enumerate(
    TRUE_DEPTHS_MM
):

    measurement = make_measurement(
        true_depth_mm * 1e-3,
        i,
    )

    state = runtime.update(
        measurement
    )

    px, py, pz = (
        state.source.position
    )

    predicted_mm = (
        pz * 1e3
    )

    depth_error = (
        predicted_mm
        - true_depth_mm
    )

    xy_error = (
        1e6
        * np.sqrt(
            (px-source_x)**2
            + (py-source_y)**2
        )
    )

    predicted_depths_mm.append(
        predicted_mm
    )

    depth_errors_mm.append(
        depth_error
    )

    xy_errors_um.append(
        xy_error
    )

    if i in [
        0,
        len(TRUE_DEPTHS_MM)//2,
        len(TRUE_DEPTHS_MM)-1,
    ]:

        example_curves.append(
            (
                true_depth_mm,
                state.metadata[
                    "depth_width_m"
                ].copy(),
            )
        )

    print(
        f"trial {i:02d}"
        f"  true_z={true_depth_mm:6.1f} mm"
        f"  inferred_z={predicted_mm:6.1f} mm"
        f"  z_error={depth_error:+6.2f} mm"
        f"  xy_error={xy_error:.6f} um"
    )


predicted_depths_mm = np.asarray(
    predicted_depths_mm
)

depth_errors_mm = np.asarray(
    depth_errors_mm
)

xy_errors_um = np.asarray(
    xy_errors_um
)


# ============================================================
# METRICS
# ============================================================

depth_mae = float(
    np.mean(
        np.abs(
            depth_errors_mm
        )
    )
)

depth_rmse = float(
    np.sqrt(
        np.mean(
            depth_errors_mm**2
        )
    )
)

depth_bias = float(
    np.mean(
        depth_errors_mm
    )
)

corr = float(
    np.corrcoef(
        TRUE_DEPTHS_MM,
        predicted_depths_mm,
    )[0,1]
)


# ============================================================
# FIGURE 1 — TRUE VS INFERRED
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    TRUE_DEPTHS_MM,
    predicted_depths_mm,
    s=55,
)

lo = min(
    TRUE_DEPTHS_MM.min(),
    predicted_depths_mm.min(),
)

hi = max(
    TRUE_DEPTHS_MM.max(),
    predicted_depths_mm.max(),
)

plt.plot(
    [lo, hi],
    [lo, hi],
    linestyle="--",
)

plt.xlabel(
    "True source depth (mm)"
)

plt.ylabel(
    "Twin inferred depth (mm)"
)

plt.title(
    "Blind downstream-to-upstream depth inference"
)

plt.tight_layout()

FIG1 = (
    "dpi_blind_depth_true_vs_predicted.png"
)

plt.savefig(
    FIG1,
    dpi=220,
)

plt.close()


# ============================================================
# FIGURE 2 — SOURCE-LIKENESS CURVES
# ============================================================

plt.figure(
    figsize=(9, 5)
)

for true_depth_mm, widths in example_curves:

    normalized = (
        widths
        / widths.min()
    )

    plt.plot(
        DEPTHS_MM,
        normalized,
        marker="o",
        markersize=3,
        label=(
            f"true z = "
            f"{true_depth_mm:.0f} mm"
        ),
    )

plt.xlabel(
    "Virtual back-propagation depth (mm)"
)

plt.ylabel(
    "RMS width / minimum RMS width"
)

plt.title(
    "Blind source-likeness search through virtual depth"
)

plt.legend()

plt.tight_layout()

FIG2 = (
    "dpi_blind_depth_score_curves.png"
)

plt.savefig(
    FIG2,
    dpi=220,
)

plt.close()


# ============================================================
# SAVE
# ============================================================

NPZ = (
    "dpi_blind_depth_twin_results.npz"
)

np.savez_compressed(
    NPZ,

    true_depths_mm=
        TRUE_DEPTHS_MM,

    predicted_depths_mm=
        predicted_depths_mm,

    depth_errors_mm=
        depth_errors_mm,

    xy_errors_um=
        xy_errors_um,

    search_depths_mm=
        DEPTHS_MM,

    depth_mae_mm=
        depth_mae,

    depth_rmse_mm=
        depth_rmse,

    depth_bias_mm=
        depth_bias,

    correlation=
        corr,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 78)
print("BLIND TWIN DEPTH RESULTS")
print("=" * 78)

print(
    "Depth MAE:",
    f"{depth_mae:.6f}",
    "mm"
)

print(
    "Depth RMSE:",
    f"{depth_rmse:.6f}",
    "mm"
)

print(
    "Depth bias:",
    f"{depth_bias:.6f}",
    "mm"
)

print(
    "True/inferred correlation:",
    f"{corr:.8f}"
)

print(
    "Mean XY localization error:",
    f"{xy_errors_um.mean():.9f}",
    "microns"
)

print()
print(
    "Twin operation:"
)

print(
    "detector -> virtual volume -> "
    "source-likeness search -> x,y,z"
)

print()
print(
    "Saved:"
)

print(FIG1)
print(FIG2)
print(NPZ)

print()
print(
    "BLIND UPSTREAM TWIN OK"
)
