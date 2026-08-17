import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# QMS PLATFORM — DEPTH / WAVELENGTH DEGENERACY
#
# Physical world:
#   true source depth z_true
#   true wavelength lambda_true
#
# Twin:
#   assumes lambda_model
#   searches z blindly by minimum spatial width
#
# Question:
#   can a wrong wavelength be compensated by a wrong depth?
# ============================================================

rng = np.random.default_rng(7102)

N = 96

LAMBDA_TRUE = 810e-9
dx = 8e-6

coord = (np.arange(N) - N//2) * dx
X, Y = np.meshgrid(coord, coord, indexing="xy")

fx = np.fft.fftfreq(N, d=dx)
fy = np.fft.fftfreq(N, d=dx)
FX, FY = np.meshgrid(fx, fy, indexing="xy")


def transfer(z, wavelength):

    return np.exp(
        -1j
        * np.pi
        * wavelength
        * z
        * (FX**2 + FY**2)
    )


def propagate(field, z, wavelength):

    return np.fft.ifft2(
        np.fft.fft2(field)
        * transfer(z, wavelength)
    )


# ============================================================
# REFERENCE FIELD
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
# RANDOM SOURCE
# ============================================================

def make_source():

    x0 = rng.uniform(-40e-6, 40e-6)
    y0 = rng.uniform(-35e-6, 35e-6)

    sigma_x = rng.uniform(24e-6, 38e-6)
    sigma_y = rng.uniform(24e-6, 38e-6)

    theta = rng.uniform(-np.pi, np.pi)
    phase0 = rng.uniform(-np.pi, np.pi)

    ct = np.cos(theta)
    st = np.sin(theta)

    xr = (
        ct*(X-x0)
        + st*(Y-y0)
    )

    yr = (
        -st*(X-x0)
        + ct*(Y-y0)
    )

    A = np.exp(
        -0.5 * (
            xr**2/sigma_x**2
            +
            yr**2/sigma_y**2
        )
    )

    field = (
        A
        * np.exp(1j*phase0)
    )

    return field, x0, y0


# ============================================================
# PHYSICAL QUADRATURE MEASUREMENT
# ============================================================

def make_measurement(
    source,
    true_depth,
    noise_fraction=0.005,
):

    sensor = propagate(
        source,
        true_depth,
        LAMBDA_TRUE,
    )

    ideal = [
        detector(sensor, R),
        detector(sensor, 1j*R),
        detector(sensor, -R),
        detector(sensor, -1j*R),
    ]

    scale = max(np.max(v) for v in ideal)

    sigma_noise = (
        noise_fraction
        * scale
    )

    noisy = []

    for I in ideal:

        J = (
            I
            + rng.normal(
                0.0,
                sigma_noise,
                I.shape,
            )
        )

        noisy.append(
            np.maximum(J, 0)
        )

    return noisy


def recover_complex(q):

    I0, I90, I180, I270 = q

    return (
        (I0-I180)
        +
        1j*(I90-I270)
    ) / (
        4*np.conj(R)
        + 1e-15
    )


# ============================================================
# SOURCE-LIKENESS METRIC
# ============================================================

def spatial_metrics(field):

    I = np.abs(field)**2

    total = I.sum() + 1e-15

    cx = float(
        np.sum(I*X) / total
    )

    cy = float(
        np.sum(I*Y) / total
    )

    width = float(
        np.sqrt(
            np.sum(
                I * (
                    (X-cx)**2
                    +
                    (Y-cy)**2
                )
            )
            / total
        )
    )

    return cx, cy, width


# ============================================================
# BLIND DEPTH SEARCH WITH MODEL WAVELENGTH
# ============================================================

SEARCH_DEPTHS_MM = np.arange(
    20.0,
    120.01,
    0.5,
)

SEARCH_DEPTHS = (
    SEARCH_DEPTHS_MM
    * 1e-3
)


def infer_depth(
    sensor_field,
    wavelength_model,
):

    widths = np.zeros(
        len(SEARCH_DEPTHS)
    )

    cx_all = np.zeros(
        len(SEARCH_DEPTHS)
    )

    cy_all = np.zeros(
        len(SEARCH_DEPTHS)
    )

    for i, z in enumerate(
        SEARCH_DEPTHS
    ):

        field = propagate(
            sensor_field,
            -z,
            wavelength_model,
        )

        cx, cy, width = spatial_metrics(
            field
        )

        widths[i] = width
        cx_all[i] = cx
        cy_all[i] = cy

    best = int(
        np.argmin(widths)
    )

    inferred_mm = (
        SEARCH_DEPTHS_MM[best]
    )

    # sub-grid interpolation
    if (
        best > 0
        and
        best < len(widths)-1
    ):

        y1 = widths[best-1]
        y2 = widths[best]
        y3 = widths[best+1]

        denom = (
            y1
            - 2*y2
            + y3
        )

        if abs(denom) > 1e-20:

            delta = (
                0.5
                * (y1-y3)
                / denom
            )

            inferred_mm += (
                0.5 * delta
            )

    return (
        inferred_mm,
        cx_all[best],
        cy_all[best],
        widths,
    )


# ============================================================
# TEST GRID
# ============================================================

MODEL_WAVELENGTHS_NM = np.array([
    780,
    790,
    800,
    805,
    810,
    815,
    820,
    830,
    840,
])

TRUE_DEPTHS_MM = np.array([
    35.0,
    50.0,
    65.0,
    80.0,
    95.0,
    110.0,
])

TRIALS = 20

depth_bias = np.zeros(
    (
        len(MODEL_WAVELENGTHS_NM),
        len(TRUE_DEPTHS_MM),
    )
)

depth_mae = np.zeros_like(
    depth_bias
)

xy_mae = np.zeros_like(
    depth_bias
)

ratio_pred_true = np.zeros_like(
    depth_bias
)


print()
print("=" * 82)
print("QMS PLATFORM — DEPTH / WAVELENGTH DEGENERACY")
print("=" * 82)

print()
print(
    "Physical wavelength:",
    LAMBDA_TRUE*1e9,
    "nm"
)

print(
    "Trials per grid point:",
    TRIALS
)

print()


for wi, wavelength_nm in enumerate(
    MODEL_WAVELENGTHS_NM
):

    wavelength_model = (
        wavelength_nm
        * 1e-9
    )

    for zi, true_depth_mm in enumerate(
        TRUE_DEPTHS_MM
    ):

        errors = []
        xy_errors = []
        ratios = []

        for trial in range(TRIALS):

            source, true_x, true_y = (
                make_source()
            )

            q = make_measurement(
                source,
                true_depth_mm*1e-3,
                noise_fraction=0.005,
            )

            sensor_rec = recover_complex(q)

            (
                inferred_mm,
                px,
                py,
                widths,
            ) = infer_depth(
                sensor_rec,
                wavelength_model,
            )

            error = (
                inferred_mm
                - true_depth_mm
            )

            xy_error = (
                1e6
                * np.sqrt(
                    (px-true_x)**2
                    +
                    (py-true_y)**2
                )
            )

            errors.append(error)
            xy_errors.append(xy_error)

            ratios.append(
                inferred_mm
                / true_depth_mm
            )

        errors = np.asarray(errors)
        xy_errors = np.asarray(xy_errors)
        ratios = np.asarray(ratios)

        depth_bias[wi, zi] = (
            np.mean(errors)
        )

        depth_mae[wi, zi] = (
            np.mean(
                np.abs(errors)
            )
        )

        xy_mae[wi, zi] = (
            np.mean(xy_errors)
        )

        ratio_pred_true[wi, zi] = (
            np.mean(ratios)
        )

    print(
        f"lambda_model={wavelength_nm:6.1f} nm"
        f"  mean depth MAE="
        f"{depth_mae[wi].mean():7.3f} mm"
        f"  mean bias="
        f"{depth_bias[wi].mean():+7.3f} mm"
        f"  mean XY="
        f"{xy_mae[wi].mean():6.3f} um"
    )


# ============================================================
# EXPECTED SCALING
#
# Fresnel phase depends primarily on lambda*z.
#
# Therefore wrong wavelength may imply:
#
#     lambda_model * z_inferred
#       ~
#     lambda_true * z_true
#
# giving approximately:
#
#     z_inferred / z_true
#       ~
#     lambda_true / lambda_model
# ============================================================

expected_ratio = (
    LAMBDA_TRUE
    /
    (
        MODEL_WAVELENGTHS_NM
        * 1e-9
    )
)

measured_ratio = (
    np.mean(
        ratio_pred_true,
        axis=1,
    )
)


# ============================================================
# FIGURE 1 — BIAS HEATMAP
# ============================================================

plt.figure(
    figsize=(9, 6)
)

im = plt.imshow(
    depth_bias,
    aspect="auto",
    origin="lower",
    extent=[
        TRUE_DEPTHS_MM[0],
        TRUE_DEPTHS_MM[-1],
        MODEL_WAVELENGTHS_NM[0],
        MODEL_WAVELENGTHS_NM[-1],
    ],
)

plt.colorbar(
    im,
    label="Mean inferred-depth bias (mm)",
)

plt.xlabel(
    "True source depth (mm)"
)

plt.ylabel(
    "Twin model wavelength (nm)"
)

plt.title(
    "Depth bias caused by wavelength mismatch"
)

plt.tight_layout()

FIG1 = (
    "dpi_depth_wavelength_bias_heatmap.png"
)

plt.savefig(
    FIG1,
    dpi=220,
)

plt.close()


# ============================================================
# FIGURE 2 — DEGENERACY SCALING
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    MODEL_WAVELENGTHS_NM,
    measured_ratio,
    marker="o",
    label="Measured inferred/true depth",
)

plt.plot(
    MODEL_WAVELENGTHS_NM,
    expected_ratio,
    linestyle="--",
    label="Expected lambda_true/lambda_model",
)

plt.axvline(
    LAMBDA_TRUE*1e9,
    linestyle=":",
)

plt.xlabel(
    "Twin model wavelength (nm)"
)

plt.ylabel(
    "Inferred depth / true depth"
)

plt.title(
    "Depth-wavelength degeneracy"
)

plt.legend()

plt.tight_layout()

FIG2 = (
    "dpi_depth_wavelength_scaling.png"
)

plt.savefig(
    FIG2,
    dpi=220,
)

plt.close()


# ============================================================
# FIGURE 3 — MEAN DEPTH MAE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    MODEL_WAVELENGTHS_NM,
    depth_mae.mean(axis=1),
    marker="o",
)

plt.axvline(
    LAMBDA_TRUE*1e9,
    linestyle="--",
    label="True wavelength",
)

plt.xlabel(
    "Twin model wavelength (nm)"
)

plt.ylabel(
    "Mean blind-depth MAE (mm)"
)

plt.title(
    "Blind depth accuracy under wavelength mismatch"
)

plt.legend()

plt.tight_layout()

FIG3 = (
    "dpi_depth_wavelength_mae.png"
)

plt.savefig(
    FIG3,
    dpi=220,
)

plt.close()


# ============================================================
# SAVE
# ============================================================

NPZ = (
    "dpi_depth_wavelength_degeneracy_results.npz"
)

np.savez_compressed(
    NPZ,

    true_wavelength_nm=
        LAMBDA_TRUE*1e9,

    model_wavelengths_nm=
        MODEL_WAVELENGTHS_NM,

    true_depths_mm=
        TRUE_DEPTHS_MM,

    depth_bias_mm=
        depth_bias,

    depth_mae_mm=
        depth_mae,

    xy_mae_um=
        xy_mae,

    measured_depth_ratio=
        measured_ratio,

    expected_depth_ratio=
        expected_ratio,

    trials=
        TRIALS,
)


# ============================================================
# REPORT
# ============================================================

matched = int(
    np.argmin(
        np.abs(
            MODEL_WAVELENGTHS_NM
            - LAMBDA_TRUE*1e9
        )
    )
)


print()
print("=" * 82)
print("DEPTH / WAVELENGTH DEGENERACY SUMMARY")
print("=" * 82)

print()

print(
    "Correct-wavelength mean depth MAE:",
    f"{depth_mae[matched].mean():.6f}",
    "mm"
)

print()

for wi, wavelength_nm in enumerate(
    MODEL_WAVELENGTHS_NM
):

    print(
        f"{wavelength_nm:6.1f} nm"
        f"  measured ratio="
        f"{measured_ratio[wi]:.6f}"
        f"  expected ratio="
        f"{expected_ratio[wi]:.6f}"
        f"  mean bias="
        f"{depth_bias[wi].mean():+.4f} mm"
    )

print()
print("Saved:")
print(FIG1)
print(FIG2)
print(FIG3)
print(NPZ)

print()
print(
    "DEPTH / WAVELENGTH DEGENERACY COMPLETE"
)
