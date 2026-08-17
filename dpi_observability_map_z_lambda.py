import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# QMS PLATFORM — SOURCE-FACING OBSERVABILITY MAP
#
# Hidden physical parameters:
#   true source depth z_true
#   true wavelength lambda_true
#
# Twin scans:
#   candidate depth z
#   candidate wavelength lambda
#
# Score:
#   source spatial compactness after back-propagation
#
# Goal:
#   visualize parameter combinations that are indistinguishable
#   or weakly distinguishable from detector data.
# ============================================================

rng = np.random.default_rng(7203)

N = 96

LAMBDA_TRUE = 810e-9
TRUE_DEPTH_MM = 82.0
TRUE_DEPTH = TRUE_DEPTH_MM * 1e-3

dx = 8e-6

coord = (np.arange(N) - N//2) * dx
X, Y = np.meshgrid(coord, coord, indexing="xy")

fx = np.fft.fftfreq(N, d=dx)
fy = np.fft.fftfreq(N, d=dx)

FX, FY = np.meshgrid(
    fx,
    fy,
    indexing="xy",
)


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
# SOURCE
# ============================================================

source_x = 24e-6
source_y = -17e-6

sigma_x = 27e-6
sigma_y = 36e-6

theta = 0.42

ct = np.cos(theta)
st = np.sin(theta)

xr = (
    ct*(X-source_x)
    + st*(Y-source_y)
)

yr = (
    -st*(X-source_x)
    + ct*(Y-source_y)
)

A = np.exp(
    -0.5 * (
        xr**2/sigma_x**2
        +
        yr**2/sigma_y**2
    )
)

phase = (
    0.35
    + 3500*X
    - 2200*Y
)

SOURCE = (
    A
    * np.exp(1j*phase)
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
        1j*(kx*X + ky*Y)
    )
)


def detector(E, ref):
    return np.abs(E + ref)**2


# ============================================================
# PHYSICAL DETECTOR DATA
# ============================================================

sensor_true = propagate(
    SOURCE,
    TRUE_DEPTH,
    LAMBDA_TRUE,
)

ideal = [
    detector(sensor_true, R),
    detector(sensor_true, 1j*R),
    detector(sensor_true, -R),
    detector(sensor_true, -1j*R),
]

noise_fraction = 0.005

scale = max(
    np.max(v)
    for v in ideal
)

sigma_noise = (
    noise_fraction
    * scale
)

measured = []

for I in ideal:

    J = (
        I
        + rng.normal(
            0.0,
            sigma_noise,
            I.shape,
        )
    )

    measured.append(
        np.maximum(J, 0)
    )


I0, I90, I180, I270 = measured

sensor_reconstructed = (
    (I0-I180)
    +
    1j*(I90-I270)
) / (
    4*np.conj(R)
    + 1e-15
)


# ============================================================
# SOURCE COMPACTNESS SCORE
# ============================================================

def spatial_width(field):

    I = np.abs(field)**2

    total = (
        I.sum()
        + 1e-15
    )

    cx = (
        np.sum(I*X)
        / total
    )

    cy = (
        np.sum(I*Y)
        / total
    )

    width = np.sqrt(
        np.sum(
            I * (
                (X-cx)**2
                +
                (Y-cy)**2
            )
        )
        / total
    )

    return float(width)


# ============================================================
# PARAMETER GRID
# ============================================================

WAVELENGTHS_NM = np.arange(
    770.0,
    851.0,
    1.0,
)

WAVELENGTHS = (
    WAVELENGTHS_NM
    * 1e-9
)

DEPTHS_MM = np.arange(
    72.0,
    93.01,
    0.25,
)

DEPTHS = (
    DEPTHS_MM
    * 1e-3
)

score = np.empty(
    (
        len(WAVELENGTHS),
        len(DEPTHS),
    )
)


print()
print("=" * 82)
print("QMS PLATFORM — Z / WAVELENGTH OBSERVABILITY MAP")
print("=" * 82)

print()
print(
    "True wavelength:",
    LAMBDA_TRUE*1e9,
    "nm"
)

print(
    "True depth:",
    TRUE_DEPTH_MM,
    "mm"
)

print()
print(
    "Scanning",
    len(WAVELENGTHS),
    "wavelengths x",
    len(DEPTHS),
    "depths"
)

print()


# ============================================================
# 2D SOURCE-LIKENESS SCAN
# ============================================================

for wi, wavelength in enumerate(
    WAVELENGTHS
):

    if wi % 10 == 0:
        print(
            f"wavelength row "
            f"{wi+1}/{len(WAVELENGTHS)}"
        )

    for zi, depth in enumerate(
        DEPTHS
    ):

        candidate_source = propagate(
            sensor_reconstructed,
            -depth,
            wavelength,
        )

        score[wi, zi] = spatial_width(
            candidate_source
        )


# ============================================================
# NORMALIZE SCORE
#
# Smaller width = more source-like.
#
# Convert to relative excess over global minimum.
# ============================================================

global_min = float(
    np.min(score)
)

relative_score = (
    score
    / global_min
    - 1.0
)


best_flat = int(
    np.argmin(score)
)

best_wi, best_zi = np.unravel_index(
    best_flat,
    score.shape,
)

best_lambda_nm = (
    WAVELENGTHS_NM[best_wi]
)

best_depth_mm = (
    DEPTHS_MM[best_zi]
)


# ============================================================
# RIDGE / DEGENERACY CURVE
#
# For every wavelength:
#   find the best source depth
# ============================================================

best_depth_by_lambda = np.zeros(
    len(WAVELENGTHS)
)

best_score_by_lambda = np.zeros(
    len(WAVELENGTHS)
)

for wi in range(
    len(WAVELENGTHS)
):

    zi = int(
        np.argmin(
            score[wi]
        )
    )

    best_depth_by_lambda[wi] = (
        DEPTHS_MM[zi]
    )

    best_score_by_lambda[wi] = (
        score[wi, zi]
    )


expected_depth_by_lambda = (
    TRUE_DEPTH_MM
    * (
        LAMBDA_TRUE*1e9
        / WAVELENGTHS_NM
    )
)


# ============================================================
# FIGURE 1 — 2D OBSERVABILITY LANDSCAPE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

im = plt.imshow(
    relative_score,
    origin="lower",
    aspect="auto",
    extent=[
        DEPTHS_MM[0],
        DEPTHS_MM[-1],
        WAVELENGTHS_NM[0],
        WAVELENGTHS_NM[-1],
    ],
)

plt.colorbar(
    im,
    label=(
        "Relative source-width excess "
        "above global optimum"
    ),
)

plt.plot(
    expected_depth_by_lambda,
    WAVELENGTHS_NM,
    linestyle="--",
    linewidth=2,
    label="Expected lambda*z degeneracy",
)

plt.scatter(
    [TRUE_DEPTH_MM],
    [LAMBDA_TRUE*1e9],
    marker="x",
    s=100,
    label="True parameters",
)

plt.scatter(
    [best_depth_mm],
    [best_lambda_nm],
    marker="o",
    s=70,
    facecolors="none",
    edgecolors="white",
    label="Global optimum",
)

plt.xlabel(
    "Candidate source depth (mm)"
)

plt.ylabel(
    "Candidate wavelength (nm)"
)

plt.title(
    "Source-facing observability landscape"
)

plt.legend()

plt.tight_layout()

FIG1 = (
    "dpi_observability_map_z_lambda.png"
)

plt.savefig(
    FIG1,
    dpi=220,
)

plt.close()


# ============================================================
# FIGURE 2 — DEGENERACY RIDGE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    WAVELENGTHS_NM,
    best_depth_by_lambda,
    marker="o",
    markersize=3,
    label="Twin best depth",
)

plt.plot(
    WAVELENGTHS_NM,
    expected_depth_by_lambda,
    linestyle="--",
    label="Expected lambda_true/lambda_model scaling",
)

plt.axvline(
    LAMBDA_TRUE*1e9,
    linestyle=":",
)

plt.axhline(
    TRUE_DEPTH_MM,
    linestyle=":",
)

plt.xlabel(
    "Candidate wavelength (nm)"
)

plt.ylabel(
    "Best source-like depth (mm)"
)

plt.title(
    "Observability degeneracy ridge"
)

plt.legend()

plt.tight_layout()

FIG2 = (
    "dpi_observability_degeneracy_ridge.png"
)

plt.savefig(
    FIG2,
    dpi=220,
)

plt.close()


# ============================================================
# FIGURE 3 — SCORE ALONG RIDGE
# ============================================================

ridge_relative = (
    best_score_by_lambda
    /
    np.min(
        best_score_by_lambda
    )
    - 1.0
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    WAVELENGTHS_NM,
    ridge_relative,
    marker="o",
    markersize=3,
)

plt.axvline(
    LAMBDA_TRUE*1e9,
    linestyle="--",
    label="True wavelength",
)

plt.xlabel(
    "Candidate wavelength (nm)"
)

plt.ylabel(
    "Relative best source-width excess"
)

plt.title(
    "How strongly does source compactness distinguish wavelength?"
)

plt.legend()

plt.tight_layout()

FIG3 = (
    "dpi_observability_ridge_flatness.png"
)

plt.savefig(
    FIG3,
    dpi=220,
)

plt.close()


# ============================================================
# SIMPLE OBSERVABILITY DIAGNOSTICS
# ============================================================

ridge_range = float(
    np.max(ridge_relative)
    - np.min(ridge_relative)
)

ridge_std = float(
    np.std(
        ridge_relative
    )
)

ridge_depth_rmse = float(
    np.sqrt(
        np.mean(
            (
                best_depth_by_lambda
                - expected_depth_by_lambda
            )**2
        )
    )
)


# ============================================================
# SAVE
# ============================================================

NPZ = (
    "dpi_observability_map_z_lambda_results.npz"
)

np.savez_compressed(
    NPZ,

    true_wavelength_nm=
        LAMBDA_TRUE*1e9,

    true_depth_mm=
        TRUE_DEPTH_MM,

    wavelengths_nm=
        WAVELENGTHS_NM,

    depths_mm=
        DEPTHS_MM,

    source_width=
        score,

    relative_score=
        relative_score,

    best_depth_by_lambda_mm=
        best_depth_by_lambda,

    expected_depth_by_lambda_mm=
        expected_depth_by_lambda,

    ridge_relative=
        ridge_relative,

    best_lambda_nm=
        best_lambda_nm,

    best_depth_mm=
        best_depth_mm,

    ridge_range=
        ridge_range,

    ridge_std=
        ridge_std,

    ridge_depth_rmse_mm=
        ridge_depth_rmse,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 82)
print("OBSERVABILITY MAP RESULTS")
print("=" * 82)

print()

print(
    "True parameters:"
)

print(
    "  wavelength:",
    LAMBDA_TRUE*1e9,
    "nm"
)

print(
    "  depth:",
    TRUE_DEPTH_MM,
    "mm"
)

print()

print(
    "Global source-width optimum:"
)

print(
    "  wavelength:",
    best_lambda_nm,
    "nm"
)

print(
    "  depth:",
    best_depth_mm,
    "mm"
)

print()

print(
    "Degeneracy-ridge RMSE from expected scaling:",
    f"{ridge_depth_rmse:.6f}",
    "mm"
)

print(
    "Relative score range along ridge:",
    f"{ridge_range:.8f}"
)

print(
    "Relative score std along ridge:",
    f"{ridge_std:.8f}"
)

print()

print(
    "Interpretation:"
)

print(
    "A very flat ridge means depth and wavelength "
    "are jointly constrained but individually weakly observable."
)

print()

print(
    "Saved:"
)

print(FIG1)
print(FIG2)
print(FIG3)
print(NPZ)

print()
print(
    "Z / WAVELENGTH OBSERVABILITY MAP COMPLETE"
)
