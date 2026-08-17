import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# QMS PLATFORM — TWO-WAVELENGTH DEPTH RECOVERY
#
# Goal:
#   break the single-wavelength z-lambda degeneracy
#
# Physical system:
#   same source depth
#   measured at two known wavelengths
#
# Twin:
#   reconstruct both detector fields
#   back-propagate each across candidate depth
#   combine source-likeness scores
#
# Expected:
#   the combined score should have a unique minimum
#   near the true physical source depth
# ============================================================

rng = np.random.default_rng(7304)

N = 96
dx = 8e-6

LAMBDA_1 = 780e-9
LAMBDA_2 = 840e-9

TRUE_DEPTH_MM = 82.37
TRUE_DEPTH = TRUE_DEPTH_MM * 1e-3

coord = (np.arange(N) - N//2) * dx

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

source_x = 22e-6
source_y = -19e-6

sigma_x = 28e-6
sigma_y = 35e-6

theta = 0.5

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
    0.4
    + 2500*X
    - 1800*Y
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


def make_measurement(
    wavelength,
    noise_fraction=0.005,
):

    sensor = propagate(
        SOURCE,
        TRUE_DEPTH,
        wavelength,
    )

    ideal = [
        detector(sensor, R),
        detector(sensor, 1j*R),
        detector(sensor, -R),
        detector(sensor, -1j*R),
    ]

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

    return measured


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
# SOURCE COMPACTNESS
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
# TWO PHYSICAL MEASUREMENTS
# ============================================================

q1 = make_measurement(
    LAMBDA_1
)

q2 = make_measurement(
    LAMBDA_2
)

sensor_1 = recover_complex(
    q1
)

sensor_2 = recover_complex(
    q2
)


# ============================================================
# BLIND DEPTH SEARCH
# ============================================================

DEPTHS_MM = np.arange(
    70.0,
    95.01,
    0.05,
)

DEPTHS = (
    DEPTHS_MM
    * 1e-3
)

width_1 = np.zeros(
    len(DEPTHS)
)

width_2 = np.zeros(
    len(DEPTHS)
)


print()
print("=" * 82)
print("QMS PLATFORM — TWO-WAVELENGTH DEPTH RECOVERY")
print("=" * 82)

print()
print(
    "Physical depth:",
    TRUE_DEPTH_MM,
    "mm"
)

print(
    "Wavelength 1:",
    LAMBDA_1*1e9,
    "nm"
)

print(
    "Wavelength 2:",
    LAMBDA_2*1e9,
    "nm"
)

print()


for i, z in enumerate(
    DEPTHS
):

    f1 = propagate(
        sensor_1,
        -z,
        LAMBDA_1,
    )

    f2 = propagate(
        sensor_2,
        -z,
        LAMBDA_2,
    )

    width_1[i] = spatial_width(
        f1
    )

    width_2[i] = spatial_width(
        f2
    )


# ============================================================
# NORMALIZE EACH CHANNEL
# ============================================================

score_1 = (
    width_1
    / np.min(width_1)
    - 1.0
)

score_2 = (
    width_2
    / np.min(width_2)
    - 1.0
)


# ============================================================
# COMBINED OBSERVABILITY SCORE
#
# Equal weighting for now
# ============================================================

combined = (
    score_1
    + score_2
)


best_index = int(
    np.argmin(combined)
)

best_depth_mm = (
    DEPTHS_MM[best_index]
)


# ============================================================
# SUB-GRID PARABOLIC FIT
# ============================================================

refined_depth_mm = (
    best_depth_mm
)

if (
    best_index > 0
    and
    best_index < len(combined)-1
):

    y1 = combined[
        best_index-1
    ]

    y2 = combined[
        best_index
    ]

    y3 = combined[
        best_index+1
    ]

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

        refined_depth_mm += (
            delta * 0.05
        )


error_mm = (
    refined_depth_mm
    - TRUE_DEPTH_MM
)


# ============================================================
# FIGURE 1 — INDIVIDUAL + COMBINED DEPTH SCORES
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    DEPTHS_MM,
    score_1,
    label=(
        f"{LAMBDA_1*1e9:.0f} nm"
    ),
)

plt.plot(
    DEPTHS_MM,
    score_2,
    label=(
        f"{LAMBDA_2*1e9:.0f} nm"
    ),
)

plt.plot(
    DEPTHS_MM,
    combined,
    linewidth=2.5,
    label="Combined two-wavelength score",
)

plt.axvline(
    TRUE_DEPTH_MM,
    linestyle="--",
    label="True depth",
)

plt.axvline(
    refined_depth_mm,
    linestyle=":",
    label=(
        f"Inferred depth = "
        f"{refined_depth_mm:.3f} mm"
    ),
)

plt.xlabel(
    "Candidate source depth (mm)"
)

plt.ylabel(
    "Relative source-width score"
)

plt.title(
    "Two-wavelength observability resolves source depth"
)

plt.legend()

plt.tight_layout()

FIG1 = (
    "dpi_two_wavelength_depth_score.png"
)

plt.savefig(
    FIG1,
    dpi=220,
)

plt.close()


# ============================================================
# MONTE CARLO ROBUSTNESS
# ============================================================

N_TRIALS = 80

errors = []


for trial in range(
    N_TRIALS
):

    q1 = make_measurement(
        LAMBDA_1,
        noise_fraction=0.01,
    )

    q2 = make_measurement(
        LAMBDA_2,
        noise_fraction=0.01,
    )

    s1 = recover_complex(
        q1
    )

    s2 = recover_complex(
        q2
    )

    a = np.zeros(
        len(DEPTHS)
    )

    b = np.zeros(
        len(DEPTHS)
    )

    for i, z in enumerate(
        DEPTHS
    ):

        a[i] = spatial_width(
            propagate(
                s1,
                -z,
                LAMBDA_1,
            )
        )

        b[i] = spatial_width(
            propagate(
                s2,
                -z,
                LAMBDA_2,
            )
        )

    sa = (
        a / np.min(a) - 1
    )

    sb = (
        b / np.min(b) - 1
    )

    c = (
        sa + sb
    )

    k = int(
        np.argmin(c)
    )

    estimate = (
        DEPTHS_MM[k]
    )

    if (
        k > 0
        and
        k < len(c)-1
    ):

        y1 = c[k-1]
        y2 = c[k]
        y3 = c[k+1]

        denom = (
            y1 - 2*y2 + y3
        )

        if abs(denom) > 1e-20:

            delta = (
                0.5
                * (y1-y3)
                / denom
            )

            estimate += (
                delta * 0.05
            )

    errors.append(
        estimate
        - TRUE_DEPTH_MM
    )


errors = np.asarray(
    errors
)

mae = float(
    np.mean(
        np.abs(errors)
    )
)

rmse = float(
    np.sqrt(
        np.mean(
            errors**2
        )
    )
)

bias = float(
    np.mean(errors)
)

p95 = float(
    np.percentile(
        np.abs(errors),
        95,
    )
)


# ============================================================
# FIGURE 2 — MONTE CARLO ERROR
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    errors,
    bins=20,
)

plt.axvline(
    0,
    linestyle="--",
)

plt.xlabel(
    "Inferred depth - true depth (mm)"
)

plt.ylabel(
    "Trials"
)

plt.title(
    "Two-wavelength blind-depth error at 1% measurement noise"
)

plt.tight_layout()

FIG2 = (
    "dpi_two_wavelength_depth_error_hist.png"
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
    "dpi_two_wavelength_depth_results.npz"
)

np.savez_compressed(
    NPZ,

    true_depth_mm=
        TRUE_DEPTH_MM,

    wavelength_1_nm=
        LAMBDA_1*1e9,

    wavelength_2_nm=
        LAMBDA_2*1e9,

    depths_mm=
        DEPTHS_MM,

    score_1=
        score_1,

    score_2=
        score_2,

    combined_score=
        combined,

    inferred_depth_mm=
        refined_depth_mm,

    error_mm=
        error_mm,

    monte_carlo_errors_mm=
        errors,

    mae_mm=
        mae,

    rmse_mm=
        rmse,

    bias_mm=
        bias,

    p95_mm=
        p95,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 82)
print("TWO-WAVELENGTH DEPTH RESULTS")
print("=" * 82)

print()

print(
    "True depth:",
    f"{TRUE_DEPTH_MM:.6f}",
    "mm"
)

print(
    "Single-run inferred depth:",
    f"{refined_depth_mm:.6f}",
    "mm"
)

print(
    "Single-run error:",
    f"{error_mm:+.6f}",
    "mm"
)

print()

print(
    "Monte Carlo trials:",
    N_TRIALS
)

print(
    "Depth MAE:",
    f"{mae:.6f}",
    "mm"
)

print(
    "Depth RMSE:",
    f"{rmse:.6f}",
    "mm"
)

print(
    "Depth bias:",
    f"{bias:+.6f}",
    "mm"
)

print(
    "95th percentile |error|:",
    f"{p95:.6f}",
    "mm"
)

print()

print(
    "Interpretation:"
)

print(
    "Two independently calibrated wavelengths constrain "
    "the common source depth rather than allowing z-lambda tradeoff."
)

print()

print("Saved:")
print(FIG1)
print(FIG2)
print(NPZ)

print()
print(
    "TWO-WAVELENGTH DEPTH RECOVERY COMPLETE"
)
