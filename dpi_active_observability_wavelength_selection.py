import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# QMS PLATFORM — ACTIVE OBSERVABILITY
#
# Goal:
#   start with one calibrated wavelength
#   evaluate candidate second wavelengths
#   choose the one that best sharpens depth observability
#
# We quantify usefulness by the curvature of the combined
# source-likeness score near its minimum.
#
# Larger curvature = sharper minimum = better depth constraint.
# ============================================================

rng = np.random.default_rng(7405)

N = 96
dx = 8e-6

TRUE_DEPTH_MM = 82.37
TRUE_DEPTH = TRUE_DEPTH_MM * 1e-3

LAMBDA_BASE = 810e-9

CANDIDATE_WAVELENGTHS_NM = np.arange(
    760.0,
    861.0,
    5.0,
)

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

source_x = 23e-6
source_y = -16e-6

sigma_x = 27e-6
sigma_y = 34e-6

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

SOURCE = (
    A
    * np.exp(
        1j * (
            0.35
            + 2200*X
            - 1700*Y
        )
    )
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
    noise_fraction=0.01,
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
# DEPTH SEARCH GRID
# ============================================================

DEPTHS_MM = np.arange(
    74.0,
    91.01,
    0.05,
)

DEPTHS = (
    DEPTHS_MM
    * 1e-3
)


def depth_score(
    sensor_field,
    wavelength,
):
    widths = np.zeros(
        len(DEPTHS)
    )

    for i, z in enumerate(
        DEPTHS
    ):
        field = propagate(
            sensor_field,
            -z,
            wavelength,
        )

        widths[i] = spatial_width(
            field
        )

    return (
        widths
        / np.min(widths)
        - 1.0
    )


# ============================================================
# BASE MEASUREMENT
# ============================================================

base_measurement = make_measurement(
    LAMBDA_BASE,
    noise_fraction=0.01,
)

base_sensor = recover_complex(
    base_measurement
)

base_score = depth_score(
    base_sensor,
    LAMBDA_BASE,
)


# ============================================================
# CANDIDATE SECOND-MEASUREMENT SELECTION
#
# Metric:
#   local curvature of combined score around minimum
#
# More curvature = sharper depth identifiability
# ============================================================

candidate_curvature = []
candidate_width_1sigma = []
candidate_depth_estimate = []


def score_curvature(
    score,
    idx,
):
    if (
        idx <= 0
        or
        idx >= len(score)-1
    ):
        return 0.0

    dz = (
        DEPTHS_MM[1]
        - DEPTHS_MM[0]
    )

    return (
        score[idx-1]
        - 2*score[idx]
        + score[idx+1]
    ) / (
        dz**2
    )


def estimate_width(
    score,
    idx,
):
    """
    Simple local uncertainty-like width from curvature.
    This is not yet a calibrated posterior sigma.
    """

    curvature = score_curvature(
        score,
        idx,
    )

    if curvature <= 0:
        return np.inf

    return float(
        1.0
        / np.sqrt(curvature)
    )


print()
print("=" * 82)
print("QMS PLATFORM — ACTIVE OBSERVABILITY")
print("=" * 82)

print()
print(
    "Base wavelength:",
    LAMBDA_BASE*1e9,
    "nm"
)

print(
    "True source depth:",
    TRUE_DEPTH_MM,
    "mm"
)

print()


for wavelength_nm in CANDIDATE_WAVELENGTHS_NM:

    wavelength = (
        wavelength_nm
        * 1e-9
    )

    candidate_measurement = (
        make_measurement(
            wavelength,
            noise_fraction=0.01,
        )
    )

    candidate_sensor = (
        recover_complex(
            candidate_measurement
        )
    )

    second_score = depth_score(
        candidate_sensor,
        wavelength,
    )

    combined = (
        base_score
        + second_score
    )

    best = int(
        np.argmin(combined)
    )

    curvature = score_curvature(
        combined,
        best,
    )

    uncertainty_width = estimate_width(
        combined,
        best,
    )

    depth_estimate = (
        DEPTHS_MM[best]
    )

    candidate_curvature.append(
        curvature
    )

    candidate_width_1sigma.append(
        uncertainty_width
    )

    candidate_depth_estimate.append(
        depth_estimate
    )

    print(
        f"candidate={wavelength_nm:6.1f} nm"
        f"  depth={depth_estimate:7.3f} mm"
        f"  curvature={curvature:10.5f}"
        f"  width_metric={uncertainty_width:8.5f}"
    )


candidate_curvature = np.asarray(
    candidate_curvature
)

candidate_width_1sigma = np.asarray(
    candidate_width_1sigma
)

candidate_depth_estimate = np.asarray(
    candidate_depth_estimate
)


# ============================================================
# BEST NEXT WAVELENGTH
# ============================================================

best_index = int(
    np.argmax(
        candidate_curvature
    )
)

best_wavelength_nm = (
    CANDIDATE_WAVELENGTHS_NM[
        best_index
    ]
)

best_depth_mm = (
    candidate_depth_estimate[
        best_index
    ]
)

best_curvature = (
    candidate_curvature[
        best_index
    ]
)


# ============================================================
# FIGURE 1 — INFORMATION VALUE OF NEXT MEASUREMENT
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    CANDIDATE_WAVELENGTHS_NM,
    candidate_curvature,
    marker="o",
)

plt.axvline(
    LAMBDA_BASE*1e9,
    linestyle="--",
    label="Existing wavelength",
)

plt.axvline(
    best_wavelength_nm,
    linestyle=":",
    label=(
        f"Selected next wavelength "
        f"{best_wavelength_nm:.0f} nm"
    ),
)

plt.xlabel(
    "Candidate next wavelength (nm)"
)

plt.ylabel(
    "Combined depth-score curvature"
)

plt.title(
    "Active observability: information value of next measurement"
)

plt.legend()

plt.tight_layout()

FIG1 = (
    "dpi_active_observability_wavelength_value.png"
)

plt.savefig(
    FIG1,
    dpi=220,
)

plt.close()


# ============================================================
# FIGURE 2 — EXPECTED DEPTH PRECISION METRIC
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    CANDIDATE_WAVELENGTHS_NM,
    candidate_width_1sigma,
    marker="o",
)

plt.axvline(
    best_wavelength_nm,
    linestyle="--",
)

plt.xlabel(
    "Candidate next wavelength (nm)"
)

plt.ylabel(
    "Local uncertainty-width metric"
)

plt.title(
    "Predicted depth uncertainty after next measurement"
)

plt.tight_layout()

FIG2 = (
    "dpi_active_observability_depth_uncertainty.png"
)

plt.savefig(
    FIG2,
    dpi=220,
)

plt.close()


# ============================================================
# VALIDATE CHOSEN MEASUREMENT
# ============================================================

chosen_lambda = (
    best_wavelength_nm
    * 1e-9
)

N_TRIALS = 100

errors_base = []
errors_active = []


def infer_from_score(score):
    k = int(
        np.argmin(score)
    )

    estimate = (
        DEPTHS_MM[k]
    )

    if (
        k > 0
        and
        k < len(score)-1
    ):
        y1 = score[k-1]
        y2 = score[k]
        y3 = score[k+1]

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

            estimate += (
                delta
                * (
                    DEPTHS_MM[1]
                    - DEPTHS_MM[0]
                )
            )

    return estimate


for trial in range(
    N_TRIALS
):

    q_base = make_measurement(
        LAMBDA_BASE,
        noise_fraction=0.01,
    )

    s_base = recover_complex(
        q_base
    )

    score_base = depth_score(
        s_base,
        LAMBDA_BASE,
    )

    estimate_base = infer_from_score(
        score_base
    )

    q_second = make_measurement(
        chosen_lambda,
        noise_fraction=0.01,
    )

    s_second = recover_complex(
        q_second
    )

    score_second = depth_score(
        s_second,
        chosen_lambda,
    )

    estimate_active = infer_from_score(
        score_base
        + score_second
    )

    errors_base.append(
        estimate_base
        - TRUE_DEPTH_MM
    )

    errors_active.append(
        estimate_active
        - TRUE_DEPTH_MM
    )


errors_base = np.asarray(
    errors_base
)

errors_active = np.asarray(
    errors_active
)


base_mae = float(
    np.mean(
        np.abs(
            errors_base
        )
    )
)

active_mae = float(
    np.mean(
        np.abs(
            errors_active
        )
    )
)

base_rmse = float(
    np.sqrt(
        np.mean(
            errors_base**2
        )
    )
)

active_rmse = float(
    np.sqrt(
        np.mean(
            errors_active**2
        )
    )
)


# ============================================================
# FIGURE 3 — BEFORE / AFTER
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.hist(
    errors_base,
    bins=20,
    alpha=0.6,
    label="Single wavelength",
)

plt.hist(
    errors_active,
    bins=20,
    alpha=0.6,
    label="Active second measurement",
)

plt.axvline(
    0,
    linestyle="--",
)

plt.xlabel(
    "Depth error (mm)"
)

plt.ylabel(
    "Trials"
)

plt.title(
    "Does active measurement selection improve depth inference?"
)

plt.legend()

plt.tight_layout()

FIG3 = (
    "dpi_active_observability_before_after.png"
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
    "dpi_active_observability_results.npz"
)

np.savez_compressed(
    NPZ,

    base_wavelength_nm=
        LAMBDA_BASE*1e9,

    candidate_wavelengths_nm=
        CANDIDATE_WAVELENGTHS_NM,

    candidate_curvature=
        candidate_curvature,

    candidate_width_metric=
        candidate_width_1sigma,

    candidate_depth_estimate_mm=
        candidate_depth_estimate,

    selected_wavelength_nm=
        best_wavelength_nm,

    selected_depth_mm=
        best_depth_mm,

    selected_curvature=
        best_curvature,

    errors_base_mm=
        errors_base,

    errors_active_mm=
        errors_active,

    base_mae_mm=
        base_mae,

    active_mae_mm=
        active_mae,

    base_rmse_mm=
        base_rmse,

    active_rmse_mm=
        active_rmse,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 82)
print("ACTIVE OBSERVABILITY RESULTS")
print("=" * 82)

print()

print(
    "Selected next wavelength:",
    best_wavelength_nm,
    "nm"
)

print(
    "Depth estimate at selection stage:",
    f"{best_depth_mm:.4f}",
    "mm"
)

print(
    "Selection curvature:",
    f"{best_curvature:.8f}"
)

print()

print(
    "Monte Carlo validation:"
)

print(
    "Single-wavelength MAE:",
    f"{base_mae:.6f}",
    "mm"
)

print(
    "Active two-measurement MAE:",
    f"{active_mae:.6f}",
    "mm"
)

print(
    "Single-wavelength RMSE:",
    f"{base_rmse:.6f}",
    "mm"
)

print(
    "Active two-measurement RMSE:",
    f"{active_rmse:.6f}",
    "mm"
)

if base_mae > 0:

    improvement = (
        100
        * (
            base_mae
            - active_mae
        )
        / base_mae
    )

    print(
        "MAE improvement:",
        f"{improvement:.2f}",
        "%"
    )

print()
print(
    "Meaning:"
)

print(
    "The twin evaluated possible future measurements "
    "and selected the one predicted to sharpen depth observability."
)

print()
print("Saved:")
print(FIG1)
print(FIG2)
print(FIG3)
print(NPZ)

print()
print(
    "ACTIVE OBSERVABILITY COMPLETE"
)
