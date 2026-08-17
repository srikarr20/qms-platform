import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# QMS PLATFORM — BLIND DEPTH ROBUSTNESS
#
# Tests:
#   - random OFF-GRID true depths
#   - varying source width
#   - ellipticity
#   - varying source phase
#   - quadrature measurement noise
#
# Twin does NOT know:
#   - true depth
#   - source width
#   - source ellipticity
#   - source phase
#
# Twin only:
#   detector -> complex field -> virtual z sweep
#            -> minimum RMS width -> inferred z
#
# This is still matched wavelength / propagation physics.
# ============================================================

rng = np.random.default_rng(7001)

N = 96
wavelength = 810e-9
dx = 8e-6

coord = (np.arange(N) - N // 2) * dx
X, Y = np.meshgrid(coord, coord, indexing="xy")

fx = np.fft.fftfreq(N, d=dx)
fy = np.fft.fftfreq(N, d=dx)
FX, FY = np.meshgrid(fx, fy, indexing="xy")


def transfer(z):
    return np.exp(
        -1j * np.pi * wavelength * z * (FX**2 + FY**2)
    )


def propagate(field, z):
    return np.fft.ifft2(
        np.fft.fft2(field) * transfer(z)
    )


# ============================================================
# QUADRATURE REFERENCE
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
# RANDOM UNKNOWN SOURCE
# ============================================================

def make_source():

    x0 = rng.uniform(-45e-6, 45e-6)
    y0 = rng.uniform(-40e-6, 40e-6)

    sigma_x = rng.uniform(
        22e-6,
        42e-6,
    )

    sigma_y = rng.uniform(
        22e-6,
        42e-6,
    )

    theta = rng.uniform(
        -np.pi,
        np.pi,
    )

    phase0 = rng.uniform(
        -np.pi,
        np.pi,
    )

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
            xr**2 / sigma_x**2
            +
            yr**2 / sigma_y**2
        )
    )

    # mild unknown phase tilt
    qx = rng.uniform(
        -8e3,
        8e3,
    )

    qy = rng.uniform(
        -8e3,
        8e3,
    )

    phase = (
        phase0
        + qx*X
        + qy*Y
    )

    field = (
        A
        * np.exp(1j*phase)
    )

    return field, x0, y0


# ============================================================
# NOISY QUADRATURE MEASUREMENT
# ============================================================

def quadrature_measurement(
    source,
    depth,
    noise_fraction,
):

    sensor = propagate(
        source,
        depth,
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
# SOURCE-LIKENESS
# ============================================================

def spatial_metrics(field):

    I = np.abs(field)**2

    total = (
        I.sum()
        + 1e-15
    )

    cx = float(
        np.sum(I*X)
        / total
    )

    cy = float(
        np.sum(I*Y)
        / total
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
# BLIND DEPTH SEARCH
#
# Use 0.5 mm virtual-depth grid.
# True depth is continuous/off-grid.
# ============================================================

SEARCH_DEPTHS_MM = np.arange(
    20.0,
    110.01,
    0.5,
)

SEARCH_DEPTHS = (
    SEARCH_DEPTHS_MM
    * 1e-3
)


def infer_source(sensor_field):

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

    # --------------------------------------------------------
    # Sub-grid parabolic interpolation around minimum
    # --------------------------------------------------------

    inferred_mm = (
        SEARCH_DEPTHS_MM[best]
    )

    if (
        best > 0
        and
        best < len(widths)-1
    ):

        y1 = widths[best-1]
        y2 = widths[best]
        y3 = widths[best+1]

        denominator = (
            y1
            - 2*y2
            + y3
        )

        if abs(denominator) > 1e-20:

            delta = (
                0.5
                * (y1-y3)
                / denominator
            )

            inferred_mm += (
                delta * 0.5
            )

    return (
        inferred_mm,
        cx_all[best],
        cy_all[best],
        widths,
    )


# ============================================================
# MONTE CARLO
# ============================================================

NOISE_LEVELS = np.array([
    0.000,
    0.001,
    0.0025,
    0.005,
    0.010,
    0.020,
    0.040,
])

TRIALS_PER_NOISE = 50

all_results = {}

example = None


print()
print("=" * 78)
print("QMS PLATFORM — BLIND DEPTH ROBUSTNESS")
print("=" * 78)

print()
print(
    "Trials per noise level:",
    TRIALS_PER_NOISE
)

print(
    "Virtual z grid:",
    SEARCH_DEPTHS_MM[1]
    - SEARCH_DEPTHS_MM[0],
    "mm"
)

print()


for noise in NOISE_LEVELS:

    depth_errors = []
    xy_errors = []

    truths = []
    predictions = []

    for trial in range(
        TRIALS_PER_NOISE
    ):

        source, true_x, true_y = (
            make_source()
        )

        # deliberately continuous/off-grid
        true_depth_mm = rng.uniform(
            30.0,
            105.0,
        )

        true_depth = (
            true_depth_mm
            * 1e-3
        )

        q = quadrature_measurement(
            source,
            true_depth,
            noise,
        )

        sensor_rec = recover_complex(
            q
        )

        (
            inferred_depth_mm,
            inferred_x,
            inferred_y,
            widths,
        ) = infer_source(
            sensor_rec
        )

        z_error = (
            inferred_depth_mm
            - true_depth_mm
        )

        xy_error = (
            1e6
            * np.sqrt(
                (inferred_x-true_x)**2
                +
                (inferred_y-true_y)**2
            )
        )

        depth_errors.append(
            z_error
        )

        xy_errors.append(
            xy_error
        )

        truths.append(
            true_depth_mm
        )

        predictions.append(
            inferred_depth_mm
        )

        if (
            example is None
            and
            noise == 0.01
        ):

            example = {
                "truth":
                    true_depth_mm,

                "prediction":
                    inferred_depth_mm,

                "widths":
                    widths.copy(),
            }

    depth_errors = np.asarray(
        depth_errors
    )

    xy_errors = np.asarray(
        xy_errors
    )

    truths = np.asarray(
        truths
    )

    predictions = np.asarray(
        predictions
    )

    mae = float(
        np.mean(
            np.abs(
                depth_errors
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                depth_errors**2
            )
        )
    )

    p95 = float(
        np.percentile(
            np.abs(depth_errors),
            95,
        )
    )

    within_1 = float(
        np.mean(
            np.abs(depth_errors)
            <= 1.0
        )
    )

    within_2 = float(
        np.mean(
            np.abs(depth_errors)
            <= 2.0
        )
    )

    corr = float(
        np.corrcoef(
            truths,
            predictions,
        )[0,1]
    )

    xy_mae = float(
        np.mean(
            xy_errors
        )
    )

    all_results[
        float(noise)
    ] = {
        "truth":
            truths,

        "prediction":
            predictions,

        "depth_errors":
            depth_errors,

        "xy_errors":
            xy_errors,

        "mae":
            mae,

        "rmse":
            rmse,

        "p95":
            p95,

        "within_1":
            within_1,

        "within_2":
            within_2,

        "corr":
            corr,

        "xy_mae":
            xy_mae,
    }

    print(
        f"noise={noise:7.4f}"
        f"  depth_MAE={mae:7.4f} mm"
        f"  RMSE={rmse:7.4f} mm"
        f"  p95={p95:7.4f} mm"
        f"  <=1mm={100*within_1:6.2f}%"
        f"  corr={corr:8.5f}"
        f"  XY={xy_mae:7.4f} um"
    )


# ============================================================
# SUMMARY ARRAYS
# ============================================================

maes = np.array([
    all_results[float(n)]["mae"]
    for n in NOISE_LEVELS
])

rmses = np.array([
    all_results[float(n)]["rmse"]
    for n in NOISE_LEVELS
])

p95s = np.array([
    all_results[float(n)]["p95"]
    for n in NOISE_LEVELS
])

within1 = np.array([
    all_results[float(n)]["within_1"]
    for n in NOISE_LEVELS
])

xy_maes = np.array([
    all_results[float(n)]["xy_mae"]
    for n in NOISE_LEVELS
])


# ============================================================
# FIGURE 1
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    100*NOISE_LEVELS,
    maes,
    marker="o",
    label="Depth MAE",
)

plt.plot(
    100*NOISE_LEVELS,
    rmses,
    marker="o",
    label="Depth RMSE",
)

plt.plot(
    100*NOISE_LEVELS,
    p95s,
    marker="o",
    label="95th percentile |error|",
)

plt.xlabel(
    "Quadrature noise (% of peak intensity)"
)

plt.ylabel(
    "Depth error (mm)"
)

plt.title(
    "Blind upstream depth robustness"
)

plt.legend()

plt.tight_layout()

FIG1 = (
    "dpi_blind_depth_robustness_error.png"
)

plt.savefig(
    FIG1,
    dpi=220,
)

plt.close()


# ============================================================
# FIGURE 2
# ============================================================

fig, axs = plt.subplots(
    1,
    2,
    figsize=(11, 4.5)
)

axs[0].plot(
    100*NOISE_LEVELS,
    100*within1,
    marker="o",
)

axs[0].set_xlabel(
    "Quadrature noise (%)"
)

axs[0].set_ylabel(
    "Trials within 1 mm (%)"
)

axs[0].set_ylim(
    -2,
    102,
)

axs[0].set_title(
    "Blind-depth success rate"
)


axs[1].plot(
    100*NOISE_LEVELS,
    xy_maes,
    marker="o",
)

axs[1].set_xlabel(
    "Quadrature noise (%)"
)

axs[1].set_ylabel(
    "Mean XY error (microns)"
)

axs[1].set_title(
    "Transverse localization"
)

plt.tight_layout()

FIG2 = (
    "dpi_blind_depth_robustness_success.png"
)

plt.savefig(
    FIG2,
    dpi=220,
)

plt.close()


# ============================================================
# FIGURE 3 — EXAMPLE SCORE CURVE
# ============================================================

if example is not None:

    plt.figure(
        figsize=(9, 5)
    )

    normalized = (
        example["widths"]
        /
        np.min(
            example["widths"]
        )
    )

    plt.plot(
        SEARCH_DEPTHS_MM,
        normalized,
    )

    plt.axvline(
        example["truth"],
        linestyle="--",
        label=(
            f"true = "
            f"{example['truth']:.2f} mm"
        ),
    )

    plt.axvline(
        example["prediction"],
        linestyle=":",
        label=(
            f"inferred = "
            f"{example['prediction']:.2f} mm"
        ),
    )

    plt.xlabel(
        "Virtual back-propagation depth (mm)"
    )

    plt.ylabel(
        "RMS width / minimum"
    )

    plt.title(
        "Example noisy blind-depth likelihood"
    )

    plt.legend()

    plt.tight_layout()

    FIG3 = (
        "dpi_blind_depth_robustness_example.png"
    )

    plt.savefig(
        FIG3,
        dpi=220,
    )

    plt.close()

else:

    FIG3 = None


# ============================================================
# SAVE
# ============================================================

NPZ = (
    "dpi_blind_depth_robustness_results.npz"
)

np.savez_compressed(
    NPZ,

    noise_levels=
        NOISE_LEVELS,

    depth_mae_mm=
        maes,

    depth_rmse_mm=
        rmses,

    depth_p95_mm=
        p95s,

    within_1mm=
        within1,

    xy_mae_um=
        xy_maes,

    search_depths_mm=
        SEARCH_DEPTHS_MM,

    trials_per_noise=
        TRIALS_PER_NOISE,
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 78)
print("ROBUSTNESS SUMMARY")
print("=" * 78)

print()

best_noise = NOISE_LEVELS[0]
worst_noise = NOISE_LEVELS[-1]

print(
    "Noiseless off-grid depth MAE:",
    f"{maes[0]:.6f}",
    "mm"
)

print(
    "At",
    f"{100*worst_noise:.1f}%",
    "noise:"
)

print(
    "  Depth MAE:",
    f"{maes[-1]:.6f}",
    "mm"
)

print(
    "  Depth RMSE:",
    f"{rmses[-1]:.6f}",
    "mm"
)

print(
    "  Within 1 mm:",
    f"{100*within1[-1]:.2f}",
    "%"
)

print(
    "  Mean XY error:",
    f"{xy_maes[-1]:.6f}",
    "microns"
)

print()
print("Saved:")
print(FIG1)
print(FIG2)

if FIG3 is not None:
    print(FIG3)

print(NPZ)

print()
print(
    "BLIND DEPTH ROBUSTNESS COMPLETE"
)
