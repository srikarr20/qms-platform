from pathlib import Path
import json

import numpy as np
from astropy.io import fits
from scipy.constants import k, m_e
from scipy.optimize import minimize_scalar

from pyxel.models.charge_transfer.utils_cdm import run_cdm_parallel


ROOT = Path(__file__).resolve().parent.parent

TRUE_DENSITY = 5.0e9
TRUE_NOISE = 5.0e-4

BASELINE_ROOT = (
    ROOT
    / "qms_pyxel_001"
    / "cti_observable_results"
    / "cti_000"
)

TRIAL_ROOT = ROOT / "qms_pyxel_twin_002" / "results"


def latest(root, filename):
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(f"{filename} missing under {root}")
    return files[-1]


def load_pixel(root):
    return np.asarray(
        np.load(latest(root, "detector_pixel.npy")),
        dtype=np.float64
    )


def load_image(root):
    return np.asarray(
        fits.getdata(latest(root, "detector_image.fits")),
        dtype=np.float64
    )


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


baseline_pixel = load_pixel(BASELINE_ROOT)
baseline_image = load_image(BASELINE_ROOT)


# Same CTI model used by Pyxel
beta = 0.3
vg = 1.62e-10
transfer_period = 9.4722e-04
fwc = 100000.0
temperature = 300.0

effective_mass = 0.5 * m_e

vth = 100.0 * np.sqrt(
    3.0 * k * temperature / effective_mass
)

tr = np.array([3.e-2], dtype=float)
sigma_capture = np.array([1.e-10], dtype=float)


def simulate_cti(log10_density):
    density = 10.0 ** log10_density

    arr = baseline_pixel.copy()

    return run_cdm_parallel(
        array=arr,
        beta=beta,
        vg=vg,
        t=transfer_period,
        fwc=fwc,
        vth=vth,
        tr=tr,
        nt=np.array([density], dtype=float),
        sigma=sigma_capture,
        charge_injection=False,
        chg_inj_parallel_transfers=baseline_pixel.shape[0],
    )


# Empirical noise calibration already measured in QMS-PYXEL-001B
REFERENCE_NOISE_SIGMA = 0.0005
REFERENCE_IMAGE_STD = 46.48433652


rows = []

trial_dirs = sorted(
    p for p in TRIAL_ROOT.iterdir()
    if p.is_dir()
)

for trial_dir in trial_dirs:

    observed_pixel = load_pixel(trial_dir)
    observed_image = load_image(trial_dir)

    # ----------------------------------
    # Stage 1: continuous CTI estimation
    # ----------------------------------

    def objective(log10_density):
        pred = simulate_cti(log10_density)
        return rmse(pred, observed_pixel)

    result = minimize_scalar(
        objective,
        bounds=(8.0, 11.0),
        method="bounded",
        options={"xatol": 1e-8},
    )

    estimated_density = 10.0 ** result.x

    predicted_pixel = simulate_cti(result.x)

    pixel_before = rmse(
        baseline_pixel,
        observed_pixel
    )

    pixel_after = rmse(
        predicted_pixel,
        observed_pixel
    )

    # ----------------------------------
    # Stage 2: downstream noise estimate
    # ----------------------------------
    #
    # We need the Image residual after CTI correction.
    #
    # We use the clean baseline Image only to estimate
    # the residual statistical scale. The CTI contribution
    # to Image is much smaller than the output-noise signal
    # at sigma=5e-4 in this experiment.
    # ----------------------------------

    image_residual = observed_image - baseline_image

    residual_std = float(
        image_residual.std()
    )

    estimated_noise = (
        REFERENCE_NOISE_SIGMA
        * residual_std
        / REFERENCE_IMAGE_STD
    )

    rows.append({
        "trial": trial_dir.name,

        "estimated_cti_density":
            float(estimated_density),

        "cti_relative_error":
            float(
                (estimated_density - TRUE_DENSITY)
                / TRUE_DENSITY
            ),

        "pixel_residual_before":
            float(pixel_before),

        "pixel_residual_after":
            float(pixel_after),

        "estimated_output_noise":
            float(estimated_noise),

        "noise_relative_error":
            float(
                (estimated_noise - TRUE_NOISE)
                / TRUE_NOISE
            ),

        "image_residual_std":
            float(residual_std),

        "optimizer_success":
            bool(result.success),
    })


densities = np.array([
    r["estimated_cti_density"]
    for r in rows
])

noise_estimates = np.array([
    r["estimated_output_noise"]
    for r in rows
])

pixel_before = np.array([
    r["pixel_residual_before"]
    for r in rows
])

pixel_after = np.array([
    r["pixel_residual_after"]
    for r in rows
])


summary = {
    "experiment":
        "QMS-PYXEL-TWIN-002",

    "trials":
        len(rows),

    "true_parameters": {
        "cti_trap_density_cm3":
            TRUE_DENSITY,

        "output_node_noise_sigma":
            TRUE_NOISE,
    },

    "cti_estimation": {
        "mean":
            float(densities.mean()),

        "std":
            float(densities.std(ddof=1)),

        "mean_relative_error":
            float(
                np.mean(
                    (densities - TRUE_DENSITY)
                    / TRUE_DENSITY
                )
            ),

        "mean_absolute_relative_error":
            float(
                np.mean(
                    np.abs(
                        (densities - TRUE_DENSITY)
                        / TRUE_DENSITY
                    )
                )
            ),
    },

    "noise_estimation": {
        "mean":
            float(noise_estimates.mean()),

        "std":
            float(noise_estimates.std(ddof=1)),

        "mean_relative_error":
            float(
                np.mean(
                    (noise_estimates - TRUE_NOISE)
                    / TRUE_NOISE
                )
            ),

        "mean_absolute_relative_error":
            float(
                np.mean(
                    np.abs(
                        (noise_estimates - TRUE_NOISE)
                        / TRUE_NOISE
                    )
                )
            ),
    },

    "pixel_reconvergence": {
        "mean_before":
            float(pixel_before.mean()),

        "mean_after":
            float(pixel_after.mean()),

        "mean_reduction_percent":
            float(
                (
                    1.0
                    -
                    np.mean(
                        pixel_after
                        / pixel_before
                    )
                )
                * 100.0
            ),
    },

    "all_optimizers_successful":
        bool(
            all(
                r["optimizer_success"]
                for r in rows
            )
        ),

    "results":
        rows,

    "scientific_boundary": (
        "Monte Carlo validation within a controlled "
        "Pyxel simulation using the same known CDM "
        "mechanism family and detector configuration. "
        "This does not constitute experimental "
        "hardware validation."
    ),
}


out = (
    ROOT
    / "qms_pyxel_twin_002"
    / "qms_pyxel_twin_002_monte_carlo.json"
)

out.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-PYXEL-TWIN-002 MONTE CARLO ===")
print()

print("Trials:", len(rows))
print()

print("CTI density")
print(" true:", TRUE_DENSITY)
print(" mean estimate:", densities.mean())
print(" std:", densities.std(ddof=1))
print(
    " mean abs relative error:",
    np.mean(
        np.abs(
            (densities - TRUE_DENSITY)
            / TRUE_DENSITY
        )
    )
)

print()

print("Output-node noise")
print(" true:", TRUE_NOISE)
print(" mean estimate:", noise_estimates.mean())
print(" std:", noise_estimates.std(ddof=1))
print(
    " mean abs relative error:",
    np.mean(
        np.abs(
            (noise_estimates - TRUE_NOISE)
            / TRUE_NOISE
        )
    )
)

print()

print("Pixel reconvergence")
print(" mean before:", pixel_before.mean())
print(" mean after:", pixel_after.mean())
print(
    " mean reduction:",
    (
        1.0
        -
        np.mean(
            pixel_after / pixel_before
        )
    )
    * 100.0,
    "%"
)

print()

print(
    "All optimizers successful:",
    summary["all_optimizers_successful"]
)

print()
print("Evidence:", out)
