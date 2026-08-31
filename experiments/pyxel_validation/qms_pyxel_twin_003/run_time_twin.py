from pathlib import Path
import json
import re
import subprocess

import numpy as np
from astropy.io import fits
from scipy.constants import k, m_e
from scipy.optimize import minimize_scalar

from pyxel.models.charge_transfer.utils_cdm import run_cdm_parallel


ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "qms_pyxel_twin_003"
OBS_ROOT = EXP / "results"
PRED_CONFIGS = EXP / "prediction_configs"
PRED_RESULTS = EXP / "prediction_results"

PRED_CONFIGS.mkdir(exist_ok=True)
PRED_RESULTS.mkdir(exist_ok=True)

BASE_CONFIG = (ROOT / "first_simulation_local.yaml").read_text()

BASELINE_ROOT = (
    ROOT
    / "qms_pyxel_001"
    / "cti_observable_results"
    / "cti_000"
)

TRUE = {
    "t00": (1.0e8,  1.0e-4),
    "t01": (5.0e8,  2.0e-4),
    "t02": (1.0e9,  3.0e-4),
    "t03": (5.0e9,  5.0e-4),
    "t04": (1.0e10, 7.0e-4),
    "t05": (2.0e10, 1.0e-3),
}


def latest(root, filename):
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(
            f"{filename} not found under {root}"
        )
    return files[-1]


def load_pixel(root):
    return np.asarray(
        np.load(latest(root, "detector_pixel.npy")),
        dtype=np.float64,
    )


def load_image(root):
    return np.asarray(
        fits.getdata(
            latest(root, "detector_image.fits")
        ),
        dtype=np.float64,
    )


def rmse(a, b):
    return float(
        np.sqrt(np.mean((a - b) ** 2))
    )


baseline_pixel = load_pixel(BASELINE_ROOT)

# ---------------------------------------
# Pyxel CDM model used by the twin
# ---------------------------------------

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


def simulate_pixel(log10_density):
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


# Empirical output-noise calibration from 001B
REFERENCE_SIGMA = 5.0e-4
REFERENCE_IMAGE_STD = 46.48433652


rows = []


for name in sorted(TRUE):

    print()
    print("=" * 60)
    print("TWIN STATE", name)
    print("=" * 60)

    obs_root = OBS_ROOT / name

    observed_pixel = load_pixel(obs_root)
    observed_image = load_image(obs_root)

    # ---------------------------------------
    # 1. Estimate CTI continuously
    # ---------------------------------------

    def objective(log_density):
        return rmse(
            simulate_pixel(log_density),
            observed_pixel
        )

    result = minimize_scalar(
        objective,
        bounds=(7.0, 11.5),
        method="bounded",
        options={"xatol": 1e-8},
    )

    density_hat = 10.0 ** result.x

    predicted_pixel = simulate_pixel(result.x)

    pixel_before = rmse(
        baseline_pixel,
        observed_pixel
    )

    pixel_after = rmse(
        predicted_pixel,
        observed_pixel
    )

    # ---------------------------------------
    # 2. Ask Pyxel for CTI-only twin image
    # ---------------------------------------

    config = BASE_CONFIG

    cti = f"""  charge_transfer:
    - name: cdm
      func: pyxel.models.charge_transfer.cdm
      enabled: true
      arguments:
        direction: parallel
        trap_release_times: [3.e-2]
        trap_densities: [{density_hat:.8f}]
        sigma: [1.e-10]
        beta: 0.3
        max_electron_volume: 1.62e-10
        transfer_period: 9.4722e-04
        charge_injection: false

"""

    config = re.sub(
        r"  charge_transfer:\s*\n",
        cti,
        config,
        count=1,
    )

    # Noise disabled: this is the twin's
    # deterministic CTI-adapted prediction.
    config = re.sub(
        r"(name:\s*output_noise.*?std_deviation:\s*)[0-9.eE+-]+",
        r"\g<1>0.0",
        config,
        flags=re.S,
    )

    config = config.replace(
        'output_folder: "output"',
        f'output_folder: "qms_pyxel_twin_003/prediction_results/{name}"'
    )

    config_path = PRED_CONFIGS / f"{name}.yaml"
    config_path.write_text(config)

    subprocess.run(
        [
            "pyxel-sim",
            "run",
            str(config_path),
        ],
        check=True,
    )

    predicted_image = load_image(
        PRED_RESULTS / name
    )

    # ---------------------------------------
    # 3. Residual downstream noise estimate
    # ---------------------------------------

    image_residual = (
        observed_image - predicted_image
    )

    residual_std = float(
        image_residual.std()
    )

    noise_hat = (
        REFERENCE_SIGMA
        * residual_std
        / REFERENCE_IMAGE_STD
    )

    true_density, true_noise = TRUE[name]

    row = {
        "time": name,

        "true_cti_density":
            true_density,

        "estimated_cti_density":
            float(density_hat),

        "cti_relative_error":
            float(
                (density_hat - true_density)
                / true_density
            ),

        "true_output_noise":
            true_noise,

        "estimated_output_noise":
            float(noise_hat),

        "noise_relative_error":
            float(
                (noise_hat - true_noise)
                / true_noise
            ),

        "pixel_residual_before":
            float(pixel_before),

        "pixel_residual_after":
            float(pixel_after),

        "image_residual_std_after_cti":
            residual_std,

        "optimizer_success":
            bool(result.success),
    }

    rows.append(row)

    print()
    print(
        "CTI:",
        f"true={true_density:.4e}",
        f"estimated={density_hat:.4e}",
    )

    print(
        "Noise:",
        f"true={true_noise:.4e}",
        f"estimated={noise_hat:.4e}",
    )

    print(
        "Pixel residual:",
        f"{pixel_before:.6e}",
        "->",
        f"{pixel_after:.6e}",
    )


cti_err = np.array([
    abs(r["cti_relative_error"])
    for r in rows
])

noise_err = np.array([
    abs(r["noise_relative_error"])
    for r in rows
])

before = np.array([
    r["pixel_residual_before"]
    for r in rows
])

after = np.array([
    r["pixel_residual_after"]
    for r in rows
])


summary = {
    "experiment":
        "QMS-PYXEL-TWIN-003",

    "title":
        "Time-evolving adaptive Pyxel detector twin",

    "states":
        len(rows),

    "mean_absolute_cti_relative_error":
        float(cti_err.mean()),

    "max_absolute_cti_relative_error":
        float(cti_err.max()),

    "mean_absolute_noise_relative_error":
        float(noise_err.mean()),

    "max_absolute_noise_relative_error":
        float(noise_err.max()),

    "mean_pixel_residual_reduction_percent":
        float(
            (
                1.0
                -
                np.mean(after / before)
            )
            * 100.0
        ),

    "results":
        rows,

    "scientific_boundary": (
        "Controlled time-indexed sequence of independent "
        "Pyxel simulations. The twin estimates parameters "
        "within the same known CDM and output-noise model "
        "families. This is a computational detector-twin "
        "validation, not experimental hardware validation."
    ),
}


evidence = (
    EXP
    / "qms_pyxel_twin_003_time_evolution.json"
)

evidence.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=" * 60)
print("QMS-PYXEL-TWIN-003 SUMMARY")
print("=" * 60)

print()
print(
    "Mean abs CTI relative error:",
    summary[
        "mean_absolute_cti_relative_error"
    ],
)

print(
    "Max abs CTI relative error:",
    summary[
        "max_absolute_cti_relative_error"
    ],
)

print()
print(
    "Mean abs noise relative error:",
    summary[
        "mean_absolute_noise_relative_error"
    ],
)

print(
    "Max abs noise relative error:",
    summary[
        "max_absolute_noise_relative_error"
    ],
)

print()
print(
    "Mean Pixel residual reduction:",
    summary[
        "mean_pixel_residual_reduction_percent"
    ],
    "%"
)

print()
print("Evidence:", evidence)
