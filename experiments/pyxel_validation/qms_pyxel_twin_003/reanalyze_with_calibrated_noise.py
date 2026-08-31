from pathlib import Path
import json

import numpy as np
from astropy.io import fits
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parent.parent

CAL_FILE = (
    ROOT
    / "qms_pyxel_twin_004"
    / "qms_pyxel_twin_004_noise_calibration.json"
)

TWIN003_FILE = (
    ROOT
    / "qms_pyxel_twin_003"
    / "qms_pyxel_twin_003_time_evolution.json"
)

OBS_ROOT = (
    ROOT
    / "qms_pyxel_twin_003"
    / "results"
)

PRED_ROOT = (
    ROOT
    / "qms_pyxel_twin_003"
    / "prediction_results"
)


def latest(root, filename):
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(
            f"{filename} not found under {root}"
        )
    return files[-1]


def load_image(root):
    return np.asarray(
        fits.getdata(
            latest(root, "detector_image.fits")
        ),
        dtype=float
    )


cal = json.loads(
    CAL_FILE.read_text()
)

std_support = np.asarray(
    cal["inverse_support"]["residual_std"],
    dtype=float
)

sigma_support = np.asarray(
    cal["inverse_support"]["sigma"],
    dtype=float
)

inverse = PchipInterpolator(
    std_support,
    sigma_support,
    extrapolate=True
)


old = json.loads(
    TWIN003_FILE.read_text()
)

rows = []

for r in old["results"]:

    name = r["time"]

    observed = load_image(
        OBS_ROOT / name
    )

    predicted_cti_only = load_image(
        PRED_ROOT / name
    )

    residual = (
        observed - predicted_cti_only
    )

    residual_std = float(
        residual.std()
    )

    noise_hat = float(
        inverse(residual_std)
    )

    # Do not allow physically meaningless
    # negative extrapolated estimate.
    noise_hat = max(noise_hat, 0.0)

    true_noise = float(
        r["true_output_noise"]
    )

    rows.append({
        "time":
            name,

        "true_cti_density":
            r["true_cti_density"],

        "estimated_cti_density":
            r["estimated_cti_density"],

        "cti_relative_error":
            r["cti_relative_error"],

        "true_output_noise":
            true_noise,

        "old_estimated_output_noise":
            r["estimated_output_noise"],

        "calibrated_estimated_output_noise":
            noise_hat,

        "calibrated_noise_relative_error":
            (
                noise_hat - true_noise
            ) / true_noise,

        "image_residual_std_after_cti":
            residual_std,
    })


noise_errors = np.asarray([
    abs(
        r["calibrated_noise_relative_error"]
    )
    for r in rows
])

old_errors = np.asarray([
    abs(
        (
            r["old_estimated_output_noise"]
            -
            r["true_output_noise"]
        )
        /
        r["true_output_noise"]
    )
    for r in rows
])


summary = {
    "experiment":
        "QMS-PYXEL-TWIN-003B",

    "title":
        "Time-evolving twin with calibrated nonlinear noise inference",

    "states":
        len(rows),

    "old_mean_absolute_noise_relative_error":
        float(old_errors.mean()),

    "calibrated_mean_absolute_noise_relative_error":
        float(noise_errors.mean()),

    "calibrated_max_absolute_noise_relative_error":
        float(noise_errors.max()),

    "results":
        rows,

    "scientific_boundary": (
        "Noise inference uses an empirical inverse calibration "
        "derived from the same Pyxel CCD configuration and scene. "
        "This is configuration-specific computational validation."
    ),
}


out = (
    ROOT
    / "qms_pyxel_twin_003"
    / "qms_pyxel_twin_003b_calibrated_noise.json"
)

out.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-PYXEL-TWIN-003B ===")
print()

for r in rows:

    print(
        r["time"],
        "true=",
        f"{r['true_output_noise']:.6e}",
        "old=",
        f"{r['old_estimated_output_noise']:.6e}",
        "calibrated=",
        f"{r['calibrated_estimated_output_noise']:.6e}",
        "error=",
        f"{100*abs(r['calibrated_noise_relative_error']):.3f}%"
    )

print()
print(
    "Old mean abs noise error:",
    f"{100*summary['old_mean_absolute_noise_relative_error']:.3f}%"
)

print(
    "Calibrated mean abs noise error:",
    f"{100*summary['calibrated_mean_absolute_noise_relative_error']:.3f}%"
)

print(
    "Calibrated max abs noise error:",
    f"{100*summary['calibrated_max_absolute_noise_relative_error']:.3f}%"
)

print()
print("Evidence:", out)
