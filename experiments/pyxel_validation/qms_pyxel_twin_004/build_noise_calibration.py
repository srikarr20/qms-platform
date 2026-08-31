from pathlib import Path
import json

import numpy as np
from astropy.io import fits
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "qms_pyxel_twin_004"

BASELINE_ROOT = (
    ROOT
    / "qms_pyxel_001"
    / "results"
    / "noise_0000"
)

LEVELS = {
    "noise_00": 1e-5,
    "noise_01": 2e-5,
    "noise_02": 5e-5,
    "noise_03": 1e-4,
    "noise_04": 2e-4,
    "noise_05": 3e-4,
    "noise_06": 5e-4,
    "noise_07": 7e-4,
    "noise_08": 1e-3,
    "noise_09": 1.5e-3,
    "noise_10": 2e-3,
}


def latest(root, filename):
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(f"{filename} missing under {root}")
    return files[-1]


def load_image(root):
    return np.asarray(
        fits.getdata(latest(root, "detector_image.fits")),
        dtype=float
    )


baseline = load_image(BASELINE_ROOT)

rows = []

for name, sigma in LEVELS.items():

    img = load_image(
        EXP / "results" / name
    )

    delta = img - baseline

    rows.append({
        "condition": name,
        "sigma": sigma,
        "residual_std": float(delta.std()),
        "residual_rmse": float(
            np.sqrt(np.mean(delta ** 2))
        ),
        "changed_fraction": float(
            np.mean(delta != 0)
        ),
    })


sigmas = np.array([
    r["sigma"]
    for r in rows
], dtype=float)

stds = np.array([
    r["residual_std"]
    for r in rows
], dtype=float)


# Ensure monotonic x-axis for inverse interpolation
order = np.argsort(stds)

std_sorted = stds[order]
sigma_sorted = sigmas[order]

# Collapse any duplicate std values if present
unique_std = []
unique_sigma = []

for s, n in zip(std_sorted, sigma_sorted):
    if not unique_std or s > unique_std[-1]:
        unique_std.append(float(s))
        unique_sigma.append(float(n))

inverse = PchipInterpolator(
    unique_std,
    unique_sigma,
    extrapolate=True
)


evidence = {
    "experiment": "QMS-PYXEL-TWIN-004",

    "title":
        "Empirical nonlinear Pyxel output-noise transfer calibration",

    "mapping":
        "output_node_noise.std_deviation -> image residual std",

    "results":
        rows,

    "inverse_support": {
        "residual_std":
            unique_std,
        "sigma":
            unique_sigma,
    },

    "scientific_boundary": (
        "Calibration is specific to this Pyxel CCD configuration, "
        "scene, amplification and ADC chain. It is not a universal "
        "detector noise calibration."
    ),
}

out = (
    EXP
    / "qms_pyxel_twin_004_noise_calibration.json"
)

out.write_text(
    json.dumps(
        evidence,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-PYXEL-TWIN-004 NOISE CALIBRATION ===")
print()

print(
    f"{'sigma':>12} "
    f"{'image std':>14} "
    f"{'image RMSE':>14} "
    f"{'changed':>10}"
)

for r in rows:
    print(
        f"{r['sigma']:12.6e} "
        f"{r['residual_std']:14.6f} "
        f"{r['residual_rmse']:14.6f} "
        f"{r['changed_fraction']:10.6f}"
    )

print()
print("Evidence:", out)
