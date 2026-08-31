import json
from pathlib import Path

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parent.parent
QMS = ROOT / "qms_pyxel_001"


def latest(root, filename):
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return files[-1]


def load_image(root):
    return np.asarray(
        fits.getdata(latest(root, "detector_image.fits")),
        dtype=float
    )


def load_pixel(root):
    return np.asarray(
        np.load(latest(root, "detector_pixel.npy")),
        dtype=float
    )


def rmse(a, b):
    return float(
        np.sqrt(
            np.mean((a - b) ** 2)
        )
    )


def image_features(delta):
    row = delta.mean(axis=1)
    col = delta.mean(axis=0)

    changed = delta != 0

    return {
        "rmse": rmse(delta, np.zeros_like(delta)),
        "mean": float(delta.mean()),
        "std": float(delta.std()),
        "changed_fraction": float(np.mean(changed)),
        "directionality": float(
            row.std() / (col.std() + 1e-12)
        ),
    }


# --------------------------------------------------
# Twin baseline
# --------------------------------------------------

baseline_root = (
    QMS /
    "cti_observable_results" /
    "cti_000"
)

baseline_image = load_image(baseline_root)
baseline_pixel = load_pixel(baseline_root)


# --------------------------------------------------
# Reference mechanism library
# --------------------------------------------------

library = []

cti_conditions = {
    "cti_1e8": 1e8,
    "cti_1e9": 1e9,
    "cti_1e10": 1e10,
    "cti_1e11": 1e11,
}

for name, value in cti_conditions.items():

    root = (
        QMS /
        "cti_observable_results" /
        name
    )

    image = load_image(root)
    pixel = load_pixel(root)

    library.append({
        "mechanism": "CTI",
        "parameter": "trap_density_cm3",
        "value": value,
        "image": image,
        "pixel": pixel,
    })


noise_conditions = {
    "noise_0005": 0.0005,
    "noise_0010": 0.001,
    "noise_0050": 0.005,
}

for name, value in noise_conditions.items():

    root = (
        QMS /
        "results" /
        name
    )

    image = load_image(root)
    pixel = load_pixel(root)

    library.append({
        "mechanism": "OUTPUT_NODE_NOISE",
        "parameter": "std_deviation",
        "value": value,
        "image": image,
        "pixel": pixel,
    })


# --------------------------------------------------
# Pick current observation
#
# For first validation, deliberately use one of our
# known degraded Pyxel runs.
# --------------------------------------------------

observation_root = (
    ROOT /
    "qms_pyxel_twin_001" /
    "blind_cti_5e9"
)

observed_image = load_image(observation_root)
observed_pixel = load_pixel(observation_root)


# --------------------------------------------------
# Divergence
# --------------------------------------------------

image_delta = observed_image - baseline_image
pixel_delta = observed_pixel - baseline_pixel

img_feat = image_features(image_delta)

pixel_rmse = rmse(
    observed_pixel,
    baseline_pixel
)


# --------------------------------------------------
# Mechanism localization
# --------------------------------------------------

if pixel_rmse > 1e-9:
    first_divergent_stage = "PIXEL"
else:
    first_divergent_stage = "IMAGE"


# --------------------------------------------------
# Reference-library model selection
# --------------------------------------------------

scores = []

for candidate in library:

    image_error = rmse(
        observed_image,
        candidate["image"]
    )

    pixel_error = rmse(
        observed_pixel,
        candidate["pixel"]
    )

    # Pixel information is extremely useful for
    # separating CTI from downstream readout noise.
    score = image_error + pixel_error

    scores.append({
        "mechanism": candidate["mechanism"],
        "parameter": candidate["parameter"],
        "value": candidate["value"],
        "score": score,
        "image_error": image_error,
        "pixel_error": pixel_error,
    })


scores.sort(
    key=lambda x: x["score"]
)

winner = scores[0]

if len(scores) > 1:
    margin = (
        scores[1]["score"] -
        scores[0]["score"]
    )
else:
    margin = 0.0


# --------------------------------------------------
# Twin state
# --------------------------------------------------

twin = {
    "twin": "QMS-PYXEL-TWIN-001",

    "status": (
        "DIVERGED"
        if img_feat["rmse"] > 0
        else "NOMINAL"
    ),

    "measurement_state": {
        "image_rmse_from_baseline":
            img_feat["rmse"],

        "pixel_rmse_from_baseline":
            pixel_rmse,

        "image_changed_fraction":
            img_feat["changed_fraction"],

        "image_directionality":
            img_feat["directionality"],

        "first_divergent_stage":
            first_divergent_stage,
    },

    "inference": {
        "mechanism":
            winner["mechanism"],

        "parameter":
            winner["parameter"],

        "estimated_value":
            winner["value"],

        "winner_score":
            winner["score"],

        "runner_up_margin":
            margin,
    },

    "top_candidates":
        scores[:5],

    "scientific_boundary": (
        "Computational detector-twin prototype. "
        "Inference is restricted to a finite library "
        "of previously simulated Pyxel mechanisms "
        "and parameter values. This is not real "
        "detector validation."
    ),
}


out = (
    ROOT /
    "qms_pyxel_twin_001" /
    "twin_state.json"
)

out.write_text(
    json.dumps(
        twin,
        indent=2
    ) + "\n"
)


print()
print("=== QMS PYXEL DETECTOR TWIN ===")
print()

print("Status:")
print(" ", twin["status"])

print()

print("Measurement state:")
for k, v in twin["measurement_state"].items():
    print(" ", k, "=", v)

print()

print("Twin inference:")
for k, v in twin["inference"].items():
    print(" ", k, "=", v)

print()

print("Top candidates:")
for c in twin["top_candidates"]:
    print(
        " ",
        c["mechanism"],
        c["value"],
        "score=",
        c["score"]
    )

print()
print("Evidence:", out)
