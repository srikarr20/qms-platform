import json
from pathlib import Path

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

LEVELS = {
    "noise_0000": 0.0,
    "noise_0005": 0.0005,
    "noise_0010": 0.001,
    "noise_0050": 0.005,
}


def load_latest_fits(folder):
    files = sorted(folder.rglob("detector_image.fits"))
    if not files:
        raise FileNotFoundError(folder)
    return np.asarray(
        fits.getdata(files[-1]),
        dtype=float
    )


def cosine_similarity(a, b):
    x = a.reshape(-1)
    y = b.reshape(-1)

    denom = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if denom == 0:
        return 0.0

    return float(
        np.dot(x, y) / denom
    )


def pearson(a, b):
    x = a.reshape(-1)
    y = b.reshape(-1)

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def rmse(a, b):
    return float(
        np.sqrt(
            np.mean((a - b) ** 2)
        )
    )


def normalized_rmse(a, b):
    scale = np.ptp(a)

    if scale == 0:
        return 0.0

    return float(
        rmse(a, b) / scale
    )


def histogram_distribution(arr, bins):
    hist, _ = np.histogram(
        arr.reshape(-1),
        bins=bins
    )

    hist = hist.astype(float)
    hist += 1e-12
    hist /= hist.sum()

    return hist


def js_divergence(p, q):
    m = 0.5 * (p + q)

    kl_pm = np.sum(
        p * np.log(p / m)
    )

    kl_qm = np.sum(
        q * np.log(q / m)
    )

    return float(
        0.5 * kl_pm
        + 0.5 * kl_qm
    )


images = {}

for name in LEVELS:
    images[name] = load_latest_fits(
        RESULTS / name
    )


baseline = images["noise_0000"]

global_min = min(
    float(img.min())
    for img in images.values()
)

global_max = max(
    float(img.max())
    for img in images.values()
)

bins = np.linspace(
    global_min,
    global_max,
    257
)

baseline_hist = histogram_distribution(
    baseline,
    bins
)


rows = []

for name, sigma in LEVELS.items():

    img = images[name]

    hist = histogram_distribution(
        img,
        bins
    )

    delta = img - baseline

    rows.append({
        "condition": name,
        "output_node_noise_sigma": sigma,

        "image_min":
            float(img.min()),

        "image_max":
            float(img.max()),

        "image_mean":
            float(img.mean()),

        "image_std":
            float(img.std()),

        "cosine_similarity_to_baseline":
            cosine_similarity(
                baseline,
                img
            ),

        "pearson_to_baseline":
            pearson(
                baseline,
                img
            ),

        "rmse_to_baseline":
            rmse(
                baseline,
                img
            ),

        "normalized_rmse_to_baseline":
            normalized_rmse(
                baseline,
                img
            ),

        "js_divergence_to_baseline":
            js_divergence(
                baseline_hist,
                hist
            ),

        "difference_mean":
            float(delta.mean()),

        "difference_std":
            float(delta.std()),
    })


evidence = {
    "experiment": "QMS-PYXEL-001B",

    "title": (
        "QMS diagnostics of Pyxel "
        "output-node-noise degradation"
    ),

    "simulator": {
        "framework": "Pyxel",
        "version": "3.0.2",
        "detector": "CCD",
    },

    "controlled_variable":
        "charge_measurement.output_node_noise.std_deviation",

    "fixed_input":
        "qms_pyxel_scene.fits",

    "results":
        rows,

    "scientific_boundary": (
        "Controlled Pyxel detector-simulation "
        "experiment using a synthetic fixed scene. "
        "Results characterize the configured Pyxel "
        "CCD pipeline and do not constitute "
        "experimental detector validation."
    ),
}


out = (
    ROOT
    / "qms_pyxel_001b_noise_diagnostics.json"
)

out.write_text(
    json.dumps(
        evidence,
        indent=2
    ) + "\n"
)


print(f"evidence: {out}")
print()
print("QMS-PYXEL-001B NOISE DIAGNOSTICS")

for r in rows:

    print()
    print(
        r["condition"],
        "sigma=",
        r["output_node_noise_sigma"]
    )

    print(
        "  cosine:",
        f"{r['cosine_similarity_to_baseline']:.8f}"
    )

    print(
        "  pearson:",
        f"{r['pearson_to_baseline']:.8f}"
    )

    print(
        "  RMSE:",
        f"{r['rmse_to_baseline']:.6f}"
    )

    print(
        "  normalized RMSE:",
        f"{r['normalized_rmse_to_baseline']:.8f}"
    )

    print(
        "  JS divergence:",
        f"{r['js_divergence_to_baseline']:.8e}"
    )

    print(
        "  image std:",
        f"{r['image_std']:.6f}"
    )

    print(
        "  difference std:",
        f"{r['difference_std']:.6f}"
    )
