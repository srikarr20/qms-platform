import json
from pathlib import Path

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "cti_results"

LEVELS = {
    "cti_000": 0.0,
    "cti_005": 5.0,
    "cti_020": 20.0,
    "cti_060": 60.0,
}


def load_latest(folder, filename):
    files = sorted(folder.rglob(filename))
    if not files:
        raise FileNotFoundError(folder / filename)

    p = files[-1]

    if p.suffix == ".fits":
        arr = fits.getdata(p)
    else:
        arr = np.load(p)

    return np.asarray(arr, dtype=float)


def cosine(a, b):
    x = a.ravel()
    y = b.ravel()

    return float(
        np.dot(x, y) /
        (np.linalg.norm(x) * np.linalg.norm(y))
    )


def pearson(a, b):
    return float(
        np.corrcoef(
            a.ravel(),
            b.ravel()
        )[0, 1]
    )


def rmse(a, b):
    return float(
        np.sqrt(
            np.mean((a - b) ** 2)
        )
    )


def nrmse(a, b):
    scale = np.ptp(a)

    if scale == 0:
        return 0.0

    return float(
        rmse(a, b) / scale
    )


def histogram(arr, bins):
    h, _ = np.histogram(
        arr.ravel(),
        bins=bins
    )

    h = h.astype(float) + 1e-12
    h /= h.sum()

    return h


def js(p, q):
    m = 0.5 * (p + q)

    return float(
        0.5 * np.sum(p * np.log(p / m))
        +
        0.5 * np.sum(q * np.log(q / m))
    )


images = {}
pixels = {}

for name in LEVELS:
    folder = RESULTS / name

    images[name] = load_latest(
        folder,
        "detector_image.fits"
    )

    pixels[name] = load_latest(
        folder,
        "detector_pixel.npy"
    )


baseline_image = images["cti_000"]
baseline_pixel = pixels["cti_000"]

global_min = min(
    img.min()
    for img in images.values()
)

global_max = max(
    img.max()
    for img in images.values()
)

bins = np.linspace(
    global_min,
    global_max,
    257
)

baseline_hist = histogram(
    baseline_image,
    bins
)


rows = []

for name, density in LEVELS.items():

    img = images[name]
    pix = pixels[name]

    image_delta = img - baseline_image
    pixel_delta = pix - baseline_pixel

    h = histogram(
        img,
        bins
    )

    rows.append({
        "condition": name,
        "trap_density": density,

        "image_cosine":
            cosine(baseline_image, img),

        "image_pearson":
            pearson(baseline_image, img),

        "image_rmse":
            rmse(baseline_image, img),

        "image_nrmse":
            nrmse(baseline_image, img),

        "image_js":
            js(baseline_hist, h),

        "image_difference_std":
            float(image_delta.std()),

        "pixel_cosine":
            cosine(baseline_pixel, pix),

        "pixel_pearson":
            pearson(baseline_pixel, pix),

        "pixel_rmse":
            rmse(baseline_pixel, pix),

        "pixel_difference_std":
            float(pixel_delta.std()),

        "pixel_mean":
            float(pix.mean()),

        "pixel_std":
            float(pix.std()),
    })


evidence = {
    "experiment": "QMS-PYXEL-001C",

    "title":
        "QMS diagnostics of Pyxel CTI degradation",

    "simulator": {
        "framework": "Pyxel",
        "version": "3.0.2",
        "detector": "CCD",
    },

    "controlled_variable":
        "charge_transfer.cdm.trap_densities",

    "cti_parameters": {
        "direction": "parallel",
        "trap_release_times": [3e-2],
        "sigma": [1e-10],
        "beta": 0.3,
        "max_electron_volume": 1.62e-10,
        "transfer_period": 9.4722e-4,
        "charge_injection": False,
    },

    "output_node_noise": 0.0,

    "results": rows,

    "scientific_boundary": (
        "Controlled Pyxel CCD simulation using "
        "a synthetic fixed scene. CTI is introduced "
        "using Pyxel's Charge Distortion Model. "
        "This is not experimental detector validation."
    ),
}


out = (
    ROOT
    / "qms_pyxel_001c_cti_diagnostics.json"
)

out.write_text(
    json.dumps(
        evidence,
        indent=2
    ) + "\n"
)


print("QMS-PYXEL-001C CTI DIAGNOSTICS")

for r in rows:

    print()
    print(
        r["condition"],
        "trap_density=",
        r["trap_density"]
    )

    print(
        " IMAGE cosine:",
        f"{r['image_cosine']:.8f}"
    )

    print(
        " IMAGE pearson:",
        f"{r['image_pearson']:.8f}"
    )

    print(
        " IMAGE RMSE:",
        f"{r['image_rmse']:.6f}"
    )

    print(
        " IMAGE NRMSE:",
        f"{r['image_nrmse']:.8f}"
    )

    print(
        " IMAGE JS:",
        f"{r['image_js']:.8e}"
    )

    print(
        " PIXEL cosine:",
        f"{r['pixel_cosine']:.8f}"
    )

    print(
        " PIXEL pearson:",
        f"{r['pixel_pearson']:.8f}"
    )

    print(
        " PIXEL RMSE:",
        f"{r['pixel_rmse']:.6f}"
    )

    print(
        " PIXEL difference std:",
        f"{r['pixel_difference_std']:.6f}"
    )


print()
print("Evidence:", out)
