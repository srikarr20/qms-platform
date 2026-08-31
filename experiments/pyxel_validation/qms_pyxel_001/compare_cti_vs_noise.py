from pathlib import Path
import numpy as np
from astropy.io import fits


ROOT = Path("qms_pyxel_001")


def latest(root, filename):
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(root / filename)
    return files[-1]


def load_fits(root):
    return np.asarray(
        fits.getdata(latest(root, "detector_image.fits")),
        dtype=float
    )


def load_npy(root):
    return np.asarray(
        np.load(latest(root, "detector_pixel.npy")),
        dtype=float
    )


def summarize_delta(delta):
    row_mean = delta.mean(axis=1)
    col_mean = delta.mean(axis=0)

    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mean": float(delta.mean()),
        "std": float(delta.std()),
        "max_abs": float(np.max(np.abs(delta))),
        "row_profile_std": float(row_mean.std()),
        "col_profile_std": float(col_mean.std()),
        "row_to_col_structure_ratio":
            float(row_mean.std() / (col_mean.std() + 1e-15)),
        "positive_fraction":
            float(np.mean(delta > 0)),
        "negative_fraction":
            float(np.mean(delta < 0)),
    }


# -------------------------
# CTI
# -------------------------

cti_base_root = ROOT / "cti_observable_results" / "cti_000"
cti_base_pixel = load_npy(cti_base_root)
cti_base_image = load_fits(cti_base_root)

cti_conditions = {
    "cti_1e8": 1e8,
    "cti_1e9": 1e9,
    "cti_1e10": 1e10,
    "cti_1e11": 1e11,
}

print("=== CTI PIXEL-BUCKET DEGRADATION ===")

for name, density in cti_conditions.items():
    root = ROOT / "cti_observable_results" / name

    pix = load_npy(root)
    img = load_fits(root)

    dp = pix - cti_base_pixel
    di = img - cti_base_image

    sp = summarize_delta(dp)
    si = summarize_delta(di)

    print()
    print(name, "density =", density)

    print(" PIXEL")
    print("  RMSE:", f"{sp['rmse']:.8f}")
    print("  std:", f"{sp['std']:.8f}")
    print("  max abs:", f"{sp['max_abs']:.8f}")
    print("  row profile std:", f"{sp['row_profile_std']:.8f}")
    print("  col profile std:", f"{sp['col_profile_std']:.8f}")
    print(
        "  row/col structure ratio:",
        f"{sp['row_to_col_structure_ratio']:.8f}"
    )
    print(
        "  positive/negative fraction:",
        f"{sp['positive_fraction']:.6f}",
        f"{sp['negative_fraction']:.6f}"
    )

    print(" IMAGE")
    print("  RMSE:", f"{si['rmse']:.8f}")
    print("  std:", f"{si['std']:.8f}")
    print("  max abs:", f"{si['max_abs']:.8f}")


# -------------------------
# Output-node noise
# -------------------------

noise_base = load_fits(
    ROOT / "results" / "noise_0000"
)

noise_conditions = {
    "noise_0005": 0.0005,
    "noise_0010": 0.001,
    "noise_0050": 0.005,
}

print()
print("=== OUTPUT-NODE-NOISE IMAGE DEGRADATION ===")

for name, sigma in noise_conditions.items():
    img = load_fits(
        ROOT / "results" / name
    )

    d = img - noise_base
    s = summarize_delta(d)

    print()
    print(name, "sigma =", sigma)
    print(" RMSE:", f"{s['rmse']:.8f}")
    print(" std:", f"{s['std']:.8f}")
    print(" max abs:", f"{s['max_abs']:.8f}")
    print(" row profile std:", f"{s['row_profile_std']:.8f}")
    print(" col profile std:", f"{s['col_profile_std']:.8f}")
    print(
        " row/col structure ratio:",
        f"{s['row_to_col_structure_ratio']:.8f}"
    )
    print(
        " positive/negative fraction:",
        f"{s['positive_fraction']:.6f}",
        f"{s['negative_fraction']:.6f}"
    )
