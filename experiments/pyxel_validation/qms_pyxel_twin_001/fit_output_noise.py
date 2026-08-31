from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent


def latest(root, filename):
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return files[-1]


def load_image(root):
    return np.asarray(
        fits.getdata(latest(root, "detector_image.fits")),
        dtype=float,
    )


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# Observed mixed detector state:
# true CTI + true output-node noise
obs_root = (
    ROOT
    / "qms_pyxel_twin_001"
    / "blind_cti_5e9_noise"
)

observed = load_image(obs_root)


# Twin prediction after CTI has already been adapted
pred_root = (
    ROOT
    / "qms_pyxel_twin_001"
    / "updated_cti_prediction"
)

predicted_after_cti = load_image(pred_root)

residual = observed - predicted_after_cti

residual_rmse = rmse(
    observed,
    predicted_after_cti
)

residual_std = float(residual.std())


# Empirical Pyxel calibration from the previously measured noise sweep.
# sigma=0.0005 produced image difference std ≈46.48433652 ADU.
reference_sigma = 0.0005
reference_std = 46.48433652

estimated_sigma = (
    reference_sigma
    * residual_std
    / reference_std
)


print()
print("=== QMS PYXEL OUTPUT-NODE NOISE ESTIMATOR ===")
print()

print("Residual after CTI adaptation:")
print("  image RMSE =", f"{residual_rmse:.10f}")
print("  image std  =", f"{residual_std:.10f}")

print()
print("Estimated output-node noise:")
print(
    "  sigma =",
    f"{estimated_sigma:.10e}"
)

print()
print("Reference calibration:")
print(
    "  sigma =",
    reference_sigma,
    "-> image std =",
    reference_std
)
