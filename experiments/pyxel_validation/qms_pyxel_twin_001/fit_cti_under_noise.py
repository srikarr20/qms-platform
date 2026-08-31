from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.constants import k, m_e
from scipy.optimize import minimize_scalar

from pyxel.models.charge_transfer.utils_cdm import run_cdm_parallel


ROOT = Path(__file__).resolve().parent.parent


def latest(root, filename):
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return files[-1]


def load_pixel(root):
    return np.asarray(
        np.load(latest(root, "detector_pixel.npy")),
        dtype=np.float64,
    )


def load_image(root):
    return np.asarray(
        fits.getdata(latest(root, "detector_image.fits")),
        dtype=np.float64,
    )


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# Clean reference
baseline_root = (
    ROOT
    / "qms_pyxel_001"
    / "cti_observable_results"
    / "cti_000"
)

baseline_pixel = load_pixel(baseline_root)
baseline_image = load_image(baseline_root)


# Noisy blind observation
obs_root = (
    ROOT
    / "qms_pyxel_twin_001"
    / "blind_cti_5e9_noise"
)

observed_pixel = load_pixel(obs_root)
observed_image = load_image(obs_root)


# CDM parameters
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
sigma = np.array([1.e-10], dtype=float)


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
        sigma=sigma,
        charge_injection=False,
        chg_inj_parallel_transfers=baseline_pixel.shape[0],
    )


def objective(log10_density):
    pred = simulate_pixel(log10_density)
    return rmse(pred, observed_pixel)


result = minimize_scalar(
    objective,
    bounds=(8.0, 11.0),
    method="bounded",
    options={"xatol": 1e-8},
)

estimated_density = 10.0 ** result.x
predicted_pixel = simulate_pixel(result.x)

pixel_before = rmse(
    baseline_pixel,
    observed_pixel
)

pixel_after = rmse(
    predicted_pixel,
    observed_pixel
)

pixel_reduction = (
    1.0 - pixel_after / pixel_before
) * 100.0


# Final image remains noisy because we have only adapted CTI,
# not the downstream output-node noise.
image_before = rmse(
    baseline_image,
    observed_image
)

print()
print("=== QMS PYXEL TWIN — CTI ESTIMATION UNDER READOUT NOISE ===")
print()

print("Estimated CTI:")
print(f"  trap density = {estimated_density:.8e} cm^-3")

print()
print("Optimizer:")
print("  success =", result.success)
print("  evaluations =", result.nfev)
print("  objective =", result.fun)

print()
print("Pixel-stage twin residual:")
print("  before CTI update =", f"{pixel_before:.10f}")
print("  after CTI update  =", f"{pixel_after:.10f}")
print("  reduction         =", f"{pixel_reduction:.6f}%")

print()
print("Final Image divergence:")
print("  image RMSE from clean baseline =", f"{image_before:.10f}")

print()
print("Interpretation:")
print(
    "  CTI is estimated from the Pixel representation before "
    "downstream output-node noise is applied."
)
print(
    "  Any remaining Image divergence therefore includes "
    "downstream measurement/readout effects not explained "
    "by the CTI model update."
)
