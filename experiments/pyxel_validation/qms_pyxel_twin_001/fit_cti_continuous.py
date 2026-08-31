from pathlib import Path

import numpy as np
from scipy.constants import k, m_e
from scipy.optimize import minimize_scalar

from pyxel.models.charge_transfer.utils_cdm import run_cdm_parallel


ROOT = Path(__file__).resolve().parent.parent


def latest(root, filename):
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(filename)
    return files[-1]


def load_pixel(root):
    return np.asarray(
        np.load(latest(root, "detector_pixel.npy")),
        dtype=np.float64,
    )


def rmse(a, b):
    return float(
        np.sqrt(
            np.mean((a - b) ** 2)
        )
    )


# Clean detector state
baseline = load_pixel(
    ROOT
    / "qms_pyxel_001"
    / "cti_observable_results"
    / "cti_000"
)

# Blind observation: true density = 5e9,
# but estimator does not use that value.
observed = load_pixel(
    ROOT
    / "qms_pyxel_twin_001"
    / "blind_cti_5e9"
)


# Same Pyxel CDM configuration
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


def simulate(log10_density):
    density = 10.0 ** log10_density

    arr = baseline.copy()

    prediction = run_cdm_parallel(
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
        chg_inj_parallel_transfers=baseline.shape[0],
    )

    return prediction


def objective(log10_density):
    prediction = simulate(log10_density)
    return rmse(prediction, observed)


# Search a continuous logarithmic CTI range
result = minimize_scalar(
    objective,
    bounds=(8.0, 11.0),
    method="bounded",
    options={
        "xatol": 1e-8
    },
)

estimated_density = 10.0 ** result.x

prediction = simulate(result.x)

baseline_error = rmse(
    baseline,
    observed
)

post_update_error = rmse(
    prediction,
    observed
)

improvement = (
    1.0 - post_update_error / baseline_error
) * 100.0


print()
print("=== QMS PYXEL CONTINUOUS CTI ESTIMATOR ===")
print()

print("Estimated trap density:")
print(f"  {estimated_density:.8e} cm^-3")

print()
print("Optimization:")
print("  success =", result.success)
print("  objective =", result.fun)
print("  evaluations =", result.nfev)

print()
print("Twin residual:")
print(
    "  before model update =",
    f"{baseline_error:.10f}"
)

print(
    "  after model update  =",
    f"{post_update_error:.10f}"
)

print(
    "  residual reduction  =",
    f"{improvement:.6f}%"
)
