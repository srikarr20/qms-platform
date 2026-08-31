from pathlib import Path
import json
import shutil

import numpy as np
from astropy.io import fits
from scipy.constants import k, m_e
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

from pyxel.models.charge_transfer.utils_cdm import run_cdm_parallel


ROOT = Path(__file__).resolve().parent.parent

INBOX = ROOT / "qms_pyxel_twin_009" / "inbox"
ARCHIVE = ROOT / "qms_pyxel_twin_009" / "archive"
STATE_DIR = ROOT / "qms_pyxel_twin_009" / "state"

BASELINE_ROOT = (
    ROOT
    / "qms_pyxel_001"
    / "cti_observable_results"
    / "cti_000"
)

CAL_FILE = (
    ROOT
    / "qms_pyxel_twin_004"
    / "qms_pyxel_twin_004_noise_calibration.json"
)

CTI_THRESHOLD = 0.20
NOISE_THRESHOLD = 0.10


def load_pixel(folder):
    return np.asarray(
        np.load(folder / "detector_pixel.npy"),
        dtype=np.float64,
    )


def load_image(folder):
    return np.asarray(
        fits.getdata(folder / "detector_image.fits"),
        dtype=np.float64,
    )


def latest(root, filename):
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(filename)
    return files[-1]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


baseline_pixel = np.asarray(
    np.load(
        latest(BASELINE_ROOT, "detector_pixel.npy")
    ),
    dtype=np.float64,
)


# ---------------------------------
# CTI estimator
# ---------------------------------

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


def simulate_cti(log_density):
    density = 10.0 ** log_density

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


def estimate_cti(observed_pixel):

    def objective(log_density):
        return rmse(
            simulate_cti(log_density),
            observed_pixel
        )

    result = minimize_scalar(
        objective,
        bounds=(7.0, 11.5),
        method="bounded",
        options={"xatol": 1e-8},
    )

    return (
        float(10.0 ** result.x),
        float(result.fun),
    )


# ---------------------------------
# Noise calibration
# ---------------------------------

cal = json.loads(CAL_FILE.read_text())

noise_inverse = PchipInterpolator(
    np.asarray(
        cal["inverse_support"]["residual_std"],
        dtype=float,
    ),
    np.asarray(
        cal["inverse_support"]["sigma"],
        dtype=float,
    ),
    extrapolate=True,
)


# ---------------------------------
# Persistent state
# ---------------------------------

state_file = STATE_DIR / "twin_state.json"

if state_file.exists():
    state = json.loads(state_file.read_text())
else:
    state = {"history": []}


def predict(history):

    if len(history) < 2:
        return None

    a = history[-2]
    b = history[-1]

    return {
        "cti":
            b["cti"] + (b["cti"] - a["cti"]),

        "noise":
            b["noise"] + (b["noise"] - a["noise"]),
    }


# ---------------------------------
# Process inbox
# ---------------------------------

measurements = sorted(
    p for p in INBOX.iterdir()
    if p.is_dir()
)

if not measurements:
    print("No measurements waiting.")
    raise SystemExit(0)


for folder in measurements:

    observed_pixel = load_pixel(folder)
    observed_image = load_image(folder)

    cti_hat, pixel_fit_residual = estimate_cti(
        observed_pixel
    )

    # First external-adapter version:
    # use closest CTI-only Pyxel prediction already available
    # from the validated TWIN-003 sequence.
    candidate_roots = sorted(
        (
            ROOT
            / "qms_pyxel_twin_003"
            / "prediction_results"
        ).iterdir()
    )

    best = None

    for candidate in candidate_roots:

        candidate_pixel = np.asarray(
            np.load(
                latest(candidate, "detector_pixel.npy")
            ),
            dtype=float,
        )

        score = rmse(
            candidate_pixel,
            observed_pixel
        )

        if best is None or score < best[0]:
            best = (score, candidate)

    cti_reference_root = best[1]

    predicted_image = np.asarray(
        fits.getdata(
            latest(
                cti_reference_root,
                "detector_image.fits"
            )
        ),
        dtype=float,
    )

    residual = observed_image - predicted_image
    residual_std = float(residual.std())

    noise_hat = float(
        noise_inverse(residual_std)
    )

    noise_hat = max(noise_hat, 0.0)

    current = {
        "cti": cti_hat,
        "noise": noise_hat,
    }

    pred = predict(state["history"])

    if pred is None:
        mode = "INITIALIZE"
        trigger = True
        cti_innovation = None
        noise_innovation = None

    else:
        cti_innovation = abs(
            pred["cti"] - current["cti"]
        ) / current["cti"]

        noise_innovation = abs(
            pred["noise"] - current["noise"]
        ) / current["noise"]

        trigger = (
            cti_innovation > CTI_THRESHOLD
            or
            noise_innovation > NOISE_THRESHOLD
        )

        mode = (
            "REESTIMATE_AND_ADAPT"
            if trigger
            else
            "ASSIMILATE"
        )

    event = {
        "measurement_id":
            folder.name,

        "predicted":
            pred,

        "cti":
            cti_hat,

        "noise":
            noise_hat,

        "pixel_fit_residual":
            pixel_fit_residual,

        "image_residual_std":
            residual_std,

        "cti_innovation_relative":
            cti_innovation,

        "noise_innovation_relative":
            noise_innovation,

        "mode":
            mode,
    }

    state["history"].append(event)

    state_file.write_text(
        json.dumps(
            state,
            indent=2
        ) + "\n"
    )

    archive_target = ARCHIVE / folder.name

    if archive_target.exists():
        shutil.rmtree(archive_target)

    shutil.move(
        str(folder),
        str(archive_target)
    )

    print()
    print("=== QMS EXTERNAL DETECTOR TWIN ===")
    print()
    print("measurement:", folder.name)
    print("CTI estimate:", cti_hat)
    print("noise estimate:", noise_hat)
    print("pixel fit residual:", pixel_fit_residual)
    print("mode:", mode)
    print("archived:", archive_target)
