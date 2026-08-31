from collections import deque
from pathlib import Path
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.glasgow_event_adapter import GlasgowCumulativeArchive


GLASGOW_ZIP = Path(
    os.environ.get(
        "GLASGOW_ZIP",
        str(
            Path.home()
            / "Desktop"
            / "Quantum-Research"
            / "experimental-data"
            / "glasgow-single-photon"
            / "dpi_lab_1"
            / "Heralded Diffraction SM.zip"
        ),
    )
)

WINDOW = 100
GRID = 32

archive = GlasgowCumulativeArchive(GLASGOW_ZIP)


def histogram_from_increment(frame):
    """
    Convert a detector-count increment into a normalized
    32x32 spatial distribution.

    The original Glasgow detector is 512x512.
    """
    h, w = frame.shape

    if h % GRID != 0 or w % GRID != 0:
        raise ValueError(
            f"Detector shape {frame.shape} is not divisible by GRID={GRID}"
        )

    bh = h // GRID
    bw = w // GRID

    reduced = frame.reshape(
        GRID, bh, GRID, bw
    ).sum(axis=(1, 3))

    total = reduced.sum()

    if total <= 0:
        return np.zeros(GRID * GRID, dtype=float)

    return (reduced / total).ravel()


def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return 0.0

    return float(
        np.dot(a, b) / (na * nb)
    )


def js_divergence(p, q):
    eps = 1e-15

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)

    kl_pm = np.sum(
        p * np.log(p / m)
    )

    kl_qm = np.sum(
        q * np.log(q / m)
    )

    return float(
        0.5 * (kl_pm + kl_qm)
    )


# ------------------------------------------------------------
# First pass: build final experimental reference distribution.
# ------------------------------------------------------------

reference_counts = None
increment_total = 0

for record in archive.iter_increments():

    inc = np.asarray(
        record["increment"],
        dtype=float,
    )

    if reference_counts is None:
        reference_counts = np.zeros_like(inc)

    reference_counts += inc
    increment_total += 1


reference = histogram_from_increment(
    reference_counts
)


# ------------------------------------------------------------
# Second pass: sequential experimental twin.
# ------------------------------------------------------------

window = deque(maxlen=WINDOW)

history = []

previous_cos = None
previous_js = None


print()
print("=== QMS-EXPERIMENTAL-TWIN-001 ===")
print("Glasgow real detector stream")
print()


for record in archive.iter_increments():

    window.append(
        np.asarray(
            record["increment"],
            dtype=float,
        )
    )

    if len(window) < WINDOW:
        continue

    aggregate = np.sum(
        np.stack(window, axis=0),
        axis=0,
    )

    distribution = histogram_from_increment(
        aggregate
    )

    cos = cosine_similarity(
        distribution,
        reference
    )

    js = js_divergence(
        distribution,
        reference
    )

    if previous_cos is None:
        delta_cos = None
        delta_js = None
        state = "INITIALIZE"

    else:

        delta_cos = cos - previous_cos
        delta_js = js - previous_js

        if delta_cos > 0 and delta_js < 0:
            state = "CONVERGING"

        elif delta_cos < 0 and delta_js > 0:
            state = "DIVERGING"

        else:
            state = "MIXED"

    event = {
        "window_end_increment":
            int(record["index"]),

        "from_name":
            record["from_name"],

        "to_name":
            record["to_name"],

        "window_size":
            WINDOW,

        "cosine_similarity_to_final_distribution":
            cos,

        "js_divergence_to_final_distribution":
            js,

        "delta_cosine":
            delta_cos,

        "delta_js":
            delta_js,

        "measurement_state":
            state,
    }

    history.append(event)

    if (
        len(history) <= 5
        or
        len(history) % 500 == 0
    ):
        print(
            f"window={record['index']:4d}",
            f"cos={cos:.6f}",
            f"JS={js:.6f}",
            f"state={state}",
        )

    previous_cos = cos
    previous_js = js


counts = {}

for event in history:
    state = event["measurement_state"]
    counts[state] = counts.get(state, 0) + 1


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-001",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "source_type":
        "real experimental cumulative detector measurements",

    "detector_increment_model":
        "D_t = F_t - F_(t-1)",

    "window_size":
        WINDOW,

    "representation_grid":
        [GRID, GRID],

    "total_detector_increments":
        increment_total,

    "evaluated_windows":
        len(history),

    "state_counts":
        counts,

    "history":
        history,

    "scientific_boundary": (
        "This experiment performs measurement-distribution "
        "state tracking on real Glasgow detector data. "
        "Cosine similarity and Jensen-Shannon divergence are "
        "computed relative to the final accumulated detector "
        "distribution. No CTI, detector-noise, hardware-health, "
        "or causal degradation parameter is inferred."
    ),
}


out = (
    Path("experiments")
    / "qms_experimental_twin_001"
    / "evidence"
    / "qms_experimental_twin_001_glasgow.json"
)

out.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)


print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print()
print(
    "Detector increments:",
    increment_total
)

print(
    "Evaluated windows:",
    len(history)
)

print(
    "State counts:",
    counts
)

print()
print("Evidence:", out)
