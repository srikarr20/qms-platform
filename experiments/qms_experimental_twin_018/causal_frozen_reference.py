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

OUT = Path(
    "experiments/qms_experimental_twin_018/"
    "evidence/qms_experimental_twin_018_causal_frozen_reference.json"
)

GRID = 32
WINDOW = 100

# Reference uses only early data.
REFERENCE_END = 399

# Calibration of innovation threshold occurs after the
# reference is frozen, but still before the target region.
CALIBRATION_END = 699


archive = GlasgowCumulativeArchive(GLASGOW_ZIP)


increments = {}

for record in archive.iter_increments():
    increments[int(record["index"])] = np.asarray(
        record["increment"],
        dtype=float,
    )


def aggregate(start, end):
    arrays = [
        increments[i]
        for i in range(start, end + 1)
        if i in increments
    ]

    if len(arrays) != end - start + 1:
        raise RuntimeError(
            f"Incomplete interval {start}:{end}"
        )

    return np.sum(
        np.stack(arrays),
        axis=0,
    )


def distribution(frame):
    h, w = frame.shape

    bh = h // GRID
    bw = w // GRID

    reduced = frame.reshape(
        GRID, bh, GRID, bw
    ).sum(axis=(1, 3))

    p = reduced.ravel().astype(float)

    total = p.sum()

    if total <= 0:
        raise RuntimeError("Zero-count distribution.")

    return p / total


def cosine(a, b):
    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )
    )


def js(p, q):
    eps = 1e-15

    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)

    return float(
        0.5 * np.sum(
            p * np.log(p / m)
        )
        +
        0.5 * np.sum(
            q * np.log(q / m)
        )
    )


# ------------------------------------------------------------
# CAUSAL REFERENCE
#
# Uses only increments 0..399.
# No later detector information enters this representation.
# ------------------------------------------------------------

reference_frame = aggregate(
    0,
    REFERENCE_END,
)

reference = distribution(
    reference_frame
)


# ------------------------------------------------------------
# Sequential observable states
# ------------------------------------------------------------

states = []

first_end = REFERENCE_END + WINDOW

for end in range(
    first_end,
    max(increments) + 1
):

    start = end - WINDOW + 1

    frame = aggregate(
        start,
        end,
    )

    p = distribution(frame)

    states.append({
        "window_start":
            start,

        "window_end":
            end,

        "cosine":
            cosine(
                p,
                reference
            ),

        "js":
            js(
                p,
                reference
            ),
    })


# ------------------------------------------------------------
# One-step prediction using ONLY previous two states.
# ------------------------------------------------------------

predictions = []

for i in range(2, len(states)):

    a = states[i - 2]
    b = states[i - 1]
    c = states[i]

    pred_cos = (
        b["cosine"]
        +
        (
            b["cosine"]
            - a["cosine"]
        )
    )

    pred_js = (
        b["js"]
        +
        (
            b["js"]
            - a["js"]
        )
    )

    dcos = (
        c["cosine"]
        - pred_cos
    )

    djs = (
        c["js"]
        - pred_js
    )

    innovation = float(
        np.sqrt(
            dcos ** 2
            +
            djs ** 2
        )
    )

    predictions.append({
        "window_end":
            c["window_end"],

        "observed_cosine":
            c["cosine"],

        "predicted_cosine":
            float(pred_cos),

        "observed_js":
            c["js"],

        "predicted_js":
            float(pred_js),

        "cosine_innovation":
            float(dcos),

        "js_innovation":
            float(djs),

        "innovation_norm":
            innovation,
    })


# ------------------------------------------------------------
# Freeze threshold using only observations through 699.
# ------------------------------------------------------------

calibration = np.asarray([
    r["innovation_norm"]
    for r in predictions
    if r["window_end"] <= CALIBRATION_END
], dtype=float)

if len(calibration) < 20:
    raise RuntimeError(
        f"Too few calibration points: {len(calibration)}"
    )

median = float(
    np.median(calibration)
)

mad = float(
    np.median(
        np.abs(
            calibration - median
        )
    )
)

p99 = float(
    np.percentile(
        calibration,
        99
    )
)

robust_threshold = float(
    median
    +
    6.0
    * 1.4826
    * mad
)

threshold = max(
    p99,
    robust_threshold
)


# ------------------------------------------------------------
# Strict future evaluation
# ------------------------------------------------------------

future = []

for r in predictions:

    if r["window_end"] <= CALIBRATION_END:
        continue

    z = (
        (
            r["innovation_norm"]
            - median
        )
        /
        (
            1.4826 * mad
        )
        if mad > 0
        else None
    )

    row = dict(r)

    row[
        "robust_z_frozen"
    ] = (
        None
        if z is None
        else float(z)
    )

    row[
        "flagged"
    ] = bool(
        r["innovation_norm"]
        > threshold
    )

    future.append(row)


flagged = [
    r for r in future
    if r["flagged"]
]


# Examine points near previous events.
targets = {}

for target in (701, 801):

    nearest = min(
        future,
        key=lambda r:
            abs(
                r["window_end"]
                - target
            )
    )

    targets[str(target)] = nearest


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-018",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "design":
        "strict causal frozen-reference prospective evaluation",

    "reference": {
        "increment_start":
            0,

        "increment_end":
            REFERENCE_END,

        "future_data_used":
            False,
    },

    "calibration": {
        "last_window":
            CALIBRATION_END,

        "samples":
            int(len(calibration)),

        "median":
            median,

        "mad":
            mad,

        "p99":
            p99,

        "robust_6mad_threshold":
            robust_threshold,

        "frozen_threshold":
            threshold,
    },

    "future_test": {
        "samples":
            len(future),

        "flags":
            len(flagged),

        "flag_fraction":
            (
                len(flagged)
                / len(future)
                if future
                else 0.0
            ),
    },

    "targets":
        targets,

    "future_results":
        future,

    "flagged_events":
        flagged,

    "first_flags":
        flagged[:30],

    "scientific_boundary": (
        "Both the measurement reference and innovation threshold "
        "are frozen using data preceding the prospective test "
        "region. One-step predictions use only prior observable "
        "states. Flags therefore contain no final-distribution "
        "reference leakage. Sliding windows still overlap and "
        "successive flags are therefore temporally dependent. "
        "Flags indicate unexpected measurement-distribution "
        "evolution only, not physical cause, detector failure, "
        "degradation, hardware health, or quantum-field dynamics."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-018 ===")
print()

print("Causal reference:")
print(
    " increments:",
    f"0-{REFERENCE_END}"
)
print(
    " future data used:",
    False
)

print()

print("Frozen calibration:")
print(
    " samples:",
    len(calibration)
)
print(
    " last window:",
    CALIBRATION_END
)
print(
    " median:",
    f"{median:.8f}"
)
print(
    " MAD:",
    f"{mad:.8f}"
)
print(
    " P99:",
    f"{p99:.8f}"
)
print(
    " threshold:",
    f"{threshold:.8f}"
)

print()

print("Strict prospective test:")
print(
    " samples:",
    len(future)
)
print(
    " flags:",
    len(flagged)
)

print()

for target in (701, 801):

    r = targets[str(target)]

    print(
        "TARGET",
        target,
        "nearest window=",
        r["window_end"]
    )

    print(
        " innovation:",
        f"{r['innovation_norm']:.8f}"
    )

    print(
        " frozen z:",
        f"{r['robust_z_frozen']:.3f}"
    )

    print(
        " FLAGGED:",
        r["flagged"]
    )

    print()


print("First causal prospective flags:")

for r in flagged[:20]:

    print(
        " window=",
        r["window_end"],
        "innovation=",
        f"{r['innovation_norm']:.8f}",
        "z=",
        f"{r['robust_z_frozen']:.3f}"
    )

print()
print("Evidence:", OUT)
