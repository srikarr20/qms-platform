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
    "experiments/qms_experimental_twin_021/"
    "evidence/qms_experimental_twin_021_baseline_robustness.json"
)

GRID = 32
WINDOW = 100
REFRACTORY = 100

REFERENCE_ENDS = [
    199,
    299,
    399,
]

CALIBRATION_ENDS = [
    599,
    699,
]


archive = GlasgowCumulativeArchive(
    GLASGOW_ZIP
)

increments = {}

for record in archive.iter_increments():

    increments[int(record["index"])] = np.asarray(
        record["increment"],
        dtype=float,
    )


MAX_INCREMENT = max(increments)


def aggregate(start, end):

    return np.sum(
        np.stack([
            increments[i]
            for i in range(start, end + 1)
        ]),
        axis=0,
    )


def distribution(frame):

    h, w = frame.shape

    bh = h // GRID
    bw = w // GRID

    reduced = frame.reshape(
        GRID,
        bh,
        GRID,
        bw,
    ).sum(axis=(1, 3))

    p = reduced.ravel().astype(float)

    return p / p.sum()


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


def run_condition(
    reference_end,
    calibration_end,
):

    reference = distribution(
        aggregate(
            0,
            reference_end,
        )
    )


    states = []

    first_end = (
        reference_end
        + WINDOW
    )

    for end in range(
        first_end,
        MAX_INCREMENT + 1,
    ):

        start = (
            end
            - WINDOW
            + 1
        )

        p = distribution(
            aggregate(
                start,
                end,
            )
        )

        states.append({
            "end":
                end,

            "cos":
                cosine(
                    p,
                    reference,
                ),

            "js":
                js(
                    p,
                    reference,
                ),
        })


    predictions = []

    for i in range(
        2,
        len(states),
    ):

        a = states[i - 2]
        b = states[i - 1]
        c = states[i]

        pred_cos = (
            b["cos"]
            +
            (
                b["cos"]
                - a["cos"]
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

        dc = (
            c["cos"]
            - pred_cos
        )

        dj = (
            c["js"]
            - pred_js
        )

        innovation = float(
            np.sqrt(
                dc ** 2
                +
                dj ** 2
            )
        )

        predictions.append({
            "window":
                c["end"],

            "innovation":
                innovation,
        })


    calibration = np.asarray([
        r["innovation"]
        for r in predictions
        if r["window"] <= calibration_end
    ], dtype=float)


    if len(calibration) < 20:

        raise RuntimeError(
            "Insufficient calibration data for "
            f"reference={reference_end}, "
            f"calibration={calibration_end}"
        )


    median = float(
        np.median(calibration)
    )

    mad = float(
        np.median(
            np.abs(
                calibration
                - median
            )
        )
    )

    p99 = float(
        np.percentile(
            calibration,
            99,
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
        robust_threshold,
    )


    future = [
        r
        for r in predictions
        if r["window"] > calibration_end
    ]

    flags = [
        r
        for r in future
        if r["innovation"] > threshold
    ]


    # ---------------------------------------------
    # Consolidate at one full measurement window.
    # ---------------------------------------------

    episodes = []
    current = None

    for event in flags:

        w = event["window"]

        if current is None:

            current = {
                "start": w,
                "end": w,
                "events": [event],
            }

            continue


        if (
            w
            - current["end"]
            <= REFRACTORY
        ):

            current["events"].append(
                event
            )

            current["end"] = w

        else:

            episodes.append(
                current
            )

            current = {
                "start": w,
                "end": w,
                "events": [event],
            }


    if current is not None:
        episodes.append(current)


    episode_summary = []

    for ep in episodes:

        peak = max(
            ep["events"],
            key=lambda r:
                r["innovation"],
        )

        episode_summary.append({
            "start":
                ep["start"],

            "end":
                ep["end"],

            "peak_window":
                peak["window"],

            "peak_innovation":
                peak["innovation"],

            "flag_count":
                len(ep["events"]),
        })


    top_flags = sorted(
        flags,
        key=lambda r:
            r["innovation"],
        reverse=True,
    )[:10]


    return {
        "reference_end":
            reference_end,

        "calibration_end":
            calibration_end,

        "calibration_samples":
            len(calibration),

        "threshold":
            threshold,

        "future_samples":
            len(future),

        "raw_flags":
            len(flags),

        "episode_count":
            len(episode_summary),

        "episodes":
            episode_summary,

        "episode_peaks": [
            e["peak_window"]
            for e in episode_summary
        ],

        "top_flag_windows": [
            r["window"]
            for r in top_flags
        ],
    }


results = []

for ref_end in REFERENCE_ENDS:

    for cal_end in CALIBRATION_ENDS:

        if cal_end <= ref_end + WINDOW:
            continue

        result = run_condition(
            ref_end,
            cal_end,
        )

        results.append(
            result
        )


# ------------------------------------------------------------
# Peak recurrence
# ------------------------------------------------------------

peak_counts = {}

for result in results:

    for peak in result[
        "episode_peaks"
    ]:

        peak_counts[peak] = (
            peak_counts.get(
                peak,
                0,
            )
            + 1
        )


recurring = sorted(
    [
        {
            "window":
                int(k),

            "conditions":
                int(v),

            "fraction":
                float(
                    v / len(results)
                ),
        }
        for k, v
        in peak_counts.items()
    ],
    key=lambda r:
        (
            -r["conditions"],
            r["window"],
        ),
)


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-021",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "design":
        "strict causal reference/calibration robustness",

    "window_size":
        WINDOW,

    "refractory":
        REFRACTORY,

    "conditions":
        len(results),

    "results":
        results,

    "peak_recurrence":
        recurring,

    "scientific_boundary": (
        "This experiment varies only causal reference and "
        "calibration periods while preserving future-only "
        "evaluation. Recurring prospective alert locations "
        "indicate robustness to these analysis choices, not "
        "physical mechanism, hardware failure, degradation, "
        "or causal attribution."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-021 ===")
print()


for r in results:

    print(
        "reference=",
        r["reference_end"],
        "calibration=",
        r["calibration_end"],
        "threshold=",
        f"{r['threshold']:.8f}",
        "flags=",
        r["raw_flags"],
        "episodes=",
        r["episode_count"],
    )

    print(
        " peaks:",
        r["episode_peaks"],
    )


print()
print("Recurring episode peaks:")

for r in recurring[:20]:

    print(
        " window=",
        r["window"],
        "conditions=",
        f"{r['conditions']}/{len(results)}",
        "fraction=",
        f"{100*r['fraction']:.1f}%"
    )


print()
print("Evidence:", OUT)
