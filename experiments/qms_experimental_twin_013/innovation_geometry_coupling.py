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

PRED_SRC = Path(
    "experiments/qms_experimental_twin_003/"
    "evidence/qms_experimental_twin_003_predictive.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_013/"
    "evidence/qms_experimental_twin_013_innovation_geometry.json"
)

WINDOW = 100
GRID = 32

archive = GlasgowCumulativeArchive(GLASGOW_ZIP)

increments = {}

for rec in archive.iter_increments():
    increments[int(rec["index"])] = np.asarray(
        rec["increment"],
        dtype=float,
    )


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
        GRID, bh, GRID, bw
    ).sum(axis=(1, 3))

    p = reduced.ravel().astype(float)
    return p / p.sum()


def geometry(frame):
    total = float(frame.sum())

    y, x = np.indices(frame.shape)

    cx = float(
        np.sum(x * frame) / total
    )

    cy = float(
        np.sum(y * frame) / total
    )

    r2 = (
        (x - cx) ** 2
        + (y - cy) ** 2
    )

    spread = float(
        np.sqrt(
            np.sum(r2 * frame) / total
        )
    )

    return {
        "total_count": total,
        "centroid_x": cx,
        "centroid_y": cy,
        "radial_spread": spread,
    }


def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return np.nan

    return float(
        np.dot(a, b) / (na * nb)
    )


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if np.sum(mask) < 3:
        return np.nan

    return float(
        np.corrcoef(
            x[mask],
            y[mask]
        )[0, 1]
    )


# ------------------------------------------------------------
# Build non-overlapping 100-frame measurement states
# ------------------------------------------------------------

states = []

start = 502

while start + WINDOW - 1 <= max(increments):

    end = start + WINDOW - 1

    if all(
        i in increments
        for i in range(start, end + 1)
    ):
        frame = aggregate(start, end)

        states.append({
            "start": start,
            "end": end,
            "frame": frame,
            "p": distribution(frame),
            "g": geometry(frame),
        })

    start += WINDOW


# ------------------------------------------------------------
# Load predictive innovations and map by window endpoint
# ------------------------------------------------------------

pred_data = json.loads(
    PRED_SRC.read_text()
)

innovation_by_end = {
    int(r["window_end_increment"]):
        float(r["innovation_norm"])
    for r in pred_data["results"]
}


rows = []

previous_delta = None


for a, b in zip(
    states[:-1],
    states[1:]
):

    delta_p = b["p"] - a["p"]

    l2 = float(
        np.linalg.norm(delta_p)
    )

    tv = float(
        0.5 * np.sum(
            np.abs(delta_p)
        )
    )

    dx = (
        b["g"]["centroid_x"]
        - a["g"]["centroid_x"]
    )

    dy = (
        b["g"]["centroid_y"]
        - a["g"]["centroid_y"]
    )

    centroid_shift = float(
        np.sqrt(
            dx ** 2 + dy ** 2
        )
    )

    spread_change = float(
        abs(
            b["g"]["radial_spread"]
            - a["g"]["radial_spread"]
        )
    )

    count_change_fraction = float(
        abs(
            b["g"]["total_count"]
            - a["g"]["total_count"]
        )
        /
        a["g"]["total_count"]
    )

    if previous_delta is None:
        reversal = np.nan
    else:
        reversal = cosine(
            previous_delta,
            -delta_p
        )

    innovation = innovation_by_end.get(
        b["end"]
    )

    rows.append({
        "window_end":
            b["end"],

        "innovation_norm":
            innovation,

        "transition_l2":
            l2,

        "total_variation":
            tv,

        "centroid_shift":
            centroid_shift,

        "spread_change":
            spread_change,

        "count_change_fraction":
            count_change_fraction,

        "reversal_cosine":
            reversal,
    })

    previous_delta = delta_p


# keep only rows with predictive innovation available
rows = [
    r for r in rows
    if r["innovation_norm"] is not None
]


metrics = [
    "transition_l2",
    "total_variation",
    "centroid_shift",
    "spread_change",
    "count_change_fraction",
    "reversal_cosine",
]


correlations = {}

innovation = [
    r["innovation_norm"]
    for r in rows
]

for metric in metrics:

    correlations[metric] = corr(
        innovation,
        [
            r[metric]
            for r in rows
        ]
    )


ranked = sorted(
    [
        {
            "metric": k,
            "correlation":
                v,
            "absolute_correlation":
                abs(v)
                if np.isfinite(v)
                else None,
        }
        for k, v in correlations.items()
    ],
    key=lambda r:
        -1
        if r["absolute_correlation"] is None
        else r["absolute_correlation"],
    reverse=True,
)


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-013",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "window_size":
        WINDOW,

    "samples":
        len(rows),

    "correlations_with_innovation":
        correlations,

    "ranked_metrics":
        ranked,

    "results":
        rows,

    "scientific_boundary": (
        "Correlations describe association between "
        "one-step predictive innovation and observable "
        "measurement-transition geometry in this Glasgow "
        "sequence. They do not establish causality, detector "
        "failure, degradation mechanism, or hardware health."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-013 ===")
print()

print(
    "Matched transitions:",
    len(rows)
)

print()
print("Correlation with innovation norm:")

for r in ranked:
    print(
        f" {r['metric']:24s}",
        "=",
        "NA"
        if r["correlation"] is None
        else f"{r['correlation']:.6f}"
    )

print()
print("Top innovation transitions:")

top = sorted(
    rows,
    key=lambda r: r["innovation_norm"],
    reverse=True,
)[:10]

for r in top:
    print(
        " end=",
        r["window_end"],
        "innovation=",
        f"{r['innovation_norm']:.8f}",
        "TV=",
        f"{r['total_variation']:.6f}",
        "centroid=",
        f"{r['centroid_shift']:.3f}",
        "spread=",
        f"{r['spread_change']:.3f}",
        "countΔ=",
        f"{100*r['count_change_fraction']:.2f}%",
        "reversal=",
        "NA"
        if not np.isfinite(r["reversal_cosine"])
        else f"{r['reversal_cosine']:.3f}"
    )

print()
print("Evidence:", OUT)
