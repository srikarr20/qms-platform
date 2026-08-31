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

OUT = Path(
    "experiments/qms_experimental_twin_012/"
    "evidence/qms_experimental_twin_012_reversal_baseline.json"
)

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


def cosine(a, b):

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return np.nan

    return float(
        np.dot(a, b) / (na * nb)
    )


# ------------------------------------------------------------
# Build consecutive non-overlapping 100-increment states.
#
# Match the region used in TWIN-011:
# 502-601, 602-701, 702-801, ...
# ------------------------------------------------------------

states = []

start = 502

while start + WINDOW - 1 <= max(increments):

    end = start + WINDOW - 1

    if all(
        i in increments
        for i in range(start, end + 1)
    ):
        states.append({
            "start": start,
            "end": end,
            "p": distribution(
                aggregate(start, end)
            ),
        })

    start += WINDOW


transitions = []

for a, b in zip(
    states[:-1],
    states[1:]
):

    transitions.append({
        "from": [a["start"], a["end"]],
        "to": [b["start"], b["end"]],
        "delta": b["p"] - a["p"],
    })


pairs = []

for i in range(len(transitions) - 1):

    d1 = transitions[i]["delta"]
    d2 = transitions[i + 1]["delta"]

    reversal = cosine(
        d1,
        -d2
    )

    direct = cosine(
        d1,
        d2
    )

    reverse_projection = float(
        np.dot(d2, -d1)
        / np.dot(d1, d1)
    )

    nz = (
        (np.abs(d1) > 1e-15)
        &
        (np.abs(d2) > 1e-15)
    )

    weights = (
        np.abs(d1[nz])
        +
        np.abs(d2[nz])
    )

    signs_reverse = (
        np.sign(d1[nz])
        ==
        -np.sign(d2[nz])
    )

    weighted_reversal = float(
        np.sum(
            weights[signs_reverse]
        )
        / np.sum(weights)
    )

    pairs.append({
        "transition_1":
            transitions[i]["to"],

        "transition_2":
            transitions[i + 1]["to"],

        "reversal_cosine":
            reversal,

        "direct_cosine":
            direct,

        "reverse_projection_fraction":
            reverse_projection,

        "magnitude_weighted_reversal":
            weighted_reversal,
    })


# ------------------------------------------------------------
# Locate the previously studied pair:
# A=502-601
# B=602-701
# C=702-801
# ------------------------------------------------------------

target = None

for r in pairs:

    if (
        r["transition_1"] == [602, 701]
        and
        r["transition_2"] == [702, 801]
    ):
        target = r
        break

if target is None:
    raise RuntimeError(
        "Target 701/801 transition pair not found."
    )


rev = np.asarray(
    [r["reversal_cosine"] for r in pairs],
    dtype=float,
)

weighted = np.asarray(
    [r["magnitude_weighted_reversal"] for r in pairs],
    dtype=float,
)


rev_percentile = float(
    100.0 * np.mean(
        rev <= target["reversal_cosine"]
    )
)

weighted_percentile = float(
    100.0 * np.mean(
        weighted
        <= target["magnitude_weighted_reversal"]
    )
)


ranked = sorted(
    pairs,
    key=lambda r: r["reversal_cosine"],
    reverse=True,
)


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-012",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "window_size":
        WINDOW,

    "state_count":
        len(states),

    "transition_pair_count":
        len(pairs),

    "target_pair":
        target,

    "target_reversal_cosine_percentile":
        rev_percentile,

    "target_weighted_reversal_percentile":
        weighted_percentile,

    "reversal_cosine_distribution": {
        "mean":
            float(np.nanmean(rev)),

        "median":
            float(np.nanmedian(rev)),

        "p90":
            float(np.nanpercentile(rev, 90)),

        "p95":
            float(np.nanpercentile(rev, 95)),

        "max":
            float(np.nanmax(rev)),
    },

    "top_reversals":
        ranked[:10],

    "scientific_boundary": (
        "This analysis compares the 701/801 transition geometry "
        "against other consecutive non-overlapping measurement-state "
        "transitions in the same Glasgow sequence. It evaluates "
        "whether partial reversal is unusual in observable detector "
        "space; it does not establish a physical mechanism or causal "
        "state transition."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-012 ===")
print()

print(
    "Non-overlapping states:",
    len(states)
)

print(
    "Consecutive transition pairs:",
    len(pairs)
)

print()

print("Target 701 -> 801 pair")

print(
    " reversal cosine:",
    f"{target['reversal_cosine']:.6f}"
)

print(
    " weighted reversal:",
    f"{100*target['magnitude_weighted_reversal']:.2f}%"
)

print(
    " reversal percentile:",
    f"{rev_percentile:.2f}%"
)

print(
    " weighted percentile:",
    f"{weighted_percentile:.2f}%"
)

print()

print("Sequence reversal distribution:")

print(
    " mean:",
    f"{np.nanmean(rev):.6f}"
)

print(
    " median:",
    f"{np.nanmedian(rev):.6f}"
)

print(
    " P90:",
    f"{np.nanpercentile(rev,90):.6f}"
)

print(
    " P95:",
    f"{np.nanpercentile(rev,95):.6f}"
)

print(
    " max:",
    f"{np.nanmax(rev):.6f}"
)

print()
print("Top reversal pairs:")

for r in ranked[:10]:

    print(
        r["transition_1"],
        "->",
        r["transition_2"],
        "cos=",
        f"{r['reversal_cosine']:.6f}",
        "weighted=",
        f"{100*r['magnitude_weighted_reversal']:.2f}%"
    )

print()
print("Evidence:", OUT)
