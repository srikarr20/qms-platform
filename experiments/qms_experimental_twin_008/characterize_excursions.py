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

TARGETS = [701, 801]
RADIUS = 5

OUT = Path(
    "experiments/qms_experimental_twin_008/"
    "evidence/qms_experimental_twin_008_excursions.json"
)

archive = GlasgowCumulativeArchive(GLASGOW_ZIP)


def geometry(frame):

    frame = np.asarray(frame, dtype=float)

    total = frame.sum()

    if total <= 0:
        return {
            "total_count": 0.0,
            "changed_pixels": 0,
            "centroid_x": None,
            "centroid_y": None,
            "radial_spread": None,
        }

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
        "total_count":
            float(total),

        "changed_pixels":
            int(np.count_nonzero(frame)),

        "centroid_x":
            cx,

        "centroid_y":
            cy,

        "radial_spread":
            spread,
    }


records = {}

for rec in archive.iter_increments():

    idx = int(rec["index"])

    if any(
        abs(idx - target) <= RADIUS
        for target in TARGETS
    ):
        records[idx] = geometry(
            rec["increment"]
        )


results = {}

for target in TARGETS:

    local = []

    for idx in range(
        target - RADIUS,
        target + RADIUS + 1
    ):

        if idx not in records:
            continue

        row = {
            "increment":
                idx,
            **records[idx],
        }

        local.append(row)

    results[str(target)] = local


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-008",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "target_excursions":
        TARGETS,

    "local_radius":
        RADIUS,

    "metrics": [
        "total_count",
        "changed_pixels",
        "centroid_x",
        "centroid_y",
        "radial_spread",
    ],

    "results":
        results,

    "scientific_boundary": (
        "This experiment characterizes detector-measurement "
        "changes around two unusually large innovation events. "
        "It does not assign a physical degradation mechanism "
        "or hardware-failure interpretation."
    ),
}

OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-008 ===")
print()

for target in TARGETS:

    print("TARGET", target)

    for r in results[str(target)]:

        marker = "*" if r["increment"] == target else " "

        print(
            marker,
            f"{r['increment']:4d}",
            "count=",
            f"{r['total_count']:.1f}",
            "pixels=",
            r["changed_pixels"],
            "cx=",
            f"{r['centroid_x']:.3f}",
            "cy=",
            f"{r['centroid_y']:.3f}",
            "spread=",
            f"{r['radial_spread']:.3f}",
        )

    print()

print("Evidence:", OUT)
