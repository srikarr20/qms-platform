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

WINDOW = 100
GRID = 32

OUT = Path(
    "experiments/qms_experimental_twin_010/"
    "evidence/qms_experimental_twin_010_spatial_localization.json"
)


archive = GlasgowCumulativeArchive(
    GLASGOW_ZIP
)


# ------------------------------------------------------------
# Load real acquisition increments
# ------------------------------------------------------------

increments = {}

for record in archive.iter_increments():

    increments[int(record["index"])] = np.asarray(
        record["increment"],
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


def grid_distribution(frame):

    h, w = frame.shape

    bh = h // GRID
    bw = w // GRID

    reduced = frame.reshape(
        GRID,
        bh,
        GRID,
        bw,
    ).sum(axis=(1, 3))

    total = reduced.sum()

    return reduced / total


results = {}


for target in TARGETS:

    current_start = target - WINDOW + 1
    current_end = target

    previous_start = current_start - WINDOW
    previous_end = current_start - 1


    previous = aggregate(
        previous_start,
        previous_end,
    )

    current = aggregate(
        current_start,
        current_end,
    )


    p_prev = grid_distribution(
        previous
    )

    p_curr = grid_distribution(
        current
    )


    # Signed normalized redistribution.
    delta = p_curr - p_prev

    abs_delta = np.abs(delta)

    total_change = float(
        abs_delta.sum()
    )


    # Rank detector cells by absolute contribution.
    flat_order = np.argsort(
        abs_delta.ravel()
    )[::-1]


    top_cells = []

    cumulative = 0.0


    for flat_idx in flat_order[:20]:

        gy, gx = np.unravel_index(
            flat_idx,
            delta.shape,
        )

        contribution = float(
            abs_delta[gy, gx]
        )

        cumulative += contribution

        top_cells.append({
            "grid_x":
                int(gx),

            "grid_y":
                int(gy),

            "pixel_x_range": [
                int(gx * (512 // GRID)),
                int(
                    (gx + 1)
                    * (512 // GRID)
                    - 1
                ),
            ],

            "pixel_y_range": [
                int(gy * (512 // GRID)),
                int(
                    (gy + 1)
                    * (512 // GRID)
                    - 1
                ),
            ],

            "previous_probability":
                float(
                    p_prev[gy, gx]
                ),

            "current_probability":
                float(
                    p_curr[gy, gx]
                ),

            "signed_change":
                float(
                    delta[gy, gx]
                ),

            "absolute_change":
                contribution,

            "fraction_of_total_absolute_change":
                (
                    contribution / total_change
                    if total_change > 0
                    else 0.0
                ),
        })


    # How concentrated is the redistribution?
    sorted_change = np.sort(
        abs_delta.ravel()
    )[::-1]

    cumulative_change = np.cumsum(
        sorted_change
    )

    def cells_for_fraction(fraction):

        if total_change <= 0:
            return 0

        idx = np.searchsorted(
            cumulative_change,
            fraction * total_change,
        )

        return int(idx + 1)


    results[str(target)] = {

        "previous_window": [
            previous_start,
            previous_end,
        ],

        "current_window": [
            current_start,
            current_end,
        ],

        "total_variation_distance":
            float(
                0.5 * total_change
            ),

        "cells_for_50_percent_change":
            cells_for_fraction(0.50),

        "cells_for_80_percent_change":
            cells_for_fraction(0.80),

        "cells_for_90_percent_change":
            cells_for_fraction(0.90),

        "top_cells":
            top_cells,
    }


summary = {

    "experiment":
        "QMS-EXPERIMENTAL-TWIN-010",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "representation":
        "normalized 32x32 detector distribution",

    "targets":
        TARGETS,

    "results":
        results,

    "scientific_boundary": (
        "This analysis localizes spatial redistribution of "
        "normalized detector-event probability around large "
        "measurement-state innovations. It identifies where "
        "the measured distribution changed, not why it changed. "
        "No detector fault, physical degradation, or causal "
        "mechanism is inferred."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-010 ===")
print()


for target in TARGETS:

    r = results[str(target)]

    print("TARGET", target)

    print(
        " total variation distance:",
        f"{r['total_variation_distance']:.6f}"
    )

    print(
        " cells for 50% of redistribution:",
        r["cells_for_50_percent_change"]
    )

    print(
        " cells for 80% of redistribution:",
        r["cells_for_80_percent_change"]
    )

    print(
        " cells for 90% of redistribution:",
        r["cells_for_90_percent_change"]
    )

    print()
    print(" Top spatial changes:")

    for cell in r["top_cells"][:10]:

        sign = (
            "+"
            if cell["signed_change"] >= 0
            else "-"
        )

        print(
            f"  grid=({cell['grid_x']:2d},"
            f"{cell['grid_y']:2d})",
            f"pixels X={cell['pixel_x_range']}",
            f"Y={cell['pixel_y_range']}",
            f"change={sign}"
            f"{abs(cell['signed_change']):.6f}",
            f"share="
            f"{100*cell['fraction_of_total_absolute_change']:.2f}%"
        )

    print()


print("Evidence:", OUT)
