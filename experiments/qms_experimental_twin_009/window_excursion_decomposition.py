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
    "experiments/qms_experimental_twin_009/"
    "evidence/qms_experimental_twin_009_window_decomposition.json"
)

archive = GlasgowCumulativeArchive(GLASGOW_ZIP)


increments = {}

for rec in archive.iter_increments():
    increments[int(rec["index"])] = np.asarray(
        rec["increment"],
        dtype=float,
    )


def aggregate(start, end):

    arrays = [
        increments[i]
        for i in range(start, end + 1)
        if i in increments
    ]

    if len(arrays) != (end - start + 1):
        raise RuntimeError(
            f"Incomplete window {start}:{end}"
        )

    return np.sum(
        np.stack(arrays),
        axis=0,
    )


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
        "changed_pixels":
            int(np.count_nonzero(frame)),
        "centroid_x": cx,
        "centroid_y": cy,
        "radial_spread": spread,
    }


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

    p = p / p.sum()
    q = q / q.sum()

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


results = {}


for target in TARGETS:

    # Twin window ending at target:
    current_start = target - WINDOW + 1
    current_end = target

    # Previous non-overlapping 100-frame window:
    previous_start = current_start - WINDOW
    previous_end = current_start - 1

    current = aggregate(
        current_start,
        current_end,
    )

    previous = aggregate(
        previous_start,
        previous_end,
    )

    g_current = geometry(current)
    g_previous = geometry(previous)

    p_current = distribution(current)
    p_previous = distribution(previous)

    comparison = {
        "cosine_previous_to_current":
            cosine(
                p_previous,
                p_current,
            ),

        "js_previous_to_current":
            js(
                p_previous,
                p_current,
            ),

        "delta_total_count":
            (
                g_current["total_count"]
                - g_previous["total_count"]
            ),

        "delta_centroid_x":
            (
                g_current["centroid_x"]
                - g_previous["centroid_x"]
            ),

        "delta_centroid_y":
            (
                g_current["centroid_y"]
                - g_previous["centroid_y"]
            ),

        "delta_radial_spread":
            (
                g_current["radial_spread"]
                - g_previous["radial_spread"]
            ),
    }

    results[str(target)] = {
        "previous_window": {
            "start": previous_start,
            "end": previous_end,
            **g_previous,
        },

        "current_window": {
            "start": current_start,
            "end": current_end,
            **g_current,
        },

        "comparison":
            comparison,
    }


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-009",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "window_size":
        WINDOW,

    "targets":
        TARGETS,

    "results":
        results,

    "scientific_boundary": (
        "This experiment characterizes large predictive "
        "innovations at the same 100-increment aggregation "
        "scale used by the experimental twin. Differences "
        "describe measurement-distribution changes only; "
        "no detector failure or physical degradation "
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
print("=== QMS-EXPERIMENTAL-TWIN-009 ===")
print()


for target in TARGETS:

    r = results[str(target)]

    print("TARGET", target)

    print(
        " previous:",
        f"{r['previous_window']['start']}-"
        f"{r['previous_window']['end']}",
        "counts=",
        f"{r['previous_window']['total_count']:.1f}",
        "cx=",
        f"{r['previous_window']['centroid_x']:.3f}",
        "cy=",
        f"{r['previous_window']['centroid_y']:.3f}",
        "spread=",
        f"{r['previous_window']['radial_spread']:.3f}",
    )

    print(
        " current :",
        f"{r['current_window']['start']}-"
        f"{r['current_window']['end']}",
        "counts=",
        f"{r['current_window']['total_count']:.1f}",
        "cx=",
        f"{r['current_window']['centroid_x']:.3f}",
        "cy=",
        f"{r['current_window']['centroid_y']:.3f}",
        "spread=",
        f"{r['current_window']['radial_spread']:.3f}",
    )

    c = r["comparison"]

    print(
        " cosine(previous,current)=",
        f"{c['cosine_previous_to_current']:.6f}"
    )

    print(
        " JS(previous,current)=",
        f"{c['js_previous_to_current']:.6f}"
    )

    print(
        " delta counts=",
        f"{c['delta_total_count']:.1f}"
    )

    print(
        " delta centroid=",
        f"({c['delta_centroid_x']:.3f}, "
        f"{c['delta_centroid_y']:.3f})"
    )

    print(
        " delta spread=",
        f"{c['delta_radial_spread']:.3f}"
    )

    print()


print("Evidence:", OUT)
