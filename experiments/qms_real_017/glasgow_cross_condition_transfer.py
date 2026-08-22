from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from qms_core.representation import effective_dimension


WINDOW_FRAMES = 100
GRID = 32
EPS = 1e-15


def frame_from_zip(zf, name):
    return np.loadtxt(
        io.BytesIO(zf.read(name))
    )


def event_coordinates(frame):
    y, x = np.nonzero(frame > 0)

    if len(x) == 0:
        return np.empty((0, 2), dtype=float)

    return np.column_stack([
        x.astype(float),
        y.astype(float),
    ])


def geometry_metrics(X):
    centered = X - np.mean(X, axis=0)

    cov = np.cov(
        centered,
        rowvar=False,
    )

    eig = np.sort(
        np.linalg.eigvalsh(cov)
    )[::-1]

    total = float(np.sum(eig))

    first_pc_fraction = (
        float(eig[0] / total)
        if total > 0
        else 0.0
    )

    radius = np.linalg.norm(
        centered,
        axis=1,
    )

    return {
        "effective_dimension":
            effective_dimension(X),

        "first_pc_fraction":
            first_pc_fraction,

        "radial_std":
            float(np.std(radius)),

        "radial_p99":
            float(np.percentile(radius, 99)),
    }


def histogram_from_events(X):
    hist, _, _ = np.histogram2d(
        X[:, 1],
        X[:, 0],
        bins=GRID,
        range=[
            [0, 512],
            [0, 512],
        ],
    )

    p = hist.ravel().astype(float)

    if np.sum(p) > 0:
        p /= np.sum(p)

    return p


def cosine_similarity(p, q):
    denom = (
        np.linalg.norm(p)
        * np.linalg.norm(q)
    )

    if denom == 0:
        return 0.0

    return float(
        np.dot(p, q) / denom
    )


def js_divergence(p, q):
    p = np.asarray(p, dtype=float) + EPS
    q = np.asarray(q, dtype=float) + EPS

    p /= np.sum(p)
    q /= np.sum(q)

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


def pearson(x, y):
    return float(
        np.corrcoef(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )[0, 1]
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--label",
        type=str,
        default="Ghost Diffraction SM",
    )

    args = parser.parse_args()

    windows = []

    with zipfile.ZipFile(args.zip) as zf:
        names = sorted(
            n
            for n in zf.namelist()
            if n.lower().endswith(".asc")
        )

        print(
            "Raw detector frames:",
            len(names),
        )

        all_clouds = []

        for start in range(
            0,
            len(names),
            WINDOW_FRAMES,
        ):
            selected = names[
                start:
                start + WINDOW_FRAMES
            ]

            clouds = []

            for name in selected:
                frame = frame_from_zip(
                    zf,
                    name,
                )

                coords = event_coordinates(
                    frame
                )

                if len(coords):
                    clouds.append(coords)

            if not clouds:
                continue

            X = np.vstack(clouds)

            all_clouds.append(X)

            windows.append({
                "window_index":
                    len(windows),

                "frame_start":
                    start + 1,

                "frame_end":
                    start + len(selected),

                "n_events":
                    len(X),

                "geometry":
                    geometry_metrics(X),

                "distribution":
                    histogram_from_events(X),
            })

    final_cloud = np.vstack(
        all_clouds
    )

    final_distribution = histogram_from_events(
        final_cloud
    )

    results = []

    for row in windows:
        p = row["distribution"]

        result = {
            "window_index":
                row["window_index"],

            "frame_start":
                row["frame_start"],

            "frame_end":
                row["frame_end"],

            "n_events":
                row["n_events"],

            **row["geometry"],

            "cosine_to_final":
                cosine_similarity(
                    p,
                    final_distribution,
                ),

            "js_divergence_to_final":
                js_divergence(
                    p,
                    final_distribution,
                ),
        }

        results.append(result)

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    index = [
        r["window_index"]
        for r in results
    ]

    cosine = [
        r["cosine_to_final"]
        for r in results
    ]

    js = [
        r["js_divergence_to_final"]
        for r in results
    ]

    ed = [
        r["effective_dimension"]
        for r in results
    ]

    pc1 = [
        r["first_pc_fraction"]
        for r in results
    ]

    summary = {
        "n_windows":
            len(results),

        "window_cosine_correlation":
            pearson(
                index,
                cosine,
            ),

        "window_js_correlation":
            pearson(
                index,
                js,
            ),

        "effective_dimension_vs_cosine":
            pearson(
                ed,
                cosine,
            ),

        "effective_dimension_vs_js":
            pearson(
                ed,
                js,
            ),

        "first_pc_fraction_vs_cosine":
            pearson(
                pc1,
                cosine,
            ),

        "first_pc_fraction_vs_js":
            pearson(
                pc1,
                js,
            ),

        "maximum_cosine_window":
            int(
                np.argmax(cosine)
            ),

        "minimum_js_window":
            int(
                np.argmin(js)
            ),

        "effective_dimension_start":
            float(ed[0]),

        "effective_dimension_end":
            float(ed[-1]),

        "first_pc_fraction_start":
            float(pc1[0]),

        "first_pc_fraction_end":
            float(pc1[-1]),
    }

    output = {
        "experiment":
            "QMS-REAL-017",

        "dataset":
            args.label,

        "source_level":
            "raw 512x512 ASC detector frames",

        "window_frames":
            WINDOW_FRAMES,

        "distribution_grid":
            GRID,

        "windows":
            results,

        "summary":
            summary,

        "interpretation_note": (
            "This experiment applies the same fixed "
            "representation and convergence diagnostics "
            "used on Heralded Diffraction SM to an "
            "independent Glasgow acquisition condition."
        ),
    }

    outdir = Path(
        "experiments/qms_real_017/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_017_ghost_diffraction_sm.json"
    )

    outfile.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print("=== SUMMARY ===")

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("Saved:", outfile)


if __name__ == "__main__":
    main()
