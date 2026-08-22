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


def histogram_from_events(X, grid=GRID):
    hist, _, _ = np.histogram2d(
        X[:, 1],
        X[:, 0],
        bins=grid,
        range=[
            [0, 512],
            [0, 512],
        ],
    )

    p = hist.ravel().astype(float)

    total = np.sum(p)

    if total > 0:
        p /= total

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
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p + EPS
    q = q + EPS

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
        "--window-frames",
        type=int,
        default=WINDOW_FRAMES,
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
            args.window_frames,
        ):
            selected = names[
                start:
                start + args.window_frames
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

    final_distribution = (
        histogram_from_events(
            final_cloud
        )
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

    window_index = [
        r["window_index"]
        for r in results
    ]

    cosine = [
        r["cosine_to_final"]
        for r in results
    ]

    jsd = [
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

    radial_std = [
        r["radial_std"]
        for r in results
    ]

    radial_p99 = [
        r["radial_p99"]
        for r in results
    ]

    summary = {
        "window_cosine_correlation":
            pearson(
                window_index,
                cosine,
            ),
        "window_js_divergence_correlation":
            pearson(
                window_index,
                jsd,
            ),
        "effective_dimension_vs_cosine":
            pearson(
                ed,
                cosine,
            ),
        "first_pc_fraction_vs_cosine":
            pearson(
                pc1,
                cosine,
            ),
        "radial_std_vs_cosine":
            pearson(
                radial_std,
                cosine,
            ),
        "radial_p99_vs_cosine":
            pearson(
                radial_p99,
                cosine,
            ),
        "effective_dimension_vs_js":
            pearson(
                ed,
                jsd,
            ),
        "first_pc_fraction_vs_js":
            pearson(
                pc1,
                jsd,
            ),
    }

    output = {
        "dataset":
            "Glasgow Heralded Diffraction SM",
        "source_level":
            "raw 512x512 ASC detector frames",
        "window_frames":
            args.window_frames,
        "distribution_grid":
            GRID,
        "reference":
            "normalized detector distribution accumulated across all 4070 frames",
        "windows":
            results,
        "summary":
            summary,
        "interpretation_note": (
            "This experiment distinguishes representation change "
            "from convergence toward the final detector distribution. "
            "Higher cosine similarity and lower Jensen-Shannon divergence "
            "indicate greater convergence. Geometry metrics remain unchanged "
            "from prior QMS experiments."
        ),
    }

    outdir = Path(
        "experiments/qms_real_007/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir
        / "qms_real_007_convergence.json"
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
