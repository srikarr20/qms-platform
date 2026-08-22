from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from qms_core.representation import effective_dimension


WINDOW_FRAMES = 100


def frame_from_zip(zf, name):
    raw = zf.read(name)

    return np.loadtxt(
        io.BytesIO(raw)
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
    if len(X) < 3:
        return None

    centered = X - np.mean(
        X,
        axis=0,
    )

    cov = np.cov(
        centered,
        rowvar=False,
    )

    eig = np.sort(
        np.linalg.eigvalsh(cov)
    )[::-1]

    total = float(np.sum(eig))

    if total <= 0:
        return None

    first_pc_fraction = float(
        eig[0] / total
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

    records = []

    with zipfile.ZipFile(args.zip) as zf:
        names = sorted(
            name
            for name in zf.namelist()
            if name.lower().endswith(".asc")
        )

        print(
            "Raw detector frames:",
            len(names),
        )

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
            event_counts = []

            for name in selected:
                frame = frame_from_zip(
                    zf,
                    name,
                )

                coords = event_coordinates(
                    frame
                )

                event_counts.append(
                    len(coords)
                )

                if len(coords):
                    clouds.append(coords)

            if not clouds:
                continue

            X = np.vstack(clouds)

            metrics = geometry_metrics(X)

            if metrics is None:
                continue

            row = {
                "window_index":
                    len(records),
                "frame_start":
                    start + 1,
                "frame_end":
                    start + len(selected),
                "n_frames":
                    len(selected),
                "n_events":
                    len(X),
                "mean_events_per_frame":
                    float(
                        np.mean(event_counts)
                    ),
                **metrics,
            }

            records.append(row)

            print(
                json.dumps(
                    row,
                    indent=2,
                )
            )

    metric_names = [
        "effective_dimension",
        "first_pc_fraction",
        "radial_std",
        "radial_p99",
    ]

    summary = {}

    for metric in metric_names:
        values = np.asarray(
            [
                r[metric]
                for r in records
            ],
            dtype=float,
        )

        summary[metric] = {
            "mean":
                float(np.mean(values)),
            "std":
                float(np.std(values)),
            "min":
                float(np.min(values)),
            "max":
                float(np.max(values)),
            "coefficient_of_variation":
                float(
                    np.std(values)
                    / abs(np.mean(values))
                )
                if np.mean(values) != 0
                else None,
        }

    output = {
        "dataset":
            "Glasgow Heralded Diffraction SM",
        "source_level":
            "raw 512x512 detector ASC frames",
        "diagnostic_mode":
            "label-free photon-event coordinate geometry",
        "window_frames":
            args.window_frames,
        "n_windows":
            len(records),
        "windows":
            records,
        "summary":
            summary,
        "interpretation_note": (
            "The four geometry diagnostics were fixed before "
            "examining Glasgow results based on the prior ISTA analysis. "
            "No QND-like ground-truth quality metric is available here, "
            "so this experiment tests cross-hardware applicability and "
            "temporal stability, not prediction of measurement fidelity."
        ),
    }

    outdir = Path(
        "experiments/qms_real_006/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_006_glasgow_raw.json"
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
