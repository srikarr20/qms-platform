from pathlib import Path
import os
import numpy as np

from adapters.glasgow_event_adapter import (
    GlasgowCumulativeArchive,
)


import os

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


CHECKPOINTS = [
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    30000,
    40000,
]


def normalized_correlation(a, b):
    x = a.ravel().astype(float)
    y = b.ravel().astype(float)

    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
        + 1e-15
    )

    return float(
        np.dot(x, y) / denom
    )


def normalized_rmse(a, b):
    diff = a.astype(float) - b.astype(float)

    rmse = np.sqrt(
        np.mean(diff**2)
    )

    scale = (
        np.sqrt(
            np.mean(
                b.astype(float)**2
            )
        )
        + 1e-15
    )

    return float(
        rmse / scale
    )


def js_divergence(a, b):
    p = a.astype(float).ravel()
    q = b.astype(float).ravel()

    p = p / (
        np.sum(p)
        + 1e-15
    )

    q = q / (
        np.sum(q)
        + 1e-15
    )

    m = 0.5 * (
        p + q
    )

    def kl(x, y):
        mask = x > 0

        return float(
            np.sum(
                x[mask]
                * np.log(
                    (
                        x[mask]
                        + 1e-15
                    )
                    /
                    (
                        y[mask]
                        + 1e-15
                    )
                )
            )
        )

    return float(
        0.5 * kl(p, m)
        + 0.5 * kl(q, m)
    )


print()
print("=" * 80)
print("QMS PLATFORM — GLASGOW REAL-EVENT CONVERGENCE")
print("=" * 80)


archive = GlasgowCumulativeArchive(
    GLASGOW_ZIP
)

increments = []

total_weight = 0

for record in archive.iter_increments():
    increments.append(
        record["increment"]
    )

    total_weight += int(
        record["added_count"]
    )


increments = np.asarray(
    increments
)

print(
    "Acquisition increments:",
    len(increments)
)

print(
    "Total event weight:",
    total_weight
)


# ============================================================
# EXPAND SPARSE COUNTS INTO UNIT EVENTS
#
# Since the Glasgow increments are integer photon counts,
# each pixel contribution is expanded into repeated unit events.
# ============================================================

events = []

for frame in increments:

    ys, xs = np.nonzero(
        frame
    )

    for y, x in zip(
        ys,
        xs,
    ):
        count = int(
            frame[y, x]
        )

        for _ in range(count):
            events.append(
                (
                    int(y),
                    int(x),
                )
            )


print(
    "Expanded unit events:",
    len(events)
)


H, W = increments.shape[1:]

final_image = np.zeros(
    (H, W),
    dtype=float,
)

for y, x in events:
    final_image[
        y,
        x,
    ] += 1.0


# ============================================================
# CHECKPOINT SWEEP
# ============================================================

results = []

running = np.zeros(
    (H, W),
    dtype=float,
)

checkpoint_set = set(
    CHECKPOINTS
)

checkpoint_set.add(
    len(events)
)


for i, (
    y,
    x,
) in enumerate(
    events,
    start=1,
):
    running[
        y,
        x,
    ] += 1.0

    if i in checkpoint_set:

        corr = normalized_correlation(
            running,
            final_image,
        )

        nrmse = normalized_rmse(
            running,
            final_image,
        )

        js = js_divergence(
            running,
            final_image,
        )

        results.append(
            (
                i,
                corr,
                nrmse,
                js,
            )
        )

        print(
            f"{i:6d} events"
            f"  corr={corr:.6f}"
            f"  NRMSE={nrmse:.6f}"
            f"  JS={js:.6f}"
        )


# ============================================================
# THRESHOLDS
# ============================================================

def first_event_count(
    metric_index,
    condition,
):
    for row in results:
        if condition(
            row[metric_index]
        ):
            return row[0]

    return None


corr90 = first_event_count(
    1,
    lambda v: v >= 0.90,
)

corr95 = first_event_count(
    1,
    lambda v: v >= 0.95,
)

corr99 = first_event_count(
    1,
    lambda v: v >= 0.99,
)

js05 = first_event_count(
    3,
    lambda v: v <= 0.05,
)

rmse025 = first_event_count(
    2,
    lambda v: v <= 0.25,
)


print()
print("=" * 80)
print("REAL EVENT CONVERGENCE SUMMARY")
print("=" * 80)

print(
    "Events to correlation >= 0.90:",
    corr90
)

print(
    "Events to correlation >= 0.95:",
    corr95
)

print(
    "Events to correlation >= 0.99:",
    corr99
)

print(
    "Events to JS <= 0.05:",
    js05
)

print(
    "Events to normalized RMSE <= 0.25:",
    rmse025
)


# ============================================================
# SAVE
# ============================================================

Path(
    "artifacts"
).mkdir(
    exist_ok=True
)

OUT = (
    "artifacts/"
    "glasgow_real_event_convergence.npz"
)

np.savez_compressed(
    OUT,

    checkpoints=
        np.asarray(
            [
                r[0]
                for r in results
            ]
        ),

    correlation=
        np.asarray(
            [
                r[1]
                for r in results
            ]
        ),

    nrmse=
        np.asarray(
            [
                r[2]
                for r in results
            ]
        ),

    js=
        np.asarray(
            [
                r[3]
                for r in results
            ]
        ),

    final_image=
        final_image,
)


print()
print(
    "Saved:",
    OUT
)

print()
print(
    "GLASGOW REAL-EVENT CONVERGENCE COMPLETE"
)
