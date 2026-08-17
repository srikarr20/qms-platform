from collections import deque
from pathlib import Path
import os

import numpy as np

from adapters.glasgow_event_adapter import (
    GlasgowCumulativeArchive,
)

from twin import (
    build_observability_layer,
)

from qms_core import (
    PlatformTwinState,
)

from adapters.aurora_dynamics_adapter import (
    enrich_with_aurora_dynamics,
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

WINDOW = 64


archive = GlasgowCumulativeArchive(
    GLASGOW_ZIP
)


# ============================================================
# STREAM REAL ACQUISITION INCREMENTS
# ============================================================

window = deque(
    maxlen=WINDOW
)

total_weight = 0.0
increment_count = 0

last_state = None


print()
print("=" * 80)
print("QMS PLATFORM — GLASGOW REAL EVENT STREAM")
print("=" * 80)


for record in archive.iter_increments():

    increment = record[
        "increment"
    ]

    window.append(
        increment
    )

    total_weight += record[
        "added_count"
    ]

    increment_count += 1


    # --------------------------------------------------------
    # Once enough acquisition states are available,
    # build a rolling detector evolution.
    #
    # IMPORTANT:
    # These are real detector increments, not reconstructed
    # upstream complex fields.
    # --------------------------------------------------------

    if len(window) >= 5:

        sequence = np.asarray(
            window
        )

        detector_state, manifold = (
            build_observability_layer(
                sequence,

                field_domain=
                    "glasgow_detector_event_increment",
            )
        )

        state = PlatformTwinState(
            measurement=None,

            detector_state=
                detector_state,

            manifold=
                manifold,

            version=
                increment_count,

            metadata={
                "dataset":
                    "Glasgow Heralded Diffraction SM",

                "data_status":
                    "real experimental detector data",

                "representation":
                    "recovered cumulative-frame increments",

                "upstream_reconstruction":
                    False,

                "rolling_window":
                    WINDOW,
            },
        )


        if (
            manifold.state.shape[0]
            >= 4
        ):
            state = (
                enrich_with_aurora_dynamics(
                    state
                )
            )

        last_state = state


    if (
        increment_count <= 5
        or
        increment_count % 500 == 0
    ):
        print(
            f"step={increment_count:04d}"
            f"  added={record['added_count']:6.1f}"
            f"  changed={record['changed_pixels']:4d}"
            f"  window={len(window):2d}"
        )


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 80)
print("GLASGOW REAL-DATA SUMMARY")
print("=" * 80)

print(
    "Recovered acquisition increments:",
    increment_count
)

print(
    "Total positive detector-event weight:",
    total_weight
)

print(
    "Rolling history:",
    len(window)
)

print(
    "Upstream reconstruction performed:",
    False
)


if last_state is not None:

    print(
        "Final detector state:",
        last_state.detector_state.data.shape
    )

    print(
        "Final observable manifold:",
        last_state.manifold.state.shape
    )

    print(
        "Dynamics ready:",
        last_state.dynamics is not None
    )

    if last_state.dynamics is not None:

        print(
            "AURORA drift:",
            last_state.dynamics.trajectory[
                "drift"
            ]
        )

        print(
            "AURORA attractor distortion:",
            last_state.dynamics.attractor[
                "distortion_score"
            ]
        )

        print(
            "AURORA mean local instability:",
            float(
                np.mean(
                    last_state.dynamics.instability
                )
            )
        )


print()
print(
    "Real-data interpretation:"
)

print(
    "Glasgow detector events"
    " -> temporal detector evolution"
    " -> observable manifold"
    " -> AURORA dynamics"
)

print(
    "No phase-aware DPI upstream field reconstruction "
    "is claimed for this dataset."
)

print()
print(
    "GLASGOW REAL EVENT PIPELINE OK"
)
