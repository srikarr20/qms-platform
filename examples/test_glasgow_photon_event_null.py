from pathlib import Path
import os
import numpy as np

from adapters.glasgow_event_adapter import (
    GlasgowCumulativeArchive,
)

from twin import (
    build_photon_event_manifold,
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
N_SHUFFLES = 200
RNG = np.random.default_rng(1511)


def dynamics_from_sequence(
    sequence,
):
    manifold = (
        build_photon_event_manifold(
            sequence
        )
    )

    state = PlatformTwinState(
        measurement=None,
        manifold=manifold,
        version=1,
    )

    state = (
        enrich_with_aurora_dynamics(
            state
        )
    )

    return np.array([
        float(
            state.dynamics.trajectory[
                "drift"
            ]
        ),

        float(
            state.dynamics.attractor[
                "distortion_score"
            ]
        ),

        float(
            np.mean(
                state.dynamics.instability
            )
        ),
    ])


archive = GlasgowCumulativeArchive(
    GLASGOW_ZIP
)

increments = np.asarray([
    record["increment"]
    for record in archive.iter_increments()
])


print()
print("=" * 82)
print("QMS PLATFORM — GLASGOW PHOTON-EVENT NULL TEST")
print("=" * 82)

print(
    "Total increments:",
    len(increments)
)

print(
    "Window:",
    WINDOW
)


# Use same final 64-step window first.
sequence = increments[
    -WINDOW:
]

observed = dynamics_from_sequence(
    sequence
)

null = np.zeros(
    (N_SHUFFLES, 3)
)


for i in range(
    N_SHUFFLES
):
    order = RNG.permutation(
        WINDOW
    )

    null[i] = dynamics_from_sequence(
        sequence[order]
    )

    if (
        i < 5
        or
        (i+1) % 50 == 0
    ):
        print(
            f"shuffle={i+1:03d}/{N_SHUFFLES}"
        )


mean = np.mean(
    null,
    axis=0,
)

std = np.std(
    null,
    axis=0,
    ddof=1,
)

z = (
    observed - mean
) / (
    std + 1e-15
)


p = np.zeros(3)

for j in range(3):
    center = mean[j]

    obs_dev = abs(
        observed[j] - center
    )

    null_dev = np.abs(
        null[:, j] - center
    )

    p[j] = (
        1
        + np.count_nonzero(
            null_dev >= obs_dev
        )
    ) / (
        N_SHUFFLES + 1
    )


names = [
    "drift",
    "attractor_distortion",
    "mean_local_instability",
]


print()
print("=" * 82)
print("PHOTON-EVENT NULL RESULTS")
print("=" * 82)


for j, name in enumerate(names):

    print()
    print(name)

    print(
        "  observed:",
        f"{observed[j]:.8f}"
    )

    print(
        "  null mean:",
        f"{mean[j]:.8f}"
    )

    print(
        "  null std:",
        f"{std[j]:.8f}"
    )

    print(
        "  z:",
        f"{z[j]:.4f}"
    )

    print(
        "  empirical p:",
        f"{p[j]:.6f}"
    )


print()
print(
    "PHOTON-EVENT TEMPORAL NULL TEST COMPLETE"
)
