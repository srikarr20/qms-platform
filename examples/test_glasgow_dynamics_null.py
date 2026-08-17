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
N_SHUFFLES = 200
RNG = np.random.default_rng(1509)


def dynamics_from_sequence(sequence):
    detector_state, manifold = (
        build_observability_layer(
            sequence,
            field_domain=
                "glasgow_detector_event_increment",
        )
    )

    state = PlatformTwinState(
        measurement=None,
        detector_state=detector_state,
        manifold=manifold,
        version=1,
        metadata={
            "dataset":
                "Glasgow Heralded Diffraction SM",
        },
    )

    state = enrich_with_aurora_dynamics(
        state
    )

    drift = float(
        state.dynamics.trajectory[
            "drift"
        ]
    )

    distortion = float(
        state.dynamics.attractor[
            "distortion_score"
        ]
    )

    instability = float(
        np.mean(
            state.dynamics.instability
        )
    )

    return (
        drift,
        distortion,
        instability,
    )


print()
print("=" * 80)
print("QMS PLATFORM — GLASGOW DYNAMICS NULL TEST")
print("=" * 80)

archive = GlasgowCumulativeArchive(
    GLASGOW_ZIP
)

increments = []

for record in archive.iter_increments():
    increments.append(
        record["increment"]
    )

increments = np.asarray(
    increments
)

print(
    "Recovered increments:",
    increments.shape
)

# ------------------------------------------------------------
# Use final 64-step real window, matching previous test.
# ------------------------------------------------------------

observed_sequence = (
    increments[-WINDOW:]
)

(
    observed_drift,
    observed_distortion,
    observed_instability,
) = dynamics_from_sequence(
    observed_sequence
)

print()
print("OBSERVED FINAL WINDOW")
print("-" * 80)

print(
    "Drift:",
    observed_drift
)

print(
    "Attractor distortion:",
    observed_distortion
)

print(
    "Mean local instability:",
    observed_instability
)


# ------------------------------------------------------------
# SHUFFLED ORDER NULL
#
# Same 64 real detector increments.
# Only temporal ordering is destroyed.
# ------------------------------------------------------------

null_drift = np.zeros(
    N_SHUFFLES
)

null_distortion = np.zeros(
    N_SHUFFLES
)

null_instability = np.zeros(
    N_SHUFFLES
)


print()
print(
    "Generating shuffled temporal-order null..."
)

for i in range(N_SHUFFLES):

    order = RNG.permutation(
        WINDOW
    )

    shuffled = (
        observed_sequence[
            order
        ]
    )

    (
        null_drift[i],
        null_distortion[i],
        null_instability[i],
    ) = dynamics_from_sequence(
        shuffled
    )

    if (
        i < 5
        or
        (i+1) % 50 == 0
    ):
        print(
            f"shuffle={i+1:03d}/{N_SHUFFLES}"
        )


def summarize(
    name,
    observed,
    null,
):
    mean = float(
        np.mean(null)
    )

    std = float(
        np.std(
            null,
            ddof=1
        )
    )

    if std > 0:
        z = (
            observed
            - mean
        ) / std
    else:
        z = np.nan

    # Two-sided empirical p-value
    center = mean

    observed_dev = abs(
        observed
        - center
    )

    null_dev = np.abs(
        null
        - center
    )

    p = (
        1
        + np.count_nonzero(
            null_dev
            >= observed_dev
        )
    ) / (
        N_SHUFFLES
        + 1
    )

    print()
    print(name)
    print(
        "  observed:",
        f"{observed:.8f}"
    )
    print(
        "  null mean:",
        f"{mean:.8f}"
    )
    print(
        "  null std:",
        f"{std:.8f}"
    )
    print(
        "  z:",
        f"{z:.4f}"
    )
    print(
        "  empirical p:",
        f"{p:.6f}"
    )

    return (
        mean,
        std,
        z,
        p,
    )


print()
print("=" * 80)
print("GLASGOW DYNAMICS NULL RESULTS")
print("=" * 80)

drift_stats = summarize(
    "drift",
    observed_drift,
    null_drift,
)

distortion_stats = summarize(
    "attractor_distortion",
    observed_distortion,
    null_distortion,
)

instability_stats = summarize(
    "mean_local_instability",
    observed_instability,
    null_instability,
)


OUT = (
    "artifacts/"
    "glasgow_dynamics_null_results.npz"
)

Path("artifacts").mkdir(
    exist_ok=True
)

np.savez_compressed(
    OUT,

    observed_drift=
        observed_drift,

    observed_distortion=
        observed_distortion,

    observed_instability=
        observed_instability,

    null_drift=
        null_drift,

    null_distortion=
        null_distortion,

    null_instability=
        null_instability,

    window=
        WINDOW,

    n_shuffles=
        N_SHUFFLES,
)


print()
print(
    "Saved:",
    OUT
)

print()
print(
    "Interpretation:"
)

print(
    "If observed metrics fall well outside the shuffled "
    "null, the AURORA dynamics depend on acquisition order."
)

print(
    "If observed metrics sit inside the null, the metric "
    "is mainly describing the distribution of sparse frames "
    "rather than real temporal ordering."
)

print()
print(
    "GLASGOW DYNAMICS NULL TEST COMPLETE"
)
