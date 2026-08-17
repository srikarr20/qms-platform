from pathlib import Path
import os
import numpy as np

from adapters.glasgow_event_adapter import (
    GlasgowCumulativeArchive,
)

from twin import build_observability_layer
from qms_core import PlatformTwinState

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
N_SHUFFLES = 100
RNG = np.random.default_rng(1510)


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
    )

    state = enrich_with_aurora_dynamics(
        state
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


def empirical_p(
    observed,
    null,
):
    mean = np.mean(
        null,
        axis=0,
    )

    observed_dev = np.abs(
        observed - mean
    )

    null_dev = np.abs(
        null - mean
    )

    p = (
        1
        + np.sum(
            null_dev >= observed_dev,
            axis=0,
        )
    ) / (
        len(null) + 1
    )

    return p


print()
print("=" * 82)
print("QMS PLATFORM — GLASGOW MULTI-WINDOW TEMPORAL NULL")
print("=" * 82)


archive = GlasgowCumulativeArchive(
    GLASGOW_ZIP
)

increments = np.asarray([
    record["increment"]
    for record in archive.iter_increments()
])

print(
    "Total increments:",
    len(increments)
)


# ============================================================
# NON-OVERLAPPING WINDOWS
# ============================================================

starts = np.arange(
    0,
    len(increments) - WINDOW + 1,
    WINDOW,
)

N_WINDOWS = len(starts)

print(
    "Window size:",
    WINDOW
)

print(
    "Windows:",
    N_WINDOWS
)

print(
    "Shuffles/window:",
    N_SHUFFLES
)


observed_metrics = np.zeros(
    (N_WINDOWS, 3)
)

pvalues = np.zeros(
    (N_WINDOWS, 3)
)

zvalues = np.zeros(
    (N_WINDOWS, 3)
)


for wi, start in enumerate(starts):

    sequence = increments[
        start:start+WINDOW
    ]

    observed = dynamics_from_sequence(
        sequence
    )

    null = np.zeros(
        (N_SHUFFLES, 3)
    )

    for si in range(
        N_SHUFFLES
    ):
        order = RNG.permutation(
            WINDOW
        )

        null[si] = (
            dynamics_from_sequence(
                sequence[order]
            )
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

    p = empirical_p(
        observed,
        null,
    )

    observed_metrics[wi] = observed
    pvalues[wi] = p
    zvalues[wi] = z

    if (
        wi < 5
        or
        (wi+1) % 10 == 0
        or
        wi == N_WINDOWS-1
    ):
        print(
            f"window={wi+1:02d}/{N_WINDOWS}"
            f"  steps={start:04d}-{start+WINDOW-1:04d}"
            f"  p_drift={p[0]:.4f}"
            f"  p_dist={p[1]:.4f}"
            f"  p_inst={p[2]:.4f}"
        )


# ============================================================
# MULTIPLE-TEST CORRECTION
#
# Bonferroni across windows separately for each metric.
# ============================================================

alpha = 0.05

bonf_threshold = (
    alpha / N_WINDOWS
)

raw_sig = (
    pvalues < alpha
)

bonf_sig = (
    pvalues < bonf_threshold
)


names = [
    "drift",
    "attractor_distortion",
    "mean_local_instability",
]


print()
print("=" * 82)
print("MULTI-WINDOW RESULTS")
print("=" * 82)

print(
    "Raw alpha:",
    alpha
)

print(
    "Bonferroni threshold:",
    f"{bonf_threshold:.8f}"
)


for mi, name in enumerate(names):

    best = int(
        np.argmin(
            pvalues[:, mi]
        )
    )

    print()
    print(name)

    print(
        "  smallest p:",
        f"{pvalues[best, mi]:.8f}"
    )

    print(
        "  window:",
        best + 1
    )

    print(
        "  start step:",
        int(
            starts[best]
        )
    )

    print(
        "  observed:",
        f"{observed_metrics[best, mi]:.8f}"
    )

    print(
        "  z:",
        f"{zvalues[best, mi]:.4f}"
    )

    print(
        "  raw p<0.05 windows:",
        int(
            np.sum(
                raw_sig[:, mi]
            )
        )
    )

    print(
        "  Bonferroni-significant windows:",
        int(
            np.sum(
                bonf_sig[:, mi]
            )
        )
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
    "glasgow_multiwindow_null_results.npz"
)

np.savez_compressed(
    OUT,

    starts=
        starts,

    observed_metrics=
        observed_metrics,

    pvalues=
        pvalues,

    zvalues=
        zvalues,

    window=
        WINDOW,

    n_shuffles=
        N_SHUFFLES,

    bonf_threshold=
        bonf_threshold,
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
    "If windows survive Bonferroni correction, "
    "the corresponding AURORA metric contains localized "
    "order-sensitive structure."
)

print(
    "If none survive, these current CKE/AURORA summaries "
    "do not provide strong evidence for temporal ordering "
    "anywhere in the acquisition."
)

print()
print(
    "GLASGOW MULTI-WINDOW NULL TEST COMPLETE"
)
