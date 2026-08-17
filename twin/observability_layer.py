import numpy as np

from qms_core import (
    DetectorState,
    ObservableManifold,
)


def compute_delta_field(field_sequence):
    """
    Temporal detector-conditioned change field.

    Input:
        field_sequence with time on axis 0:
            (T, ...)
    Output:
        absolute temporal difference:
            (T-1, ...)
    """

    V = np.asarray(field_sequence)

    if V.shape[0] < 2:
        raise ValueError(
            "Need at least two field states."
        )

    return np.abs(
        np.diff(
            V,
            axis=0,
        )
    )


def compute_cke(delta_field):
    """
    AURORA-compatible observable construction.

    C = spatial variability / mean activity
    K = temporal variability between successive detector states
    E = total detector activity
    """

    delta_field = np.asarray(
        delta_field
    )

    T = delta_field.shape[0]

    C = []
    K = []
    E = []

    for t in range(T):

        frame = delta_field[t]

        mean_frame = np.mean(frame)

        C.append(
            np.std(frame)
            /
            (
                mean_frame
                + 1e-6
            )
        )

        if t > 0:
            K.append(
                np.std(
                    frame
                    - delta_field[t-1]
                )
            )
        else:
            K.append(0.0)

        E.append(
            np.sum(frame)
        )

    return (
        np.asarray(C),
        np.asarray(K),
        np.asarray(E),
    )


def normalize_signal(x):
    x = np.asarray(x)

    return (
        x
        - np.mean(x)
    ) / (
        np.std(x)
        + 1e-6
    )


def build_manifold(
    C,
    K,
    E,
):
    """
    Construct X(t) = [C, K, E].
    """

    Cn = normalize_signal(C)
    Kn = normalize_signal(K)
    En = normalize_signal(E)

    return np.vstack([
        Cn,
        Kn,
        En,
    ]).T


def build_observability_layer(
    field_sequence,
    field_domain="reconstructed_field",
):
    """
    Convert reconstructed field evolution into:

        DetectorState
            +
        ObservableManifold

    This mirrors the canonical AURORA transformation
    without importing the medical/application pipeline.
    """

    delta_field = compute_delta_field(
        field_sequence
    )

    C, K, E = compute_cke(
        delta_field
    )

    X = build_manifold(
        C,
        K,
        E,
    )

    detector_state = DetectorState(
        data=delta_field,
        detector_type="temporal_delta_field",
        metadata={
            "source_domain":
                field_domain,

            "construction":
                "absolute temporal field difference",
        },
    )

    manifold = ObservableManifold(
        state=X,

        names=[
            "C",
            "K",
            "E",
        ],

        metadata={
            "C":
                C,

            "K":
                K,

            "E":
                E,

            "construction":
                "AURORA-compatible normalized CKE manifold",
        },
    )

    return (
        detector_state,
        manifold,
    )
