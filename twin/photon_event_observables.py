import numpy as np

from qms_core import ObservableManifold


def photon_event_features(
    increment,
):
    """
    Extract compact spatial statistics from one
    photon/event increment frame.
    """

    I = np.asarray(
        increment,
        dtype=float,
    )

    if I.ndim != 2:
        raise ValueError(
            f"Expected 2D increment frame, got {I.shape}"
        )

    total = float(
        np.sum(I)
    )

    H, W = I.shape

    yy, xx = np.indices(
        (H, W)
    )

    if total <= 0:
        return np.array([
            0.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ])

    xc = float(
        np.sum(
            I * xx
        ) / total
    )

    yc = float(
        np.sum(
            I * yy
        ) / total
    )

    dx = xx - xc
    dy = yy - yc

    sigma_x = float(
        np.sqrt(
            np.sum(
                I * dx**2
            ) / total
        )
    )

    sigma_y = float(
        np.sqrt(
            np.sum(
                I * dy**2
            ) / total
        )
    )

    radial_rms = float(
        np.sqrt(
            np.sum(
                I * (
                    dx**2
                    + dy**2
                )
            ) / total
        )
    )

    return np.array([
        total,
        xc,
        yc,
        sigma_x,
        sigma_y,
        radial_rms,
    ])


def build_photon_event_manifold(
    increment_sequence,
):
    """
    Convert photon-event increments into a
    low-dimensional physical event trajectory.
    """

    features = np.asarray([
        photon_event_features(
            frame
        )
        for frame in increment_sequence
    ])

    names = [
        "event_count",
        "centroid_x",
        "centroid_y",
        "sigma_x",
        "sigma_y",
        "radial_rms",
    ]

    # Replace possible NaNs from empty increments
    # with column means.
    for j in range(
        features.shape[1]
    ):
        col = features[:, j]

        mask = np.isfinite(
            col
        )

        if np.any(mask):
            fill = float(
                np.mean(
                    col[mask]
                )
            )
        else:
            fill = 0.0

        col[~mask] = fill

        features[:, j] = col

    mean = np.mean(
        features,
        axis=0,
    )

    std = np.std(
        features,
        axis=0,
    )

    normalized = (
        features - mean
    ) / (
        std + 1e-8
    )

    return ObservableManifold(
        state=normalized,

        names=names,

        metadata={
            "raw_features":
                features,

            "feature_mean":
                mean,

            "feature_std":
                std,

            "representation":
                "photon_event_spatial_state",
        },
    )
