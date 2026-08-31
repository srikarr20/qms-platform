import numpy as np


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (a - b) ** 2
            )
        )
    )


def residual_features(delta):
    """
    Generic 2-D detector residual diagnostics.

    Directionality is the ratio of row-mean variation
    to column-mean variation. It is a descriptive
    representation metric, not a universal physical
    mechanism classifier.
    """
    delta = np.asarray(
        delta,
        dtype=float,
    )

    row = delta.mean(axis=1)
    col = delta.mean(axis=0)

    changed = (
        delta != 0
    )

    return {
        "rmse":
            rmse(
                delta,
                np.zeros_like(delta),
            ),

        "mean":
            float(
                delta.mean()
            ),

        "std":
            float(
                delta.std()
            ),

        "changed_fraction":
            float(
                np.mean(
                    changed
                )
            ),

        "directionality":
            float(
                row.std()
                /
                (
                    col.std()
                    + 1e-12
                )
            ),
    }
