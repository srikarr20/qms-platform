from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FrozenInnovationCalibration:
    samples: int
    median: float
    mad: float
    p_quantile: float
    robust_threshold: float
    threshold: float
    mad_scale: float
    robust_multiplier: float


def calibrate_frozen_threshold(
    innovations: Iterable[float],
    *,
    percentile: float = 99.0,
    mad_scale: float = 1.4826,
    robust_multiplier: float = 6.0,
    minimum_samples: int = 20,
) -> FrozenInnovationCalibration:
    """
    Build the frozen innovation threshold used by the
    causal Glasgow replay.

    threshold = max(
        requested percentile,
        median + robust_multiplier * mad_scale * MAD
    )

    Calibration must be performed on pre-test data by
    the caller.
    """
    values = np.asarray(
        list(innovations),
        dtype=float,
    )

    if len(values) < minimum_samples:
        raise RuntimeError(
            "Too few innovation samples for "
            f"calibration: {len(values)}"
        )

    median = float(
        np.median(values)
    )

    mad = float(
        np.median(
            np.abs(
                values
                - median
            )
        )
    )

    p = float(
        np.percentile(
            values,
            percentile,
        )
    )

    robust_threshold = float(
        median
        + robust_multiplier
        * mad_scale
        * mad
    )

    threshold = max(
        p,
        robust_threshold,
    )

    return FrozenInnovationCalibration(
        samples=int(
            len(values)
        ),
        median=median,
        mad=mad,
        p_quantile=p,
        robust_threshold=(
            robust_threshold
        ),
        threshold=float(
            threshold
        ),
        mad_scale=float(
            mad_scale
        ),
        robust_multiplier=float(
            robust_multiplier
        ),
    )


def frozen_robust_z(
    value: float,
    calibration: FrozenInnovationCalibration,
):
    """
    Empirical robust standardized residual.

    This quantity must not be interpreted as Gaussian
    sigma significance.
    """
    if calibration.mad <= 0:
        return None

    return float(
        (
            float(value)
            - calibration.median
        )
        /
        (
            calibration.mad_scale
            * calibration.mad
        )
    )
