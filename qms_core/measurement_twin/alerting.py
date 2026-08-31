from dataclasses import dataclass
from typing import Optional

from .calibration import (
    FrozenInnovationCalibration,
    frozen_robust_z,
)


@dataclass(frozen=True)
class MeasurementAlert:
    window_end: int
    innovation_norm: float
    robust_z_frozen: Optional[float]
    flagged: bool


def evaluate_prediction(
    prediction,
    calibration: FrozenInnovationCalibration,
) -> MeasurementAlert:
    """
    Evaluate one future prediction against a frozen
    pre-test calibration.
    """
    value = float(
        prediction.innovation_norm
    )

    return MeasurementAlert(
        window_end=int(
            prediction.window_end
        ),

        innovation_norm=value,

        robust_z_frozen=(
            frozen_robust_z(
                value,
                calibration,
            )
        ),

        flagged=bool(
            value
            > calibration.threshold
        ),
    )


def evaluate_future_predictions(
    predictions,
    calibration,
    *,
    calibration_end: int,
):
    """
    Evaluate only predictions occurring after the
    frozen calibration boundary.
    """
    alerts = []

    for prediction in predictions:

        if (
            prediction.window_end
            <= calibration_end
        ):
            continue

        alerts.append(
            evaluate_prediction(
                prediction,
                calibration,
            )
        )

    return alerts
