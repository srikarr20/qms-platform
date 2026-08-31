from dataclasses import dataclass
from typing import Optional

from .state import (
    aggregate_increment_window,
    normalized_grid_distribution,
    build_measurement_states,
)

from .prediction import (
    predict_sequence,
)

from .calibration import (
    calibrate_frozen_threshold,
)

from .alerting import (
    evaluate_future_predictions,
)

from .episodes import (
    consolidate_alerts,
)


@dataclass(frozen=True)
class CausalReplayResult:
    reference_end: int
    calibration_end: int
    threshold: float
    calibration_samples: int
    future_samples: int
    raw_flags: int
    episodes: tuple


def run_causal_replay(
    increments,
    *,
    reference_start: int = 0,
    reference_end: int,
    calibration_end: int,
    last_increment: Optional[int] = None,
    window: int = 100,
    grid: int = 32,
    refractory: int = 100,
    percentile: float = 99.0,
    robust_multiplier: float = 6.0,
):
    """
    End-to-end causal measurement-state replay.

    Reference:
        uses only reference_start..reference_end

    Calibration:
        predictions are frozen through calibration_end

    Test:
        only predictions after calibration_end are
        evaluated as future alerts.
    """
    if last_increment is None:
        last_increment = max(
            increments
        )

    if calibration_end <= (
        reference_end
        + window
    ):
        raise ValueError(
            "Calibration period must contain "
            "post-reference prediction samples."
        )

    reference_frame = (
        aggregate_increment_window(
            increments,
            reference_start,
            reference_end,
        )
    )

    reference = (
        normalized_grid_distribution(
            reference_frame,
            grid=grid,
        )
    )

    states = (
        build_measurement_states(
            increments,
            reference,
            first_window_end=(
                reference_end
                + window
            ),
            last_window_end=(
                last_increment
            ),
            window=window,
            grid=grid,
        )
    )

    predictions = (
        predict_sequence(
            states
        )
    )

    calibration_values = [
        p.innovation_norm
        for p in predictions
        if p.window_end
        <= calibration_end
    ]

    calibration = (
        calibrate_frozen_threshold(
            calibration_values,
            percentile=percentile,
            robust_multiplier=(
                robust_multiplier
            ),
        )
    )

    alerts = (
        evaluate_future_predictions(
            predictions,
            calibration,
            calibration_end=(
                calibration_end
            ),
        )
    )

    episodes = (
        consolidate_alerts(
            alerts,
            refractory=refractory,
        )
    )

    return {
        "reference":
            reference,

        "states":
            states,

        "predictions":
            predictions,

        "calibration":
            calibration,

        "alerts":
            alerts,

        "episodes":
            episodes,

        "summary":
            CausalReplayResult(
                reference_end=(
                    reference_end
                ),

                calibration_end=(
                    calibration_end
                ),

                threshold=(
                    calibration.threshold
                ),

                calibration_samples=(
                    calibration.samples
                ),

                future_samples=(
                    len(alerts)
                ),

                raw_flags=(
                    sum(
                        a.flagged
                        for a in alerts
                    )
                ),

                episodes=tuple(
                    episodes
                ),
            ),
    }
