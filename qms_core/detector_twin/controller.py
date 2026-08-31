from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class DetectorPrediction:
    values: dict


@dataclass(frozen=True)
class DetectorInnovation:
    relative: dict
    trigger_reestimate: bool
    mode: str


def constant_velocity_predict(
    history,
    *,
    fields,
) -> Optional[DetectorPrediction]:
    """
    Predict next detector state using the last two
    inferred states.

    history entries may contain additional metadata;
    only requested fields are used.
    """
    if len(history) < 2:
        return None

    a = history[-2]
    b = history[-1]

    predicted = {}

    for field in fields:

        predicted[field] = (
            b[field]
            + (
                b[field]
                - a[field]
            )
        )

    return DetectorPrediction(
        values=predicted
    )


def relative_innovation(
    predicted: Mapping[str, float],
    observed: Mapping[str, float],
    *,
    thresholds: Mapping[str, float],
    epsilon: float = 1e-15,
) -> DetectorInnovation:
    """
    Relative prediction innovation with configurable
    parameter-specific adaptation thresholds.
    """
    relative = {}
    trigger = False

    for field, threshold in (
        thresholds.items()
    ):

        denominator = max(
            abs(
                float(
                    observed[field]
                )
            ),
            epsilon,
        )

        value = abs(
            float(
                predicted[field]
            )
            - float(
                observed[field]
            )
        ) / denominator

        relative[field] = (
            float(value)
        )

        if value > threshold:
            trigger = True

    return DetectorInnovation(
        relative=relative,
        trigger_reestimate=trigger,
        mode=(
            "REESTIMATE_AND_ADAPT"
            if trigger
            else "ASSIMILATE"
        ),
    )


def assess_detector_state(
    history,
    current,
    *,
    fields,
    thresholds,
):
    """
    Complete prediction/gating operation used by the
    Pyxel streaming twins.
    """
    prediction = (
        constant_velocity_predict(
            history,
            fields=fields,
        )
    )

    if prediction is None:

        return {
            "predicted": None,
            "relative_innovation": {
                field: None
                for field in fields
            },
            "trigger_reestimate":
                True,
            "mode":
                "INITIALIZE",
        }

    innovation = (
        relative_innovation(
            prediction.values,
            current,
            thresholds=thresholds,
        )
    )

    return {
        "predicted":
            prediction.values,

        "relative_innovation":
            innovation.relative,

        "trigger_reestimate":
            innovation.trigger_reestimate,

        "mode":
            innovation.mode,
    }
