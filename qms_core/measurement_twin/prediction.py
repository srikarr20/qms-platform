from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .state import MeasurementState


@dataclass(frozen=True)
class MeasurementPrediction:
    window_end: int

    observed_cosine: float
    predicted_cosine: float
    cosine_innovation: float

    observed_js: float
    predicted_js: float
    js_innovation: float

    innovation_norm: float


def constant_velocity_prediction(
    previous_previous: MeasurementState,
    previous: MeasurementState,
    current: MeasurementState,
) -> MeasurementPrediction:
    """
    One-step constant-velocity extrapolation in
    cosine/JS observable-state space.
    """
    predicted_cosine = (
        previous.cosine_similarity
        + (
            previous.cosine_similarity
            - previous_previous.cosine_similarity
        )
    )

    predicted_js = (
        previous.js_divergence
        + (
            previous.js_divergence
            - previous_previous.js_divergence
        )
    )

    cosine_innovation = (
        current.cosine_similarity
        - predicted_cosine
    )

    js_innovation = (
        current.js_divergence
        - predicted_js
    )

    innovation_norm = float(
        np.sqrt(
            cosine_innovation ** 2
            + js_innovation ** 2
        )
    )

    return MeasurementPrediction(
        window_end=(
            current.window_end
        ),

        observed_cosine=float(
            current.cosine_similarity
        ),

        predicted_cosine=float(
            predicted_cosine
        ),

        cosine_innovation=float(
            cosine_innovation
        ),

        observed_js=float(
            current.js_divergence
        ),

        predicted_js=float(
            predicted_js
        ),

        js_innovation=float(
            js_innovation
        ),

        innovation_norm=(
            innovation_norm
        ),
    )


def predict_sequence(
    states: Sequence[
        MeasurementState
    ],
):
    """
    Produce causal one-step predictions using only
    the previous two observable states.
    """
    predictions = []

    for i in range(
        2,
        len(states),
    ):
        predictions.append(
            constant_velocity_prediction(
                states[i - 2],
                states[i - 1],
                states[i],
            )
        )

    return predictions
