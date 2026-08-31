from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class AdaptationResult:
    parameter_value: float
    fit_mse: float


def estimate_parameter_from_trajectory(
    measurements,
    x_start: np.ndarray,
    candidate_values: Iterable[float],
    *,
    transition_builder: Callable[
        [float],
        np.ndarray,
    ],
    C: np.ndarray,
) -> AdaptationResult:
    """
    Estimate one supported dynamical parameter by matching
    deterministic model trajectories to an observed window.

    Extracted from the validated QMS-QT-005A adaptive
    resynchronization experiment.
    """
    y = np.asarray(
        measurements,
        dtype=float,
    ).reshape(-1)

    C = np.asarray(
        C,
        dtype=float,
    )

    best_value = None
    best_mse = np.inf

    for value in candidate_values:
        A = transition_builder(
            float(value)
        )

        x = np.asarray(
            x_start,
            dtype=float,
        ).copy()

        predicted = []

        for _ in range(
            len(measurements)
        ):
            x = A @ x

            measurement = (
                C @ x
            )

            predicted.extend(
                np.asarray(
                    measurement
                ).reshape(-1)
            )

        prediction = np.asarray(
            predicted,
            dtype=float,
        )

        mse = float(
            np.mean(
                (
                    y
                    - prediction
                ) ** 2
            )
        )

        if mse < best_mse:
            best_mse = mse
            best_value = float(
                value
            )

    if best_value is None:
        raise ValueError(
            "Candidate parameter grid is empty."
        )

    return AdaptationResult(
        parameter_value=best_value,
        fit_mse=float(
            best_mse
        ),
    )
