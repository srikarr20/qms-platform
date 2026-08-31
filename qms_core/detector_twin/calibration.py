from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass
class MonotonicInverseCalibration:
    observed_support: np.ndarray
    parameter_support: np.ndarray

    def __post_init__(self):

        x = np.asarray(
            self.observed_support,
            dtype=float,
        )

        y = np.asarray(
            self.parameter_support,
            dtype=float,
        )

        if len(x) != len(y):
            raise ValueError(
                "Support arrays must have equal length."
            )

        order = np.argsort(x)

        x = x[order]
        y = y[order]

        unique_x = []
        unique_y = []

        for observed, parameter in zip(
            x,
            y,
        ):
            if (
                not unique_x
                or observed > unique_x[-1]
            ):
                unique_x.append(
                    float(observed)
                )
                unique_y.append(
                    float(parameter)
                )

        if len(unique_x) < 2:
            raise ValueError(
                "At least two unique calibration points are required."
            )

        self.observed_support = (
            np.asarray(
                unique_x,
                dtype=float,
            )
        )

        self.parameter_support = (
            np.asarray(
                unique_y,
                dtype=float,
            )
        )

        self._inverse = (
            PchipInterpolator(
                self.observed_support,
                self.parameter_support,
                extrapolate=True,
            )
        )

    def estimate(
        self,
        observed_value: float,
        *,
        floor: Optional[float] = 0.0,
    ) -> float:

        value = float(
            self._inverse(
                float(
                    observed_value
                )
            )
        )

        if floor is not None:
            value = max(
                value,
                float(floor),
            )

        return value


def build_inverse_calibration(
    observed_support,
    parameter_support,
):
    return MonotonicInverseCalibration(
        observed_support=np.asarray(
            observed_support,
            dtype=float,
        ),
        parameter_support=np.asarray(
            parameter_support,
            dtype=float,
        ),
    )
