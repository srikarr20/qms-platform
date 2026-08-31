from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ResidualStatistics:
    mean: float
    std: float
    rms: float
    mean_abs: float
    lag1_autocorrelation: float


@dataclass(frozen=True)
class DivergenceAssessment:
    pre: ResidualStatistics
    post: ResidualStatistics
    rms_ratio: float
    mean_abs_ratio: float
    lag1_change: float
    classification: str


def lag1_autocorrelation(
    values: Iterable[float],
) -> float:
    x = np.asarray(
        list(values),
        dtype=float,
    )

    if len(x) < 3:
        return 0.0

    a = x[:-1] - np.mean(x[:-1])
    b = x[1:] - np.mean(x[1:])

    denominator = np.sqrt(
        np.sum(a * a)
        * np.sum(b * b)
    )

    if denominator <= 0:
        return 0.0

    return float(
        np.sum(a * b)
        / denominator
    )


def summarize_residuals(
    values: Iterable[float],
) -> ResidualStatistics:
    x = np.asarray(
        list(values),
        dtype=float,
    )

    return ResidualStatistics(
        mean=float(np.mean(x)),
        std=float(np.std(x)),
        rms=float(
            np.sqrt(
                np.mean(x ** 2)
            )
        ),
        mean_abs=float(
            np.mean(
                np.abs(x)
            )
        ),
        lag1_autocorrelation=(
            lag1_autocorrelation(x)
        ),
    )


def classify_residual_structure(
    pre_values,
    post_values,
    *,
    nominal_rms_ratio: float = 1.20,
    measurement_rms_ratio: float = 3.0,
    lag1_threshold: float = 0.20,
) -> DivergenceAssessment:
    """
    Transparent residual-structure classifier extracted
    from QMS-QT-004B.

    Thresholds are configurable because the validated values
    are simulation-specific and are not universal physical
    detector thresholds.
    """
    pre = summarize_residuals(
        pre_values
    )

    post = summarize_residuals(
        post_values
    )

    rms_ratio = (
        post.rms / pre.rms
        if pre.rms > 0
        else np.inf
    )

    mean_abs_ratio = (
        post.mean_abs / pre.mean_abs
        if pre.mean_abs > 0
        else np.inf
    )

    lag1_change = (
        post.lag1_autocorrelation
        - pre.lag1_autocorrelation
    )

    if rms_ratio < nominal_rms_ratio:
        classification = "nominal"

    elif (
        rms_ratio >= measurement_rms_ratio
        and abs(
            post.lag1_autocorrelation
        ) < lag1_threshold
    ):
        classification = (
            "measurement_system_change"
        )

    elif (
        rms_ratio >= nominal_rms_ratio
        and abs(
            post.lag1_autocorrelation
        ) >= lag1_threshold
    ):
        classification = (
            "physical_or_model_change"
        )

    else:
        classification = "ambiguous"

    return DivergenceAssessment(
        pre=pre,
        post=post,
        rms_ratio=float(rms_ratio),
        mean_abs_ratio=float(
            mean_abs_ratio
        ),
        lag1_change=float(
            lag1_change
        ),
        classification=classification,
    )
