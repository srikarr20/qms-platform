import numpy as np

from .reconstruction import propagate
from .source_search import spatial_metrics


def z_lambda_map(
    sensor_field,
    depths,
    wavelengths,
    FX,
    FY,
    X,
    Y,
):
    score = np.empty(
        (
            len(wavelengths),
            len(depths),
        )
    )

    for wi, wavelength in enumerate(wavelengths):
        for zi, depth in enumerate(depths):
            field = propagate(
                sensor_field,
                FX,
                FY,
                wavelength,
                -depth,
            )

            _, _, width = spatial_metrics(
                field,
                X,
                Y,
            )

            score[wi, zi] = width

    return score


def best_depth_per_wavelength(
    score,
    depths,
):
    indices = np.argmin(
        score,
        axis=1,
    )

    return np.asarray([
        depths[i]
        for i in indices
    ])


def score_curvature(
    score,
    index,
    step,
):
    if (
        index <= 0
        or index >= len(score) - 1
    ):
        return 0.0

    return (
        score[index - 1]
        - 2 * score[index]
        + score[index + 1]
    ) / (
        step**2
    )
