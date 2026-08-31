from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class MeasurementState:
    window_start: int
    window_end: int
    cosine_similarity: float
    js_divergence: float


def aggregate_increment_window(
    increments: Mapping[int, np.ndarray],
    start: int,
    end: int,
) -> np.ndarray:
    """
    Aggregate detector increments over an inclusive interval.

    Raises when the requested interval is incomplete.
    """
    arrays = [
        np.asarray(
            increments[i],
            dtype=float,
        )
        for i in range(start, end + 1)
        if i in increments
    ]

    expected = end - start + 1

    if len(arrays) != expected:
        raise RuntimeError(
            f"Incomplete increment interval "
            f"{start}:{end}; expected {expected}, "
            f"found {len(arrays)}."
        )

    return np.sum(
        np.stack(arrays),
        axis=0,
    )


def normalized_grid_distribution(
    frame: np.ndarray,
    *,
    grid: int = 32,
) -> np.ndarray:
    """
    Reduce a 2-D detector frame to a normalized grid
    distribution.

    The frame dimensions must be divisible by grid.
    """
    frame = np.asarray(
        frame,
        dtype=float,
    )

    if frame.ndim != 2:
        raise ValueError(
            "Detector frame must be 2-D."
        )

    h, w = frame.shape

    if (
        h % grid != 0
        or w % grid != 0
    ):
        raise ValueError(
            f"Frame shape {frame.shape} is not "
            f"divisible by grid={grid}."
        )

    bh = h // grid
    bw = w // grid

    reduced = frame.reshape(
        grid,
        bh,
        grid,
        bw,
    ).sum(
        axis=(1, 3)
    )

    p = reduced.ravel().astype(
        float
    )

    total = float(
        p.sum()
    )

    if total <= 0:
        raise RuntimeError(
            "Zero-count detector distribution."
        )

    return p / total


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator <= 0:
        raise ValueError(
            "Cosine similarity requires "
            "non-zero vectors."
        )

    return float(
        np.dot(a, b)
        / denominator
    )


def jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
    *,
    epsilon: float = 1e-15,
) -> float:
    """
    Natural-log Jensen-Shannon divergence matching
    the validated Glasgow experiments.
    """
    p = np.asarray(
        p,
        dtype=float,
    )

    q = np.asarray(
        q,
        dtype=float,
    )

    p = np.clip(
        p,
        epsilon,
        None,
    )

    q = np.clip(
        q,
        epsilon,
        None,
    )

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (
        p + q
    )

    return float(
        0.5
        * np.sum(
            p
            * np.log(
                p / m
            )
        )
        +
        0.5
        * np.sum(
            q
            * np.log(
                q / m
            )
        )
    )


def build_measurement_states(
    increments,
    reference,
    *,
    first_window_end: int,
    last_window_end: int,
    window: int = 100,
    grid: int = 32,
):
    """
    Construct sequential observable measurement states
    relative to a frozen reference distribution.
    """
    reference = np.asarray(
        reference,
        dtype=float,
    )

    states = []

    for end in range(
        first_window_end,
        last_window_end + 1,
    ):
        start = (
            end
            - window
            + 1
        )

        frame = (
            aggregate_increment_window(
                increments,
                start,
                end,
            )
        )

        distribution = (
            normalized_grid_distribution(
                frame,
                grid=grid,
            )
        )

        states.append(
            MeasurementState(
                window_start=start,
                window_end=end,
                cosine_similarity=(
                    cosine_similarity(
                        distribution,
                        reference,
                    )
                ),
                js_divergence=(
                    jensen_shannon_divergence(
                        distribution,
                        reference,
                    )
                ),
            )
        )

    return states
