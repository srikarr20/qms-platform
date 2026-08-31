from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from .residuals import rmse


@dataclass(frozen=True)
class StageDivergence:
    first_divergent_stage: Optional[str]
    stage_rmse: dict
    diverged: bool


def first_divergent_stage(
    observed: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    *,
    stage_order,
    tolerance: float = 1e-9,
) -> StageDivergence:
    """
    Identify the first detector representation whose
    RMSE from baseline exceeds a supplied tolerance.

    Example stage_order:
        ["PIXEL", "IMAGE"]
    """
    stage_rmse = {}
    first = None

    for stage in stage_order:

        if (
            stage not in observed
            or stage not in baseline
        ):
            continue

        value = rmse(
            observed[stage],
            baseline[stage],
        )

        stage_rmse[
            stage
        ] = value

        if (
            first is None
            and value > tolerance
        ):
            first = stage

    return StageDivergence(
        first_divergent_stage=first,
        stage_rmse=stage_rmse,
        diverged=bool(
            first is not None
        ),
    )
