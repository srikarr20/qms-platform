from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .residuals import rmse


@dataclass(frozen=True)
class MechanismCandidate:
    mechanism: str
    parameter: str
    value: Any
    score: float
    stage_errors: dict


@dataclass(frozen=True)
class MechanismInference:
    winner: MechanismCandidate
    ranked_candidates: list
    runner_up_margin: float


def rank_mechanism_library(
    observed: Mapping[str, object],
    library: Iterable[Mapping],
    *,
    stages=("IMAGE", "PIXEL"),
) -> MechanismInference:
    """
    Rank a finite mechanism/parameter library against
    current detector representations.

    Each library element should contain:
        mechanism
        parameter
        value
        representations: {stage: array}

    This is finite-library model selection only.
    """
    scores = []

    for candidate in library:

        representations = (
            candidate[
                "representations"
            ]
        )

        stage_errors = {}
        total = 0.0

        for stage in stages:

            if (
                stage not in observed
                or stage not in representations
            ):
                continue

            err = rmse(
                observed[stage],
                representations[stage],
            )

            stage_errors[
                stage
            ] = err

            total += err

        scores.append(
            MechanismCandidate(
                mechanism=(
                    candidate[
                        "mechanism"
                    ]
                ),
                parameter=(
                    candidate[
                        "parameter"
                    ]
                ),
                value=(
                    candidate[
                        "value"
                    ]
                ),
                score=float(
                    total
                ),
                stage_errors=(
                    stage_errors
                ),
            )
        )

    if not scores:
        raise ValueError(
            "Mechanism library is empty."
        )

    scores.sort(
        key=lambda x:
            x.score
    )

    margin = (
        scores[1].score
        - scores[0].score
        if len(scores) > 1
        else 0.0
    )

    return MechanismInference(
        winner=scores[0],
        ranked_candidates=scores,
        runner_up_margin=float(
            margin
        ),
    )
