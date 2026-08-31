from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence

import numpy as np

from .identification import (
    build_window_matrix,
    candidate_parameter_sets,
)


@dataclass(frozen=True)
class MeasurementRecommendation:
    name: str
    score: float
    closest_model_family: str
    closest_parameters: Dict[str, float]
    mean_family_separation: float
    median_family_separation: float


def orthonormal_trajectory_basis(
    H: np.ndarray,
    *,
    rtol: float = 1e-12,
) -> np.ndarray:
    """
    Orthonormal basis for the trajectory subspace
    spanned by an observation operator.
    """
    u, s, _ = np.linalg.svd(
        np.asarray(
            H,
            dtype=float,
        ),
        full_matrices=False,
    )

    if len(s) == 0:
        return np.empty(
            (
                H.shape[0],
                0,
            )
        )

    tolerance = (
        rtol * s[0]
    )

    rank = int(
        np.sum(
            s > tolerance
        )
    )

    return u[:, :rank]


def subspace_mismatch(
    true_basis: np.ndarray,
    model_basis: np.ndarray,
) -> float:
    """
    Fractional trajectory-subspace mismatch used by
    QMS-QT-006G.

    Zero means the true trajectory subspace is fully
    representable by the candidate-model subspace.
    """
    projection = (
        model_basis
        @ (
            model_basis.T
            @ true_basis
        )
    )

    residual = (
        true_basis
        - projection
    )

    return float(
        np.linalg.norm(
            residual,
            ord="fro",
        ) ** 2
        / true_basis.shape[1]
    )


def recommend_measurement_configuration(
    configurations: Mapping[
        str,
        np.ndarray,
    ],
    *,
    true_parameters: Mapping[
        str,
        float,
    ],
    base_parameters: Mapping[
        str,
        float,
    ],
    model_families: Mapping[
        str,
        Sequence[str],
    ],
    grids,
    transition_builder: Callable[
        [Mapping[str, float]],
        np.ndarray,
    ],
    window: int,
):
    """
    Rank candidate measurement configurations using
    worst-case separation from the closest supported
    wrong-model family.

    This is a model-specific geometric recommendation,
    not a universal information metric.
    """
    results = []

    A_true = transition_builder(
        true_parameters
    )

    for (
        config_name,
        C,
    ) in configurations.items():

        H_true = build_window_matrix(
            A_true,
            C,
            window,
        )

        Q_true = (
            orthonormal_trajectory_basis(
                H_true
            )
        )

        family_results = []

        for (
            family,
            parameter_names,
        ) in model_families.items():

            best_mismatch = np.inf
            best_parameters = None

            for changes in (
                candidate_parameter_sets(
                    parameter_names,
                    grids,
                )
            ):
                params = dict(
                    base_parameters
                )
                params.update(
                    changes
                )

                A_model = (
                    transition_builder(
                        params
                    )
                )

                H_model = (
                    build_window_matrix(
                        A_model,
                        C,
                        window,
                    )
                )

                Q_model = (
                    orthonormal_trajectory_basis(
                        H_model
                    )
                )

                mismatch = (
                    subspace_mismatch(
                        Q_true,
                        Q_model,
                    )
                )

                if (
                    mismatch
                    < best_mismatch
                ):
                    best_mismatch = (
                        mismatch
                    )
                    best_parameters = (
                        dict(changes)
                    )

            family_results.append(
                (
                    family,
                    float(
                        best_mismatch
                    ),
                    best_parameters,
                )
            )

        closest = min(
            family_results,
            key=lambda r: r[1],
        )

        values = np.asarray(
            [
                r[1]
                for r
                in family_results
            ],
            dtype=float,
        )

        results.append(
            MeasurementRecommendation(
                name=config_name,
                score=float(
                    closest[1]
                ),
                closest_model_family=(
                    closest[0]
                ),
                closest_parameters=(
                    closest[2]
                ),
                mean_family_separation=(
                    float(
                        np.mean(
                            values
                        )
                    )
                ),
                median_family_separation=(
                    float(
                        np.median(
                            values
                        )
                    )
                ),
            )
        )

    return sorted(
        results,
        key=lambda r: r.score,
        reverse=True,
    )
