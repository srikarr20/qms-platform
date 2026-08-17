import numpy as np

from .observability import score_curvature


def choose_next_measurement(
    base_score,
    candidate_scores,
    candidate_values,
    step,
):
    """
    candidate_scores:
        iterable of 1D depth-score arrays

    candidate_values:
        e.g. wavelengths
    """

    curvatures = []
    depth_indices = []

    for score in candidate_scores:
        combined = base_score + score

        best = int(
            np.argmin(combined)
        )

        curvature = score_curvature(
            combined,
            best,
            step,
        )

        curvatures.append(
            curvature
        )

        depth_indices.append(
            best
        )

    curvatures = np.asarray(
        curvatures
    )

    best_candidate = int(
        np.argmax(curvatures)
    )

    return {
        "candidate_index": best_candidate,
        "candidate_value": candidate_values[
            best_candidate
        ],
        "curvatures": curvatures,
        "depth_indices": np.asarray(
            depth_indices
        ),
    }
