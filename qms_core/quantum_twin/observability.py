from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class ObservabilityDiagnostics:
    rank: int
    nullity: int
    singular_values: np.ndarray
    condition_number_nonzero: Optional[float]
    null_space_basis: np.ndarray


def observability_matrix(
    F: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """
    Classical finite-dimensional observability matrix:

        O = [C; CF; CF^2; ...; CF^(n-1)]
    """
    F = np.asarray(F, dtype=float)
    C = np.asarray(C, dtype=float)

    n = F.shape[0]

    return np.vstack(
        [
            C @ np.linalg.matrix_power(F, k)
            for k in range(n)
        ]
    )


def null_space(
    A: np.ndarray,
    *,
    rtol: float = 1e-10,
):
    """
    Return null-space basis, singular values, and numerical rank.
    """
    A = np.asarray(A, dtype=float)

    _, s, vh = np.linalg.svd(
        A,
        full_matrices=True,
    )

    if len(s) == 0:
        return (
            np.empty((A.shape[1], 0)),
            s,
            0,
        )

    tol = rtol * s[0]
    rank = int(
        np.sum(
            s > tol
        )
    )

    basis = vh[rank:].T

    return basis, s, rank


def analyze_observability(
    F: np.ndarray,
    C: np.ndarray,
    *,
    rtol: float = 1e-10,
) -> ObservabilityDiagnostics:
    """
    Compute rank, nullity, singular spectrum,
    non-zero condition number, and null-space basis.
    """
    O = observability_matrix(
        F,
        C,
    )

    basis, s, rank = null_space(
        O,
        rtol=rtol,
    )

    tol = (
        rtol * s[0]
        if len(s)
        else rtol
    )

    nonzero = s[
        s > tol
    ]

    condition_number = (
        float(
            nonzero.max()
            / nonzero.min()
        )
        if len(nonzero)
        else None
    )

    return ObservabilityDiagnostics(
        rank=rank,
        nullity=int(
            F.shape[0]
            - rank
        ),
        singular_values=s,
        condition_number_nonzero=condition_number,
        null_space_basis=basis,
    )


def build_discrete_observation_matrix(
    F: np.ndarray,
    C: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Construct the stacked finite-time observation operator:

        H = [C exp(F t_0);
             C exp(F t_1);
             ...
             C exp(F t_k)]
    """
    F = np.asarray(F, dtype=float)
    C = np.asarray(C, dtype=float)

    rows = []

    for t in np.asarray(
        times,
        dtype=float,
    ):
        rows.append(
            C
            @ expm(
                F * float(t)
            )
        )

    return np.vstack(rows)


def null_overlap(
    H: np.ndarray,
    x: np.ndarray,
    *,
    rtol: float = 1e-10,
) -> float:
    """
    Norm of the component of x lying in null(H).
    """
    basis, _, _ = null_space(
        H,
        rtol=rtol,
    )

    if basis.size == 0:
        return 0.0

    x = np.asarray(
        x,
        dtype=float,
    )

    projection = (
        basis
        @ (
            basis.T
            @ x
        )
    )

    return float(
        np.linalg.norm(
            projection
        )
    )
