import numpy as np


def symplectic_form_two_mode() -> np.ndarray:
    """
    Canonical two-mode quadrature symplectic form
    for [x1, p1, x2, p2].
    """
    return np.array(
        [
            [0.0,  1.0,  0.0,  0.0],
            [-1.0, 0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  1.0],
            [0.0,  0.0, -1.0,  0.0],
        ],
        dtype=float,
    )


def gaussian_physicality(
    covariance: np.ndarray,
    *,
    tolerance: float = 1e-9,
):
    """
    Computational Gaussian uncertainty check:

        V + i Omega / 2 >= 0

    Returns the minimum Hermitian eigenvalue and
    a Boolean physicality indicator.
    """
    V = np.asarray(
        covariance,
        dtype=float,
    )

    omega = (
        symplectic_form_two_mode()
    )

    matrix = (
        V
        + 0.5j * omega
    )

    eigenvalues = (
        np.linalg.eigvalsh(
            matrix
        )
    )

    minimum = float(
        np.min(
            eigenvalues
        )
    )

    return {
        "min_uncertainty_eigenvalue":
            minimum,

        "physical":
            bool(
                minimum
                >= -tolerance
            ),
    }
