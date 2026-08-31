from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def build_two_mode_drift(
    coupling: float,
    *,
    omega1: float = 1.00,
    omega2: float = 1.20,
    gamma1: float = 0.08,
    gamma2: float = 0.06,
) -> np.ndarray:
    """
    Build the validated QMS two-mode linear drift matrix.

    State ordering:
        x = [x1, p1, x2, p2]^T

    This is a finite-dimensional computational model and does not
    represent a reconstructed physical quantum field.
    """
    g = float(coupling)

    return np.array(
        [
            [-gamma1,  omega1,   0.0,      0.0],
            [-omega1, -gamma1,  g,        0.0],
            [0.0,       0.0,    -gamma2,  omega2],
            [g,         0.0,    -omega2, -gamma2],
        ],
        dtype=float,
    )


def default_drive_matrix() -> np.ndarray:
    """
    Drive coupling used by the validated QT baseline.
    The scalar drive enters the p1 equation.
    """
    return np.array(
        [
            [0.0],
            [1.0],
            [0.0],
            [0.0],
        ],
        dtype=float,
    )


def default_diffusion(
    *,
    gamma1: float = 0.08,
    gamma2: float = 0.06,
) -> np.ndarray:
    """
    Diffusion matrix used by the QT-001 Gaussian baseline.
    """
    return np.diag(
        [
            gamma1,
            gamma1,
            gamma2,
            gamma2,
        ]
    )


def gaussian_rhs(
    t: float,
    z: np.ndarray,
    F: np.ndarray,
    B: np.ndarray,
    drive: Callable[[float], float],
    D: np.ndarray,
) -> np.ndarray:
    """
    Coupled first- and second-moment Gaussian dynamics.

    z = [mu, vec(V)]
    """
    n = F.shape[0]

    mu = z[:n]
    V = z[n:].reshape(n, n)

    u = np.array([drive(t)], dtype=float)

    dmu = F @ mu + B @ u
    dV = F @ V + V @ F.T + D

    return np.concatenate(
        [
            dmu,
            dV.ravel(),
        ]
    )


def propagate_gaussian_state(
    F: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    drive: Callable[[float], float],
    mu0: np.ndarray,
    V0: np.ndarray,
    times: np.ndarray,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-11,
):
    """
    Propagate Gaussian first and second moments over supplied times.
    """
    times = np.asarray(times, dtype=float)

    z0 = np.concatenate(
        [
            np.asarray(mu0, dtype=float),
            np.asarray(V0, dtype=float).ravel(),
        ]
    )

    solution = solve_ivp(
        gaussian_rhs,
        (float(times[0]), float(times[-1])),
        z0,
        t_eval=times,
        args=(F, B, drive, D),
        rtol=rtol,
        atol=atol,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    n = F.shape[0]

    mu_t = solution.y[:n].T
    V_t = solution.y[n:].T.reshape(
        -1,
        n,
        n,
    )

    return solution, mu_t, V_t


def discretize_linear_dynamics(
    F: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Exact discrete-time transition matrix exp(F*dt).
    """
    return expm(
        np.asarray(F, dtype=float)
        * float(dt)
    )
