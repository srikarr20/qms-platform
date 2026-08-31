from dataclasses import dataclass

import numpy as np


def reconstruct_state(
    H: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Minimum-norm linear state reconstruction using
    the Moore-Penrose pseudoinverse.
    """
    return (
        np.linalg.pinv(
            np.asarray(
                H,
                dtype=float,
            )
        )
        @ np.asarray(
            y,
            dtype=float,
        )
    )


@dataclass
class KalmanState:
    mean: np.ndarray
    covariance: np.ndarray


@dataclass
class KalmanUpdate:
    predicted_mean: np.ndarray
    predicted_covariance: np.ndarray
    predicted_measurement: np.ndarray
    residual: np.ndarray
    gain: np.ndarray
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray


def kalman_predict(
    state: KalmanState,
    A: np.ndarray,
    Q: np.ndarray,
) -> KalmanState:
    """
    Linear Gaussian prediction step.
    """
    A = np.asarray(
        A,
        dtype=float,
    )

    Q = np.asarray(
        Q,
        dtype=float,
    )

    mean = (
        A
        @ state.mean
    )

    covariance = (
        A
        @ state.covariance
        @ A.T
        + Q
    )

    return KalmanState(
        mean=mean,
        covariance=covariance,
    )


def kalman_update(
    predicted: KalmanState,
    measurement: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
) -> KalmanUpdate:
    """
    Linear Gaussian measurement update.
    """
    C = np.asarray(
        C,
        dtype=float,
    )

    R = np.asarray(
        R,
        dtype=float,
    )

    y = np.atleast_1d(
        np.asarray(
            measurement,
            dtype=float,
        )
    )

    predicted_measurement = (
        C
        @ predicted.mean
    )

    residual = (
        y
        - predicted_measurement
    )

    innovation_covariance = (
        C
        @ predicted.covariance
        @ C.T
        + R
    )

    gain = (
        predicted.covariance
        @ C.T
        @ np.linalg.inv(
            innovation_covariance
        )
    )

    posterior_mean = (
        predicted.mean
        + (
            gain
            @ residual
        ).reshape(-1)
    )

    identity = np.eye(
        predicted.mean.size
    )

    posterior_covariance = (
        (
            identity
            - gain @ C
        )
        @ predicted.covariance
    )

    return KalmanUpdate(
        predicted_mean=predicted.mean,
        predicted_covariance=predicted.covariance,
        predicted_measurement=predicted_measurement,
        residual=residual,
        gain=gain,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
    )


def kalman_step(
    state: KalmanState,
    measurement: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
):
    """
    Convenience predict + update operation.
    """
    predicted = kalman_predict(
        state,
        A,
        Q,
    )

    update = kalman_update(
        predicted,
        measurement,
        C,
        R,
    )

    posterior = KalmanState(
        mean=update.posterior_mean,
        covariance=update.posterior_covariance,
    )

    return posterior, update
