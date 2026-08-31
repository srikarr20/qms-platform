import numpy as np

from qms_core.quantum_twin import (
    build_two_mode_drift,
    analyze_observability,
    build_discrete_observation_matrix,
    reconstruct_state,
    gaussian_physicality,
)


def test_qt_observability_baseline():
    F = build_two_mode_drift(
        0.18
    )

    C = np.array([
        [1.0, 0.0, 0.0, 0.0]
    ])

    result = analyze_observability(
        F,
        C,
    )

    assert result.rank == 4
    assert result.nullity == 0

    assert np.isclose(
        result.condition_number_nonzero,
        12.324776748897756,
        rtol=1e-10,
    )


def test_qt_noiseless_state_reconstruction():
    F = build_two_mode_drift(
        0.18
    )

    C = np.array([
        [1.0, 0.0, 0.0, 0.0]
    ])

    times = np.linspace(
        0.0,
        8.0,
        81,
    )

    H = build_discrete_observation_matrix(
        F,
        C,
        times,
    )

    truth = np.array([
        0.7,
        -0.2,
        0.5,
        0.45,
    ])

    estimate = reconstruct_state(
        H,
        H @ truth,
    )

    assert np.allclose(
        estimate,
        truth,
        atol=1e-10,
    )


def test_vacuum_covariance_is_physical():
    result = gaussian_physicality(
        0.5 * np.eye(4)
    )

    assert result["physical"]
