import numpy as np

from qms_core.measurement_twin import (
    MeasurementState,
    predict_sequence,
    calibrate_frozen_threshold,
    consolidate_alerts,
    MeasurementAlert,
)


def test_constant_velocity_measurement_prediction():
    states = [
        MeasurementState(
            0, 100, 0.90, 0.10
        ),
        MeasurementState(
            1, 101, 0.91, 0.09
        ),
        MeasurementState(
            2, 102, 0.92, 0.08
        ),
    ]

    result = predict_sequence(
        states
    )

    assert len(result) == 1

    assert np.isclose(
        result[0].innovation_norm,
        0.0,
        atol=1e-12,
    )


def test_frozen_calibration():
    values = [
        0.001,
        0.002,
        0.003,
        0.002,
        0.001,
    ] * 4

    calibration = (
        calibrate_frozen_threshold(
            values
        )
    )

    assert (
        calibration.samples
        == 20
    )

    assert (
        calibration.threshold
        > calibration.median
    )


def test_alert_episode_consolidation():
    alerts = [
        MeasurementAlert(
            701, 0.10, 20.0, True
        ),
        MeasurementAlert(
            702, 0.09, 18.0, True
        ),
        MeasurementAlert(
            950, 0.11, 22.0, True
        ),
        MeasurementAlert(
            951, 0.12, 24.0, True
        ),
    ]

    episodes = consolidate_alerts(
        alerts,
        refractory=100,
    )

    assert len(episodes) == 2

    assert (
        episodes[0].peak_window
        == 701
    )

    assert (
        episodes[1].peak_window
        == 951
    )
