import numpy as np

from qms_core.detector_twin import (
    first_divergent_stage,
    build_inverse_calibration,
    assess_detector_state,
)


def test_stage_localization():
    baseline = np.zeros(
        (8, 8)
    )

    pixel = baseline.copy()
    pixel[2:5, :] = -1.0

    image = pixel + 0.1

    result = first_divergent_stage(
        {
            "PIXEL": pixel,
            "IMAGE": image,
        },
        {
            "PIXEL": baseline,
            "IMAGE": baseline,
        },
        stage_order=[
            "PIXEL",
            "IMAGE",
        ],
    )

    assert (
        result.first_divergent_stage
        == "PIXEL"
    )


def test_monotonic_inverse_calibration():
    calibration = (
        build_inverse_calibration(
            observed_support=[
                10.0,
                20.0,
                40.0,
            ],
            parameter_support=[
                1e-4,
                2e-4,
                4e-4,
            ],
        )
    )

    assert np.isclose(
        calibration.estimate(
            30.0
        ),
        3e-4,
    )


def test_detector_predict_assimilate():
    history = [
        {
            "cti": 1e9,
            "noise": 2e-4,
        },
        {
            "cti": 2e9,
            "noise": 3e-4,
        },
    ]

    result = assess_detector_state(
        history,
        {
            "cti": 3e9,
            "noise": 4e-4,
        },
        fields=[
            "cti",
            "noise",
        ],
        thresholds={
            "cti": 0.20,
            "noise": 0.10,
        },
    )

    assert result["mode"] == "ASSIMILATE"
