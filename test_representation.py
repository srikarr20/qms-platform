import numpy as np
import pytest

from qms_core.representation import (
    participation_ratio,
    covariance_spectrum,
    effective_dimension,
    explained_variance_fraction,
    observable_sensitivity,
)


def test_participation_ratio_single_component():
    assert np.isclose(
        participation_ratio(np.array([1.0, 0.0, 0.0])),
        1.0,
    )


def test_participation_ratio_equal_components():
    assert np.isclose(
        participation_ratio(np.array([1.0, 1.0, 1.0])),
        3.0,
    )


def test_effective_dimension_rank_one():
    x = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ])

    assert np.isclose(effective_dimension(x), 1.0)


def test_explained_variance_rank_one():
    x = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ])

    assert np.isclose(
        explained_variance_fraction(x, 1),
        1.0,
    )


def test_observable_sensitivity_linear():
    control = np.array([0.0, 1.0, 2.0, 3.0])
    observable = np.array([1.0, 3.0, 5.0, 7.0])

    result = observable_sensitivity(observable, control)

    assert np.isclose(result["correlation"], 1.0)
    assert np.isclose(result["slope"], 2.0)
    assert np.isclose(result["intercept"], 1.0)


def test_constant_observable():
    control = np.arange(5, dtype=float)
    observable = np.ones(5)

    result = observable_sensitivity(observable, control)

    assert result["correlation"] == 0.0
    assert np.isclose(result["slope"], 0.0)


def test_invalid_negative_spectrum():
    with pytest.raises(ValueError):
        participation_ratio(np.array([1.0, -1.0]))


def test_covariance_requires_multiple_samples():
    with pytest.raises(ValueError):
        covariance_spectrum(np.array([[1.0, 2.0]]))


def test_participation_ratio_small_scale_is_scale_invariant():
    spectrum = np.array([
        8.55747873e-09,
        8.40866300e-10,
    ])

    expected = (
        np.sum(spectrum) ** 2
        / np.sum(spectrum ** 2)
    )

    assert np.isclose(
        participation_ratio(spectrum),
        expected,
    )
