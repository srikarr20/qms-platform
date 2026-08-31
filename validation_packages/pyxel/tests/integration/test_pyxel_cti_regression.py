from pathlib import Path
import os

import pytest


PYXEL_VALIDATION_ROOT = os.environ.get(
    "QMS_PYXEL_VALIDATION_ROOT"
)


def test_pyxel_blind_cti_recovery():

    pytest.importorskip(
        "pyxel",
        reason="Pyxel is required for this regression.",
    )

    if not PYXEL_VALIDATION_ROOT:
        pytest.skip(
            "Set QMS_PYXEL_VALIDATION_ROOT to the "
            "external Pyxel validation-data directory."
        )

    root = Path(
        PYXEL_VALIDATION_ROOT
    ).expanduser().resolve()

    from qms_core.detector_twin.io import load_pixel

    from integrations.pyxel import (
        estimate_parallel_cti_density,
    )

    baseline_root = (
        root
        / "qms_pyxel_001"
        / "cti_observable_results"
        / "cti_000"
    )

    observed_root = (
        root
        / "qms_pyxel_twin_001"
        / "blind_cti_5e9"
    )

    if (
        not baseline_root.exists()
        or not observed_root.exists()
    ):
        pytest.skip(
            "Expected Pyxel validation directories "
            "were not found under "
            "QMS_PYXEL_VALIDATION_ROOT."
        )

    baseline = load_pixel(
        baseline_root
    )

    observed = load_pixel(
        observed_root
    )

    result = (
        estimate_parallel_cti_density(
            baseline,
            observed,
            log10_bounds=(8.0, 11.0),
        )
    )

    expected_density = 5e9

    relative_error = abs(
        result.trap_density
        - expected_density
    ) / expected_density

    assert result.optimization_success
    assert relative_error < 1e-4
    assert result.residual_rmse < 1e-5
