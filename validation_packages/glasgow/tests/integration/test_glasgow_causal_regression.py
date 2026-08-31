from pathlib import Path
import os

import numpy as np
import pytest

from integrations.glasgow import (
    GlasgowCumulativeArchive,
)

from qms_core.measurement_twin import (
    run_causal_replay,
)


GLASGOW_ZIP = Path(
    os.environ.get(
        "GLASGOW_ZIP",
        str(
            Path.home()
            / "Desktop"
            / "Quantum-Research"
            / "experimental-data"
            / "glasgow-single-photon"
            / "dpi_lab_1"
            / "Heralded Diffraction SM.zip"
        ),
    )
)


def test_glasgow_causal_replay():

    if not GLASGOW_ZIP.exists():
        pytest.skip(
            "Glasgow experimental archive "
            "is not available."
        )

    archive = GlasgowCumulativeArchive(
        GLASGOW_ZIP
    )

    increments = {}

    for record in (
        archive.iter_increments()
    ):
        increments[
            int(record["index"])
        ] = np.asarray(
            record["increment"],
            dtype=float,
        )

    result = run_causal_replay(
        increments,
        reference_start=0,
        reference_end=399,
        calibration_end=699,
        window=100,
        grid=32,
        refractory=100,
    )

    summary = result["summary"]
    calibration = result[
        "calibration"
    ]
    alerts = result["alerts"]
    episodes = result[
        "episodes"
    ]

    assert (
        calibration.samples
        == 199
    )

    assert np.isclose(
        calibration.median,
        0.00448429,
        atol=1e-8,
    )

    assert np.isclose(
        calibration.mad,
        0.00229512,
        atol=1e-8,
    )

    assert np.isclose(
        calibration.threshold,
        0.02490077,
        atol=1e-8,
    )

    assert (
        summary.future_samples
        == 3369
    )

    assert (
        summary.raw_flags
        == 64
    )

    peaks = [
        episode.peak_window
        for episode in episodes
    ]

    assert peaks == [
        951,
        1598,
        2316,
        2761,
        3458,
        3875,
    ]

    for target in (
        701,
        801,
    ):

        nearest = min(
            alerts,
            key=lambda alert:
                abs(
                    alert.window_end
                    - target
                ),
        )

        assert (
            nearest.window_end
            == target
        )

        assert nearest.flagged
