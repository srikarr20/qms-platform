import os
import sys
from pathlib import Path
import numpy as np

from qms_core import DetectorDiagnostics


DEFAULT_QMS_ROOT = (
    Path.home()
    / "Desktop"
    / "Quantum-Research"
    / "quantum-measurement-stack"
    / "repositories"
    / "quantum-measurement-stack"
)


def resolve_qms_root(explicit_root=None):
    """
    Resolve external QMS repository path.

    Priority:
        1. explicit constructor/function path
        2. QMS_RUNTIME_ROOT environment variable
        3. known local fallback
    """

    if explicit_root is not None:
        root = Path(
            explicit_root
        ).expanduser()

        if root.exists():
            return root

        raise FileNotFoundError(
            f"Explicit QMS root does not exist: {root}"
        )


    env_root = os.environ.get(
        "QMS_RUNTIME_ROOT"
    )

    if env_root:
        root = Path(
            env_root
        ).expanduser()

        if root.exists():
            return root

        raise FileNotFoundError(
            "QMS_RUNTIME_ROOT is set but does not exist: "
            f"{root}"
        )


    if DEFAULT_QMS_ROOT.exists():
        return DEFAULT_QMS_ROOT


    raise FileNotFoundError(
        "Could not locate quantum-measurement-stack. "
        "Pass qms_root explicitly or set QMS_RUNTIME_ROOT."
    )


def load_compute_visibility(
    qms_root=None,
):
    root = resolve_qms_root(
        qms_root
    )

    root_str = str(root)

    if root_str not in sys.path:
        sys.path.insert(
            0,
            root_str,
        )

    from qmctb.diagnostics.visibility import (
        compute_visibility,
    )

    return (
        compute_visibility,
        root,
    )


def enrich_with_qms_diagnostics(
    platform_state,
    qms_root=None,
):
    """
    Add QMS detector diagnostics to PlatformTwinState.

    External QMS location is resolved dynamically.
    """

    (
        compute_visibility,
        resolved_root,
    ) = load_compute_visibility(
        qms_root
    )

    measurement = (
        platform_state.measurement
    )

    data = np.asarray(
        measurement.data
    )

    diagnostics = (
        platform_state.detector_diagnostics
        or DetectorDiagnostics()
    )

    if (
        getattr(
            measurement,
            "modality",
            None,
        )
        == "quadrature"
    ):
        if data.shape[0] != 4:
            raise ValueError(
                "Quadrature measurement must contain "
                "I0, I90, I180, I270."
            )

        labels = [
            "I0",
            "I90",
            "I180",
            "I270",
        ]

        visibilities = {}
        uncertainties = {}

        for label, frame in zip(
            labels,
            data,
        ):
            V, sigma_V = (
                compute_visibility(
                    frame
                )
            )

            visibilities[
                label
            ] = float(V)

            uncertainties[
                label
            ] = float(
                sigma_V
            )

        values = np.asarray(
            list(
                visibilities.values()
            )
        )

        diagnostics.visibility = float(
            np.mean(values)
        )

        diagnostics.metadata.update({
            "quadrature_visibility":
                visibilities,

            "quadrature_visibility_uncertainty":
                uncertainties,

            "visibility_mean":
                float(
                    np.mean(values)
                ),

            "visibility_min":
                float(
                    np.min(values)
                ),

            "visibility_max":
                float(
                    np.max(values)
                ),

            "qms_diagnostic":
                (
                    "qmctb.diagnostics.visibility."
                    "compute_visibility"
                ),

            "qms_repository":
                str(
                    resolved_root
                ),

            "visibility_input":
                (
                    "individual quadrature "
                    "interferograms"
                ),
        })

    else:
        V, sigma_V = (
            compute_visibility(
                data
            )
        )

        diagnostics.visibility = float(
            V
        )

        diagnostics.metadata.update({
            "visibility_uncertainty":
                float(
                    sigma_V
                ),

            "qms_diagnostic":
                (
                    "qmctb.diagnostics.visibility."
                    "compute_visibility"
                ),

            "qms_repository":
                str(
                    resolved_root
                ),

            "visibility_input":
                "measurement intensity",
        })

    platform_state.detector_diagnostics = (
        diagnostics
    )

    return platform_state
