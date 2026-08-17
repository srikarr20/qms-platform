from qms_core import (
    DetectorDiagnostics,
    ReconstructedField,
    UpstreamEstimate,
    ObservabilityState,
    PlatformTwinState,
)


def adapt_dpi_twin_state(state):
    """
    Convert DPI TwinState into the platform V2 contract.

    DPI contributes:
        reconstructed detector field
        upstream source estimate
        depth observability
    """

    source = state.source
    x, y, z = source.position

    upstream = UpstreamEstimate(
        x=float(x),
        y=float(y),
        z=float(z),
        field=None,
        confidence=getattr(
            source,
            "confidence",
            None,
        ),
        metadata={
            "source_metadata":
                getattr(
                    source,
                    "metadata",
                    {},
                ),
            "origin":
                "dpi",
        },
    )

    reconstructed_field = ReconstructedField(
        data=state.detector_field.field,
        domain="optical_complex_detector_field",
        timestamp=state.measurement.timestamp,
        metadata={
            "wavelength":
                state.detector_field.wavelength,

            "pixel_spacing":
                state.detector_field.pixel_spacing,

            "measurement_plane_z":
                state.detector_field.z,

            "origin":
                "dpi",
        },
    )

    depth_score = None

    if state.observables:
        observable = state.observables[0]

        metadata = getattr(
            observable,
            "metadata",
            {},
        )

        depth_score = metadata.get(
            "width_curve"
        )

    observability = ObservabilityState(
        depth_score=depth_score,
        metadata={
            "method":
                "blind-depth-source-width",
        },
    )

    diagnostics = DetectorDiagnostics(
        metadata={
            "detector_id":
                state.measurement.detector_id,

            "modality":
                state.measurement.modality,
        },
    )

    return PlatformTwinState(
        measurement=state.measurement,

        detector_diagnostics=diagnostics,

        reconstructed_field=reconstructed_field,

        upstream=upstream,

        observability=observability,

        version=state.version,

        metadata={
            "source":
                "QuantumMeasurementTwin",

            "direction":
                "detector-to-upstream",
        },
    )
