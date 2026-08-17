from qms_core import (
    QuadratureMeasurementAdapter,
)

from twin import (
    QuantumMeasurementTwin,
    MeasurementTwinPlatform,
)

from adapters.dpi_reconstruction_adapter import (
    DPIReconstructionAdapter,
)

from adapters.mri_kspace_adapter import (
    MRIKSpaceMeasurementAdapter,
    MRIKSpaceReconstructionAdapter,
)


def create_optical_platform(
    *,
    n,
    dx,
    wavelength,
    depths,
    reference,
    detector_id="optical-4q",
    min_dynamics_states=4,
    recorder=None,
    metadata=None,
):
    """
    Build a complete optical/QMS/DPI/AURORA platform.
    """

    dpi_twin = QuantumMeasurementTwin(
        n=n,
        dx=dx,
        wavelength=wavelength,
        depths=depths,
        reference=reference,
    )

    reconstruction_adapter = (
        DPIReconstructionAdapter(
            dpi_twin=dpi_twin
        )
    )

    measurement_adapter = (
        QuadratureMeasurementAdapter(
            detector_id=detector_id,
            metadata={
                "wavelength": wavelength,
                "pixel_spacing": dx,
                **(metadata or {}),
            },
        )
    )

    return MeasurementTwinPlatform(
        reconstruction_adapter=
            reconstruction_adapter,

        measurement_adapter=
            measurement_adapter,

        min_dynamics_states=
            min_dynamics_states,

        recorder=
            recorder,
    )


def create_mri_platform(
    *,
    detector_id="mri-kspace",
    min_dynamics_states=4,
    recorder=None,
    metadata=None,
):
    """
    Build a single-coil Cartesian MRI/AURORA platform.
    """

    measurement_adapter = (
        MRIKSpaceMeasurementAdapter(
            detector_id=detector_id,
            metadata=metadata,
        )
    )

    reconstruction_adapter = (
        MRIKSpaceReconstructionAdapter()
    )

    return MeasurementTwinPlatform(
        reconstruction_adapter=
            reconstruction_adapter,

        measurement_adapter=
            measurement_adapter,

        min_dynamics_states=
            min_dynamics_states,

        recorder=
            recorder,
    )
