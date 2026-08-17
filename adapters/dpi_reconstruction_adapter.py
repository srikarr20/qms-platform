from qms_core import ReconstructionAdapter

from adapters.dpi_twin_adapter import (
    adapt_dpi_twin_state,
)


class DPIReconstructionAdapter(
    ReconstructionAdapter
):
    """
    Optical/DPI implementation of the platform reconstruction
    interface.

    Measurement
        -> QuantumMeasurementTwin
        -> DPI TwinState
        -> PlatformTwinState
    """

    def __init__(
        self,
        dpi_twin,
    ):
        self.dpi_twin = dpi_twin


    def reconstruct(
        self,
        measurement,
    ):
        dpi_state = (
            self.dpi_twin.update(
                measurement
            )
        )

        platform_state = (
            adapt_dpi_twin_state(
                dpi_state
            )
        )

        return (
            platform_state,
            dpi_state.detector_field.field,
        )


    def reset(self):
        if hasattr(
            self.dpi_twin,
            "version",
        ):
            self.dpi_twin.version = 0

        if hasattr(
            self.dpi_twin,
            "state",
        ):
            self.dpi_twin.state = None
