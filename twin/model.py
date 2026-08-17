import numpy as np

from qms_core import (
    Measurement,
    ComplexFieldState,
    VirtualVolumeState,
    SourceEstimate,
    ObservableEstimate,
    TwinState,
)

from dpi.reconstruction import (
    make_frequency_grid,
    recover_4q_field,
)

from dpi.source_search import (
    blind_depth_search,
)


class QuantumMeasurementTwin:
    """
    Central detector-driven upstream twin.

    Pipeline:
        measurement
        -> detector complex field
        -> virtual upstream volume
        -> blind source inference
        -> TwinState
    """

    def __init__(
        self,
        n,
        dx,
        wavelength,
        depths,
        reference,
    ):
        self.n = n
        self.dx = dx
        self.wavelength = wavelength
        self.depths = np.asarray(depths)
        self.reference = reference

        coord = (
            np.arange(n)
            - n // 2
        ) * dx

        self.X, self.Y = np.meshgrid(
            coord,
            coord,
            indexing="xy",
        )

        self.FX, self.FY = make_frequency_grid(
            n,
            dx,
        )

        self.state = None
        self.version = 0


    def update(
        self,
        measurement: Measurement,
    ):
        if measurement.modality != "quadrature":
            raise ValueError(
                "Current prototype expects quadrature data."
            )

        I0, I90, I180, I270 = measurement.data

        sensor_field = recover_4q_field(
            I0,
            I90,
            I180,
            I270,
            self.reference,
        )

        search = blind_depth_search(
            sensor_field,
            self.depths,
            self.FX,
            self.FY,
            self.wavelength,
            self.X,
            self.Y,
        )

        detector_state = ComplexFieldState(
            field=sensor_field,
            wavelength=self.wavelength,
            pixel_spacing=self.dx,
            z=0.0,
            timestamp=measurement.timestamp,
        )

        volume_state = VirtualVolumeState(
            field=search["volume"],
            depths=self.depths,
            wavelength=self.wavelength,
            pixel_spacing=self.dx,
            timestamp=measurement.timestamp,
            metadata={
                "direction":
                    "downstream-to-upstream",
            },
        )

        source = SourceEstimate(
            position=(
                search["x"],
                search["y"],
                search["best_depth"],
            ),
            metadata={
                "method":
                    "blind minimum-width search",
            },
        )

        depth_observable = ObservableEstimate(
            name="source_depth",
            value=search["best_depth"],
            best_depth=search["best_depth"],
            metadata={
                "width_curve":
                    search["widths"],
            },
        )

        self.version += 1

        self.state = TwinState(
            measurement=measurement,
            detector_field=detector_state,
            virtual_volume=volume_state,
            source=source,
            observables=[
                depth_observable
            ],
            version=self.version,
            metadata={
                "architecture":
                    (
                        "measurement -> reconstruction -> "
                        "virtual volume -> source inference"
                    ),
            },
        )

        return self.state
