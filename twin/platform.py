import numpy as np

from twin.observability_layer import (
    build_observability_layer,
)

from adapters.qms_diagnostics_adapter import (
    enrich_with_qms_diagnostics,
)

from adapters.aurora_dynamics_adapter import (
    enrich_with_aurora_dynamics,
)


class MeasurementTwinPlatform:
    """
    Modality-aware measurement-twin runtime.

    Public API:

        state = platform.ingest(raw)

    Generic flow:

        raw
        -> MeasurementAdapter
        -> Measurement
        -> ReconstructionAdapter
        -> ReconstructedField
        -> temporal field evolution
        -> DetectorState
        -> ObservableManifold
        -> AURORA dynamics
        -> PlatformTwinState

    Temporal handling:

        Optical/DPI:
            one reconstructed field per measurement;
            history accumulates over successive updates.

        MRI:
            one reconstruction may already contain an entire
            temporal 4D field sequence.
    """

    def __init__(
        self,
        reconstruction_adapter,
        min_dynamics_states=4,
        recorder=None,
        measurement_adapter=None,
        max_history=64,
    ):
        self.reconstruction_adapter = (
            reconstruction_adapter
        )

        self.measurement_adapter = (
            measurement_adapter
        )

        self.min_dynamics_states = (
            min_dynamics_states
        )

        self.recorder = recorder

        # Used for modalities where each measurement produces
        # one field snapshot, e.g. current optical DPI.
        self.field_history = []

        self.state_history = []

        self.current_state = None

        self.measurements_processed = 0

        self.max_history = max_history


    def ingest(
        self,
        raw,
    ):
        """
        Convert raw modality-specific input to Measurement
        and execute the complete twin pipeline.
        """

        if self.measurement_adapter is None:
            raise ValueError(
                "No measurement_adapter configured."
            )

        measurement = (
            self.measurement_adapter.to_measurement(
                raw
            )
        )

        return self.update(
            measurement
        )


    def update(
        self,
        measurement,
    ):
        # ====================================================
        # 1. MODALITY-SPECIFIC INVERSE RECONSTRUCTION
        # ====================================================

        (
            platform_state,
            reconstructed_data,
        ) = self.reconstruction_adapter.reconstruct(
            measurement
        )

        self.measurements_processed += 1


        # ====================================================
        # 2. DETECTOR DIAGNOSTICS
        #
        # Current QMS visibility diagnostic is specifically
        # meaningful for optical quadrature interferograms.
        #
        # Do NOT apply it blindly to MRI k-space.
        # ====================================================

        if (
            getattr(
                measurement,
                "modality",
                None,
            )
            == "quadrature"
        ):
            platform_state = (
                enrich_with_qms_diagnostics(
                    platform_state
                )
            )


        # ====================================================
        # 3. DETERMINE TEMPORAL FIELD SEQUENCE
        # ====================================================

        reconstructed_data = np.asarray(
            reconstructed_data
        )

        reconstructed_field = (
            platform_state.reconstructed_field
        )

        metadata = {}

        if reconstructed_field is not None:
            metadata = (
                reconstructed_field.metadata
                or {}
            )

        time_axis = metadata.get(
            "time_axis",
            None,
        )

        temporal_sequence = None


        # ----------------------------------------------------
        # CASE A:
        # Reconstruction already contains temporal sequence.
        #
        # MRI adapter:
        #     (T,H,W,Z)
        # ----------------------------------------------------

        if (
            time_axis == 0
            and
            reconstructed_data.shape[0] >= 2
        ):
            temporal_sequence = (
                reconstructed_data
            )

            platform_state.metadata[
                "temporal_source"
            ] = "reconstruction_internal_sequence"


        # ----------------------------------------------------
        # CASE B:
        # One reconstructed field per measurement.
        #
        # Optical DPI:
        #     measurement 1 -> Psi_1
        #     measurement 2 -> Psi_2
        #     ...
        # ----------------------------------------------------

        else:
            self.field_history.append(
                reconstructed_data.copy()
            )

            if (
                self.max_history is not None
                and
                len(self.field_history) > self.max_history
            ):
                self.field_history = self.field_history[
                    -self.max_history:
                ]

            if len(self.field_history) >= 2:
                temporal_sequence = np.asarray(
                    self.field_history
                )

            platform_state.metadata[
                "temporal_source"
            ] = "stream_history"


        # ====================================================
        # 4. SHARED OBSERVABILITY LAYER
        # ====================================================

        if (
            temporal_sequence is not None
            and
            temporal_sequence.shape[0] >= 2
        ):
            (
                detector_state,
                manifold,
            ) = build_observability_layer(
                temporal_sequence,

                field_domain=(
                    reconstructed_field.domain
                    if reconstructed_field
                    is not None
                    else "reconstructed_field"
                ),
            )

            platform_state.detector_state = (
                detector_state
            )

            platform_state.manifold = (
                manifold
            )


        # ====================================================
        # 5. AURORA DYNAMICAL OBSERVABILITY
        # ====================================================

        if (
            platform_state.manifold
            is not None
            and
            platform_state.manifold.state.shape[0]
            >= self.min_dynamics_states
        ):
            platform_state = (
                enrich_with_aurora_dynamics(
                    platform_state
                )
            )


        # ====================================================
        # 6. RUNTIME STATE
        # ====================================================

        platform_state.metadata.update({
            "runtime":
                "MeasurementTwinPlatform",

            "measurements_processed":
                self.measurements_processed,

            "dynamics_ready":
                (
                    platform_state.dynamics
                    is not None
                ),
        })

        self.current_state = (
            platform_state
        )

        self.state_history.append(
            platform_state
        )


        # ====================================================
        # 7. PERSISTENCE
        # ====================================================

        if self.recorder is not None:
            self.recorder.record(
                platform_state
            )


        return platform_state


    def reset(
        self,
    ):
        self.field_history = []

        self.state_history = []

        self.current_state = None

        self.measurements_processed = 0

        if hasattr(
            self.reconstruction_adapter,
            "reset",
        ):
            self.reconstruction_adapter.reset()


    @property
    def measurement_count(
        self,
    ):
        return self.measurements_processed


    @property
    def dynamics_ready(
        self,
    ):
        return (
            self.current_state
            is not None
            and
            self.current_state.dynamics
            is not None
        )
