import numpy as np

from qms_core import (
    Measurement,
    ReconstructedField,
    PlatformTwinState,
    ReconstructionAdapter,
)


class MRIReconstructedMeasurementAdapter:
    """
    Ingest an already reconstructed 4D MRI volume.

    Expected raw input:
        V.shape == (H, W, Z, T)
    """

    def __init__(
        self,
        detector_id="mri-reconstructed-volume",
        metadata=None,
    ):
        self.detector_id = detector_id
        self.metadata = dict(metadata or {})


    def to_measurement(self, raw):
        V = np.asarray(raw)

        if V.ndim != 4:
            raise ValueError(
                f"Expected reconstructed MRI volume "
                f"(H,W,Z,T), got {V.shape}"
            )

        return Measurement(
            data=V,
            modality="mri_reconstructed_4d",
            detector_id=self.detector_id,
            metadata=self.metadata.copy(),
        )


class MRIReconstructedFieldAdapter(
    ReconstructionAdapter
):
    """
    Platform reconstruction adapter for an MRI volume that
    has ALREADY been reconstructed from acquisition data.

    Important:
        This does not perform k-space -> image reconstruction.

    It translates:
        V(H,W,Z,T)
            ->
        ReconstructedField(T,H,W,Z)
    """

    def __init__(self):
        self.version = 0


    def reconstruct(
        self,
        measurement,
    ):
        if (
            measurement.modality
            != "mri_reconstructed_4d"
        ):
            raise ValueError(
                "MRIReconstructedFieldAdapter expects "
                "modality='mri_reconstructed_4d'"
            )

        V = np.asarray(
            measurement.data
        )

        if V.ndim != 4:
            raise ValueError(
                f"Expected 4D MRI field, got {V.shape}"
            )

        # AURORA convention:
        #     (H,W,Z,T)
        #
        # Platform temporal convention:
        #     (T,H,W,Z)

        V_time_first = np.moveaxis(
            V,
            -1,
            0,
        )

        self.version += 1

        reconstructed = ReconstructedField(
            data=V_time_first,

            domain="mri_reconstructed_volume",

            metadata={
                "original_shape":
                    list(V.shape),

                "platform_shape":
                    list(V_time_first.shape),

                "time_axis":
                    0,

                "source":
                    "already_reconstructed_mri",

                "raw_mri_inverse_reconstruction":
                    False,
            },
        )

        state = PlatformTwinState(
            measurement=measurement,

            reconstructed_field=
                reconstructed,

            version=self.version,

            metadata={
                "modality":
                    "MRI",

                "reconstruction_stage":
                    "post-reconstruction",

                "note":
                    (
                        "Input is reconstructed MRI volume; "
                        "raw k-space inversion is not performed."
                    ),
            },
        )

        return (
            state,
            V_time_first,
        )


    def reset(self):
        self.version = 0
