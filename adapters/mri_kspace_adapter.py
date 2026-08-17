import numpy as np

from qms_core import (
    Measurement,
    ReconstructedField,
    PlatformTwinState,
    ReconstructionAdapter,
)


class MRIKSpaceMeasurementAdapter:
    """
    Adapter for single-coil Cartesian MRI k-space.

    Expected shape:

        (T, Z, KY, KX)

    complex-valued.
    """

    def __init__(
        self,
        detector_id="mri-kspace-singlecoil",
        metadata=None,
    ):
        self.detector_id = detector_id
        self.metadata = dict(metadata or {})


    def to_measurement(self, raw):
        kspace = np.asarray(raw)

        if kspace.ndim != 4:
            raise ValueError(
                "Expected k-space shape "
                "(T,Z,KY,KX), "
                f"got {kspace.shape}"
            )

        if not np.iscomplexobj(kspace):
            raise ValueError(
                "MRI k-space input must be complex-valued."
            )

        return Measurement(
            data=kspace,
            modality="mri_kspace_cartesian_singlecoil",
            detector_id=self.detector_id,
            metadata=self.metadata.copy(),
        )


class MRIKSpaceReconstructionAdapter(
    ReconstructionAdapter
):
    """
    Minimal Cartesian MRI inverse reconstruction.

    Input:
        k-space (T,Z,KY,KX)

    Reconstruction:
        centered 2D inverse FFT independently for every
        time frame and z-slice.

    Platform output:
        reconstructed field (T,H,W,Z)
    """

    def __init__(self):
        self.version = 0


    @staticmethod
    def reconstruct_slice(kspace_slice):
        return np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(
                    kspace_slice
                )
            )
        )


    def reconstruct(
        self,
        measurement,
    ):
        if (
            measurement.modality
            != "mri_kspace_cartesian_singlecoil"
        ):
            raise ValueError(
                "MRIKSpaceReconstructionAdapter expects "
                "mri_kspace_cartesian_singlecoil"
            )

        kspace = np.asarray(
            measurement.data
        )

        T, Z, KY, KX = kspace.shape

        image = np.empty(
            (T, Z, KY, KX),
            dtype=np.complex128,
        )

        for t in range(T):
            for z in range(Z):
                image[t, z] = (
                    self.reconstruct_slice(
                        kspace[t, z]
                    )
                )

        # Convert:
        #
        # (T,Z,H,W)
        #      ->
        # (T,H,W,Z)

        field = np.transpose(
            image,
            (0, 2, 3, 1),
        )

        self.version += 1

        reconstructed = ReconstructedField(
            data=field,
            domain="mri_complex_image_volume",
            metadata={
                "input_domain":
                    "cartesian_kspace",

                "input_shape":
                    list(kspace.shape),

                "output_shape":
                    list(field.shape),

                "time_axis":
                    0,

                "contains_temporal_sequence":
                    True,

                "coil_count":
                    1,

                "reconstruction":
                    "centered_2d_ifft_per_slice",

                "raw_mri_inverse_reconstruction":
                    True,
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
                    "kspace-to-image",

                "reconstruction_model":
                    "single-coil Cartesian inverse FFT",
            },
        )

        return state, field


    def reset(self):
        self.version = 0
