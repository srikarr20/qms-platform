from abc import ABC, abstractmethod

from .state import Measurement


class MeasurementAdapter(ABC):
    """
    Convert modality-specific raw input into the common
    qms_core.Measurement contract.
    """

    @abstractmethod
    def to_measurement(self, raw):
        raise NotImplementedError


class QuadratureMeasurementAdapter(MeasurementAdapter):
    """
    Adapter for optical 4-quadrature detector data.

    Expected raw form:
        array-like shape (4, H, W)
        ordered as:
            I0, I90, I180, I270
    """

    def __init__(
        self,
        detector_id="quadrature-detector",
        metadata=None,
    ):
        self.detector_id = detector_id
        self.metadata = dict(
            metadata or {}
        )

    def to_measurement(self, raw):
        return Measurement(
            data=raw,
            modality="quadrature",
            detector_id=self.detector_id,
            metadata=self.metadata.copy(),
        )
