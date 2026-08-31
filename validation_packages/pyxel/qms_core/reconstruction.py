from abc import ABC, abstractmethod


class ReconstructionAdapter(ABC):
    """
    Convert a Measurement into a PlatformTwinState-compatible
    reconstructed state.
    """

    @abstractmethod
    def reconstruct(self, measurement):
        raise NotImplementedError
