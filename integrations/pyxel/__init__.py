"""
QMS integration for ESA Pyxel.

This package contains Pyxel-specific detector-model
adaptation and inverse-estimation utilities.

Pyxel is an external dependency and is not required
to use the provider-independent QMS core.
"""

from .pyxel_cti import (
    PyxelCTIConfig,
    CTIEstimate,
    simulate_parallel_cti,
    estimate_parallel_cti_density,
)

__all__ = [
    "PyxelCTIConfig",
    "CTIEstimate",
    "simulate_parallel_cti",
    "estimate_parallel_cti_density",
]
