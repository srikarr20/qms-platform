from __future__ import annotations

import numpy as np


def participation_ratio(values, eps=1e-15):
    """Return the participation ratio of a non-negative spectrum.

    The calculation is scale invariant. Values are normalized by the
    largest spectral component before evaluating the ratio so that valid
    small-magnitude covariance spectra are not mistaken for zero.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")

    if np.any(values < -eps):
        raise ValueError("values must be non-negative")

    # Clamp tiny negative numerical round-off.
    values = np.maximum(values, 0.0)

    maximum = float(np.max(values)) if values.size else 0.0

    if maximum == 0.0:
        return 0.0

    scaled = values / maximum

    numerator = float(np.sum(scaled)) ** 2
    denominator = float(np.sum(scaled ** 2))

    if denominator == 0.0:
        return 0.0

    return numerator / denominator

def covariance_spectrum(
    measurements: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    """
    Return descending eigenvalues of the sample covariance matrix.

    Expected shape:
        (n_samples, n_features)
    """
    x = np.asarray(measurements, dtype=float)

    if x.ndim != 2:
        raise ValueError("measurements must be a 2D array")

    if x.shape[0] < 2:
        raise ValueError("at least two samples are required")

    if center:
        x = x - np.mean(x, axis=0, keepdims=True)

    singular_values = np.linalg.svd(
        x,
        full_matrices=False,
        compute_uv=False,
    )

    eigenvalues = (singular_values ** 2) / (x.shape[0] - 1)

    return np.sort(eigenvalues)[::-1]


def effective_dimension(
    measurements: np.ndarray,
    *,
    center: bool = True,
) -> float:
    """
    Estimate representation effective dimension using the
    participation ratio of the covariance spectrum.
    """
    spectrum = covariance_spectrum(
        measurements,
        center=center,
    )

    return participation_ratio(spectrum)


def explained_variance_fraction(
    measurements: np.ndarray,
    n_components: int,
    *,
    center: bool = True,
) -> float:
    """
    Fraction of covariance variance explained by the first
    n_components principal directions.
    """
    spectrum = covariance_spectrum(
        measurements,
        center=center,
    )

    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    total = np.sum(spectrum)

    if total <= 0:
        return 0.0

    k = min(n_components, len(spectrum))

    return float(np.sum(spectrum[:k]) / total)


def observable_sensitivity(
    observable: np.ndarray,
    control: np.ndarray,
) -> dict[str, float]:
    """
    Basic sensitivity of a scalar observable to an experimental
    control/state variable.

    Returns Pearson correlation and a linear least-squares slope.

    This is a diagnostic, not by itself evidence of causality or
    predictive validity.
    """
    y = np.asarray(observable, dtype=float)
    x = np.asarray(control, dtype=float)

    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("observable and control must be 1D arrays")

    if len(y) != len(x):
        raise ValueError("observable and control must have equal length")

    if len(x) < 2:
        raise ValueError("at least two samples are required")

    if np.std(x) == 0 or np.std(y) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(x, y)[0, 1])

    A = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    return {
        "correlation": correlation,
        "slope": float(slope),
        "intercept": float(intercept),
    }
