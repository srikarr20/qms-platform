from pathlib import Path
import numpy as np

from qms_core import (
    Measurement,
    ComplexFieldState,
    VirtualVolumeState,
    SourceEstimate,
    ObservableEstimate,
    TwinState,
)

# ============================================================
# QMS PLATFORM — DPI TWIN DEMO
#
# Forward:
#   source -> propagation -> downstream detector
#
# Inverse:
#   quadrature measurement -> complex detector field
#   -> virtual upstream propagation -> source estimate
# ============================================================

rng = np.random.default_rng(2001)

# ------------------------------------------------------------
# Physics
# ------------------------------------------------------------

N = 96
wavelength = 810e-9
dx = 8e-6
source_depth = 0.100

coord = (np.arange(N) - N // 2) * dx

X, Y = np.meshgrid(
    coord,
    coord,
    indexing="xy"
)

fx = np.fft.fftfreq(
    N,
    d=dx
)

fy = np.fft.fftfreq(
    N,
    d=dx
)

FX, FY = np.meshgrid(
    fx,
    fy,
    indexing="xy"
)


def transfer(z):

    return np.exp(
        -1j
        * np.pi
        * wavelength
        * z
        * (
            FX**2
            + FY**2
        )
    )


def propagate(field, H):

    return np.fft.ifft2(
        np.fft.fft2(field)
        * H
    )


# ------------------------------------------------------------
# Hidden upstream source
# ------------------------------------------------------------

source_x = 35e-6
source_y = -25e-6

sigma = 30e-6

source_true = np.exp(
    -(
        (X - source_x)**2
        + (Y - source_y)**2
    )
    / (
        2 * sigma**2
    )
)

source_true = source_true.astype(
    complex
)

source_true *= np.exp(
    1j * 0.35
)

# ------------------------------------------------------------
# Forward physical propagation
# ------------------------------------------------------------

H_forward = transfer(
    source_depth
)

sensor_true = propagate(
    source_true,
    H_forward
)

# ------------------------------------------------------------
# 4-quadrature measurement
# ------------------------------------------------------------

reference_amp = 1.5

kx = 2*np.pi / 160e-6
ky = 2*np.pi / 210e-6

R = (
    reference_amp
    * np.exp(
        1j
        * (
            kx*X
            + ky*Y
        )
    )
)


def detector(E, ref):

    return np.abs(
        E + ref
    )**2


I0 = detector(
    sensor_true,
    R
)

I90 = detector(
    sensor_true,
    1j*R
)

I180 = detector(
    sensor_true,
    -R
)

I270 = detector(
    sensor_true,
    -1j*R
)

measurements = np.stack([
    I0,
    I90,
    I180,
    I270,
])

measurement = Measurement(
    data=measurements,
    modality="quadrature",
    detector_id="simulated-4Q-01",
    metadata={
        "wavelength": wavelength,
        "pixel_spacing": dx,
        "source_depth_truth": source_depth,
    },
)

# ------------------------------------------------------------
# DPI detector-field reconstruction
# ------------------------------------------------------------

Qc = I0 - I180
Qs = I90 - I270

sensor_reconstructed = (
    Qc + 1j*Qs
) / (
    4*np.conj(R)
    + 1e-15
)

detector_field = ComplexFieldState(
    field=sensor_reconstructed,
    wavelength=wavelength,
    pixel_spacing=dx,
    z=0.0,
    metadata={
        "reconstruction": "4-quadrature"
    },
)

# ------------------------------------------------------------
# DPI virtual propagation volume
# ------------------------------------------------------------

depths = np.arange(
    0.0,
    source_depth + 0.0001,
    0.005
)

volume = np.empty(
    (
        len(depths),
        N,
        N
    ),
    dtype=np.complex64
)

for i, z in enumerate(depths):

    H_back = transfer(
        -z
    )

    volume[i] = propagate(
        sensor_reconstructed,
        H_back
    )

virtual_volume = VirtualVolumeState(
    field=volume,
    depths=depths,
    wavelength=wavelength,
    pixel_spacing=dx,
    metadata={
        "direction": "downstream-to-upstream",
        "method": "Fresnel transfer function",
    },
)

# ------------------------------------------------------------
# Simple source-plane localization
# ------------------------------------------------------------

source_plane = volume[-1]

I_source = np.abs(
    source_plane
)**2

total = (
    I_source.sum()
    + 1e-15
)

cx = float(
    np.sum(
        I_source * X
    ) / total
)

cy = float(
    np.sum(
        I_source * Y
    ) / total
)

source_estimate = SourceEstimate(
    position=(
        cx,
        cy,
        source_depth,
    ),
    confidence=1.0,
    metadata={
        "method": "virtual source-plane centroid"
    },
)

observable = ObservableEstimate(
    name="source_position",
    value=source_estimate.position,
    confidence=source_estimate.confidence,
    best_depth=source_depth,
)

# ------------------------------------------------------------
# Assemble shared twin state
# ------------------------------------------------------------

twin = TwinState(
    measurement=measurement,
    detector_field=detector_field,
    virtual_volume=virtual_volume,
    source=source_estimate,
    observables=[
        observable
    ],
    version=1,
    metadata={
        "pipeline": (
            "measurement -> complex field -> "
            "virtual volume -> source estimate"
        )
    },
)

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

position_error_um = (
    1e6
    * np.sqrt(
        (cx - source_x)**2
        + (cy - source_y)**2
    )
)

field_error = (
    np.linalg.norm(
        source_plane - source_true
    )
    /
    (
        np.linalg.norm(
            source_true
        )
        + 1e-15
    )
)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print()
print("=" * 72)
print("QMS PLATFORM — DPI TWIN INTEGRATION")
print("=" * 72)

print(
    "Measurement:",
    twin.measurement.modality
)

print(
    "Detector:",
    twin.measurement.detector_id
)

print(
    "Virtual planes:",
    len(
        twin.virtual_volume.depths
    )
)

print(
    "Virtual depth range:",
    twin.virtual_volume.depths[0],
    "to",
    twin.virtual_volume.depths[-1],
    "m"
)

print()
print(
    "True source:",
    (
        source_x,
        source_y,
        source_depth,
    )
)

print(
    "Recovered source:",
    twin.source.position
)

print(
    "Position error:",
    f"{position_error_um:.6f}",
    "microns"
)

print(
    "Source-plane complex error:",
    f"{field_error:.10f}"
)

print()
print(
    "Pipeline:",
    twin.metadata["pipeline"]
)

print()
print("DPI TWIN STATE OK")
