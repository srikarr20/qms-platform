import numpy as np


def make_frequency_grid(n, dx):
    fx = np.fft.fftfreq(n, d=dx)
    fy = np.fft.fftfreq(n, d=dx)
    return np.meshgrid(fx, fy, indexing="xy")


def fresnel_transfer(FX, FY, wavelength, z):
    return np.exp(
        -1j
        * np.pi
        * wavelength
        * z
        * (FX**2 + FY**2)
    )


def propagate(field, FX, FY, wavelength, z):
    H = fresnel_transfer(
        FX,
        FY,
        wavelength,
        z,
    )

    return np.fft.ifft2(
        np.fft.fft2(field)
        * H
    )


def recover_4q_field(
    I0,
    I90,
    I180,
    I270,
    reference,
):
    return (
        (I0 - I180)
        + 1j * (I90 - I270)
    ) / (
        4 * np.conj(reference)
        + 1e-15
    )
