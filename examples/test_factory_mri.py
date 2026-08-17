import numpy as np
import qms_platform as qmp


T = 12
H = 64
W = 64
Z = 6

x = np.linspace(-1.0, 1.0, H)
y = np.linspace(-1.0, 1.0, W)
z = np.linspace(-1.0, 1.0, Z)

X, Y, ZG = np.meshgrid(
    x,
    y,
    z,
    indexing="ij",
)

truth = np.zeros(
    (T, H, W, Z),
    dtype=np.complex128,
)


for t in range(T):

    phase_t = 2*np.pi*t/T

    cx = 0.20*np.sin(phase_t)
    cy = 0.14*np.cos(phase_t)

    magnitude = np.exp(
        -(
            (X-cx)**2/(2*0.24**2)
            +
            (Y-cy)**2/(2*0.20**2)
            +
            ZG**2/(2*0.50**2)
        )
    )

    truth[t] = (
        magnitude
        * np.exp(
            1j * (
                0.20
                + 0.12*X
                - 0.08*Y
            )
        )
    )


truth_tzhw = np.transpose(
    truth,
    (0, 3, 1, 2),
)

kspace = np.empty_like(
    truth_tzhw
)


for t in range(T):
    for zi in range(Z):

        kspace[t, zi] = (
            np.fft.fftshift(
                np.fft.fft2(
                    np.fft.ifftshift(
                        truth_tzhw[
                            t,
                            zi,
                        ]
                    )
                )
            )
        )


platform = qmp.create_mri_platform(
    detector_id="factory-mri"
)

state = platform.ingest(
    kspace
)

reconstructed = (
    state.reconstructed_field.data
)

error = (
    np.linalg.norm(
        reconstructed - truth
    )
    /
    (
        np.linalg.norm(truth)
        + 1e-15
    )
)


print("=" * 72)
print("QMS PLATFORM — MRI FACTORY TEST")
print("=" * 72)

print(
    "Reconstruction error:",
    f"{error:.12e}"
)

print(
    "Manifold:",
    state.manifold.state.shape
)

print(
    "Dynamics ready:",
    state.dynamics is not None
)

print()
print("MRI FACTORY OK")
