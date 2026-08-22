import itertools

import numpy as np


BASIS_FAMILY = {
    "H": "Z",
    "V": "Z",
    "D": "X",
    "A": "X",
    "R": "Y",
    "L": "Y",
}

CONTEXTS = [
    "XX", "XY", "XZ",
    "YX", "YY", "YZ",
    "ZX", "ZY", "ZZ",
]


def ket(label: str) -> np.ndarray:
    h = np.array([1.0, 0.0], dtype=complex)
    v = np.array([0.0, 1.0], dtype=complex)

    return {
        "H": h,
        "V": v,
        "D": (h + v) / np.sqrt(2),
        "A": (h - v) / np.sqrt(2),
        "R": (h + 1j * v) / np.sqrt(2),
        "L": (h - 1j * v) / np.sqrt(2),
    }[label]


def projector(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def measurement_projector(labels) -> np.ndarray:
    psi = np.kron(
        ket(labels[0]),
        ket(labels[1]),
    )
    return projector(psi)


def measurement_context(labels) -> str:
    return (
        BASIS_FAMILY[labels[0]]
        + BASIS_FAMILY[labels[1]]
    )


def all_measurements():
    labels = ["H", "V", "D", "A", "R", "L"]

    output = []

    for pair in itertools.product(labels, repeat=2):
        output.append(
            {
                "basis": pair,
                "context": measurement_context(pair),
                "projector": measurement_projector(pair),
            }
        )

    return output


MEASUREMENTS = all_measurements()


def random_pure_state(
    rng: np.random.Generator,
) -> np.ndarray:
    psi = (
        rng.normal(size=4)
        + 1j * rng.normal(size=4)
    )

    psi /= np.linalg.norm(psi)
    return psi


def build_exact_system(
    rho: np.ndarray,
    retained_contexts: set[str],
):
    filtered = [
        item
        for item in MEASUREMENTS
        if item["context"] in retained_contexts
    ]

    a = np.asarray(
        [
            item["projector"].T.reshape(-1)
            for item in filtered
        ],
        dtype=complex,
    )

    b = np.asarray(
        [
            np.real(
                np.trace(
                    rho @ item["projector"]
                )
            )
            for item in filtered
        ],
        dtype=float,
    )

    return a, b


def add_gaussian_noise(
    b: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
):
    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=b.shape,
    )

    return b + noise, noise


def reconstruct(
    a: np.ndarray,
    b: np.ndarray,
):
    rho_vec, _, rank, _ = np.linalg.lstsq(
        a,
        b,
        rcond=None,
    )

    return rho_vec, rank


def null_projector(
    a: np.ndarray,
    tolerance: float = 1e-12,
):
    _, singular, vh = np.linalg.svd(
        a,
        full_matrices=True,
    )

    rank = np.sum(
        singular > tolerance
    )

    ns = vh[rank:].conj().T

    if ns.shape[1] == 0:
        return np.zeros(
            (a.shape[1], a.shape[1]),
            dtype=complex,
        )

    return ns @ ns.conj().T


def main():
    rng = np.random.default_rng(20260817)

    n_states = 200

    noise_levels = [
        0.0,
        0.001,
        0.005,
        0.010,
        0.020,
        0.050,
    ]

    print(
        "QMS-QST-005A — Noisy observability decomposition"
    )
    print("random states:", n_states)
    print()

    print(
        "sigma   total_err   null_err    noise_err   "
        "pred_err    residual"
    )
    print(
        "-----------------------------------------------------------"
    )

    for sigma in noise_levels:
        total_errors = []
        null_errors = []
        noise_errors = []
        predicted_errors = []
        residuals = []

        for _ in range(n_states):
            psi = random_pure_state(rng)
            rho = projector(psi)
            x_true = rho.reshape(-1)

            removed = rng.choice(CONTEXTS)

            retained = set(CONTEXTS)
            retained.remove(removed)

            a, b_exact = build_exact_system(
                rho,
                retained,
            )

            b_noisy, epsilon = add_gaussian_noise(
                b_exact,
                sigma,
                rng,
            )

            x_hat, _ = reconstruct(
                a,
                b_noisy,
            )

            actual_error_vector = (
                x_hat - x_true
            )

            p_null = null_projector(a)

            null_component = (
                -p_null @ x_true
            )

            a_pinv = np.linalg.pinv(a)

            noise_component = (
                a_pinv @ epsilon
            )

            predicted_error_vector = (
                null_component
                + noise_component
            )

            total_error = np.linalg.norm(
                actual_error_vector
            )

            null_error = np.linalg.norm(
                null_component
            )

            noise_error = np.linalg.norm(
                noise_component
            )

            predicted_error = np.linalg.norm(
                predicted_error_vector
            )

            residual = np.linalg.norm(
                actual_error_vector
                - predicted_error_vector
            )

            total_errors.append(total_error)
            null_errors.append(null_error)
            noise_errors.append(noise_error)
            predicted_errors.append(
                predicted_error
            )
            residuals.append(residual)

        print(
            f"{sigma:<8.3f}"
            f"{np.mean(total_errors):<12.8f}"
            f"{np.mean(null_errors):<12.8f}"
            f"{np.mean(noise_errors):<12.8f}"
            f"{np.mean(predicted_errors):<12.8f}"
            f"{np.mean(residuals):.3e}"
        )

    print()
    print(
        "Model:"
    )
    print(
        "x_hat - x_true = "
        "-P_null x_true + A^+ epsilon"
    )


if __name__ == "__main__":
    main()
