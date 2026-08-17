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
    """
    Haar-random two-qubit pure state.
    """
    vector = (
        rng.normal(size=4)
        + 1j * rng.normal(size=4)
    )

    vector /= np.linalg.norm(vector)

    return vector


def measurement_rows(rho: np.ndarray):
    rows = []

    for item in MEASUREMENTS:
        probability = float(
            np.real(
                np.trace(
                    rho @ item["projector"]
                )
            )
        )

        rows.append(
            {
                **item,
                "probability": probability,
            }
        )

    return rows


def build_system(
    rows,
    retained_contexts,
):
    filtered = [
        row
        for row in rows
        if row["context"] in retained_contexts
    ]

    a = np.asarray(
        [
            row["projector"].T.reshape(-1)
            for row in filtered
        ],
        dtype=complex,
    )

    b = np.asarray(
        [
            row["probability"]
            for row in filtered
        ],
        dtype=complex,
    )

    return a, b


def reconstruct(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, int]:
    rho_vec, _, rank, _ = np.linalg.lstsq(
        a,
        b,
        rcond=None,
    )

    rho = rho_vec.reshape(4, 4)

    rho = (
        rho + rho.conj().T
    ) / 2

    tr = np.trace(rho)

    if abs(tr) > 1e-12:
        rho /= tr

    return rho, rank


def null_space(
    a: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    _, singular, vh = np.linalg.svd(
        a,
        full_matrices=True,
    )

    rank = np.sum(
        singular > tolerance
    )

    return vh[rank:].conj().T


def target_null_overlap(
    a: np.ndarray,
    rho: np.ndarray,
) -> float:
    ns = null_space(a)

    if ns.shape[1] == 0:
        return 0.0

    target = rho.reshape(-1)

    projection = (
        ns
        @ (
            ns.conj().T
            @ target
        )
    )

    return float(
        np.linalg.norm(projection)
    )


def reconstruction_error(
    reconstructed: np.ndarray,
    target: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            reconstructed - target
        )
    )


def main():
    rng = np.random.default_rng(20260817)

    n_states = 500

    overlaps = []
    errors = []
    absolute_differences = []

    per_context = {
        context: {
            "overlaps": [],
            "errors": [],
        }
        for context in CONTEXTS
    }

    rank_values = []

    print(
        "QMS-QST-004D — Random-state observability validation"
    )
    print("random pure states:", n_states)
    print("removed contexts per state:", len(CONTEXTS))
    print(
        "total reconstructions:",
        n_states * len(CONTEXTS),
    )
    print()

    for _ in range(n_states):
        psi = random_pure_state(rng)
        rho_target = projector(psi)

        rows = measurement_rows(
            rho_target
        )

        for removed in CONTEXTS:
            retained = set(CONTEXTS)
            retained.remove(removed)

            a, b = build_system(
                rows,
                retained,
            )

            rho_reconstructed, rank = reconstruct(
                a,
                b,
            )

            overlap = target_null_overlap(
                a,
                rho_target,
            )

            error = reconstruction_error(
                rho_reconstructed,
                rho_target,
            )

            overlaps.append(overlap)
            errors.append(error)

            absolute_differences.append(
                abs(error - overlap)
            )

            per_context[removed][
                "overlaps"
            ].append(overlap)

            per_context[removed][
                "errors"
            ].append(error)

            rank_values.append(rank)

    overlaps = np.asarray(
        overlaps,
        dtype=float,
    )

    errors = np.asarray(
        errors,
        dtype=float,
    )

    absolute_differences = np.asarray(
        absolute_differences,
        dtype=float,
    )

    correlation = np.corrcoef(
        overlaps,
        errors,
    )[0, 1]

    slope, intercept = np.polyfit(
        overlaps,
        errors,
        1,
    )

    r_squared = correlation ** 2

    print("Global validation")
    print("-----------------")
    print(
        "rank values:",
        sorted(set(rank_values)),
    )
    print(
        "mean null overlap:",
        f"{np.mean(overlaps):.12f}",
    )
    print(
        "mean reconstruction error:",
        f"{np.mean(errors):.12f}",
    )
    print(
        "mean |error - overlap|:",
        f"{np.mean(absolute_differences):.12e}",
    )
    print(
        "max |error - overlap|:",
        f"{np.max(absolute_differences):.12e}",
    )
    print(
        "Pearson correlation:",
        f"{correlation:.12f}",
    )
    print(
        "linear fit slope:",
        f"{slope:.12f}",
    )
    print(
        "linear fit intercept:",
        f"{intercept:.12e}",
    )
    print(
        "R^2:",
        f"{r_squared:.12f}",
    )

    print()
    print("Per-context summary")
    print("-------------------")
    print(
        "removed  overlap_mean  error_mean  "
        "mean_abs_delta"
    )
    print(
        "----------------------------------------------"
    )

    for removed in CONTEXTS:
        ctx_overlaps = np.asarray(
            per_context[removed]["overlaps"],
            dtype=float,
        )

        ctx_errors = np.asarray(
            per_context[removed]["errors"],
            dtype=float,
        )

        delta = np.abs(
            ctx_errors - ctx_overlaps
        )

        print(
            f"{removed:<9}"
            f"{np.mean(ctx_overlaps):<14.8f}"
            f"{np.mean(ctx_errors):<12.8f}"
            f"{np.mean(delta):.3e}"
        )

    tolerance = 1e-10

    passed = (
        np.max(absolute_differences) < tolerance
        and abs(slope - 1.0) < tolerance
        and abs(intercept) < tolerance
        and r_squared > 1.0 - tolerance
    )

    print()
    print(
        "Observability-loss validation:",
        "PASS" if passed else "FAIL",
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
