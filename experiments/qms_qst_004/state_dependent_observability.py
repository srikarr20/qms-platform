import itertools
from pathlib import Path

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

    states = {
        "H": h,
        "V": v,
        "D": (h + v) / np.sqrt(2),
        "A": (h - v) / np.sqrt(2),
        "R": (h + 1j * v) / np.sqrt(2),
        "L": (h - 1j * v) / np.sqrt(2),
    }

    return states[label]


def projector(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def product_state(a: str, b: str) -> np.ndarray:
    psi = np.kron(
        ket(a),
        ket(b),
    )
    return psi / np.linalg.norm(psi)


def bell_phi_plus() -> np.ndarray:
    hh = product_state("H", "H")
    vv = product_state("V", "V")
    psi = (hh + vv) / np.sqrt(2)
    return psi / np.linalg.norm(psi)


def bell_psi_minus() -> np.ndarray:
    hv = product_state("H", "V")
    vh = product_state("V", "H")
    psi = (hv - vh) / np.sqrt(2)
    return psi / np.linalg.norm(psi)


def generic_state() -> np.ndarray:
    """
    A deliberately non-symmetric two-qubit pure state.
    """
    psi = np.array(
        [
            np.sqrt(0.40),
            np.sqrt(0.30),
            1j * np.sqrt(0.20),
            np.sqrt(0.10),
        ],
        dtype=complex,
    )

    return psi / np.linalg.norm(psi)


def measurement_projector(labels: tuple[str, str]) -> np.ndarray:
    psi = np.kron(
        ket(labels[0]),
        ket(labels[1]),
    )
    return projector(psi)


def context_for_labels(labels: tuple[str, str]) -> str:
    return (
        BASIS_FAMILY[labels[0]]
        + BASIS_FAMILY[labels[1]]
    )


def all_measurements():
    labels = ["H", "V", "D", "A", "R", "L"]

    result = []

    for a, b in itertools.product(labels, repeat=2):
        pair = (a, b)

        result.append(
            {
                "basis": pair,
                "context": context_for_labels(pair),
                "projector": measurement_projector(pair),
            }
        )

    return result


def born_probability(
    rho: np.ndarray,
    measurement: np.ndarray,
) -> float:
    return float(
        np.real(
            np.trace(
                rho @ measurement
            )
        )
    )


def measurements_for_state(
    rho: np.ndarray,
):
    rows = []

    for item in all_measurements():
        probability = born_probability(
            rho,
            item["projector"],
        )

        rows.append(
            {
                **item,
                "probability": probability,
            }
        )

    return rows


def reconstruct(
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

    return rho, a, rank


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


def analyse_state(
    name: str,
    psi: np.ndarray,
):
    reference = projector(psi)
    rows = measurements_for_state(reference)

    ref_vec = reference.reshape(-1)

    results = []

    for removed in CONTEXTS:
        retained = set(CONTEXTS)
        retained.remove(removed)

        rho, a, rank = reconstruct(
            rows,
            retained,
        )

        fidelity = float(
            np.real(
                np.trace(
                    rho @ reference
                )
            )
        )

        error = float(
            np.linalg.norm(
                rho - reference
            )
        )

        ns = null_space(a)

        nullity = ns.shape[1]

        if nullity > 0:
            projection = (
                ns
                @ (
                    ns.conj().T
                    @ ref_vec
                )
            )

            overlap = float(
                np.linalg.norm(
                    projection
                )
            )
        else:
            overlap = 0.0

        results.append(
            {
                "removed": removed,
                "rank": rank,
                "nullity": nullity,
                "fidelity": fidelity,
                "error": error,
                "overlap": overlap,
            }
        )

    results.sort(
        key=lambda x: (
            x["fidelity"],
            -x["overlap"],
        )
    )

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        "removed  rank  nullity  fidelity   "
        "error      target_null_overlap"
    )

    print(
        "------------------------------------------------------------"
    )

    for result in results:
        print(
            f"{result['removed']:<9}"
            f"{result['rank']:<6}"
            f"{result['nullity']:<9}"
            f"{result['fidelity']:<11.6f}"
            f"{result['error']:<11.6f}"
            f"{result['overlap']:.6f}"
        )

    critical = [
        result["removed"]
        for result in results
        if result["fidelity"] < 1.0 - 1e-10
    ]

    print()
    print(
        "critical contexts:",
        critical if critical else "none",
    )


def main():
    states = {
        "|HH>": product_state("H", "H"),
        "|DD>": product_state("D", "D"),
        "|Phi+>": bell_phi_plus(),
        "|Psi->": bell_psi_minus(),
        "|Generic>": generic_state(),
    }

    print(
        "QMS-QST-004C — State-dependent observability"
    )

    print(
        "One measurement context is removed at a time."
    )

    for name, psi in states.items():
        analyse_state(
            name,
            psi,
        )


if __name__ == "__main__":
    main()
