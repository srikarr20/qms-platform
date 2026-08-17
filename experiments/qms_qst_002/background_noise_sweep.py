import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "qms_qst_001"
    / "data"
    / "kwiat_2qubit_reference.json"
)


BASIS_FAMILY = {
    "H": "Z",
    "V": "Z",
    "D": "X",
    "A": "X",
    "R": "Y",
    "L": "Y",
}


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
    return np.outer(
        state,
        np.conjugate(state),
    )


def measurement_projector(
    labels: list[str],
) -> np.ndarray:

    psi = np.kron(
        ket(labels[0]),
        ket(labels[1]),
    )

    return projector(psi)


def context(
    labels: list[str],
) -> tuple[str, str]:

    return (
        BASIS_FAMILY[labels[0]],
        BASIS_FAMILY[labels[1]],
    )


def load_rows() -> list[dict]:

    with DATA_PATH.open() as f:
        return json.load(f)["data"]


def bell_reference() -> np.ndarray:

    hh = np.kron(
        ket("H"),
        ket("H"),
    )

    vv = np.kron(
        ket("V"),
        ket("V"),
    )

    psi = (hh + vv) / np.sqrt(2)

    return projector(psi)


def degraded_rows(
    rows: list[dict],
    signal_scale: float,
    background: float,
    rng: np.random.Generator,
) -> list[dict]:

    result = []

    for row in rows:

        ideal = float(
            row["counts"][-1]
        )

        expected = (
            signal_scale * ideal
            + background
        )

        measured = int(
            rng.poisson(expected)
        )

        result.append(
            {
                "basis": row["basis"],
                "count": measured,
            }
        )

    return result


def normalized_measurements(
    rows: list[dict],
):

    totals = defaultdict(float)

    for row in rows:
        totals[
            context(row["basis"])
        ] += row["count"]

    output = []

    for row in rows:

        ctx = context(
            row["basis"]
        )

        total = totals[ctx]

        if total <= 0:
            return None

        probability = (
            row["count"] / total
        )

        output.append(
            (
                measurement_projector(
                    row["basis"]
                ),
                probability,
            )
        )

    return output


def reconstruct(
    measurements,
):

    a = np.asarray(
        [
            m.T.reshape(-1)
            for m, _ in measurements
        ],
        dtype=complex,
    )

    b = np.asarray(
        [
            p
            for _, p in measurements
        ],
        dtype=complex,
    )

    rho_vec, _, rank, _ = (
        np.linalg.lstsq(
            a,
            b,
            rcond=None,
        )
    )

    rho = rho_vec.reshape(
        4,
        4,
    )

    rho = (
        rho + rho.conj().T
    ) / 2

    rho = rho / np.trace(rho)

    return (
        rho,
        rank,
        np.linalg.cond(a),
    )


def metrics(
    rho: np.ndarray,
    reference: np.ndarray,
) -> dict:

    eigenvalues = (
        np.linalg.eigvalsh(rho)
    )

    fidelity = float(
        np.real(
            np.trace(
                rho @ reference
            )
        )
    )

    purity = float(
        np.real(
            np.trace(
                rho @ rho
            )
        )
    )

    error = float(
        np.linalg.norm(
            rho - reference
        )
    )

    return {
        "fidelity": fidelity,
        "purity": purity,
        "min_eigenvalue": float(
            np.min(eigenvalues)
        ),
        "frobenius_error": error,
    }


def run_level(
    rows,
    reference,
    background,
    trials,
):

    results = []

    for seed in range(trials):

        rng = (
            np.random.default_rng(
                seed
            )
        )

        degraded = degraded_rows(
            rows,
            signal_scale=1.0,
            background=background,
            rng=rng,
        )

        measurements = (
            normalized_measurements(
                degraded
            )
        )

        if measurements is None:
            continue

        rho, rank, condition = (
            reconstruct(
                measurements
            )
        )

        result = metrics(
            rho,
            reference,
        )

        result["rank"] = rank
        result["condition"] = condition

        results.append(result)

    if not results:
        raise RuntimeError(
            "No valid trials"
        )

    def summarize(name):

        values = np.asarray(
            [
                r[name]
                for r in results
            ],
            dtype=float,
        )

        return (
            float(np.mean(values)),
            float(np.std(values)),
        )

    return {
        "background": background,
        "trials": len(results),
        "fidelity": summarize(
            "fidelity"
        ),
        "purity": summarize(
            "purity"
        ),
        "min_eigenvalue": summarize(
            "min_eigenvalue"
        ),
        "frobenius_error": summarize(
            "frobenius_error"
        ),
    }


def main():

    rows = load_rows()
    reference = bell_reference()

    backgrounds = [
        0.0,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
    ]

    trials = 200

    print(
        "QMS-QST-002B — "
        "Background-count degradation"
    )

    print(
        "trials per level:",
        trials,
    )

    print()

    print(
        "background  F_mean    F_std     "
        "purity    min_eig    error"
    )

    print(
        "----------------------------------------------------------"
    )

    for background in backgrounds:

        result = run_level(
            rows,
            reference,
            background,
            trials,
        )

        f_mean, f_std = (
            result["fidelity"]
        )

        p_mean, _ = (
            result["purity"]
        )

        e_mean, _ = (
            result[
                "min_eigenvalue"
            ]
        )

        error_mean, _ = (
            result[
                "frobenius_error"
            ]
        )

        print(
            f"{background:<12.1f}"
            f"{f_mean:<10.6f}"
            f"{f_std:<10.6f}"
            f"{p_mean:<10.6f}"
            f"{e_mean:<11.6f}"
            f"{error_mean:.6f}"
        )


if __name__ == "__main__":
    main()
