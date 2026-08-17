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
    return np.outer(state, np.conjugate(state))


def measurement_projector(labels: list[str]) -> np.ndarray:
    psi = np.kron(
        ket(labels[0]),
        ket(labels[1]),
    )
    return projector(psi)


def context(labels: list[str]) -> tuple[str, str]:
    return (
        BASIS_FAMILY[labels[0]],
        BASIS_FAMILY[labels[1]],
    )


def load_rows() -> list[dict]:
    with DATA_PATH.open() as f:
        return json.load(f)["data"]


def bell_reference() -> np.ndarray:
    hh = np.kron(ket("H"), ket("H"))
    vv = np.kron(ket("V"), ket("V"))

    psi = (hh + vv) / np.sqrt(2)
    return projector(psi)


def noisy_rows(
    rows: list[dict],
    scale: float,
    rng: np.random.Generator,
) -> list[dict]:
    result = []

    for row in rows:
        original = float(row["counts"][-1])

        # Scale acquisition level before Poisson sampling.
        lam = original * scale
        noisy_count = int(rng.poisson(lam))

        result.append(
            {
                "basis": row["basis"],
                "count": noisy_count,
            }
        )

    return result


def normalized_measurements(rows: list[dict]):
    totals = defaultdict(float)

    for row in rows:
        totals[context(row["basis"])] += row["count"]

    result = []

    for row in rows:
        ctx = context(row["basis"])
        total = totals[ctx]

        if total <= 0:
            return None

        result.append(
            (
                measurement_projector(row["basis"]),
                row["count"] / total,
            )
        )

    return result


def reconstruct(measurements) -> tuple[np.ndarray, int, float]:
    a = np.asarray(
        [m.T.reshape(-1) for m, _ in measurements],
        dtype=complex,
    )

    b = np.asarray(
        [p for _, p in measurements],
        dtype=complex,
    )

    rho_vec, _, rank, _ = np.linalg.lstsq(
        a,
        b,
        rcond=None,
    )

    rho = rho_vec.reshape(4, 4)

    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)

    return rho, rank, np.linalg.cond(a)


def metrics(
    rho: np.ndarray,
    reference: np.ndarray,
) -> dict:
    eig = np.linalg.eigvalsh(rho)

    fidelity = float(
        np.real(
            np.trace(rho @ reference)
        )
    )

    purity = float(
        np.real(
            np.trace(rho @ rho)
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
        "min_eigenvalue": float(np.min(eig)),
        "frobenius_error": error,
    }


def run_level(
    rows: list[dict],
    reference: np.ndarray,
    scale: float,
    trials: int,
) -> dict:
    results = []

    for seed in range(trials):
        rng = np.random.default_rng(seed)

        degraded = noisy_rows(
            rows,
            scale,
            rng,
        )

        measurements = normalized_measurements(
            degraded
        )

        if measurements is None:
            continue

        rho, rank, condition = reconstruct(
            measurements
        )

        m = metrics(rho, reference)
        m["rank"] = rank
        m["condition"] = condition

        results.append(m)

    if not results:
        raise RuntimeError(
            f"No valid trials at scale {scale}"
        )

    def stat(name: str):
        values = np.array(
            [r[name] for r in results],
            dtype=float,
        )
        return (
            float(np.mean(values)),
            float(np.std(values)),
        )

    return {
        "scale": scale,
        "trials": len(results),
        "fidelity": stat("fidelity"),
        "purity": stat("purity"),
        "min_eigenvalue": stat("min_eigenvalue"),
        "frobenius_error": stat("frobenius_error"),
        "rank": results[0]["rank"],
        "condition": results[0]["condition"],
    }


def main() -> None:
    rows = load_rows()
    reference = bell_reference()

    scales = [
        10.0,
        5.0,
        2.0,
        1.0,
        0.5,
        0.2,
        0.1,
        0.05,
    ]

    trials = 100

    print("QMS-QST-002A — Poisson count-noise degradation")
    print("trials per level:", trials)
    print()
    print(
        "scale   F_mean    F_std     "
        "purity    min_eig    error"
    )
    print(
        "-------------------------------------------------------"
    )

    for scale in scales:
        result = run_level(
            rows,
            reference,
            scale,
            trials,
        )

        f_mean, f_std = result["fidelity"]
        p_mean, _ = result["purity"]
        e_mean, _ = result["min_eigenvalue"]
        err_mean, _ = result["frobenius_error"]

        print(
            f"{scale:<7.2f}"
            f"{f_mean:<10.6f}"
            f"{f_std:<10.6f}"
            f"{p_mean:<10.6f}"
            f"{e_mean:<11.6f}"
            f"{err_mean:.6f}"
        )

    print()
    print("Note:")
    print(
        "Linear inversion is unconstrained, so noisy data may "
        "produce non-positive density matrices."
    )


if __name__ == "__main__":
    main()
