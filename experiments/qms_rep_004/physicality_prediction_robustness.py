from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DATA = Path(
    "experiments/qms_qst_001/data/"
    "kwiat_2qubit_reference.json"
)

SEED = 42
TRIALS = 200

COUNT_SCALES = [3.0, 1.0, 0.3, 0.1]
BACKGROUNDS = [0.0, 1.0, 5.0, 10.0, 20.0]


def ket(label: str) -> np.ndarray:
    states = {
        "H": np.array([1, 0], dtype=complex),
        "V": np.array([0, 1], dtype=complex),
        "D": np.array([1, 1], dtype=complex) / np.sqrt(2),
        "A": np.array([1, -1], dtype=complex) / np.sqrt(2),
        "R": np.array([1, 1j], dtype=complex) / np.sqrt(2),
        "L": np.array([1, -1j], dtype=complex) / np.sqrt(2),
    }
    return states[label]


def projector(basis) -> np.ndarray:
    psi = np.kron(
        ket(basis[0]),
        ket(basis[1]),
    )
    return np.outer(psi, psi.conj())


def context(basis):
    axis = {
        "H": "Z",
        "V": "Z",
        "D": "X",
        "A": "X",
        "R": "Y",
        "L": "Y",
    }
    return axis[basis[0]], axis[basis[1]]


def probabilities_for_state(psi, bases):
    rho = np.outer(psi, psi.conj())

    probs = []

    for basis in bases:
        p = np.real(
            np.trace(
                rho @ projector(basis)
            )
        )
        probs.append(max(float(p), 0.0))

    return np.asarray(probs)


def normalized_measurements(rows):
    totals = {}

    for row in rows:
        ctx = context(row["basis"])
        totals[ctx] = totals.get(ctx, 0.0) + row["count"]

    measurements = []

    for row in rows:
        ctx = context(row["basis"])
        total = totals[ctx]

        if total <= 0:
            return None

        measurements.append(
            (
                projector(row["basis"]),
                row["count"] / total,
            )
        )

    return measurements


def measurement_vector(measurements):
    return np.asarray(
        [p for _, p in measurements],
        dtype=float,
    )


def reconstruct(measurements):
    A = np.asarray(
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

    rho_vec = np.linalg.pinv(A) @ b

    rho = rho_vec.reshape(4, 4)
    rho = (rho + rho.conj().T) / 2

    trace = np.trace(rho)

    if abs(trace) > 0:
        rho = rho / trace

    return rho


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def random_pure_state(rng):
    x = (
        rng.normal(size=4)
        + 1j * rng.normal(size=4)
    )

    return x / np.linalg.norm(x)


def bell_state():
    hh = np.kron(
        ket("H"),
        ket("H"),
    )
    vv = np.kron(
        ket("V"),
        ket("V"),
    )

    return (hh + vv) / np.sqrt(2)


def build_reference_rows(psi, bases, nominal_total=200.0):
    probs = probabilities_for_state(
        psi,
        bases,
    )

    context_totals = {}

    for basis, p in zip(bases, probs):
        ctx = context(basis)
        context_totals[ctx] = (
            context_totals.get(ctx, 0.0)
            + p
        )

    rows = []

    for basis, p in zip(bases, probs):
        ctx = context(basis)

        conditional = (
            p / context_totals[ctx]
            if context_totals[ctx] > 0
            else 0.0
        )

        rows.append(
            {
                "basis": basis,
                "ideal_count":
                    conditional * nominal_total,
            }
        )

    return rows


def run_noise_condition(
    reference_rows,
    rng,
    *,
    count_scale=1.0,
    background=0.0,
):
    noisy_rows = []

    for row in reference_rows:
        lam = max(
            row["ideal_count"] * count_scale
            + background,
            0.0,
        )

        noisy_rows.append(
            {
                "basis": row["basis"],
                "count": int(
                    rng.poisson(lam)
                ),
            }
        )

    return normalized_measurements(
        noisy_rows
    )


def analyze_family(
    state_name,
    psi,
    bases,
    rng,
):
    reference_rows = build_reference_rows(
        psi,
        bases,
    )

    ideal_measurements = (
        normalized_measurements(
            [
                {
                    "basis": row["basis"],
                    "count": row["ideal_count"],
                }
                for row in reference_rows
            ]
        )
    )

    ideal_vector = measurement_vector(
        ideal_measurements
    )

    records = []

    for scale in COUNT_SCALES:
        for _ in range(TRIALS):
            measurements = run_noise_condition(
                reference_rows,
                rng,
                count_scale=scale,
                background=0.0,
            )

            if measurements is None:
                continue

            records.append(
                evaluate(
                    state_name,
                    "poisson",
                    scale,
                    measurements,
                    ideal_vector,
                )
            )

    for background in BACKGROUNDS:
        for _ in range(TRIALS):
            measurements = run_noise_condition(
                reference_rows,
                rng,
                count_scale=1.0,
                background=background,
            )

            if measurements is None:
                continue

            records.append(
                evaluate(
                    state_name,
                    "background",
                    background,
                    measurements,
                    ideal_vector,
                )
            )

    return records


def evaluate(
    state_name,
    noise_type,
    noise_level,
    measurements,
    ideal_vector,
):
    vector = measurement_vector(
        measurements
    )

    rho = reconstruct(
        measurements
    )

    similarity = cosine_similarity(
        vector,
        ideal_vector,
    )

    min_eigenvalue = float(
        np.min(
            np.linalg.eigvalsh(rho)
        )
    )

    return {
        "state": state_name,
        "noise_type": noise_type,
        "noise_level": float(noise_level),
        "similarity": similarity,
        "min_eigenvalue": min_eigenvalue,
        "nonphysical": bool(
            min_eigenvalue < -1e-10
        ),
    }


def summarize(records):
    similarity = np.asarray(
        [r["similarity"] for r in records]
    )

    min_eig = np.asarray(
        [r["min_eigenvalue"] for r in records]
    )

    return {
        "n_records": len(records),
        "similarity_min_eigenvalue_correlation":
            float(
                np.corrcoef(
                    similarity,
                    min_eig,
                )[0, 1]
            ),
        "mean_similarity":
            float(np.mean(similarity)),
        "mean_min_eigenvalue":
            float(np.mean(min_eig)),
        "nonphysical_fraction":
            float(
                np.mean(
                    [
                        r["nonphysical"]
                        for r in records
                    ]
                )
            ),
    }


def main():
    raw = json.loads(
        DATA.read_text()
    )

    bases = [
        row["basis"]
        for row in raw["data"]
    ]

    rng = np.random.default_rng(
        SEED
    )

    states = {
        "bell_phi_plus": bell_state(),
        "random_1": random_pure_state(rng),
        "random_2": random_pure_state(rng),
        "random_3": random_pure_state(rng),
        "random_4": random_pure_state(rng),
        "random_5": random_pure_state(rng),
    }

    all_records = []
    per_state = {}

    for name, psi in states.items():
        records = analyze_family(
            name,
            psi,
            bases,
            rng,
        )

        all_records.extend(records)
        per_state[name] = summarize(
            records
        )

    overall = summarize(
        all_records
    )

    result = {
        "overall": overall,
        "per_state": per_state,
    }

    outdir = Path(
        "experiments/qms_rep_004/evidence"
    )
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        outdir /
        "qms_rep_004_summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print(
        "Saved:",
        outdir /
        "qms_rep_004_summary.json",
    )


if __name__ == "__main__":
    main()
