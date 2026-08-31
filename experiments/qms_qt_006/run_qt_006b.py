import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

DT = 0.05
CHANGE_STEP = 600
IDENTIFY_STEP = 750
TRIALS = 100

PROCESS_SIGMA = 1e-4
MEAS_SIGMA = 1e-3

BASE = {
    "omega1": 1.00,
    "omega2": 1.20,
    "gamma1": 0.08,
    "gamma2": 0.06,
    "g": 0.18,
}

C = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])

SWEEPS = {
    "coupling_change": {
        "parameter": "g",
        "values": [0.17, 0.16, 0.14, 0.12, 0.10],
    },
    "frequency_change": {
        "parameter": "omega2",
        "values": [1.18, 1.16, 1.14, 1.10, 1.08],
    },
    "damping_change": {
        "parameter": "gamma2",
        "values": [0.065, 0.07, 0.08, 0.09, 0.10],
    },
}

HYPOTHESIS_GRIDS = {
    "coupling_change": {
        "parameter": "g",
        "values": np.linspace(0.06, 0.20, 71),
    },
    "frequency_change": {
        "parameter": "omega2",
        "values": np.linspace(1.00, 1.30, 61),
    },
    "damping_change": {
        "parameter": "gamma2",
        "values": np.linspace(0.03, 0.14, 56),
    },
}


def build_F(params):
    w1 = params["omega1"]
    w2 = params["omega2"]
    g1 = params["gamma1"]
    g2 = params["gamma2"]
    g = params["g"]

    return np.array([
        [-g1,  w1, 0.0, 0.0],
        [-w1, -g1, g,   0.0],
        [0.0,  0.0, -g2, w2],
        [g,    0.0, -w2, -g2],
    ])


def build_window_matrix(A, length):
    return np.vstack([
        C @ np.linalg.matrix_power(A, k)
        for k in range(1, length + 1)
    ])


def simulate_window(parameter, value, seed):
    rng = np.random.default_rng(seed)

    Q = PROCESS_SIGMA**2 * np.eye(4)

    x = np.array(
        [0.7, -0.2, 0.5, 0.45],
        dtype=float
    )

    measurements = []

    for k in range(IDENTIFY_STEP):

        params = BASE.copy()

        if k >= CHANGE_STEP:
            params[parameter] = value

        A = expm(build_F(params) * DT)

        x = (
            A @ x
            + rng.multivariate_normal(
                np.zeros(4), Q
            )
        )

        y = (
            C @ x
            + rng.normal(
                scale=MEAS_SIGMA,
                size=2
            )
        )

        if k >= CHANGE_STEP:
            measurements.append(y.copy())

    return measurements


def fit_hypothesis(measurements, name):
    y = np.asarray(measurements).reshape(-1)

    grid = HYPOTHESIS_GRIDS[name]
    parameter = grid["parameter"]

    best_value = None
    best_mse = np.inf

    for value in grid["values"]:

        params = BASE.copy()
        params[parameter] = float(value)

        A = expm(build_F(params) * DT)

        H = build_window_matrix(
            A,
            len(measurements)
        )

        x_start = np.linalg.pinv(H) @ y
        y_hat = H @ x_start

        mse = float(
            np.mean((y - y_hat) ** 2)
        )

        if mse < best_mse:
            best_mse = mse
            best_value = float(value)

    return {
        "hypothesis": name,
        "best_value": best_value,
        "best_mse": best_mse,
    }


results = []

condition_index = 0

for true_case, sweep in SWEEPS.items():

    parameter = sweep["parameter"]

    for true_value in sweep["values"]:

        predictions = []
        margins = []
        parameter_errors = []

        for trial in range(TRIALS):

            measurements = simulate_window(
                parameter,
                true_value,
                seed=(
                    20260824
                    + condition_index * 10000
                    + trial
                ),
            )

            fits = [
                fit_hypothesis(
                    measurements,
                    h
                )
                for h in HYPOTHESIS_GRIDS
            ]

            fits = sorted(
                fits,
                key=lambda r: r["best_mse"]
            )

            winner = fits[0]
            runner = fits[1]

            predictions.append(
                winner["hypothesis"]
            )

            margin = (
                runner["best_mse"]
                - winner["best_mse"]
            ) / winner["best_mse"]

            margins.append(
                float(margin)
            )

            if winner["hypothesis"] == true_case:
                parameter_errors.append(
                    abs(
                        winner["best_value"]
                        - true_value
                    )
                )

        correct = np.array([
            p == true_case
            for p in predictions
        ])

        results.append({
            "true_case": true_case,
            "parameter": parameter,
            "true_value": true_value,

            "identification_rate":
                float(np.mean(correct)),

            "prediction_counts":
                dict(Counter(predictions)),

            "mean_winner_margin":
                float(np.mean(margins)),

            "median_winner_margin":
                float(np.median(margins)),

            "mean_parameter_error_when_correct":
                (
                    float(np.mean(parameter_errors))
                    if parameter_errors
                    else None
                ),
        })

        condition_index += 1


evidence = {
    "experiment": "QMS-QT-006B",

    "title": (
        "Detectability boundary for "
        "internal dynamical mechanism identification"
    ),

    "measurement_channels": [
        "x1",
        "x2",
    ],

    "baseline_parameters": BASE,

    "results": results,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "mechanism-identification sweep. "
        "Candidate parameter families are known "
        "in advance. Identification rates apply "
        "only to this model, noise level, "
        "measurement architecture and finite "
        "observation window."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_006b_mechanism_boundary.json"
)

out.write_text(
    json.dumps(
        evidence,
        indent=2
    ) + "\n"
)


print(f"evidence: {out}")
print()
print(
    "QMS-QT-006B MECHANISM IDENTIFICATION BOUNDARY"
)

for r in results:

    print()
    print(
        f"{r['true_case']:20s}",
        f"{r['parameter']}={r['true_value']}"
    )

    print(
        "  identification:",
        f"{100 * r['identification_rate']:.1f}%"
    )

    print(
        "  predictions:",
        r["prediction_counts"]
    )

    print(
        "  winner margin:",
        f"{r['mean_winner_margin']:.4f}"
    )

    print(
        "  parameter error:",
        r[
            "mean_parameter_error_when_correct"
        ]
    )
