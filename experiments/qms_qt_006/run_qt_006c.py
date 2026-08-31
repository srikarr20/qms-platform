import json
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


TRUE_CASES = {
    "g_gamma2": {
        "g": 0.12,
        "gamma2": 0.08,
    },

    "g_omega2": {
        "g": 0.12,
        "omega2": 1.14,
    },

    "omega2_gamma2": {
        "omega2": 1.14,
        "gamma2": 0.08,
    },
}


GRIDS = {
    "g": np.linspace(0.08, 0.20, 31),
    "omega2": np.linspace(1.06, 1.26, 21),
    "gamma2": np.linspace(0.04, 0.10, 16),
}


MODEL_FAMILIES = {
    "g_only": ["g"],
    "omega2_only": ["omega2"],
    "gamma2_only": ["gamma2"],
    "g_gamma2": ["g", "gamma2"],
    "g_omega2": ["g", "omega2"],
    "omega2_gamma2": ["omega2", "gamma2"],
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
    ], dtype=float)


def build_window_matrix(A, length):
    return np.vstack([
        C @ np.linalg.matrix_power(A, k)
        for k in range(1, length + 1)
    ])


def simulate_window(changes, seed):
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
            params.update(changes)

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


def candidate_parameter_sets(family):
    params = MODEL_FAMILIES[family]

    if len(params) == 1:
        p = params[0]

        for v in GRIDS[p]:
            yield {p: float(v)}

    elif len(params) == 2:
        p1, p2 = params

        for v1 in GRIDS[p1]:
            for v2 in GRIDS[p2]:
                yield {
                    p1: float(v1),
                    p2: float(v2),
                }


def fit_family(measurements, family):
    y = np.asarray(
        measurements,
        dtype=float
    ).reshape(-1)

    best_mse = np.inf
    best_params = None

    for changes in candidate_parameter_sets(family):

        params = BASE.copy()
        params.update(changes)

        A = expm(build_F(params) * DT)

        H = build_window_matrix(
            A,
            len(measurements)
        )

        x_start = np.linalg.pinv(H) @ y
        y_hat = H @ x_start

        mse = float(
            np.mean(
                (y - y_hat) ** 2
            )
        )

        if mse < best_mse:
            best_mse = mse
            best_params = changes.copy()

    return {
        "family": family,
        "best_mse": best_mse,
        "best_params": best_params,
    }


# -------------------------------------------------------
# Baseline adequacy calibration
#
# Fit the correct baseline model to unchanged data.
# We use the distribution of MSE as a reference for
# whether a fitted changed-model is still plausible.
# -------------------------------------------------------

baseline_mses = []

for trial in range(300):

    measurements = simulate_window(
        {},
        19000000 + trial
    )

    y = np.asarray(
        measurements
    ).reshape(-1)

    A = expm(
        build_F(BASE) * DT
    )

    H = build_window_matrix(
        A,
        len(measurements)
    )

    x_start = np.linalg.pinv(H) @ y
    y_hat = H @ x_start

    baseline_mses.append(
        float(
            np.mean(
                (y - y_hat) ** 2
            )
        )
    )


ADEQUACY_THRESHOLD = float(
    np.quantile(
        baseline_mses,
        0.99
    )
)


trial_results = []


for case_index, (true_case, changes) in enumerate(
    TRUE_CASES.items()
):

    for trial in range(TRIALS):

        measurements = simulate_window(
            changes,
            (
                20260824
                + case_index * 10000
                + trial
            )
        )

        fits = [
            fit_family(
                measurements,
                family
            )
            for family
            in MODEL_FAMILIES
        ]

        fits = sorted(
            fits,
            key=lambda x: x["best_mse"]
        )

        winner = fits[0]
        runner = fits[1]

        adequate = (
            winner["best_mse"]
            <= ADEQUACY_THRESHOLD
        )

        margin = float(
            (
                runner["best_mse"]
                - winner["best_mse"]
            )
            / winner["best_mse"]
        )

        trial_results.append({
            "true_case": true_case,

            "predicted_family":
                winner["family"],

            "best_params":
                winner["best_params"],

            "best_mse":
                winner["best_mse"],

            "adequate":
                bool(adequate),

            "runner_up":
                runner["family"],

            "relative_margin":
                margin,
        })


aggregate = {}

for true_case in TRUE_CASES:

    subset = [
        r
        for r in trial_results
        if r["true_case"] == true_case
    ]

    correct = [
        r["predicted_family"] == true_case
        for r in subset
    ]

    adequate = [
        r["adequate"]
        for r in subset
    ]

    correct_and_adequate = [
        (
            r["predicted_family"] == true_case
            and r["adequate"]
        )
        for r in subset
    ]

    counts = {}

    for r in subset:
        name = r["predicted_family"]
        counts[name] = (
            counts.get(name, 0) + 1
        )

    margins = [
        r["relative_margin"]
        for r in subset
    ]

    aggregate[true_case] = {
        "trials": TRIALS,

        "correct_family_rate":
            float(np.mean(correct)),

        "adequate_fit_rate":
            float(np.mean(adequate)),

        "correct_and_adequate_rate":
            float(
                np.mean(
                    correct_and_adequate
                )
            ),

        "prediction_counts":
            counts,

        "mean_relative_margin":
            float(np.mean(margins)),
    }


evidence = {
    "experiment": "QMS-QT-006C",

    "title": (
        "Multi-parameter dynamical mechanism "
        "identification and model adequacy"
    ),

    "measurement_channels": [
        "x1",
        "x2",
    ],

    "true_cases":
        TRUE_CASES,

    "model_families":
        MODEL_FAMILIES,

    "adequacy_calibration": {
        "baseline_trials": 300,
        "quantile": 0.99,
        "mse_threshold":
            ADEQUACY_THRESHOLD,
    },

    "aggregate":
        aggregate,

    "trial_results":
        trial_results,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "multi-parameter model-selection test. "
        "Candidate model families and parameter "
        "ranges are known in advance. Adequacy "
        "threshold is calibrated only to this "
        "simulated measurement architecture, "
        "noise level and observation window."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_006c_multi_parameter_identification.json"
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
    "QMS-QT-006C MULTI-PARAMETER IDENTIFICATION"
)

print(
    "baseline adequacy MSE threshold:",
    f"{ADEQUACY_THRESHOLD:.6e}"
)

for case, r in aggregate.items():

    print()
    print(case)

    print(
        "  correct family:",
        f"{100 * r['correct_family_rate']:.1f}%"
    )

    print(
        "  adequate fit:",
        f"{100 * r['adequate_fit_rate']:.1f}%"
    )

    print(
        "  correct + adequate:",
        f"{100 * r['correct_and_adequate_rate']:.1f}%"
    )

    print(
        "  predictions:",
        r["prediction_counts"]
    )

    print(
        "  winner margin:",
        f"{r['mean_relative_margin']:.4f}"
    )
