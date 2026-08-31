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

# -----------------------------------------------------------
# Controlled perturbations
# -----------------------------------------------------------

TRUE_CASES = {
    "coupling_change": {
        "parameter": "g",
        "value": 0.10,
    },

    "frequency_change": {
        "parameter": "omega2",
        "value": 1.08,
    },

    "damping_change": {
        "parameter": "gamma2",
        "value": 0.10,
    },
}


# -----------------------------------------------------------
# Candidate hypothesis grids
#
# Only one parameter is allowed to vary per candidate family.
# -----------------------------------------------------------

HYPOTHESIS_GRIDS = {
    "coupling_change": {
        "parameter": "g",
        "values": np.linspace(
            0.06, 0.20, 71
        ),
    },

    "frequency_change": {
        "parameter": "omega2",
        "values": np.linspace(
            1.00, 1.30, 61
        ),
    },

    "damping_change": {
        "parameter": "gamma2",
        "values": np.linspace(
            0.03, 0.14, 56
        ),
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
    ], dtype=float)


def build_window_matrix(A, length):
    return np.vstack([
        C @ np.linalg.matrix_power(A, k)
        for k in range(1, length + 1)
    ])


def simulate_window(
    case_name,
    seed,
):
    rng = np.random.default_rng(seed)

    Q = (
        PROCESS_SIGMA ** 2
        * np.eye(4)
    )

    x = np.array(
        [0.7, -0.2, 0.5, 0.45],
        dtype=float
    )

    measurements = []

    case = TRUE_CASES[case_name]

    for k in range(IDENTIFY_STEP):

        params = BASE.copy()

        if k >= CHANGE_STEP:
            params[
                case["parameter"]
            ] = case["value"]

        A = expm(
            build_F(params) * DT
        )

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
            measurements.append(
                y.copy()
            )

    return measurements


def fit_hypothesis(
    measurements,
    hypothesis_name,
):
    y = np.asarray(
        measurements,
        dtype=float
    ).reshape(-1)

    grid = (
        HYPOTHESIS_GRIDS[
            hypothesis_name
        ]
    )

    parameter = grid["parameter"]

    best_value = None
    best_mse = np.inf
    best_state = None

    all_fits = []

    for value in grid["values"]:

        params = BASE.copy()
        params[parameter] = float(value)

        A = expm(
            build_F(params) * DT
        )

        H = build_window_matrix(
            A,
            len(measurements)
        )

        # Jointly fit unknown state at
        # beginning of observation window.
        x_start = (
            np.linalg.pinv(H) @ y
        )

        y_hat = H @ x_start

        mse = float(
            np.mean(
                (y - y_hat) ** 2
            )
        )

        all_fits.append({
            "value": float(value),
            "mse": mse,
        })

        if mse < best_mse:
            best_mse = mse
            best_value = float(value)
            best_state = x_start.copy()

    return {
        "hypothesis":
            hypothesis_name,

        "parameter":
            parameter,

        "best_value":
            best_value,

        "best_mse":
            best_mse,

        "best_state":
            best_state.tolist(),

        "fit_curve":
            all_fits,
    }


trial_results = []


for case_index, true_case in enumerate(
    TRUE_CASES
):

    for trial in range(TRIALS):

        measurements = simulate_window(
            true_case,
            seed=(
                20260824
                + case_index * 10000
                + trial
            ),
        )

        fits = [
            fit_hypothesis(
                measurements,
                hypothesis_name
            )
            for hypothesis_name
            in HYPOTHESIS_GRIDS
        ]

        fits_sorted = sorted(
            fits,
            key=lambda r:
                r["best_mse"]
        )

        winner = fits_sorted[0]
        runner_up = fits_sorted[1]

        relative_margin = float(
            (
                runner_up["best_mse"]
                - winner["best_mse"]
            )
            / winner["best_mse"]
        )

        trial_results.append({
            "true_case":
                true_case,

            "predicted_case":
                winner["hypothesis"],

            "predicted_parameter":
                winner["parameter"],

            "predicted_value":
                winner["best_value"],

            "winning_mse":
                winner["best_mse"],

            "runner_up":
                runner_up["hypothesis"],

            "runner_up_mse":
                runner_up["best_mse"],

            "relative_margin":
                relative_margin,

            "all_hypothesis_fits":
                fits,
        })


# -----------------------------------------------------------
# Aggregate results
# -----------------------------------------------------------

aggregate = {}

for true_case, config in TRUE_CASES.items():

    subset = [
        r
        for r in trial_results
        if r["true_case"] == true_case
    ]

    predicted = [
        r["predicted_case"]
        for r in subset
    ]

    correct = np.array([
        p == true_case
        for p in predicted
    ])

    correct_values = np.array([
        r["predicted_value"]
        for r in subset
        if r["predicted_case"] == true_case
    ])

    true_value = config["value"]

    if len(correct_values) > 0:
        value_abs_error = np.abs(
            correct_values - true_value
        )

        mean_value = float(
            np.mean(correct_values)
        )

        std_value = float(
            np.std(correct_values)
        )

        mean_abs_value_error = float(
            np.mean(value_abs_error)
        )

    else:
        mean_value = None
        std_value = None
        mean_abs_value_error = None

    margins = np.array([
        r["relative_margin"]
        for r in subset
    ])

    aggregate[true_case] = {
        "true_parameter":
            config["parameter"],

        "true_value":
            true_value,

        "trials":
            TRIALS,

        "correct_identification_rate":
            float(np.mean(correct)),

        "prediction_counts":
            dict(
                Counter(predicted)
            ),

        "mean_predicted_value_when_correct":
            mean_value,

        "std_predicted_value_when_correct":
            std_value,

        "mean_absolute_parameter_error_when_correct":
            mean_abs_value_error,

        "mean_relative_margin":
            float(np.mean(margins)),

        "median_relative_margin":
            float(np.median(margins)),
    }


# Confusion matrix
case_names = list(TRUE_CASES.keys())

confusion = {
    true_case: {
        predicted_case: 0
        for predicted_case
        in case_names
    }
    for true_case in case_names
}

for r in trial_results:
    confusion[
        r["true_case"]
    ][
        r["predicted_case"]
    ] += 1


evidence = {
    "experiment":
        "QMS-QT-006A",

    "title": (
        "External identification of alternative "
        "internal dynamical mechanisms"
    ),

    "measurement_channels": [
        "x1",
        "x2",
    ],

    "baseline_parameters":
        BASE,

    "true_cases":
        TRUE_CASES,

    "candidate_hypotheses": {
        name: {
            "parameter":
                h["parameter"],

            "grid_min":
                float(
                    np.min(
                        h["values"]
                    )
                ),

            "grid_max":
                float(
                    np.max(
                        h["values"]
                    )
                ),

            "grid_count":
                int(
                    len(
                        h["values"]
                    )
                ),
        }
        for name, h
        in HYPOTHESIS_GRIDS.items()
    },

    "aggregate":
        aggregate,

    "confusion_matrix":
        confusion,

    "trial_results":
        trial_results,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "model-selection experiment. Candidate "
        "mechanisms and parameter families are "
        "known in advance. Successful classification "
        "would establish discrimination only among "
        "these controlled candidate hypotheses, "
        "not general quantum-system identification."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_006a_mechanism_identification.json"
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
    "QMS-QT-006A DYNAMICAL MECHANISM IDENTIFICATION"
)

for case, r in aggregate.items():

    print()
    print(case)

    print(
        "  true:",
        r["true_parameter"],
        "=",
        r["true_value"]
    )

    print(
        "  identification rate:",
        f"{100 * r['correct_identification_rate']:.1f}%"
    )

    print(
        "  prediction counts:",
        r["prediction_counts"]
    )

    print(
        "  estimated value when correct:",
        r[
            "mean_predicted_value_when_correct"
        ],
        "±",
        r[
            "std_predicted_value_when_correct"
        ],
    )

    print(
        "  mean parameter error when correct:",
        r[
            "mean_absolute_parameter_error_when_correct"
        ],
    )

    print(
        "  mean winner margin:",
        f"{r['mean_relative_margin']:.4f}"
    )

print()
print("CONFUSION MATRIX")

for true_case, row in confusion.items():
    print(
        f"  {true_case}: {row}"
    )
