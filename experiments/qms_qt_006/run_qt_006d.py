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
TRIALS = 200

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

UNKNOWN_CASES = {
    "omega1_change": {
        "omega1": 0.90,
    },

    "gamma1_change": {
        "gamma1": 0.12,
    },

    "omega1_gamma1_change": {
        "omega1": 0.92,
        "gamma1": 0.11,
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
    ])


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

    else:
        p1, p2 = params

        for v1 in GRIDS[p1]:
            for v2 in GRIDS[p2]:
                yield {
                    p1: float(v1),
                    p2: float(v2),
                }


def fit_family(measurements, family):
    y = np.asarray(measurements).reshape(-1)

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
            np.mean((y - y_hat) ** 2)
        )

        if mse < best_mse:
            best_mse = mse
            best_params = changes.copy()

    return {
        "family": family,
        "best_mse": best_mse,
        "best_params": best_params,
    }


# ----------------------------------------
# Calibrate adequacy threshold from nominal
# ----------------------------------------

baseline_mses = []

for trial in range(500):

    measurements = simulate_window(
        {},
        seed=18000000 + trial
    )

    y = np.asarray(measurements).reshape(-1)

    A = expm(build_F(BASE) * DT)

    H = build_window_matrix(
        A,
        len(measurements)
    )

    x_start = np.linalg.pinv(H) @ y
    y_hat = H @ x_start

    baseline_mses.append(
        float(
            np.mean((y - y_hat) ** 2)
        )
    )


ADEQUACY_THRESHOLD = float(
    np.quantile(
        baseline_mses,
        0.99
    )
)


results = {}


for case_index, (case_name, changes) in enumerate(
    UNKNOWN_CASES.items()
):

    predicted_families = []
    rejected = []
    best_mses = []
    margins = []

    for trial in range(TRIALS):

        measurements = simulate_window(
            changes,
            seed=(
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
            for family in MODEL_FAMILIES
        ]

        fits = sorted(
            fits,
            key=lambda r: r["best_mse"]
        )

        winner = fits[0]
        runner = fits[1]

        is_rejected = (
            winner["best_mse"]
            > ADEQUACY_THRESHOLD
        )

        margin = (
            runner["best_mse"]
            - winner["best_mse"]
        ) / winner["best_mse"]

        predicted_families.append(
            winner["family"]
        )

        rejected.append(
            is_rejected
        )

        best_mses.append(
            winner["best_mse"]
        )

        margins.append(
            margin
        )

    results[case_name] = {
        "true_changes":
            changes,

        "trials":
            TRIALS,

        "unknown_rejection_rate":
            float(
                np.mean(rejected)
            ),

        "forced_prediction_counts":
            dict(
                Counter(
                    predicted_families
                )
            ),

        "mean_best_mse":
            float(
                np.mean(best_mses)
            ),

        "mean_best_mse_over_threshold":
            float(
                np.mean(best_mses)
                / ADEQUACY_THRESHOLD
            ),

        "mean_winner_margin":
            float(
                np.mean(margins)
            ),
    }


evidence = {
    "experiment":
        "QMS-QT-006D",

    "title": (
        "Unknown dynamical mechanism "
        "rejection by model adequacy"
    ),

    "candidate_model_library":
        MODEL_FAMILIES,

    "unknown_test_cases":
        UNKNOWN_CASES,

    "adequacy_threshold":
        ADEQUACY_THRESHOLD,

    "results":
        results,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "out-of-library model test. Rejection "
        "performance applies only to the tested "
        "unknown mechanisms, model library, "
        "measurement architecture, noise level "
        "and observation window."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_006d_unknown_mechanism_rejection.json"
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
    "QMS-QT-006D UNKNOWN MECHANISM REJECTION"
)

print(
    "adequacy threshold:",
    f"{ADEQUACY_THRESHOLD:.6e}"
)

for case, r in results.items():

    print()
    print(case)

    print(
        "  true changes:",
        r["true_changes"]
    )

    print(
        "  unknown rejection:",
        f"{100 * r['unknown_rejection_rate']:.1f}%"
    )

    print(
        "  forced winners:",
        r["forced_prediction_counts"]
    )

    print(
        "  best MSE / threshold:",
        f"{r['mean_best_mse_over_threshold']:.3f}"
    )

    print(
        "  winner margin:",
        f"{r['mean_winner_margin']:.4f}"
    )
