import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

DT = 0.05
STEPS = 1400
CHANGE_STEP = 600
ADAPT_STEP = 750
TRIALS = 100

BASE_G = 0.18
TRUE_G_AFTER = 0.08

PROCESS_SIGMA = 1e-4
MEAS_SIGMA = 1e-3

C = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])

G_CANDIDATES = np.linspace(0.04, 0.20, 81)


def build_F(g):
    omega1 = 1.00
    omega2 = 1.20
    gamma1 = 0.08
    gamma2 = 0.06

    return np.array([
        [-gamma1,  omega1, 0.0,   0.0],
        [-omega1, -gamma1, g,     0.0],
        [0.0,       0.0, -gamma2, omega2],
        [g,          0.0, -omega2, -gamma2],
    ], dtype=float)


def build_window_matrix(A, length):
    return np.vstack([
        C @ np.linalg.matrix_power(A, k)
        for k in range(1, length + 1)
    ])


def fit_g_and_state(measurements):
    y = np.asarray(measurements).reshape(-1)

    best_g = None
    best_state = None
    best_mse = np.inf

    for g in G_CANDIDATES:

        A = expm(build_F(g) * DT)

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
            best_g = float(g)
            best_state = x_start.copy()

    return best_g, best_state, best_mse


def kalman_step(
    x_hat,
    P,
    A,
    y,
    Q,
    R,
):
    x_pred = A @ x_hat
    P_pred = A @ P @ A.T + Q

    residual = y - C @ x_pred

    S = C @ P_pred @ C.T + R

    K = (
        P_pred
        @ C.T
        @ np.linalg.inv(S)
    )

    x_new = (
        x_pred
        + K @ residual
    )

    P_new = (
        np.eye(4) - K @ C
    ) @ P_pred

    return x_new, P_new


def run_trial(seed):

    rng = np.random.default_rng(seed)

    Q = (
        PROCESS_SIGMA ** 2
        * np.eye(4)
    )

    R = (
        MEAS_SIGMA ** 2
        * np.eye(2)
    )

    x_true = np.array(
        [0.7, -0.2, 0.5, 0.45],
        dtype=float
    )

    # Fixed twin
    x_fixed = np.zeros(4)
    P_fixed = np.eye(4)

    # Parameter-only twin
    x_param = np.zeros(4)
    P_param = np.eye(4)

    # Parameter + state recovery twin
    x_reset = np.zeros(4)
    P_reset = np.eye(4)

    g_param = BASE_G
    g_reset = BASE_G

    measurements = []

    fixed_errors = []
    param_errors = []
    reset_errors = []

    estimated_g = None

    for k in range(STEPS):

        true_g = (
            BASE_G
            if k < CHANGE_STEP
            else TRUE_G_AFTER
        )

        A_true = expm(
            build_F(true_g) * DT
        )

        x_true = (
            A_true @ x_true
            + rng.multivariate_normal(
                np.zeros(4), Q
            )
        )

        y = (
            C @ x_true
            + rng.normal(
                scale=MEAS_SIGMA,
                size=2
            )
        )

        measurements.append(
            y.copy()
        )

        # Fixed
        A_fixed = expm(
            build_F(BASE_G) * DT
        )

        x_fixed, P_fixed = kalman_step(
            x_fixed,
            P_fixed,
            A_fixed,
            y,
            Q,
            R,
        )

        # Parameter-only
        A_param = expm(
            build_F(g_param) * DT
        )

        x_param, P_param = kalman_step(
            x_param,
            P_param,
            A_param,
            y,
            Q,
            R,
        )

        # Parameter + state recovery
        A_reset = expm(
            build_F(g_reset) * DT
        )

        x_reset, P_reset = kalman_step(
            x_reset,
            P_reset,
            A_reset,
            y,
            Q,
            R,
        )

        if k == ADAPT_STEP:

            y_window = measurements[
                CHANGE_STEP:ADAPT_STEP
            ]

            (
                estimated_g,
                reconstructed_start,
                _
            ) = fit_g_and_state(
                y_window
            )

            g_param = estimated_g
            g_reset = estimated_g

            A_est = expm(
                build_F(estimated_g) * DT
            )

            x_reset = (
                np.linalg.matrix_power(
                    A_est,
                    len(y_window)
                )
                @ reconstructed_start
            )

            P_reset = 0.1 * np.eye(4)

        fixed_errors.append(
            float(
                np.linalg.norm(
                    x_fixed - x_true
                )
            )
        )

        param_errors.append(
            float(
                np.linalg.norm(
                    x_param - x_true
                )
            )
        )

        reset_errors.append(
            float(
                np.linalg.norm(
                    x_reset - x_true
                )
            )
        )

    evaluation = slice(
        ADAPT_STEP + 100,
        STEPS
    )

    return {
        "estimated_g":
            estimated_g,

        "fixed_error":
            float(
                np.mean(
                    fixed_errors[evaluation]
                )
            ),

        "parameter_only_error":
            float(
                np.mean(
                    param_errors[evaluation]
                )
            ),

        "parameter_plus_state_error":
            float(
                np.mean(
                    reset_errors[evaluation]
                )
            ),
    }


results = [
    run_trial(20260824 + i)
    for i in range(TRIALS)
]


def mean(name):
    return float(
        np.mean([
            r[name]
            for r in results
        ])
    )


estimated_g = np.array([
    r["estimated_g"]
    for r in results
])

fixed = np.array([
    r["fixed_error"]
    for r in results
])

param = np.array([
    r["parameter_only_error"]
    for r in results
])

reset = np.array([
    r["parameter_plus_state_error"]
    for r in results
])


summary = {
    "true_g":
        TRUE_G_AFTER,

    "mean_estimated_g":
        float(np.mean(estimated_g)),

    "std_estimated_g":
        float(np.std(estimated_g)),

    "mean_absolute_g_error":
        float(
            np.mean(
                np.abs(
                    estimated_g
                    - TRUE_G_AFTER
                )
            )
        ),

    "fixed_error":
        float(np.mean(fixed)),

    "parameter_only_error":
        float(np.mean(param)),

    "parameter_plus_state_error":
        float(np.mean(reset)),

    "parameter_only_improvement":
        float(
            np.mean(
                1.0 - param / fixed
            )
        ),

    "parameter_plus_state_improvement":
        float(
            np.mean(
                1.0 - reset / fixed
            )
        ),
}


evidence = {
    "experiment": "QMS-QT-005E",

    "title": (
        "Adaptive twin recovery under "
        "improved measurement architecture"
    ),

    "measurement_channels": [
        "x1",
        "x2"
    ],

    "summary":
        summary,

    "trials":
        TRIALS,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "recovery experiment using two abstract "
        "external quadrature observables and a "
        "known candidate model family. "
        "No physical autonomous quantum-field "
        "twin is claimed."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_005e_improved_recovery.json"
)

out.write_text(
    json.dumps(
        evidence,
        indent=2
    ) + "\n"
)


print(f"evidence: {out}")
print()
print("QMS-QT-005E IMPROVED RECOVERY")

print(
    "true g:",
    TRUE_G_AFTER
)

print(
    "estimated g:",
    f"{summary['mean_estimated_g']:.6f}",
    "±",
    f"{summary['std_estimated_g']:.6f}",
)

print(
    "mean |g error|:",
    f"{summary['mean_absolute_g_error']:.6e}"
)

print()

print(
    "fixed twin error:",
    f"{summary['fixed_error']:.6e}"
)

print(
    "parameter-only error:",
    f"{summary['parameter_only_error']:.6e}"
)

print(
    "parameter + state error:",
    f"{summary['parameter_plus_state_error']:.6e}"
)

print()

print(
    "parameter-only improvement:",
    f"{100 * summary['parameter_only_improvement']:.2f}%"
)

print(
    "parameter + state improvement:",
    f"{100 * summary['parameter_plus_state_improvement']:.2f}%"
)
