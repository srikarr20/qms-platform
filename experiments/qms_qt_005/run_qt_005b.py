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

C = np.array([[1.0, 0.0, 0.0, 0.0]])

G_CANDIDATES = np.linspace(
    0.04, 0.20, 81
)


def build_F(g):
    omega1 = 1.00
    omega2 = 1.20
    gamma1 = 0.08
    gamma2 = 0.06

    return np.array([
        [-gamma1, omega1, 0.0, 0.0],
        [-omega1, -gamma1, g, 0.0],
        [0.0, 0.0, -gamma2, omega2],
        [g, 0.0, -omega2, -gamma2],
    ])


def build_window_matrix(A, length):
    rows = []

    for k in range(1, length + 1):
        rows.append(
            C @ np.linalg.matrix_power(A, k)
        )

    return np.vstack(rows)


def estimate_g_and_start_state(
    measurements,
):
    best_g = None
    best_state = None
    best_mse = np.inf

    y = np.asarray(measurements)

    for g in G_CANDIDATES:

        A = expm(build_F(g) * DT)

        H = build_window_matrix(
            A,
            len(y)
        )

        x_start = (
            np.linalg.pinv(H) @ y
        )

        prediction = H @ x_start

        mse = float(
            np.mean(
                (y - prediction) ** 2
            )
        )

        if mse < best_mse:
            best_mse = mse
            best_g = float(g)
            best_state = x_start.copy()

    return best_g, best_state, best_mse


def propagate_state(
    x,
    A,
    steps,
):
    return (
        np.linalg.matrix_power(A, steps)
        @ x
    )


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
        + (K @ residual).reshape(-1)
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

    R = np.array([
        [MEAS_SIGMA ** 2]
    ])

    x_true = np.array(
        [0.7, -0.2, 0.5, 0.45],
        dtype=float,
    )

    # -----------------------------------------
    # Three twins
    # -----------------------------------------

    x_fixed = np.zeros(4)
    P_fixed = np.eye(4)

    x_param = np.zeros(4)
    P_param = np.eye(4)

    x_reset = np.zeros(4)
    P_reset = np.eye(4)

    g_param = BASE_G
    g_reset = BASE_G

    measurements = []

    fixed_errors = []
    param_errors = []
    reset_errors = []

    fixed_mode2 = []
    param_mode2 = []
    reset_mode2 = []

    estimated_g = None
    fit_mse = None

    for k in range(STEPS):

        # -------------------------------------
        # True environment
        # -------------------------------------

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

        y = float(
            C @ x_true
            + rng.normal(
                scale=MEAS_SIGMA
            )
        )

        measurements.append(y)

        # -------------------------------------
        # Fixed twin
        # -------------------------------------

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

        # -------------------------------------
        # Parameter-only adaptive twin
        # -------------------------------------

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

        # -------------------------------------
        # Parameter + state-reset twin
        # -------------------------------------

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

        # -------------------------------------
        # Adaptation
        # -------------------------------------

        if k == ADAPT_STEP:

            y_window = measurements[
                CHANGE_STEP:ADAPT_STEP
            ]

            (
                estimated_g,
                reconstructed_start,
                fit_mse,
            ) = estimate_g_and_start_state(
                y_window
            )

            g_param = estimated_g
            g_reset = estimated_g

            A_est = expm(
                build_F(estimated_g) * DT
            )

            # Reconstruct state at start of
            # post-change window, then propagate
            # it to the present adaptation time.
            x_reset = propagate_state(
                reconstructed_start,
                A_est,
                len(y_window),
            )

            # Deliberately reset uncertainty to
            # acknowledge state reinitialization.
            P_reset = 0.1 * np.eye(4)

        # -------------------------------------
        # Truth-only validation
        # -------------------------------------

        fixed_errors.append(
            np.linalg.norm(
                x_fixed - x_true
            )
        )

        param_errors.append(
            np.linalg.norm(
                x_param - x_true
            )
        )

        reset_errors.append(
            np.linalg.norm(
                x_reset - x_true
            )
        )

        fixed_mode2.append(
            np.linalg.norm(
                x_fixed[2:] - x_true[2:]
            )
        )

        param_mode2.append(
            np.linalg.norm(
                x_param[2:] - x_true[2:]
            )
        )

        reset_mode2.append(
            np.linalg.norm(
                x_reset[2:] - x_true[2:]
            )
        )

    evaluation = slice(
        ADAPT_STEP + 100,
        STEPS,
    )

    def mean_region(x):
        return float(
            np.mean(
                np.asarray(x)[evaluation]
            )
        )

    return {
        "estimated_g":
            estimated_g,

        "fit_mse":
            fit_mse,

        "fixed_error":
            mean_region(fixed_errors),

        "parameter_only_error":
            mean_region(param_errors),

        "parameter_plus_reset_error":
            mean_region(reset_errors),

        "fixed_mode2_error":
            mean_region(fixed_mode2),

        "parameter_only_mode2_error":
            mean_region(param_mode2),

        "parameter_plus_reset_mode2_error":
            mean_region(reset_mode2),
    }


results = [
    run_trial(20260824 + trial)
    for trial in range(TRIALS)
]


def avg(name):
    return float(
        np.mean([
            r[name]
            for r in results
        ])
    )


summary = {
    "true_g_after":
        TRUE_G_AFTER,

    "mean_estimated_g":
        avg("estimated_g"),

    "mean_absolute_g_error":
        float(
            np.mean([
                abs(
                    r["estimated_g"]
                    - TRUE_G_AFTER
                )
                for r in results
            ])
        ),

    "fixed_error":
        avg("fixed_error"),

    "parameter_only_error":
        avg("parameter_only_error"),

    "parameter_plus_reset_error":
        avg("parameter_plus_reset_error"),

    "fixed_mode2_error":
        avg("fixed_mode2_error"),

    "parameter_only_mode2_error":
        avg(
            "parameter_only_mode2_error"
        ),

    "parameter_plus_reset_mode2_error":
        avg(
            "parameter_plus_reset_mode2_error"
        ),
}


evidence = {
    "experiment":
        "QMS-QT-005B",

    "title": (
        "Adaptive twin recovery with "
        "parameter estimation and state reinitialization"
    ),

    "summary":
        summary,

    "trials":
        TRIALS,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "experiment using a known candidate model "
        "family and retrospective finite-window "
        "reconstruction at the adaptation event. "
        "This is not experimental autonomous "
        "quantum-field resynchronization."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_005b_state_reinitialization.json"
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
    "QMS-QT-005B ADAPTIVE STATE RECOVERY"
)

print(
    "true g:",
    TRUE_G_AFTER
)

print(
    "estimated g:",
    f"{summary['mean_estimated_g']:.6f}"
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
    "parameter + state reset error:",
    f"{summary['parameter_plus_reset_error']:.6e}"
)

print()

print(
    "fixed mode2 error:",
    f"{summary['fixed_mode2_error']:.6e}"
)

print(
    "parameter-only mode2 error:",
    f"{summary['parameter_only_mode2_error']:.6e}"
)

print(
    "parameter + state reset mode2:",
    f"{summary['parameter_plus_reset_mode2_error']:.6e}"
)
