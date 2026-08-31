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


def estimate_g_from_window(
    measurements,
    x_start,
    candidate_grid,
):
    """
    Estimate coupling by finding the candidate model
    whose deterministic predicted x1 trajectory best
    matches the observed measurement window.

    This is a controlled computational baseline, not
    a general quantum-system-identification method.
    """
    best_g = None
    best_mse = np.inf

    for g in candidate_grid:
        A = expm(build_F(g) * DT)

        x = x_start.copy()
        predictions = []

        for _ in range(len(measurements)):
            x = A @ x
            predictions.append(float(C @ x))

        mse = float(
            np.mean(
                (
                    np.asarray(measurements)
                    - np.asarray(predictions)
                ) ** 2
            )
        )

        if mse < best_mse:
            best_mse = mse
            best_g = float(g)

    return best_g, best_mse


def kalman_step(
    x_hat,
    P,
    A_model,
    y,
    Q,
    R,
):
    x_pred = A_model @ x_hat
    P_pred = A_model @ P @ A_model.T + Q

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

    return x_new, P_new, float(residual[0])


def run_trial(seed):
    rng = np.random.default_rng(seed)

    Q = (PROCESS_SIGMA ** 2) * np.eye(4)
    R = np.array([[MEAS_SIGMA ** 2]])

    x_true = np.array(
        [0.7, -0.2, 0.5, 0.45],
        dtype=float,
    )

    # Fixed-model twin
    x_fixed = np.zeros(4)
    P_fixed = np.eye(4)

    # Adaptive twin
    x_adapt = np.zeros(4)
    P_adapt = np.eye(4)

    fixed_g = BASE_G
    adaptive_g = BASE_G

    fixed_errors = []
    adaptive_errors = []

    fixed_mode2_errors = []
    adaptive_mode2_errors = []

    measurements = []
    adaptive_state_history = []

    estimated_g = None
    fit_mse = None

    for k in range(STEPS):

        # ------------------------------------------------
        # True environment
        # ------------------------------------------------

        g_true = (
            BASE_G
            if k < CHANGE_STEP
            else TRUE_G_AFTER
        )

        A_true = expm(build_F(g_true) * DT)

        x_true = (
            A_true @ x_true
            + rng.multivariate_normal(
                np.zeros(4), Q
            )
        )

        y = float(
            C @ x_true
            + rng.normal(scale=MEAS_SIGMA)
        )

        measurements.append(y)

        # ------------------------------------------------
        # Fixed twin: never changes model
        # ------------------------------------------------

        A_fixed = expm(
            build_F(fixed_g) * DT
        )

        x_fixed, P_fixed, _ = kalman_step(
            x_fixed,
            P_fixed,
            A_fixed,
            y,
            Q,
            R,
        )

        # ------------------------------------------------
        # Adaptive twin
        # ------------------------------------------------

        # Estimate new coupling once, after a finite
        # post-change observation window.
        if k == ADAPT_STEP:

            window_start = CHANGE_STEP
            window_end = ADAPT_STEP

            y_window = measurements[
                window_start:window_end
            ]

            # Use the adaptive twin state immediately
            # before the change as the starting estimate.
            x_window_start = (
                adaptive_state_history[
                    CHANGE_STEP - 1
                ].copy()
            )

            estimated_g, fit_mse = (
                estimate_g_from_window(
                    y_window,
                    x_window_start,
                    G_CANDIDATES,
                )
            )

            adaptive_g = estimated_g

        A_adapt = expm(
            build_F(adaptive_g) * DT
        )

        x_adapt, P_adapt, _ = kalman_step(
            x_adapt,
            P_adapt,
            A_adapt,
            y,
            Q,
            R,
        )

        adaptive_state_history.append(
            x_adapt.copy()
        )

        # ------------------------------------------------
        # Truth-only validation metrics
        # ------------------------------------------------

        fixed_errors.append(
            float(
                np.linalg.norm(
                    x_fixed - x_true
                )
            )
        )

        adaptive_errors.append(
            float(
                np.linalg.norm(
                    x_adapt - x_true
                )
            )
        )

        fixed_mode2_errors.append(
            float(
                np.linalg.norm(
                    x_fixed[2:] - x_true[2:]
                )
            )
        )

        adaptive_mode2_errors.append(
            float(
                np.linalg.norm(
                    x_adapt[2:] - x_true[2:]
                )
            )
        )

    post_adapt = slice(
        ADAPT_STEP + 100,
        STEPS,
    )

    return {
        "estimated_g": estimated_g,
        "fit_mse": fit_mse,

        "fixed_post_error": float(
            np.mean(
                fixed_errors[post_adapt]
            )
        ),

        "adaptive_post_error": float(
            np.mean(
                adaptive_errors[post_adapt]
            )
        ),

        "fixed_post_mode2_error": float(
            np.mean(
                fixed_mode2_errors[post_adapt]
            )
        ),

        "adaptive_post_mode2_error": float(
            np.mean(
                adaptive_mode2_errors[post_adapt]
            )
        ),
    }


trial_results = []

for trial in range(TRIALS):
    trial_results.append(
        run_trial(
            20260824 + trial
        )
    )


estimated_gs = np.array([
    r["estimated_g"]
    for r in trial_results
])

fixed_errors = np.array([
    r["fixed_post_error"]
    for r in trial_results
])

adaptive_errors = np.array([
    r["adaptive_post_error"]
    for r in trial_results
])

fixed_mode2 = np.array([
    r["fixed_post_mode2_error"]
    for r in trial_results
])

adaptive_mode2 = np.array([
    r["adaptive_post_mode2_error"]
    for r in trial_results
])


error_reduction = (
    1.0
    - adaptive_errors / fixed_errors
)

mode2_reduction = (
    1.0
    - adaptive_mode2 / fixed_mode2
)


summary = {
    "true_g_before": BASE_G,
    "true_g_after": TRUE_G_AFTER,

    "mean_estimated_g": float(
        np.mean(estimated_gs)
    ),

    "std_estimated_g": float(
        np.std(estimated_gs)
    ),

    "mean_absolute_g_error": float(
        np.mean(
            np.abs(
                estimated_gs
                - TRUE_G_AFTER
            )
        )
    ),

    "mean_fixed_post_error": float(
        np.mean(fixed_errors)
    ),

    "mean_adaptive_post_error": float(
        np.mean(adaptive_errors)
    ),

    "mean_state_error_reduction_fraction":
        float(np.mean(error_reduction)),

    "mean_fixed_mode2_error": float(
        np.mean(fixed_mode2)
    ),

    "mean_adaptive_mode2_error": float(
        np.mean(adaptive_mode2)
    ),

    "mean_mode2_error_reduction_fraction":
        float(np.mean(mode2_reduction)),
}


evidence = {
    "experiment": "QMS-QT-005A",

    "title": (
        "Adaptive resynchronization of a "
        "causal virtual quantum twin"
    ),

    "model": {
        "initial_coupling": BASE_G,
        "true_coupling_after_change":
            TRUE_G_AFTER,

        "change_step": CHANGE_STEP,
        "adaptation_step": ADAPT_STEP,

        "candidate_grid_min":
            float(G_CANDIDATES.min()),

        "candidate_grid_max":
            float(G_CANDIDATES.max()),

        "candidate_count":
            int(len(G_CANDIDATES)),
    },

    "trials": TRIALS,

    "summary": summary,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "parameter-adaptation experiment. "
        "The coupling estimator uses a bounded "
        "candidate grid and known model family. "
        "This does not establish general autonomous "
        "quantum-system identification or a physical "
        "self-correcting quantum-field twin."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_005a_adaptive_twin.json"
)

out.write_text(
    json.dumps(evidence, indent=2)
    + "\n"
)


print(f"evidence: {out}")
print()
print("QMS-QT-005A ADAPTIVE TWIN")

print(
    "true coupling:",
    BASE_G,
    "->",
    TRUE_G_AFTER,
)

print(
    "estimated coupling:",
    f"{summary['mean_estimated_g']:.6f}",
    "±",
    f"{summary['std_estimated_g']:.6f}",
)

print(
    "mean |g error|:",
    f"{summary['mean_absolute_g_error']:.6e}",
)

print()
print(
    "fixed twin state error:",
    f"{summary['mean_fixed_post_error']:.6e}",
)

print(
    "adaptive twin state error:",
    f"{summary['mean_adaptive_post_error']:.6e}",
)

print(
    "state error reduction:",
    f"{100 * summary['mean_state_error_reduction_fraction']:.2f}%"
)

print()
print(
    "fixed twin mode2 error:",
    f"{summary['mean_fixed_mode2_error']:.6e}",
)

print(
    "adaptive twin mode2 error:",
    f"{summary['mean_adaptive_mode2_error']:.6e}",
)

print(
    "mode2 error reduction:",
    f"{100 * summary['mean_mode2_error_reduction_fraction']:.2f}%"
)
