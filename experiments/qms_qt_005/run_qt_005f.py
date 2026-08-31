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

PROCESS_SIGMA = 1e-4
MEAS_SIGMA = 1e-3

C = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])

G_CANDIDATES = np.linspace(0.04, 0.20, 81)

PERTURBATIONS = [
    0.16,
    0.14,
    0.12,
    0.10,
    0.08,
    0.06,
]


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


def fit_g(measurements):
    y = np.asarray(measurements).reshape(-1)

    best_g = None
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
            np.mean((y - y_hat) ** 2)
        )

        if mse < best_mse:
            best_mse = mse
            best_g = float(g)

    return best_g, best_mse


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


def run_trial(true_g_after, seed):

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

    # Adaptive twin
    x_adapt = np.zeros(4)
    P_adapt = np.eye(4)

    adaptive_g = BASE_G

    measurements = []

    fixed_errors = []
    adaptive_errors = []

    fixed_mode2_errors = []
    adaptive_mode2_errors = []

    estimated_g = None

    for k in range(STEPS):

        true_g = (
            BASE_G
            if k < CHANGE_STEP
            else true_g_after
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

        # Fixed twin
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

        # Adaptive twin
        A_adapt = expm(
            build_F(adaptive_g) * DT
        )

        x_adapt, P_adapt = kalman_step(
            x_adapt,
            P_adapt,
            A_adapt,
            y,
            Q,
            R,
        )

        # Parameter adaptation
        if k == ADAPT_STEP:

            y_window = measurements[
                CHANGE_STEP:ADAPT_STEP
            ]

            estimated_g, _ = fit_g(
                y_window
            )

            adaptive_g = estimated_g

        # Truth-only validation
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

    evaluation = slice(
        ADAPT_STEP + 100,
        STEPS
    )

    def avg_region(values):
        return float(
            np.mean(
                np.asarray(values)[evaluation]
            )
        )

    return {
        "true_g_after":
            true_g_after,

        "estimated_g":
            estimated_g,

        "absolute_g_error":
            abs(
                estimated_g
                - true_g_after
            ),

        "fixed_error":
            avg_region(fixed_errors),

        "adaptive_error":
            avg_region(adaptive_errors),

        "fixed_mode2_error":
            avg_region(
                fixed_mode2_errors
            ),

        "adaptive_mode2_error":
            avg_region(
                adaptive_mode2_errors
            ),
    }


all_results = []

for condition_index, true_g_after in enumerate(PERTURBATIONS):

    trials = []

    for trial in range(TRIALS):

        trials.append(
            run_trial(
                true_g_after,
                20260824
                + 10000 * condition_index
                + trial
            )
        )

    estimates = np.array([
        r["estimated_g"]
        for r in trials
    ])

    g_errors = np.array([
        r["absolute_g_error"]
        for r in trials
    ])

    fixed = np.array([
        r["fixed_error"]
        for r in trials
    ])

    adaptive = np.array([
        r["adaptive_error"]
        for r in trials
    ])

    fixed_mode2 = np.array([
        r["fixed_mode2_error"]
        for r in trials
    ])

    adaptive_mode2 = np.array([
        r["adaptive_mode2_error"]
        for r in trials
    ])

    improvement = (
        1.0 - adaptive / fixed
    )

    mode2_improvement = (
        1.0
        - adaptive_mode2 / fixed_mode2
    )

    all_results.append({
        "true_g_after":
            true_g_after,

        "trials":
            TRIALS,

        "mean_estimated_g":
            float(np.mean(estimates)),

        "std_estimated_g":
            float(np.std(estimates)),

        "mean_absolute_g_error":
            float(np.mean(g_errors)),

        "fraction_within_0.01":
            float(
                np.mean(g_errors <= 0.01)
            ),

        "mean_fixed_error":
            float(np.mean(fixed)),

        "mean_adaptive_error":
            float(np.mean(adaptive)),

        "mean_error_improvement_fraction":
            float(np.mean(improvement)),

        "fraction_trials_improved":
            float(
                np.mean(
                    adaptive < fixed
                )
            ),

        "mean_fixed_mode2_error":
            float(
                np.mean(fixed_mode2)
            ),

        "mean_adaptive_mode2_error":
            float(
                np.mean(adaptive_mode2)
            ),

        "mean_mode2_improvement_fraction":
            float(
                np.mean(
                    mode2_improvement
                )
            ),
    })


evidence = {
    "experiment": "QMS-QT-005F",

    "title": (
        "Adaptive twin recovery across "
        "internal coupling perturbation magnitude"
    ),

    "baseline_coupling":
        BASE_G,

    "measurement_channels": [
        "x1",
        "x2",
    ],

    "results":
        all_results,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "robustness study using two abstract "
        "external observables, a known model "
        "family, and bounded grid-search "
        "parameter estimation. Results do not "
        "establish general physical quantum-field "
        "twin adaptation."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_005f_recovery_robustness.json"
)

out.write_text(
    json.dumps(
        evidence,
        indent=2
    ) + "\n"
)


print(f"evidence: {out}")
print()
print("QMS-QT-005F RECOVERY ROBUSTNESS")

for r in all_results:

    print()
    print(
        "true g:",
        f"{r['true_g_after']:.3f}"
    )

    print(
        "  estimated:",
        f"{r['mean_estimated_g']:.6f}",
        "±",
        f"{r['std_estimated_g']:.6f}"
    )

    print(
        "  mean |g error|:",
        f"{r['mean_absolute_g_error']:.6e}"
    )

    print(
        "  within ±0.01:",
        f"{100 * r['fraction_within_0.01']:.1f}%"
    )

    print(
        "  fixed error:",
        f"{r['mean_fixed_error']:.6e}"
    )

    print(
        "  adaptive error:",
        f"{r['mean_adaptive_error']:.6e}"
    )

    print(
        "  mean improvement:",
        f"{100 * r['mean_error_improvement_fraction']:.2f}%"
    )

    print(
        "  trials improved:",
        f"{100 * r['fraction_trials_improved']:.1f}%"
    )

    print(
        "  mode2 improvement:",
        f"{100 * r['mean_mode2_improvement_fraction']:.2f}%"
    )
