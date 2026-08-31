import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

DT = 0.05
STEPS = 1200
CHANGE_STEP = 600
TRIALS = 100

C = np.array([[1.0, 0.0, 0.0, 0.0]])

PROCESS_SIGMA = 1e-4
BASE_MEAS_SIGMA = 1e-3


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


def run_trial(case_name, seed):
    rng = np.random.default_rng(seed)

    g_model = 0.18
    F_model = build_F(g_model)
    A_model = expm(F_model * DT)

    x_true = np.array([0.7, -0.2, 0.5, 0.45], dtype=float)
    x_hat = np.zeros(4)

    P = np.eye(4)
    Q = (PROCESS_SIGMA ** 2) * np.eye(4)

    residuals = []
    normalized_residuals = []
    errors = []
    mode2_errors = []
    posterior_trace = []

    for k in range(STEPS):

        # ---------------------------------------------
        # True physical environment
        # ---------------------------------------------

        if case_name == "baseline":
            g_true = 0.18
            meas_sigma = BASE_MEAS_SIGMA

        elif case_name == "physical_change":
            g_true = (
                0.18 if k < CHANGE_STEP
                else 0.08
            )
            meas_sigma = BASE_MEAS_SIGMA

        elif case_name == "measurement_change":
            g_true = 0.18
            meas_sigma = (
                BASE_MEAS_SIGMA
                if k < CHANGE_STEP
                else 1e-2
            )

        else:
            raise ValueError(case_name)

        A_true = expm(build_F(g_true) * DT)

        process_noise = rng.multivariate_normal(
            np.zeros(4), Q
        )

        x_true = A_true @ x_true + process_noise

        y = (
            C @ x_true
            + rng.normal(scale=meas_sigma)
        )

        # ---------------------------------------------
        # Twin prediction
        #
        # Twin intentionally continues assuming the
        # baseline physical model and baseline sensor.
        # ---------------------------------------------

        x_pred = A_model @ x_hat
        P_pred = A_model @ P @ A_model.T + Q

        R_assumed = np.array([
            [BASE_MEAS_SIGMA ** 2]
        ])

        y_pred = C @ x_pred
        residual = y - y_pred

        S = C @ P_pred @ C.T + R_assumed

        normalized_residual = (
            residual[0] / np.sqrt(S[0, 0])
        )

        K = (
            P_pred
            @ C.T
            @ np.linalg.inv(S)
        )

        x_hat = (
            x_pred
            + (K @ residual).reshape(-1)
        )

        P = (
            np.eye(4) - K @ C
        ) @ P_pred

        residuals.append(float(residual[0]))
        normalized_residuals.append(
            float(normalized_residual)
        )

        errors.append(
            float(np.linalg.norm(x_hat - x_true))
        )

        mode2_errors.append(
            float(
                np.linalg.norm(
                    x_hat[2:] - x_true[2:]
                )
            )
        )

        posterior_trace.append(
            float(np.trace(P))
        )

    pre = slice(CHANGE_STEP - 300, CHANGE_STEP)
    post = slice(CHANGE_STEP, CHANGE_STEP + 300)

    def summarize(values, region):
        x = np.asarray(values)[region]
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "mean_abs": float(np.mean(np.abs(x))),
            "rms": float(np.sqrt(np.mean(x**2))),
        }

    return {
        "case": case_name,

        "residual_pre": summarize(residuals, pre),
        "residual_post": summarize(residuals, post),

        "normalized_residual_pre":
            summarize(normalized_residuals, pre),

        "normalized_residual_post":
            summarize(normalized_residuals, post),

        "state_error_pre":
            summarize(errors, pre),

        "state_error_post":
            summarize(errors, post),

        "mode2_error_pre":
            summarize(mode2_errors, pre),

        "mode2_error_post":
            summarize(mode2_errors, post),

        "posterior_trace_pre":
            summarize(posterior_trace, pre),

        "posterior_trace_post":
            summarize(posterior_trace, post),
    }


cases = [
    "baseline",
    "physical_change",
    "measurement_change",
]

aggregate = {}

for case_index, case in enumerate(cases):

    trial_results = []

    for trial in range(TRIALS):
        trial_results.append(
            run_trial(
                case,
                20260824
                + 10000 * case_index
                + trial
            )
        )

    def average_metric(section, field):
        return float(
            np.mean([
                r[section][field]
                for r in trial_results
            ])
        )

    aggregate[case] = {
        "trials": TRIALS,

        "normalized_residual_pre_rms":
            average_metric(
                "normalized_residual_pre",
                "rms"
            ),

        "normalized_residual_post_rms":
            average_metric(
                "normalized_residual_post",
                "rms"
            ),

        "residual_pre_rms":
            average_metric(
                "residual_pre",
                "rms"
            ),

        "residual_post_rms":
            average_metric(
                "residual_post",
                "rms"
            ),

        "state_error_pre_mean":
            average_metric(
                "state_error_pre",
                "mean"
            ),

        "state_error_post_mean":
            average_metric(
                "state_error_post",
                "mean"
            ),

        "mode2_error_pre_mean":
            average_metric(
                "mode2_error_pre",
                "mean"
            ),

        "mode2_error_post_mean":
            average_metric(
                "mode2_error_post",
                "mean"
            ),

        "posterior_trace_pre_mean":
            average_metric(
                "posterior_trace_pre",
                "mean"
            ),

        "posterior_trace_post_mean":
            average_metric(
                "posterior_trace_post",
                "mean"
            ),
    }


evidence = {
    "experiment": "QMS-QT-004A",
    "title": (
        "Twin divergence under physical-model "
        "and measurement-system perturbations"
    ),

    "baseline_model": {
        "coupling": 0.18,
        "measurement_noise_sigma":
            BASE_MEAS_SIGMA,
    },

    "change_step": CHANGE_STEP,

    "perturbations": {
        "physical_change": {
            "true_coupling_after_change": 0.08,
            "twin_assumed_coupling": 0.18,
        },

        "measurement_change": {
            "true_measurement_noise_after_change":
                1e-2,
            "twin_assumed_measurement_noise":
                BASE_MEAS_SIGMA,
        },
    },

    "aggregate": aggregate,

    "scientific_boundary": (
        "Computational finite-mode perturbation "
        "experiment only. This experiment tests "
        "whether different mismatch mechanisms "
        "produce distinguishable diagnostic signatures; "
        "it does not yet implement a validated "
        "failure-attribution classifier."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_004a_twin_divergence.json"
)

out.write_text(
    json.dumps(evidence, indent=2) + "\n"
)

print(f"evidence: {out}")
print()
print("QMS-QT-004A TWIN DIVERGENCE")

for case, r in aggregate.items():

    print()
    print(case)

    print(
        "  normalized residual RMS:",
        f"{r['normalized_residual_pre_rms']:.4f}",
        "->",
        f"{r['normalized_residual_post_rms']:.4f}",
    )

    print(
        "  total state error:",
        f"{r['state_error_pre_mean']:.6e}",
        "->",
        f"{r['state_error_post_mean']:.6e}",
    )

    print(
        "  mode2 error:",
        f"{r['mode2_error_pre_mean']:.6e}",
        "->",
        f"{r['mode2_error_post_mean']:.6e}",
    )

    print(
        "  twin covariance trace:",
        f"{r['posterior_trace_pre_mean']:.6e}",
        "->",
        f"{r['posterior_trace_post_mean']:.6e}",
    )
