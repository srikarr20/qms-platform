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
TRIALS = 200

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


def lag1_autocorr(x):
    x = np.asarray(x, dtype=float)

    if len(x) < 3:
        return 0.0

    a = x[:-1] - np.mean(x[:-1])
    b = x[1:] - np.mean(x[1:])

    denom = np.sqrt(
        np.sum(a * a) * np.sum(b * b)
    )

    if denom <= 0:
        return 0.0

    return float(np.sum(a * b) / denom)


def summarize(x):
    x = np.asarray(x, dtype=float)

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
        "mean_abs": float(np.mean(np.abs(x))),
        "lag1_autocorrelation": lag1_autocorr(x),
    }


def run_trial(case_name, seed):
    rng = np.random.default_rng(seed)

    baseline_g = 0.18

    A_model = expm(
        build_F(baseline_g) * DT
    )

    x_true = np.array(
        [0.7, -0.2, 0.5, 0.45],
        dtype=float
    )

    x_hat = np.zeros(4)
    P = np.eye(4)

    Q = (PROCESS_SIGMA ** 2) * np.eye(4)

    normalized_residuals = []

    # Truth-only diagnostics retained for scientific
    # validation, but NOT used by attribution logic.
    true_state_errors = []

    for k in range(STEPS):

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

        A_true = expm(
            build_F(g_true) * DT
        )

        process_noise = rng.multivariate_normal(
            np.zeros(4), Q
        )

        x_true = (
            A_true @ x_true
            + process_noise
        )

        y = (
            C @ x_true
            + rng.normal(scale=meas_sigma)
        )

        # Twin prediction
        x_pred = A_model @ x_hat
        P_pred = (
            A_model @ P @ A_model.T + Q
        )

        R_assumed = np.array([
            [BASE_MEAS_SIGMA ** 2]
        ])

        y_pred = C @ x_pred
        residual = y - y_pred

        S = (
            C @ P_pred @ C.T
            + R_assumed
        )

        normalized_residual = (
            residual[0] / np.sqrt(S[0, 0])
        )

        normalized_residuals.append(
            float(normalized_residual)
        )

        # State update
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

        true_state_errors.append(
            float(
                np.linalg.norm(
                    x_hat - x_true
                )
            )
        )

    pre = normalized_residuals[
        CHANGE_STEP - 300:
        CHANGE_STEP
    ]

    post = normalized_residuals[
        CHANGE_STEP:
        CHANGE_STEP + 300
    ]

    pre_stats = summarize(pre)
    post_stats = summarize(post)

    rms_ratio = (
        post_stats["rms"]
        / pre_stats["rms"]
    )

    lag1_change = (
        post_stats["lag1_autocorrelation"]
        - pre_stats["lag1_autocorrelation"]
    )

    mean_abs_ratio = (
        post_stats["mean_abs"]
        / pre_stats["mean_abs"]
    )

    # --------------------------------------------------
    # Transparent external-only attribution rule
    #
    # No hidden true-state information is used.
    # --------------------------------------------------

    if rms_ratio < 1.20:
        classification = "nominal"

    elif rms_ratio >= 3.0 and abs(
        post_stats["lag1_autocorrelation"]
    ) < 0.20:
        classification = (
            "measurement_system_change"
        )

    elif (
        rms_ratio >= 1.20
        and abs(
            post_stats[
                "lag1_autocorrelation"
            ]
        ) >= 0.20
    ):
        classification = (
            "physical_or_model_change"
        )

    else:
        classification = "ambiguous"

    truth_post_error = float(
        np.mean(
            true_state_errors[
                CHANGE_STEP:
                CHANGE_STEP + 300
            ]
        )
    )

    return {
        "case": case_name,
        "pre": pre_stats,
        "post": post_stats,
        "rms_ratio": float(rms_ratio),
        "mean_abs_ratio": float(
            mean_abs_ratio
        ),
        "lag1_change": float(
            lag1_change
        ),
        "classification":
            classification,

        # Validation only.
        "truth_only_post_state_error":
            truth_post_error,
    }


cases = [
    "baseline",
    "physical_change",
    "measurement_change",
]

all_results = {}
confusion = {}

for case_index, case in enumerate(cases):

    trials = []

    for trial in range(TRIALS):
        trials.append(
            run_trial(
                case,
                20260824
                + 10000 * case_index
                + trial
            )
        )

    labels = [
        r["classification"]
        for r in trials
    ]

    counts = {
        label: labels.count(label)
        for label in sorted(set(labels))
    }

    all_results[case] = {
        "trials": TRIALS,

        "mean_rms_ratio": float(
            np.mean([
                r["rms_ratio"]
                for r in trials
            ])
        ),

        "std_rms_ratio": float(
            np.std([
                r["rms_ratio"]
                for r in trials
            ])
        ),

        "mean_post_lag1": float(
            np.mean([
                r["post"][
                    "lag1_autocorrelation"
                ]
                for r in trials
            ])
        ),

        "std_post_lag1": float(
            np.std([
                r["post"][
                    "lag1_autocorrelation"
                ]
                for r in trials
            ])
        ),

        "classification_counts":
            counts,

        "mean_truth_only_post_state_error":
            float(
                np.mean([
                    r[
                        "truth_only_post_state_error"
                    ]
                    for r in trials
                ])
            ),
    }

    confusion[case] = counts


evidence = {
    "experiment": "QMS-QT-004B",

    "title": (
        "Observable failure attribution "
        "from residual structure"
    ),

    "external_diagnostics": [
        "normalized innovation RMS",
        "normalized innovation mean absolute value",
        "normalized innovation lag-1 autocorrelation",
    ],

    "attribution_rule": {
        "nominal":
            "RMS ratio < 1.20",

        "measurement_system_change":
            (
                "RMS ratio >= 3.0 and "
                "|post lag-1 autocorrelation| < 0.20"
            ),

        "physical_or_model_change":
            (
                "RMS ratio >= 1.20 and "
                "|post lag-1 autocorrelation| >= 0.20"
            ),

        "otherwise":
            "ambiguous",
    },

    "important_note": (
        "The attribution rule uses only external "
        "innovation statistics. Hidden true-state "
        "error is retained only for offline validation."
    ),

    "results": all_results,
    "confusion_counts": confusion,

    "scientific_boundary": (
        "Controlled computational finite-mode test. "
        "Thresholds are provisional and specific to "
        "this simulated environment; no universal "
        "failure classifier is established."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_004b_failure_attribution.json"
)

out.write_text(
    json.dumps(evidence, indent=2)
    + "\n"
)

print(f"evidence: {out}")
print()
print(
    "QMS-QT-004B OBSERVABLE FAILURE ATTRIBUTION"
)

for case, r in all_results.items():

    print()
    print(case)

    print(
        "  RMS ratio:",
        f"{r['mean_rms_ratio']:.4f}",
        "±",
        f"{r['std_rms_ratio']:.4f}",
    )

    print(
        "  post lag1:",
        f"{r['mean_post_lag1']:.4f}",
        "±",
        f"{r['std_post_lag1']:.4f}",
    )

    print(
        "  classifications:",
        r["classification_counts"],
    )

    print(
        "  truth-only post state error:",
        f"{r['mean_truth_only_post_state_error']:.6e}",
    )
