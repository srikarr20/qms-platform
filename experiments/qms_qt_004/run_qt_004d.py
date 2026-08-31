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

CALIBRATION_TRIALS = 500
TEST_TRIALS = 100

C = np.array([[1.0, 0.0, 0.0, 0.0]])

PROCESS_SIGMA = 1e-4
BASE_G = 0.18
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

    a = x[:-1] - np.mean(x[:-1])
    b = x[1:] - np.mean(x[1:])

    denom = np.sqrt(
        np.sum(a * a) * np.sum(b * b)
    )

    if denom <= 0:
        return 0.0

    return float(np.sum(a * b) / denom)


def run_trial(
    seed,
    physical_g_after=None,
    measurement_sigma_after=None,
):
    rng = np.random.default_rng(seed)

    A_model = expm(build_F(BASE_G) * DT)

    x_true = np.array(
        [0.7, -0.2, 0.5, 0.45],
        dtype=float,
    )

    x_hat = np.zeros(4)
    P = np.eye(4)

    Q = (PROCESS_SIGMA ** 2) * np.eye(4)

    innovations = []

    for k in range(STEPS):

        g_true = BASE_G
        meas_sigma = BASE_MEAS_SIGMA

        if (
            physical_g_after is not None
            and k >= CHANGE_STEP
        ):
            g_true = physical_g_after

        if (
            measurement_sigma_after is not None
            and k >= CHANGE_STEP
        ):
            meas_sigma = measurement_sigma_after

        A_true = expm(build_F(g_true) * DT)

        x_true = (
            A_true @ x_true
            + rng.multivariate_normal(
                np.zeros(4), Q
            )
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

        residual = y - C @ x_pred

        S = (
            C @ P_pred @ C.T
            + R_assumed
        )

        innovation = (
            residual[0]
            / np.sqrt(S[0, 0])
        )

        innovations.append(float(innovation))

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

    pre = np.asarray(
        innovations[
            CHANGE_STEP - 300:
            CHANGE_STEP
        ]
    )

    post = np.asarray(
        innovations[
            CHANGE_STEP:
            CHANGE_STEP + 300
        ]
    )

    pre_rms = np.sqrt(np.mean(pre**2))
    post_rms = np.sqrt(np.mean(post**2))

    return {
        "rms_ratio": float(
            post_rms / pre_rms
        ),

        "post_abs_lag1": float(
            abs(lag1_autocorr(post))
        ),
    }


# ============================================================
# 1. CALIBRATION — BASELINE ONLY
# ============================================================

baseline_rms = []
baseline_lag = []

for trial in range(CALIBRATION_TRIALS):

    r = run_trial(
        seed=20260824 + trial
    )

    baseline_rms.append(r["rms_ratio"])
    baseline_lag.append(r["post_abs_lag1"])


# 99th percentile nominal thresholds
rms_threshold = float(
    np.quantile(baseline_rms, 0.99)
)

lag_threshold = float(
    np.quantile(baseline_lag, 0.99)
)


print("BASELINE-CALIBRATED THRESHOLDS")
print(
    "  RMS ratio 99th percentile:",
    f"{rms_threshold:.6f}"
)
print(
    "  |lag1| 99th percentile:",
    f"{lag_threshold:.6f}"
)


# ============================================================
# 2. TEST PERTURBATIONS
# ============================================================

tests = []

# Physical coupling sweep
for g_after in [
    0.175,
    0.170,
    0.165,
    0.160,
    0.150,
    0.140,
    0.130,
    0.120,
    0.110,
    0.100,
    0.090,
    0.080,
]:
    tests.append({
        "type": "physical_change",
        "value": g_after,
    })


# Measurement noise sweep
for sigma_after in [
    0.0012,
    0.0015,
    0.0020,
    0.0025,
    0.0030,
    0.0040,
    0.0050,
    0.0075,
    0.0100,
]:
    tests.append({
        "type": "measurement_change",
        "value": sigma_after,
    })


results = []

for test_index, test in enumerate(tests):

    anomaly_flags = []
    rms_flags = []
    lag_flags = []

    rms_values = []
    lag_values = []

    for trial in range(TEST_TRIALS):

        seed = (
            30300000
            + test_index * 10000
            + trial
        )

        if test["type"] == "physical_change":

            r = run_trial(
                seed,
                physical_g_after=test["value"],
            )

        else:

            r = run_trial(
                seed,
                measurement_sigma_after=test["value"],
            )

        rms_flag = (
            r["rms_ratio"] > rms_threshold
        )

        lag_flag = (
            r["post_abs_lag1"] > lag_threshold
        )

        # Any statistically unusual innovation behavior
        anomaly = rms_flag or lag_flag

        anomaly_flags.append(anomaly)
        rms_flags.append(rms_flag)
        lag_flags.append(lag_flag)

        rms_values.append(
            r["rms_ratio"]
        )

        lag_values.append(
            r["post_abs_lag1"]
        )


    results.append({
        "type": test["type"],
        "value": test["value"],
        "trials": TEST_TRIALS,

        "detection_rate": float(
            np.mean(anomaly_flags)
        ),

        "rms_detection_rate": float(
            np.mean(rms_flags)
        ),

        "lag_detection_rate": float(
            np.mean(lag_flags)
        ),

        "mean_rms_ratio": float(
            np.mean(rms_values)
        ),

        "mean_abs_lag1": float(
            np.mean(lag_values)
        ),
    })


evidence = {
    "experiment": "QMS-QT-004D",

    "title": (
        "Baseline-calibrated detectability "
        "boundary for quantum-twin divergence"
    ),

    "calibration": {
        "baseline_trials":
            CALIBRATION_TRIALS,

        "quantile": 0.99,

        "rms_ratio_threshold":
            rms_threshold,

        "absolute_lag1_threshold":
            lag_threshold,
    },

    "test_trials_per_condition":
        TEST_TRIALS,

    "results": results,

    "scientific_boundary": (
        "Detection thresholds are calibrated "
        "only to this finite computational model. "
        "Detection rates characterize this simulated "
        "test bench and are not universal quantum "
        "field or detector thresholds."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_004d_detectability_boundary.json"
)

out.write_text(
    json.dumps(evidence, indent=2)
    + "\n"
)


print()
print("QMS-QT-004D DETECTABILITY BOUNDARY")

for r in results:

    print(
        f"{r['type']:20s} "
        f"value={r['value']:<7} "
        f"detect={r['detection_rate']:.2f} "
        f"RMS={r['rms_detection_rate']:.2f} "
        f"lag={r['lag_detection_rate']:.2f}"
    )
