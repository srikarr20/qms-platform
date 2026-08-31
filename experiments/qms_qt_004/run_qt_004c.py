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


def classify(pre, post):
    pre = np.asarray(pre)
    post = np.asarray(post)

    pre_rms = np.sqrt(np.mean(pre**2))
    post_rms = np.sqrt(np.mean(post**2))

    rms_ratio = post_rms / pre_rms
    post_lag1 = lag1_autocorr(post)

    # SAME rule as QT-004B
    if rms_ratio < 1.20:
        label = "nominal"

    elif (
        rms_ratio >= 3.0
        and abs(post_lag1) < 0.20
    ):
        label = "measurement_system_change"

    elif (
        rms_ratio >= 1.20
        and abs(post_lag1) >= 0.20
    ):
        label = "physical_or_model_change"

    else:
        label = "ambiguous"

    return label, rms_ratio, post_lag1


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

        innovations.append(
            float(
                residual[0]
                / np.sqrt(S[0, 0])
            )
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

    pre = innovations[
        CHANGE_STEP - 300:
        CHANGE_STEP
    ]

    post = innovations[
        CHANGE_STEP:
        CHANGE_STEP + 300
    ]

    return classify(pre, post)


experiments = []

# Baseline
experiments.append({
    "type": "baseline",
    "value": None,
})

# Physical/model changes
for g_after in [
    0.16,
    0.14,
    0.12,
    0.10,
    0.08,
]:
    experiments.append({
        "type": "physical_change",
        "value": g_after,
    })

# Measurement-system changes
for sigma_after in [
    0.002,
    0.003,
    0.005,
    0.010,
]:
    experiments.append({
        "type": "measurement_change",
        "value": sigma_after,
    })


results = []

for exp_index, exp in enumerate(experiments):

    labels = []
    rms_ratios = []
    lag1_values = []

    for trial in range(TRIALS):

        seed = (
            20260824
            + exp_index * 10000
            + trial
        )

        if exp["type"] == "baseline":
            label, rms, lag1 = run_trial(seed)

        elif exp["type"] == "physical_change":
            label, rms, lag1 = run_trial(
                seed,
                physical_g_after=exp["value"],
            )

        elif exp["type"] == "measurement_change":
            label, rms, lag1 = run_trial(
                seed,
                measurement_sigma_after=exp["value"],
            )

        labels.append(label)
        rms_ratios.append(rms)
        lag1_values.append(lag1)

    counts = {
        label: labels.count(label)
        for label in sorted(set(labels))
    }

    results.append({
        "type": exp["type"],
        "value": exp["value"],
        "trials": TRIALS,

        "mean_rms_ratio": float(
            np.mean(rms_ratios)
        ),

        "std_rms_ratio": float(
            np.std(rms_ratios)
        ),

        "mean_post_lag1": float(
            np.mean(lag1_values)
        ),

        "std_post_lag1": float(
            np.std(lag1_values)
        ),

        "classification_counts": counts,
    })


evidence = {
    "experiment": "QMS-QT-004C",

    "title": (
        "Robustness of external-only twin "
        "failure attribution across perturbation magnitude"
    ),

    "classifier": (
        "Unchanged QT-004B residual RMS / "
        "lag-1 autocorrelation rule"
    ),

    "results": results,

    "scientific_boundary": (
        "Finite computational robustness test only. "
        "Thresholds remain provisional and are not "
        "claimed to generalize across physical platforms."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_004c_attribution_robustness.json"
)

out.write_text(
    json.dumps(evidence, indent=2)
    + "\n"
)

print(f"evidence: {out}")
print()
print("QMS-QT-004C ATTRIBUTION ROBUSTNESS")

for r in results:

    print()
    print(
        f"{r['type']:20s}",
        f"value={r['value']}",
    )

    print(
        "  RMS ratio:",
        f"{r['mean_rms_ratio']:.4f}",
        "±",
        f"{r['std_rms_ratio']:.4f}",
    )

    print(
        "  lag1:",
        f"{r['mean_post_lag1']:.4f}",
        "±",
        f"{r['std_post_lag1']:.4f}",
    )

    print(
        "  classifications:",
        r["classification_counts"],
    )
