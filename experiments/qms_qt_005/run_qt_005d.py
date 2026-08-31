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
ADAPT_STEP = 750
TRIALS = 200

BASE_G = 0.18
TRUE_G_AFTER = 0.08

PROCESS_SIGMA = 1e-4
MEAS_SIGMA = 1e-3

G_CANDIDATES = np.linspace(0.04, 0.20, 81)

MEASUREMENT_CONFIGS = {
    "x1_only": np.array([
        [1.0, 0.0, 0.0, 0.0],
    ]),

    "x1_x2": np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]),
}


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
    ], dtype=float)


def build_window_matrix(A, C, length):
    rows = []

    for k in range(1, length + 1):
        rows.append(
            C @ np.linalg.matrix_power(A, k)
        )

    return np.vstack(rows)


def fit_candidate_grid(y, C):
    y = np.asarray(y, dtype=float).reshape(-1)

    fits = []

    for g in G_CANDIDATES:
        A = expm(build_F(g) * DT)

        H = build_window_matrix(
            A,
            C,
            len(y) // C.shape[0]
        )

        x_start = np.linalg.pinv(H) @ y
        y_hat = H @ x_start

        mse = float(
            np.mean((y - y_hat) ** 2)
        )

        fits.append({
            "g": float(g),
            "mse": mse,
        })

    best = min(
        fits,
        key=lambda r: r["mse"]
    )

    return best, fits


def simulate_window(seed, C):
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

    for k in range(ADAPT_STEP):

        g_true = (
            BASE_G
            if k < CHANGE_STEP
            else TRUE_G_AFTER
        )

        A = expm(
            build_F(g_true) * DT
        )

        x = (
            A @ x
            + rng.multivariate_normal(
                np.zeros(4), Q
            )
        )

        noise = rng.normal(
            scale=MEAS_SIGMA,
            size=C.shape[0]
        )

        y = C @ x + noise

        if k >= CHANGE_STEP:
            measurements.extend(
                y.tolist()
            )

    return measurements


all_results = {}


for config_name, C in MEASUREMENT_CONFIGS.items():

    trials = []

    for trial in range(TRIALS):

        y = simulate_window(
            20260824 + trial,
            C
        )

        best, fits = fit_candidate_grid(
            y,
            C
        )

        mses = np.array([
            r["mse"]
            for r in fits
        ])

        best_mse = float(np.min(mses))

        near_threshold = (
            best_mse * 1.01
        )

        near = [
            r["g"]
            for r in fits
            if r["mse"] <= near_threshold
        ]

        trials.append({
            "estimated_g":
                best["g"],

            "absolute_error":
                abs(
                    best["g"]
                    - TRUE_G_AFTER
                ),

            "near_optimal_width":
                float(
                    max(near) - min(near)
                ),
        })

    estimates = np.array([
        r["estimated_g"]
        for r in trials
    ])

    errors = np.array([
        r["absolute_error"]
        for r in trials
    ])

    widths = np.array([
        r["near_optimal_width"]
        for r in trials
    ])

    counts = Counter(
        round(float(g), 6)
        for g in estimates
    )

    all_results[config_name] = {
        "channels":
            int(C.shape[0]),

        "mean_estimated_g":
            float(np.mean(estimates)),

        "median_estimated_g":
            float(np.median(estimates)),

        "std_estimated_g":
            float(np.std(estimates)),

        "mean_absolute_error":
            float(np.mean(errors)),

        "median_absolute_error":
            float(np.median(errors)),

        "fraction_within_0.01":
            float(
                np.mean(errors <= 0.01)
            ),

        "fraction_within_0.02":
            float(
                np.mean(errors <= 0.02)
            ),

        "mean_near_optimal_width":
            float(np.mean(widths)),

        "most_common_estimates": [
            {
                "g": g,
                "count": count,
            }
            for g, count
            in counts.most_common(5)
        ],
    }


evidence = {
    "experiment": "QMS-QT-005D",

    "title": (
        "Measurement-architecture dependence "
        "of joint parameter-state identifiability"
    ),

    "true_g":
        TRUE_G_AFTER,

    "measurement_configurations": {
        name: C.tolist()
        for name, C
        in MEASUREMENT_CONFIGS.items()
    },

    "results":
        all_results,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "measurement-architecture comparison. "
        "The additional channel is an abstract "
        "external observable in this model and "
        "does not represent a demonstrated "
        "physical detector implementation."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_005d_measurement_architecture.json"
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
    "QMS-QT-005D MEASUREMENT ARCHITECTURE"
)

for name, r in all_results.items():

    print()
    print(name)

    print(
        "  channels:",
        r["channels"]
    )

    print(
        "  mean estimated g:",
        f"{r['mean_estimated_g']:.6f}"
    )

    print(
        "  std estimated g:",
        f"{r['std_estimated_g']:.6f}"
    )

    print(
        "  mean |g error|:",
        f"{r['mean_absolute_error']:.6f}"
    )

    print(
        "  within ±0.01:",
        f"{100 * r['fraction_within_0.01']:.1f}%"
    )

    print(
        "  within ±0.02:",
        f"{100 * r['fraction_within_0.02']:.1f}%"
    )

    print(
        "  near-optimal width:",
        f"{r['mean_near_optimal_width']:.6f}"
    )

    print(
        "  common estimates:",
        r["most_common_estimates"]
    )
