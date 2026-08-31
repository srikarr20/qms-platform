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


def build_window_matrix(A, length):
    return np.vstack([
        C @ np.linalg.matrix_power(A, k)
        for k in range(1, length + 1)
    ])


def fit_candidate_grid(y):
    y = np.asarray(y, dtype=float)

    fits = []

    for g in G_CANDIDATES:
        A = expm(build_F(g) * DT)
        H = build_window_matrix(A, len(y))

        x_start = np.linalg.pinv(H) @ y
        y_hat = H @ x_start

        mse = float(
            np.mean((y - y_hat) ** 2)
        )

        fits.append({
            "g": float(g),
            "mse": mse,
            "x_start": x_start.tolist(),
        })

    best = min(fits, key=lambda r: r["mse"])

    return best, fits


def simulate_window(seed):
    rng = np.random.default_rng(seed)

    Q = (PROCESS_SIGMA ** 2) * np.eye(4)

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

        A = expm(build_F(g_true) * DT)

        x = (
            A @ x
            + rng.multivariate_normal(
                np.zeros(4), Q
            )
        )

        y = float(
            C @ x
            + rng.normal(scale=MEAS_SIGMA)
        )

        measurements.append(y)

    return measurements[
        CHANGE_STEP:ADAPT_STEP
    ]


trial_results = []

for trial in range(TRIALS):

    y = simulate_window(
        20260824 + trial
    )

    best, fits = fit_candidate_grid(y)

    mses = np.array([
        r["mse"]
        for r in fits
    ])

    sorted_idx = np.argsort(mses)

    best_mse = mses[sorted_idx[0]]
    second_mse = mses[sorted_idx[1]]

    # Width of near-optimal region:
    # candidate g values with MSE within 1% of best.
    threshold = best_mse * 1.01

    near_optimal = [
        r["g"]
        for r in fits
        if r["mse"] <= threshold
    ]

    trial_results.append({
        "trial": trial,

        "estimated_g":
            best["g"],

        "absolute_g_error":
            abs(
                best["g"]
                - TRUE_G_AFTER
            ),

        "best_mse":
            best_mse,

        "second_best_mse":
            second_mse,

        "relative_gap_second_best":
            float(
                (second_mse - best_mse)
                / best_mse
            ),

        "near_optimal_g_min":
            float(min(near_optimal)),

        "near_optimal_g_max":
            float(max(near_optimal)),

        "near_optimal_width":
            float(
                max(near_optimal)
                - min(near_optimal)
            ),

        "near_optimal_count":
            len(near_optimal),

        "fit_surface": [
            {
                "g": r["g"],
                "mse": r["mse"],
            }
            for r in fits
        ],
    })


estimated = np.array([
    r["estimated_g"]
    for r in trial_results
])

abs_errors = np.array([
    r["absolute_g_error"]
    for r in trial_results
])

widths = np.array([
    r["near_optimal_width"]
    for r in trial_results
])

g_counts = Counter(
    round(float(g), 6)
    for g in estimated
)


summary = {
    "true_g": TRUE_G_AFTER,

    "mean_estimated_g":
        float(np.mean(estimated)),

    "median_estimated_g":
        float(np.median(estimated)),

    "std_estimated_g":
        float(np.std(estimated)),

    "min_estimated_g":
        float(np.min(estimated)),

    "max_estimated_g":
        float(np.max(estimated)),

    "mean_absolute_g_error":
        float(np.mean(abs_errors)),

    "median_absolute_g_error":
        float(np.median(abs_errors)),

    "fraction_within_0.01":
        float(
            np.mean(
                abs_errors <= 0.01
            )
        ),

    "fraction_within_0.02":
        float(
            np.mean(
                abs_errors <= 0.02
            )
        ),

    "mean_near_optimal_width":
        float(np.mean(widths)),

    "median_near_optimal_width":
        float(np.median(widths)),

    "most_common_estimates": [
        {
            "g": g,
            "count": count,
        }
        for g, count
        in g_counts.most_common(10)
    ],
}


evidence = {
    "experiment": "QMS-QT-005C",

    "title": (
        "Joint parameter-state identifiability "
        "audit from a single external channel"
    ),

    "measurement":
        "x1(t) only",

    "window_samples":
        ADAPT_STEP - CHANGE_STEP,

    "candidate_grid": {
        "minimum":
            float(G_CANDIDATES.min()),

        "maximum":
            float(G_CANDIDATES.max()),

        "count":
            len(G_CANDIDATES),

        "spacing":
            float(
                G_CANDIDATES[1]
                - G_CANDIDATES[0]
            ),
    },

    "summary":
        summary,

    "trials":
        trial_results,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "identifiability audit. The test jointly "
        "fits coupling and initial state from one "
        "external measurement channel. Results "
        "characterize this model, window, noise "
        "level and candidate grid only."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_005c_joint_identifiability.json"
)

out.write_text(
    json.dumps(
        evidence,
        indent=2
    ) + "\n"
)


print(f"evidence: {out}")
print()
print("QMS-QT-005C JOINT IDENTIFIABILITY")

print(
    "true g:",
    TRUE_G_AFTER
)

print(
    "estimated g mean:",
    f"{summary['mean_estimated_g']:.6f}"
)

print(
    "estimated g median:",
    f"{summary['median_estimated_g']:.6f}"
)

print(
    "estimated g std:",
    f"{summary['std_estimated_g']:.6f}"
)

print(
    "estimated g range:",
    f"{summary['min_estimated_g']:.3f}",
    "to",
    f"{summary['max_estimated_g']:.3f}"
)

print(
    "mean |g error|:",
    f"{summary['mean_absolute_g_error']:.6f}"
)

print(
    "median |g error|:",
    f"{summary['median_absolute_g_error']:.6f}"
)

print(
    "within ±0.01:",
    f"{100 * summary['fraction_within_0.01']:.1f}%"
)

print(
    "within ±0.02:",
    f"{100 * summary['fraction_within_0.02']:.1f}%"
)

print(
    "mean near-optimal width:",
    f"{summary['mean_near_optimal_width']:.6f}"
)

print()
print("MOST COMMON ESTIMATES")

for r in summary["most_common_estimates"]:
    print(
        f"  g={r['g']:.3f}: "
        f"{r['count']} trials"
    )
