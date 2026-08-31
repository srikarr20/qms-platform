import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

DT = 0.05
WINDOW = 150
MEAS_SIGMA = 1e-3

BASE = {
    "omega1": 1.00,
    "omega2": 1.20,
    "gamma1": 0.08,
    "gamma2": 0.06,
    "g": 0.18,
}

TARGET_CHANGE = {
    "gamma1": 0.12,
}

# State at start of the diagnostic window.
# Fixed representative state for this controlled test.
X_START = np.array(
    [0.7, -0.2, 0.5, 0.45],
    dtype=float,
)

OBSERVABLES = {
    "x1": np.array([1., 0., 0., 0.]),
    "p1": np.array([0., 1., 0., 0.]),
    "x2": np.array([0., 0., 1., 0.]),
    "p2": np.array([0., 0., 0., 1.]),
}

CANDIDATE_ADDITIONS = [
    "p1",
    "x2",
    "p2",
]

GRIDS = {
    "g": np.linspace(0.08, 0.20, 31),
    "omega2": np.linspace(1.06, 1.26, 21),
    "gamma2": np.linspace(0.04, 0.10, 16),
}

MODEL_FAMILIES = {
    "g_only": ["g"],
    "omega2_only": ["omega2"],
    "gamma2_only": ["gamma2"],
    "g_gamma2": ["g", "gamma2"],
    "g_omega2": ["g", "omega2"],
    "omega2_gamma2": ["omega2", "gamma2"],
}


def build_F(params):
    w1 = params["omega1"]
    w2 = params["omega2"]
    g1 = params["gamma1"]
    g2 = params["gamma2"]
    g = params["g"]

    return np.array([
        [-g1,  w1, 0., 0.],
        [-w1, -g1, g,  0.],
        [0.,   0., -g2, w2],
        [g,    0., -w2, -g2],
    ])


def build_H(A, C):
    rows = []
    Ak = np.eye(4)

    for _ in range(WINDOW):
        Ak = A @ Ak
        rows.append(C @ Ak)

    return np.vstack(rows)


def candidates(family):
    params = MODEL_FAMILIES[family]

    if len(params) == 1:
        p = params[0]

        for v in GRIDS[p]:
            yield {p: float(v)}

    else:
        p1, p2 = params

        for v1 in GRIDS[p1]:
            for v2 in GRIDS[p2]:
                yield {
                    p1: float(v1),
                    p2: float(v2),
                }


results = []


for addition in CANDIDATE_ADDITIONS:

    C = np.vstack([
        OBSERVABLES["x1"],
        OBSERVABLES[addition],
    ])

    true_params = BASE.copy()
    true_params.update(TARGET_CHANGE)

    A_true = expm(
        build_F(true_params) * DT
    )

    H_true = build_H(A_true, C)

    y_true = H_true @ X_START

    family_scores = []

    for family in MODEL_FAMILIES:

        best_mse = np.inf
        best_params = None

        for change in candidates(family):

            params = BASE.copy()
            params.update(change)

            A_wrong = expm(
                build_F(params) * DT
            )

            H_wrong = build_H(
                A_wrong,
                C
            )

            # Wrong model is allowed to choose
            # its best possible initial state.
            x_wrong = (
                np.linalg.pinv(H_wrong)
                @ y_true
            )

            residual = (
                y_true
                - H_wrong @ x_wrong
            )

            mse = float(
                np.mean(residual**2)
            )

            if mse < best_mse:
                best_mse = mse
                best_params = change.copy()

        noise_normalized = (
            best_mse
            / MEAS_SIGMA**2
        )

        family_scores.append({
            "family":
                family,

            "best_mse":
                best_mse,

            "noise_normalized_mse":
                float(noise_normalized),

            "best_params":
                best_params,
        })

    # Conservative recommendation:
    # how distinguishable is the MOST
    # confusable wrong model?
    hardest = min(
        family_scores,
        key=lambda r:
            r["noise_normalized_mse"]
    )

    results.append({
        "addition":
            addition,

        "channels": [
            "x1",
            addition,
        ],

        "worst_case_noise_normalized_score":
            hardest[
                "noise_normalized_mse"
            ],

        "closest_wrong_family":
            hardest["family"],

        "closest_wrong_params":
            hardest["best_params"],

        "mean_noise_normalized_score":
            float(
                np.mean([
                    r[
                        "noise_normalized_mse"
                    ]
                    for r in family_scores
                ])
            ),

        "family_scores":
            family_scores,
    })


ranking = sorted(
    results,
    key=lambda r:
        r[
            "worst_case_noise_normalized_score"
        ],
    reverse=True,
)


evidence = {
    "experiment":
        "QMS-QT-006H",

    "title": (
        "Noise-aware measurement recommendation "
        "from finite-window model discriminability"
    ),

    "starting_channel":
        "x1",

    "target_change":
        TARGET_CHANGE,

    "measurement_noise_sigma":
        MEAS_SIGMA,

    "representative_start_state":
        X_START.tolist(),

    "ranking":
        ranking,

    "scientific_boundary": (
        "Finite deterministic computational "
        "measurement-design test. The score "
        "depends on the representative state, "
        "known target ambiguity, finite window, "
        "noise scale and candidate model library. "
        "It is not a universal measurement-value metric."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_006h_noise_aware_recommendation.json"
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
    "QMS-QT-006H NOISE-AWARE RECOMMENDATION"
)

for i, r in enumerate(ranking, 1):

    print()
    print(
        f"{i}. add {r['addition']}"
    )

    print(
        "   score:",
        f"{r['worst_case_noise_normalized_score']:.6f}"
    )

    print(
        "   mean score:",
        f"{r['mean_noise_normalized_score']:.6f}"
    )

    print(
        "   closest wrong family:",
        r["closest_wrong_family"]
    )

    print(
        "   closest wrong params:",
        r["closest_wrong_params"]
    )
