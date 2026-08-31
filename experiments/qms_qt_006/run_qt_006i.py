import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

DT = 0.05
CHANGE_STEP = 600
WINDOW = 150

PROCESS_SIGMA = 1e-4
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

X_INITIAL = np.array([
    0.7, -0.2, 0.5, 0.45
], dtype=float)

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
    ], dtype=float)


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


# ============================================================
# REACHABLE STATE DISTRIBUTION AT CHANGE POINT
# ============================================================

A_base = expm(
    build_F(BASE) * DT
)

Q = PROCESS_SIGMA**2 * np.eye(4)

mu = X_INITIAL.copy()
P = np.zeros((4, 4))

for _ in range(CHANGE_STEP):

    mu = A_base @ mu

    P = (
        A_base @ P @ A_base.T
        + Q
    )


print("STATE DISTRIBUTION AT CHANGE")
print(
    "  mean norm:",
    f"{np.linalg.norm(mu):.6e}"
)
print(
    "  covariance trace:",
    f"{np.trace(P):.6e}"
)


# ============================================================
# STATE-DISTRIBUTION-AWARE MODEL SEPARATION
# ============================================================

results = []

true_params = BASE.copy()
true_params.update(TARGET_CHANGE)

A_true = expm(
    build_F(true_params) * DT
)


for addition in CANDIDATE_ADDITIONS:

    C = np.vstack([
        OBSERVABLES["x1"],
        OBSERVABLES[addition],
    ])

    H_true = build_H(
        A_true,
        C
    )

    family_scores = []

    for family in MODEL_FAMILIES:

        best_score = np.inf
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

            # Orthogonal projector onto trajectories
            # representable by the wrong model.
            Hpinv = np.linalg.pinv(H_wrong)

            residual_operator = (
                H_true
                - H_wrong @ Hpinv @ H_true
            )

            # Expected mismatch for x ~ N(mu, P):
            #
            # E ||R x||² =
            # ||R mu||² + tr(R P R^T)
            mean_term = float(
                np.linalg.norm(
                    residual_operator @ mu
                ) ** 2
            )

            covariance_term = float(
                np.trace(
                    residual_operator
                    @ P
                    @ residual_operator.T
                )
            )

            expected_energy = (
                mean_term
                + covariance_term
            )

            number_measurements = (
                WINDOW * C.shape[0]
            )

            expected_mse = (
                expected_energy
                / number_measurements
            )

            score = (
                expected_mse
                / MEAS_SIGMA**2
            )

            if score < best_score:
                best_score = score
                best_params = change.copy()

        family_scores.append({
            "family": family,
            "expected_noise_normalized_mismatch":
                float(best_score),
            "closest_parameters":
                best_params,
        })

    hardest = min(
        family_scores,
        key=lambda r:
            r[
                "expected_noise_normalized_mismatch"
            ]
    )

    scores = np.array([
        r[
            "expected_noise_normalized_mismatch"
        ]
        for r in family_scores
    ])

    results.append({
        "addition":
            addition,

        "measurement_channels": [
            "x1",
            addition,
        ],

        "worst_case_expected_score":
            float(
                hardest[
                    "expected_noise_normalized_mismatch"
                ]
            ),

        "mean_expected_score":
            float(np.mean(scores)),

        "closest_wrong_family":
            hardest["family"],

        "closest_wrong_parameters":
            hardest[
                "closest_parameters"
            ],

        "family_scores":
            family_scores,
    })


ranking = sorted(
    results,
    key=lambda r:
        r["worst_case_expected_score"],
    reverse=True,
)


evidence = {
    "experiment":
        "QMS-QT-006I",

    "title": (
        "State-distribution-aware recommendation "
        "of additional measurement observables"
    ),

    "starting_channel":
        "x1",

    "candidate_additions":
        CANDIDATE_ADDITIONS,

    "target_change":
        TARGET_CHANGE,

    "state_distribution_at_change": {
        "mean":
            mu.tolist(),

        "mean_norm":
            float(
                np.linalg.norm(mu)
            ),

        "covariance":
            P.tolist(),

        "covariance_trace":
            float(np.trace(P)),
    },

    "ranking":
        ranking,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "measurement-design analysis. The score "
        "averages deterministic model mismatch "
        "over the nominal state distribution at "
        "the change point. It does not use the "
        "QT-006F rejection rates as an optimization "
        "target and is not a universal measurement "
        "utility metric."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_006i_state_aware_recommendation.json"
)

out.write_text(
    json.dumps(
        evidence,
        indent=2
    ) + "\n"
)


print()
print(f"evidence: {out}")
print()
print(
    "QMS-QT-006I STATE-AWARE RECOMMENDATION"
)

for i, r in enumerate(
    ranking,
    start=1
):

    print()
    print(
        f"{i}. add {r['addition']}"
    )

    print(
        "   expected score:",
        f"{r['worst_case_expected_score']:.6e}"
    )

    print(
        "   mean score:",
        f"{r['mean_expected_score']:.6e}"
    )

    print(
        "   closest wrong family:",
        r["closest_wrong_family"]
    )

    print(
        "   closest wrong parameters:",
        r["closest_wrong_parameters"]
    )
