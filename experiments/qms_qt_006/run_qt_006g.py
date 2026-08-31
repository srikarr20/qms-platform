import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

DT = 0.05
WINDOW = 150

BASE = {
    "omega1": 1.00,
    "omega2": 1.20,
    "gamma1": 0.08,
    "gamma2": 0.06,
    "g": 0.18,
}

# The unresolved out-of-library mechanism
TARGET_CHANGE = {
    "gamma1": 0.12,
}

OBSERVABLES = {
    "x1": np.array([1.0, 0.0, 0.0, 0.0]),
    "p1": np.array([0.0, 1.0, 0.0, 0.0]),
    "x2": np.array([0.0, 0.0, 1.0, 0.0]),
    "p2": np.array([0.0, 0.0, 0.0, 1.0]),
}

# Start with x1 and ask which ONE observable
# should be added next.
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
        [-g1,  w1, 0.0, 0.0],
        [-w1, -g1, g,   0.0],
        [0.0,  0.0, -g2, w2],
        [g,    0.0, -w2, -g2],
    ], dtype=float)


def measurement_matrix(addition):
    return np.vstack([
        OBSERVABLES["x1"],
        OBSERVABLES[addition],
    ])


def build_window_matrix(A, C):
    rows = []
    Ak = np.eye(4)

    for _ in range(WINDOW):
        Ak = A @ Ak
        rows.append(C @ Ak)

    return np.vstack(rows)


def orthonormal_basis(H):
    u, s, _ = np.linalg.svd(
        H,
        full_matrices=False
    )

    tol = 1e-12 * s[0]
    rank = int(np.sum(s > tol))

    return u[:, :rank]


def candidate_parameter_sets(family):
    params = MODEL_FAMILIES[family]

    if len(params) == 1:
        p = params[0]

        for v in GRIDS[p]:
            yield {
                p: float(v)
            }

    else:
        p1, p2 = params

        for v1 in GRIDS[p1]:
            for v2 in GRIDS[p2]:
                yield {
                    p1: float(v1),
                    p2: float(v2),
                }


def subspace_mismatch(Q_true, Q_model):
    """
    Measures how much of the true trajectory
    subspace cannot be represented by the
    candidate model trajectory subspace.

    0 means complete subspace overlap.
    Larger means stronger geometric separation.
    """

    projection = (
        Q_model
        @ (Q_model.T @ Q_true)
    )

    residual = (
        Q_true - projection
    )

    return float(
        np.linalg.norm(
            residual,
            ord="fro"
        ) ** 2
        / Q_true.shape[1]
    )


results = []


for addition in CANDIDATE_ADDITIONS:

    C = measurement_matrix(addition)

    # -----------------------------------------
    # True changed dynamics
    # -----------------------------------------

    true_params = BASE.copy()
    true_params.update(TARGET_CHANGE)

    A_true = expm(
        build_F(true_params) * DT
    )

    H_true = build_window_matrix(
        A_true,
        C
    )

    Q_true = orthonormal_basis(
        H_true
    )

    family_results = []

    # -----------------------------------------
    # Find closest wrong model in each family
    # -----------------------------------------

    for family in MODEL_FAMILIES:

        best_mismatch = np.inf
        best_params = None

        for changes in candidate_parameter_sets(
            family
        ):

            params = BASE.copy()
            params.update(changes)

            A_model = expm(
                build_F(params) * DT
            )

            H_model = build_window_matrix(
                A_model,
                C
            )

            Q_model = orthonormal_basis(
                H_model
            )

            mismatch = subspace_mismatch(
                Q_true,
                Q_model
            )

            if mismatch < best_mismatch:
                best_mismatch = mismatch
                best_params = changes.copy()

        family_results.append({
            "family": family,
            "minimum_subspace_mismatch":
                float(best_mismatch),
            "closest_parameters":
                best_params,
        })

    # Hardest wrong model to distinguish
    closest_wrong = min(
        family_results,
        key=lambda r:
            r["minimum_subspace_mismatch"]
    )

    mismatch_values = np.array([
        r["minimum_subspace_mismatch"]
        for r in family_results
    ])

    results.append({
        "addition":
            addition,

        "measurement_channels": [
            "x1",
            addition,
        ],

        # Conservative score:
        # separation from the most confusable
        # known wrong model family.
        "worst_case_separation_score":
            float(
                closest_wrong[
                    "minimum_subspace_mismatch"
                ]
            ),

        "closest_wrong_family":
            closest_wrong["family"],

        "closest_wrong_parameters":
            closest_wrong[
                "closest_parameters"
            ],

        "mean_family_separation":
            float(
                np.mean(mismatch_values)
            ),

        "median_family_separation":
            float(
                np.median(mismatch_values)
            ),

        "family_results":
            family_results,
    })


ranking = sorted(
    results,
    key=lambda r:
        r["worst_case_separation_score"],
    reverse=True,
)


evidence = {
    "experiment":
        "QMS-QT-006G",

    "title": (
        "Geometry-based recommendation "
        "of an additional measurement observable"
    ),

    "starting_measurement": [
        "x1"
    ],

    "candidate_additions":
        CANDIDATE_ADDITIONS,

    "target_unresolved_mechanism":
        TARGET_CHANGE,

    "recommendation_metric": (
        "Worst-case trajectory-subspace "
        "separation between the target dynamics "
        "and the closest model in the existing "
        "candidate library."
    ),

    "ranking":
        ranking,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "measurement-design analysis. The target "
        "mechanism, candidate observable set and "
        "wrong-model library are specified in "
        "advance. The score is a geometric "
        "heuristic for this model and is not a "
        "universal measure of experimental "
        "information or quantum measurement value."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_006g_measurement_recommendation.json"
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
    "QMS-QT-006G MEASUREMENT RECOMMENDATION"
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
        "   channels:",
        r["measurement_channels"]
    )

    print(
        "   worst-case separation:",
        f"{r['worst_case_separation_score']:.6e}"
    )

    print(
        "   mean separation:",
        f"{r['mean_family_separation']:.6e}"
    )

    print(
        "   closest wrong family:",
        r["closest_wrong_family"]
    )

    print(
        "   closest wrong parameters:",
        r["closest_wrong_parameters"]
    )
