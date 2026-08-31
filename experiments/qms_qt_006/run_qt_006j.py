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
IDENTIFY_STEP = 750
WINDOW = IDENTIFY_STEP - CHANGE_STEP

PROCESS_SIGMA = 1e-4
MEAS_SIGMA = 1e-3

MC_TRIALS = 100
BASELINE_TRIALS = 200

BASE = {
    "omega1": 1.00,
    "omega2": 1.20,
    "gamma1": 0.08,
    "gamma2": 0.06,
    "g": 0.18,
}

X_INITIAL = np.array(
    [0.7, -0.2, 0.5, 0.45],
    dtype=float
)

UNKNOWN_CASES = {
    "gamma1_change": {
        "gamma1": 0.12,
    },

    "omega1_change": {
        "omega1": 0.90,
    },

    "omega1_gamma1_change": {
        "omega1": 0.92,
        "gamma1": 0.11,
    },
}

OBSERVABLES = {
    "x1": np.array([1., 0., 0., 0.]),
    "p1": np.array([0., 1., 0., 0.]),
    "x2": np.array([0., 0., 1., 0.]),
    "p2": np.array([0., 0., 0., 1.]),
}

ADDITIONS = [
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
# Reachable state distribution at the change point
# ============================================================

A_base = expm(build_F(BASE) * DT)
Q_process = PROCESS_SIGMA**2 * np.eye(4)

mu = X_INITIAL.copy()
P_state = np.zeros((4, 4))

for _ in range(CHANGE_STEP):
    mu = A_base @ mu

    P_state = (
        A_base
        @ P_state
        @ A_base.T
        + Q_process
    )


# ============================================================
# State-aware recommendation
# ============================================================

def recommendation_score(target_change, addition):

    C = np.vstack([
        OBSERVABLES["x1"],
        OBSERVABLES[addition],
    ])

    true_params = BASE.copy()
    true_params.update(target_change)

    A_true = expm(
        build_F(true_params) * DT
    )

    H_true = build_H(A_true, C)

    family_best_scores = []

    for family in MODEL_FAMILIES:

        best = np.inf

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

            # Residual operator after allowing
            # the wrong model its best initial state.
            residual_operator = (
                H_true
                - H_wrong
                @ np.linalg.pinv(H_wrong)
                @ H_true
            )

            mean_term = float(
                np.linalg.norm(
                    residual_operator @ mu
                ) ** 2
            )

            covariance_term = float(
                np.trace(
                    residual_operator
                    @ P_state
                    @ residual_operator.T
                )
            )

            expected_mse = (
                mean_term
                + covariance_term
            ) / (
                WINDOW * C.shape[0]
            )

            score = (
                expected_mse
                / MEAS_SIGMA**2
            )

            if score < best:
                best = score

        family_best_scores.append(best)

    # Conservative score:
    # separation from hardest wrong family.
    return float(
        np.min(family_best_scores)
    )


# ============================================================
# Monte Carlo validation machinery
# ============================================================

def precompute_models(C):
    out = {}

    for family in MODEL_FAMILIES:
        models = []

        for change in candidates(family):

            params = BASE.copy()
            params.update(change)

            A = expm(
                build_F(params) * DT
            )

            H = build_H(A, C)

            models.append({
                "H": H,
                "H_pinv":
                    np.linalg.pinv(H),
            })

        out[family] = models

    return out


def simulate_window(C, target_change, seed):
    rng = np.random.default_rng(seed)

    x = X_INITIAL.copy()

    changed_params = BASE.copy()
    changed_params.update(target_change)

    A_changed = expm(
        build_F(changed_params) * DT
    )

    measurements = []

    for k in range(IDENTIFY_STEP):

        A = (
            A_base
            if k < CHANGE_STEP
            else A_changed
        )

        x = (
            A @ x
            + rng.multivariate_normal(
                np.zeros(4),
                Q_process
            )
        )

        y = (
            C @ x
            + rng.normal(
                scale=MEAS_SIGMA,
                size=C.shape[0]
            )
        )

        if k >= CHANGE_STEP:
            measurements.extend(
                y.tolist()
            )

    return np.asarray(
        measurements
    )


def fit_best_wrong_model(
    y,
    precomputed,
):
    best_mse = np.inf
    best_family = None

    for family, models in precomputed.items():

        family_best = np.inf

        for model in models:

            x_start = (
                model["H_pinv"] @ y
            )

            residual = (
                y
                - model["H"] @ x_start
            )

            mse = float(
                residual @ residual
                / len(residual)
            )

            if mse < family_best:
                family_best = mse

        if family_best < best_mse:
            best_mse = family_best
            best_family = family

    return best_family, best_mse


def calibrate_threshold(C, seed_offset):

    H_base = build_H(
        A_base,
        C
    )

    Hpinv = np.linalg.pinv(
        H_base
    )

    mses = []

    for trial in range(BASELINE_TRIALS):

        y = simulate_window(
            C,
            {},
            seed_offset + trial
        )

        x_start = Hpinv @ y

        residual = (
            y
            - H_base @ x_start
        )

        mses.append(
            float(
                residual @ residual
                / len(residual)
            )
        )

    return float(
        np.quantile(
            mses,
            0.99
        )
    )


# ============================================================
# Prospective recommendation + independent validation
# ============================================================

case_results = {}

for case_index, (
    case_name,
    target_change
) in enumerate(
    UNKNOWN_CASES.items()
):

    print()
    print(
        "CASE:",
        case_name,
        target_change
    )

    # ----------------------------------------
    # Recommendation stage
    # ----------------------------------------

    recommendation_scores = {}

    for addition in ADDITIONS:

        score = recommendation_score(
            target_change,
            addition
        )

        recommendation_scores[
            addition
        ] = score

    predicted_ranking = sorted(
        ADDITIONS,
        key=lambda a:
            recommendation_scores[a],
        reverse=True,
    )

    print(
        "  predicted ranking:",
        predicted_ranking
    )

    # ----------------------------------------
    # Independent MC validation
    # ----------------------------------------

    mc_results = {}

    for addition_index, addition in enumerate(
        ADDITIONS
    ):

        C = np.vstack([
            OBSERVABLES["x1"],
            OBSERVABLES[addition],
        ])

        print(
            "  validating:",
            addition
        )

        precomputed = precompute_models(C)

        threshold = calibrate_threshold(
            C,
            seed_offset=(
                11000000
                + case_index * 100000
                + addition_index * 10000
            )
        )

        rejected = []
        winners = []

        for trial in range(MC_TRIALS):

            y = simulate_window(
                C,
                target_change,
                seed=(
                    22000000
                    + case_index * 100000
                    + addition_index * 10000
                    + trial
                )
            )

            winner, best_mse = (
                fit_best_wrong_model(
                    y,
                    precomputed
                )
            )

            rejected.append(
                best_mse > threshold
            )

            winners.append(winner)

        mc_results[addition] = {
            "rejection_rate":
                float(
                    np.mean(rejected)
                ),

            "adequacy_threshold":
                threshold,

            "forced_winner_counts":
                dict(
                    Counter(winners)
                ),
        }

    empirical_ranking = sorted(
        ADDITIONS,
        key=lambda a:
            mc_results[a][
                "rejection_rate"
            ],
        reverse=True,
    )

    predicted_best = (
        predicted_ranking[0]
    )

    empirical_best = (
        empirical_ranking[0]
    )

    case_results[case_name] = {
        "target_change":
            target_change,

        "recommendation_scores":
            recommendation_scores,

        "predicted_ranking":
            predicted_ranking,

        "mc_results":
            mc_results,

        "empirical_ranking":
            empirical_ranking,

        "top_recommendation_correct":
            bool(
                predicted_best
                == empirical_best
            ),
    }


top_correct = [
    r["top_recommendation_correct"]
    for r in case_results.values()
]


evidence = {
    "experiment":
        "QMS-QT-006J",

    "title": (
        "Prospective validation of "
        "state-aware measurement recommendations"
    ),

    "unknown_cases":
        UNKNOWN_CASES,

    "starting_channel":
        "x1",

    "candidate_additions":
        ADDITIONS,

    "state_distribution_at_change": {
        "mean":
            mu.tolist(),

        "mean_norm":
            float(
                np.linalg.norm(mu)
            ),

        "covariance_trace":
            float(
                np.trace(P_state)
            ),
    },

    "results":
        case_results,

    "summary": {
        "cases":
            len(case_results),

        "top_recommendations_correct":
            int(
                np.sum(top_correct)
            ),

        "top_recommendation_accuracy":
            float(
                np.mean(top_correct)
            ),
    },

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "prospective validation. Recommendations "
        "use known unresolved mechanisms, known "
        "candidate observable choices, a fixed "
        "candidate wrong-model library and the "
        "nominal reachable state distribution. "
        "Monte Carlo rejection rates provide "
        "independent computational validation only."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_006j_recommender_validation.json"
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
    "QMS-QT-006J RECOMMENDER VALIDATION"
)

for case, r in case_results.items():

    print()
    print(case)

    print(
        "  predicted ranking:",
        r["predicted_ranking"]
    )

    print(
        "  empirical ranking:",
        r["empirical_ranking"]
    )

    for addition in ADDITIONS:

        print(
            f"    {addition}: "
            f"score="
            f"{r['recommendation_scores'][addition]:.6e} "
            f"rejection="
            f"{100 * r['mc_results'][addition]['rejection_rate']:.1f}%"
        )

    print(
        "  top recommendation correct:",
        r[
            "top_recommendation_correct"
        ]
    )

print()
print(
    "TOP RECOMMENDATION ACCURACY:",
    f"{100 * evidence['summary']['top_recommendation_accuracy']:.1f}%"
)
