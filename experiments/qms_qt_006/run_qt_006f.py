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

TRIALS = 200
BASELINE_TRIALS = 300

PROCESS_SIGMA = 1e-4
MEAS_SIGMA = 1e-3

BASE = {
    "omega1": 1.00,
    "omega2": 1.20,
    "gamma1": 0.08,
    "gamma2": 0.06,
    "g": 0.18,
}

UNKNOWN_CHANGE = {
    "gamma1": 0.12,
}

MEASUREMENT_CONFIGS = {
    "x1_only": np.array([
        [1.0, 0.0, 0.0, 0.0],
    ]),

    "x1_x2": np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]),

    "x1_p1": np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]),

    "x1_p2": np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]),

    "x1_x2_p1": np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]),

    "x1_x2_p2": np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]),

    "all_quadratures": np.eye(4),
}

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


def build_window_matrix(A, C):
    rows = []
    Ak = np.eye(4)

    for _ in range(WINDOW):
        Ak = A @ Ak
        rows.append(C @ Ak)

    return np.vstack(rows)


def candidate_parameter_sets(family):
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


def simulate_window(C, changed, seed):
    rng = np.random.default_rng(seed)

    Q = PROCESS_SIGMA**2 * np.eye(4)

    x = np.array(
        [0.7, -0.2, 0.5, 0.45],
        dtype=float
    )

    A_base = expm(build_F(BASE) * DT)

    changed_params = BASE.copy()
    changed_params.update(UNKNOWN_CHANGE)

    A_changed = expm(
        build_F(changed_params) * DT
    )

    measurements = []

    for k in range(IDENTIFY_STEP):

        A = (
            A_changed
            if changed and k >= CHANGE_STEP
            else A_base
        )

        x = (
            A @ x
            + rng.multivariate_normal(
                np.zeros(4), Q
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
            measurements.extend(y.tolist())

    return np.asarray(measurements)


def precompute_models(C):
    result = {}

    for family in MODEL_FAMILIES:
        models = []

        for changes in candidate_parameter_sets(family):

            params = BASE.copy()
            params.update(changes)

            A = expm(
                build_F(params) * DT
            )

            H = build_window_matrix(A, C)

            models.append({
                "params": changes,
                "H": H,
                "H_pinv": np.linalg.pinv(H),
            })

        result[family] = models

    return result


def fit_family(y, family, precomputed):
    best_mse = np.inf
    best_params = None

    for model in precomputed[family]:

        x_start = model["H_pinv"] @ y
        residual = (
            y - model["H"] @ x_start
        )

        mse = float(
            residual @ residual
            / len(residual)
        )

        if mse < best_mse:
            best_mse = mse
            best_params = model["params"]

    return {
        "family": family,
        "mse": best_mse,
        "params": best_params,
    }


results = {}


for config_index, (config_name, C) in enumerate(
    MEASUREMENT_CONFIGS.items()
):

    print()
    print(
        "Precomputing:",
        config_name,
        f"({C.shape[0]} channels)"
    )

    precomputed = precompute_models(C)

    # ----------------------------------------
    # Nominal adequacy calibration
    # ----------------------------------------

    A_base = expm(
        build_F(BASE) * DT
    )

    H_base = build_window_matrix(
        A_base,
        C
    )

    H_base_pinv = np.linalg.pinv(
        H_base
    )

    baseline_mses = []

    for trial in range(BASELINE_TRIALS):

        y = simulate_window(
            C,
            changed=False,
            seed=(
                17000000
                + config_index * 10000
                + trial
            )
        )

        x_start = H_base_pinv @ y
        residual = (
            y - H_base @ x_start
        )

        baseline_mses.append(
            float(
                residual @ residual
                / len(residual)
            )
        )

    threshold = float(
        np.quantile(
            baseline_mses,
            0.99
        )
    )

    # ----------------------------------------
    # Unknown-mechanism rejection
    # ----------------------------------------

    rejected = []
    winners = []
    ratios = []
    margins = []

    for trial in range(TRIALS):

        y = simulate_window(
            C,
            changed=True,
            seed=(
                20260824
                + config_index * 10000
                + trial
            )
        )

        fits = [
            fit_family(
                y,
                family,
                precomputed
            )
            for family in MODEL_FAMILIES
        ]

        fits.sort(
            key=lambda r: r["mse"]
        )

        winner = fits[0]
        runner = fits[1]

        rejected.append(
            winner["mse"] > threshold
        )

        winners.append(
            winner["family"]
        )

        ratios.append(
            winner["mse"] / threshold
        )

        margins.append(
            (
                runner["mse"]
                - winner["mse"]
            )
            / winner["mse"]
        )

    results[config_name] = {
        "channels":
            int(C.shape[0]),

        "measurement_matrix":
            C.tolist(),

        "adequacy_threshold":
            threshold,

        "unknown_rejection_rate":
            float(
                np.mean(rejected)
            ),

        "forced_prediction_counts":
            dict(
                Counter(winners)
            ),

        "mean_best_mse_over_threshold":
            float(
                np.mean(ratios)
            ),

        "median_best_mse_over_threshold":
            float(
                np.median(ratios)
            ),

        "mean_winner_margin":
            float(
                np.mean(margins)
            ),
    }


# Rank by rejection capability
ranking = sorted(
    [
        {
            "configuration": name,
            **r
        }
        for name, r in results.items()
    ],
    key=lambda r:
        r["unknown_rejection_rate"],
    reverse=True,
)


evidence = {
    "experiment":
        "QMS-QT-006F",

    "title": (
        "External observable information ranking "
        "for unknown-mechanism discrimination"
    ),

    "unknown_change":
        UNKNOWN_CHANGE,

    "candidate_library":
        MODEL_FAMILIES,

    "results":
        results,

    "ranking":
        ranking,

    "scientific_boundary": (
        "Finite linear Gaussian computational "
        "measurement-channel ranking. Results "
        "apply only to the tested gamma1 change, "
        "candidate model library, noise level, "
        "observation window and abstract "
        "quadrature measurement configurations."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_006f_measurement_information_ranking.json"
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
    "QMS-QT-006F MEASUREMENT INFORMATION RANKING"
)

for i, r in enumerate(ranking, start=1):

    print()
    print(
        f"{i}. {r['configuration']}"
    )

    print(
        "   channels:",
        r["channels"]
    )

    print(
        "   unknown rejection:",
        f"{100 * r['unknown_rejection_rate']:.1f}%"
    )

    print(
        "   best MSE / threshold:",
        f"{r['mean_best_mse_over_threshold']:.3f}"
    )

    print(
        "   forced winners:",
        r["forced_prediction_counts"]
    )

    print(
        "   winner margin:",
        f"{r['mean_winner_margin']:.4f}"
    )
