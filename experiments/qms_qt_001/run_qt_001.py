import json
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from qms_core.quantum_twin import (
    build_two_mode_drift,
    default_drive_matrix,
    default_diffusion,
    propagate_gaussian_state,
    analyze_observability,
    gaussian_physicality,
)


ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Two-mode continuously driven baseline
# x = [x1, p1, x2, p2]^T
# ------------------------------------------------------------

omega1 = 1.00
omega2 = 1.20
gamma1 = 0.08
gamma2 = 0.06
coupling = 0.18

F = build_two_mode_drift(
    coupling,
    omega1=omega1,
    omega2=omega2,
    gamma1=gamma1,
    gamma2=gamma2,
)

B = default_drive_matrix()

D = default_diffusion(
    gamma1=gamma1,
    gamma2=gamma2,
)

mu0 = np.zeros(4)
V0 = 0.5 * np.eye(4)


def drive(t):
    return 0.35 * np.cos(
        0.95 * t
    )


measurement_configs = {
    "all_quadratures":
        np.eye(4),

    "x_each_mode":
        np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]),

    "mode1_only":
        np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]),

    "single_x1":
        np.array([
            [1.0, 0.0, 0.0, 0.0],
        ]),

    "sum_x_modes":
        np.array([
            [1.0, 0.0, 1.0, 0.0],
        ]),
}


t_eval = np.linspace(
    0.0,
    40.0,
    2001,
)

solution, mu_t, V_t = (
    propagate_gaussian_state(
        F,
        B,
        D,
        drive,
        mu0,
        V0,
        t_eval,
    )
)


config_results = {}

for name, C in (
    measurement_configs.items()
):

    diagnostics = (
        analyze_observability(
            F,
            C,
        )
    )

    y = (
        mu_t
        @ C.T
    )

    config_results[name] = {
        "measurement_dimension":
            int(
                C.shape[0]
            ),

        "observability_shape":
            [
                int(
                    C.shape[0]
                    * F.shape[0]
                ),
                int(
                    F.shape[0]
                ),
            ],

        "rank":
            diagnostics.rank,

        "nullity":
            diagnostics.nullity,

        "singular_values":
            [
                float(v)
                for v
                in diagnostics.singular_values
            ],

        "condition_number_nonzero":
            diagnostics.condition_number_nonzero,

        "null_space_basis":
            diagnostics.null_space_basis.tolist(),

        "measurement_start":
            y[0].tolist(),

        "measurement_end":
            y[-1].tolist(),
    }


physicality = [
    gaussian_physicality(V)
    for V in V_t
]

min_uncertainty_eigenvalue = min(
    p[
        "min_uncertainty_eigenvalue"
    ]
    for p in physicality
)

all_physical = all(
    p["physical"]
    for p in physicality
)


evidence = {
    "experiment":
        "QMS-QT-001",

    "status":
        "computational_baseline",

    "field_model": {
        "modes":
            2,

        "quadratures":
            [
                "x1",
                "p1",
                "x2",
                "p2",
            ],

        "state_dimension":
            4,

        "omega1":
            omega1,

        "omega2":
            omega2,

        "gamma1":
            gamma1,

        "gamma2":
            gamma2,

        "coupling":
            coupling,

        "drive_description":
            "0.35*cos(0.95*t) applied to p1",
    },

    "simulation": {
        "t_start":
            float(
                t_eval[0]
            ),

        "t_end":
            float(
                t_eval[-1]
            ),

        "samples":
            int(
                len(t_eval)
            ),

        "solver_success":
            bool(
                solution.success
            ),

        "initial_mean":
            mu0.tolist(),

        "final_mean":
            mu_t[-1].tolist(),
    },

    "gaussian_physicality": {
        "all_time_points_physical":
            all_physical,

        "minimum_uncertainty_eigenvalue":
            float(
                min_uncertainty_eigenvalue
            ),
    },

    "measurement_configurations":
        config_results,

    "scientific_boundary": (
        "Finite two-mode Gaussian computational model only; "
        "not experimental quantum-field tomography or a "
        "physical quantum twin."
    ),
}


out = (
    EVIDENCE_DIR
    / "qms_qt_001_baseline.json"
)

out.write_text(
    json.dumps(
        evidence,
        indent=2,
    )
    + "\n"
)


print(
    f"evidence: {out}"
)

print(
    f"solver_success: "
    f"{solution.success}"
)

print(
    f"all_covariances_physical: "
    f"{all_physical}"
)

print(
    "minimum_uncertainty_eigenvalue:",
    f"{min_uncertainty_eigenvalue:.6e}",
)

print()
print(
    "DYNAMICAL OBSERVABILITY"
)

for (
    name,
    result,
) in config_results.items():

    print(
        f"{name:20s}"
        f" rank={result['rank']}"
        f" nullity={result['nullity']}"
        f" cond="
        f"{result['condition_number_nonzero']:.6g}"
    )
