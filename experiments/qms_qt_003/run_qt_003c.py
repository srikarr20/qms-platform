import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)


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


def observability_matrix(Ad, C):
    n = Ad.shape[0]
    return np.vstack([
        C @ np.linalg.matrix_power(Ad, k)
        for k in range(n)
    ])


dt = 0.05
steps = 800
burn = steps // 2
trials = 100

C = np.array([[1.0, 0.0, 0.0, 0.0]])

couplings = [
    0.0,
    0.001,
    0.01,
    0.05,
    0.18,
]

measurement_noise_levels = [
    1e-4,
    1e-3,
    1e-2,
]

process_sigma = 1e-4

results = []

for g in couplings:

    F = build_F(g)
    Ad = expm(F * dt)

    O = observability_matrix(Ad, C)
    s = np.linalg.svd(O, compute_uv=False)

    tol = 1e-10 * s[0]
    rank = int(np.sum(s > tol))

    for meas_sigma in measurement_noise_levels:

        trial_total = []
        trial_mode1 = []
        trial_mode2 = []
        trial_cov = []

        for trial in range(trials):

            rng = np.random.default_rng(
                20260824 + trial
                + int(g * 1_000_000)
                + int(meas_sigma * 10_000_000)
            )

            Q = (process_sigma ** 2) * np.eye(4)
            R = np.array([[meas_sigma ** 2]])

            x_true = np.array(
                [0.7, -0.2, 0.5, 0.45],
                dtype=float
            )

            x_hat = np.zeros(4)
            P = np.eye(4)

            total_errors = []
            mode1_errors = []
            mode2_errors = []
            cov_traces = []

            for _ in range(steps):

                process_noise = rng.multivariate_normal(
                    np.zeros(4), Q
                )

                x_true = Ad @ x_true + process_noise

                y = (
                    C @ x_true
                    + rng.normal(scale=meas_sigma)
                )

                # Predict
                x_pred = Ad @ x_hat
                P_pred = Ad @ P @ Ad.T + Q

                # Update
                residual = y - C @ x_pred

                S = C @ P_pred @ C.T + R
                K = P_pred @ C.T @ np.linalg.inv(S)

                x_hat = (
                    x_pred
                    + (K @ residual).reshape(-1)
                )

                P = (
                    np.eye(4) - K @ C
                ) @ P_pred

                total_errors.append(
                    np.linalg.norm(x_hat - x_true)
                )

                mode1_errors.append(
                    np.linalg.norm(
                        x_hat[:2] - x_true[:2]
                    )
                )

                mode2_errors.append(
                    np.linalg.norm(
                        x_hat[2:] - x_true[2:]
                    )
                )

                cov_traces.append(np.trace(P))

            trial_total.append(
                float(np.mean(total_errors[burn:]))
            )

            trial_mode1.append(
                float(np.mean(mode1_errors[burn:]))
            )

            trial_mode2.append(
                float(np.mean(mode2_errors[burn:]))
            )

            trial_cov.append(
                float(np.mean(cov_traces[burn:]))
            )

        results.append({
            "coupling": g,
            "measurement_noise_sigma": meas_sigma,
            "rank": rank,
            "trials": trials,

            "mean_total_error": float(
                np.mean(trial_total)
            ),
            "std_total_error": float(
                np.std(trial_total)
            ),

            "mean_mode1_error": float(
                np.mean(trial_mode1)
            ),
            "std_mode1_error": float(
                np.std(trial_mode1)
            ),

            "mean_mode2_error": float(
                np.mean(trial_mode2)
            ),
            "std_mode2_error": float(
                np.std(trial_mode2)
            ),

            "mean_posterior_covariance_trace": float(
                np.mean(trial_cov)
            ),
        })


evidence = {
    "experiment": "QMS-QT-003C",
    "title": (
        "Monte Carlo validation of causal hidden-mode "
        "tracking in the virtual quantum twin"
    ),
    "external_measurement": "x1(t) only",
    "trials_per_condition": trials,
    "dt": dt,
    "steps": steps,
    "process_noise_sigma": process_sigma,
    "results": results,
    "scientific_boundary": (
        "Finite linear Gaussian computational twin only. "
        "No experimental quantum-field reconstruction is claimed."
    ),
}

out = (
    EVIDENCE_DIR
    / "qms_qt_003c_monte_carlo_twin.json"
)

out.write_text(
    json.dumps(evidence, indent=2) + "\n"
)

print(f"evidence: {out}")
print()
print("QMS-QT-003C MONTE CARLO TWIN")

for r in results:
    print(
        f"g={r['coupling']:6.3f} "
        f"noise={r['measurement_noise_sigma']:.4g} "
        f"rank={r['rank']} "
        f"mode1={r['mean_mode1_error']:.6e}"
        f"±{r['std_mode1_error']:.2e} "
        f"mode2={r['mean_mode2_error']:.6e}"
        f"±{r['std_mode2_error']:.2e} "
        f"total={r['mean_total_error']:.6e}"
    )
