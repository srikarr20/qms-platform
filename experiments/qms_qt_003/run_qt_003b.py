import json
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.linalg import expm


ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(20260824)


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

C = np.array([[1.0, 0.0, 0.0, 0.0]])

couplings = [0.0, 0.001, 0.01, 0.05, 0.18]
measurement_noise_levels = [1e-4, 1e-3, 1e-2]

process_sigma = 1e-4

results = []

for g in couplings:
    F = build_F(g)
    Ad = expm(F * dt)

    O = observability_matrix(Ad, C)
    s = np.linalg.svd(O, compute_uv=False)
    tol = 1e-10 * s[0]
    rank = int(np.sum(s > tol))
    nonzero = s[s > tol]
    cond = float(nonzero.max() / nonzero.min())

    for meas_sigma in measurement_noise_levels:

        Q = (process_sigma ** 2) * np.eye(4)
        R = np.array([[meas_sigma ** 2]])

        x_true = np.array([0.7, -0.2, 0.5, 0.45], dtype=float)

        # Twin starts with deliberately imperfect knowledge.
        x_hat = np.zeros(4)
        P = np.eye(4)

        errors = []
        residuals = []
        covariance_traces = []

        mode1_errors = []
        mode2_errors = []

        for _ in range(steps):

            # ----- physical environment -----
            process_noise = rng.multivariate_normal(
                np.zeros(4), Q
            )
            x_true = Ad @ x_true + process_noise

            measurement_noise = rng.normal(scale=meas_sigma)
            y = C @ x_true + measurement_noise

            # ----- twin prediction -----
            x_pred = Ad @ x_hat
            P_pred = Ad @ P @ Ad.T + Q

            # ----- predicted external measurement -----
            y_pred = C @ x_pred
            residual = y - y_pred

            # ----- causal measurement update -----
            S = C @ P_pred @ C.T + R
            K = P_pred @ C.T @ np.linalg.inv(S)

            x_hat = x_pred + (K @ residual).reshape(-1)

            P = (np.eye(4) - K @ C) @ P_pred

            err = np.linalg.norm(x_hat - x_true)

            errors.append(float(err))
            residuals.append(float(residual[0]))
            covariance_traces.append(float(np.trace(P)))

            mode1_errors.append(
                float(np.linalg.norm(x_hat[:2] - x_true[:2]))
            )
            mode2_errors.append(
                float(np.linalg.norm(x_hat[2:] - x_true[2:]))
            )

        burn = steps // 2

        steady_errors = np.array(errors[burn:])
        steady_mode1 = np.array(mode1_errors[burn:])
        steady_mode2 = np.array(mode2_errors[burn:])
        steady_residuals = np.array(residuals[burn:])
        steady_cov = np.array(covariance_traces[burn:])

        results.append({
            "coupling": g,
            "measurement_noise_sigma": meas_sigma,
            "process_noise_sigma": process_sigma,
            "rank": rank,
            "observability_condition_number": cond,
            "mean_steady_state_error": float(
                np.mean(steady_errors)
            ),
            "mean_mode1_error": float(
                np.mean(steady_mode1)
            ),
            "mean_mode2_error": float(
                np.mean(steady_mode2)
            ),
            "mean_abs_residual": float(
                np.mean(np.abs(steady_residuals))
            ),
            "mean_posterior_covariance_trace": float(
                np.mean(steady_cov)
            ),
            "final_true_state": x_true.tolist(),
            "final_estimated_state": x_hat.tolist(),
        })


evidence = {
    "experiment": "QMS-QT-003B",
    "title": "Causal real-time two-mode virtual quantum twin baseline",
    "state": ["x1", "p1", "x2", "p2"],
    "external_measurement": "x1(t) only",
    "estimator": (
        "Discrete linear Gaussian causal state estimator "
        "(Kalman-style computational baseline)"
    ),
    "dt": dt,
    "steps": steps,
    "results": results,
    "scientific_boundary": (
        "Computational finite-mode Gaussian state-estimation baseline. "
        "This is not an experimental quantum filter and does not establish "
        "real-time reconstruction of a physical quantum electromagnetic field."
    ),
}

out = EVIDENCE_DIR / "qms_qt_003b_realtime_twin.json"
out.write_text(json.dumps(evidence, indent=2) + "\n")

print(f"evidence: {out}")
print()
print("QMS-QT-003B REAL-TIME TWIN")

for r in results:
    print(
        f"g={r['coupling']:6.3f} "
        f"noise={r['measurement_noise_sigma']:.4g} "
        f"rank={r['rank']} "
        f"cond={r['observability_condition_number']:.3g} "
        f"err={r['mean_steady_state_error']:.6e} "
        f"mode1={r['mean_mode1_error']:.6e} "
        f"mode2={r['mean_mode2_error']:.6e} "
        f"P={r['mean_posterior_covariance_trace']:.6e}"
    )
