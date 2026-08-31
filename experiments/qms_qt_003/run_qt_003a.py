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


def build_discrete_observation_matrix(F, C, times):
    rows = []
    for t in times:
        Phi = expm(F * t)
        rows.append(C @ Phi)
    return np.vstack(rows)


def reconstruct_state(H, y):
    xhat = np.linalg.pinv(H) @ y
    return xhat


rng = np.random.default_rng(20260824)

C = np.array([[1.0, 0.0, 0.0, 0.0]])

couplings = [
    0.0,
    0.001,
    0.005,
    0.01,
    0.05,
    0.10,
    0.18,
]

noise_levels = [
    0.0,
    1e-4,
    1e-3,
    1e-2,
]

times = np.linspace(0.0, 8.0, 81)

trials = 200

all_results = []

for g in couplings:
    F = build_F(g)
    H = build_discrete_observation_matrix(F, C, times)

    s = np.linalg.svd(H, compute_uv=False)
    tol = 1e-10 * s[0]
    rank = int(np.sum(s > tol))

    nonzero = s[s > tol]
    cond = (
        float(nonzero.max() / nonzero.min())
        if len(nonzero)
        else None
    )

    nullity = int(4 - rank)

    for sigma in noise_levels:
        errors = []
        null_overlaps = []

        for _ in range(trials):
            x0 = rng.normal(size=4)
            x0 = x0 / np.linalg.norm(x0)

            y_clean = H @ x0
            noise = rng.normal(scale=sigma, size=y_clean.shape)
            y = y_clean + noise

            xhat = reconstruct_state(H, y)

            err = np.linalg.norm(xhat - x0)
            errors.append(float(err))

            # Projection onto null space of H
            u, sv, vh = np.linalg.svd(H, full_matrices=True)
            r = int(np.sum(sv > 1e-10 * sv[0]))
            null_basis = vh[r:].T

            if null_basis.size == 0:
                null_overlap = 0.0
            else:
                proj = null_basis @ (null_basis.T @ x0)
                null_overlap = float(np.linalg.norm(proj))

            null_overlaps.append(null_overlap)

        result = {
            "coupling": g,
            "noise_sigma": sigma,
            "rank": rank,
            "nullity": nullity,
            "condition_number_nonzero": cond,
            "sigma_min": float(s.min()),
            "mean_reconstruction_error": float(np.mean(errors)),
            "std_reconstruction_error": float(np.std(errors)),
            "max_reconstruction_error": float(np.max(errors)),
            "mean_null_overlap": float(np.mean(null_overlaps)),
        }

        all_results.append(result)


evidence = {
    "experiment": "QMS-QT-003A",
    "title": "Hidden field-mode reconstruction from a single external channel",
    "state": ["x1", "p1", "x2", "p2"],
    "measurement": "x1(t) only",
    "time_window": {
        "start": float(times[0]),
        "end": float(times[-1]),
        "samples": int(len(times)),
    },
    "trials_per_condition": trials,
    "results": all_results,
    "scientific_boundary": (
        "Finite linear two-mode computational state-estimation test only; "
        "not experimental quantum-field reconstruction."
    ),
}

out = EVIDENCE_DIR / "qms_qt_003a_state_reconstruction.json"
out.write_text(json.dumps(evidence, indent=2) + "\n")

print(f"evidence: {out}")
print()
print("QMS-QT-003A STATE RECONSTRUCTION")

for r in all_results:
    print(
        f"g={r['coupling']:7.3f} "
        f"noise={r['noise_sigma']:.4g} "
        f"rank={r['rank']} "
        f"nullity={r['nullity']} "
        f"cond={r['condition_number_nonzero']:.6g} "
        f"mean_err={r['mean_reconstruction_error']:.6e} "
        f"null={r['mean_null_overlap']:.6e}"
    )
