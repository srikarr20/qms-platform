import json
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)


def observability_matrix(F, C):
    n = F.shape[0]
    return np.vstack([
        C @ np.linalg.matrix_power(F, k)
        for k in range(n)
    ])


def analyze(F, C):
    O = observability_matrix(F, C)

    u, s, vh = np.linalg.svd(O, full_matrices=True)

    tol = 1e-10 * s[0]
    rank = int(np.sum(s > tol))
    nullity = int(F.shape[0] - rank)

    nonzero = s[s > tol]

    cond = (
        float(nonzero.max() / nonzero.min())
        if len(nonzero)
        else None
    )

    null_basis = vh[rank:].T

    return {
        "rank": rank,
        "nullity": nullity,
        "singular_values": [float(v) for v in s],
        "condition_number_nonzero": cond,
        "null_space_basis": null_basis.tolist(),
    }


omega1 = 1.00
omega2 = 1.20
gamma1 = 0.08
gamma2 = 0.06

C = np.array([
    [1.0, 0.0, 0.0, 0.0]
])

couplings = [
    0.0,
    0.001,
    0.005,
    0.01,
    0.05,
    0.10,
    0.18,
]

results = []

for g in couplings:

    F = np.array([
        [-gamma1,  omega1, 0.0,   0.0],
        [-omega1, -gamma1, g,     0.0],
        [0.0,       0.0, -gamma2, omega2],
        [g,          0.0, -omega2, -gamma2],
    ])

    r = analyze(F, C)

    results.append({
        "coupling": g,
        **r,
    })


evidence = {
    "experiment": "QMS-QT-002",
    "title": "Field-mode observability under dynamical coupling",
    "measurement": "single external x1 quadrature",
    "state": ["x1", "p1", "x2", "p2"],
    "results": results,
    "scientific_boundary": (
        "Finite two-mode linear Gaussian computational model only."
    ),
}


out = EVIDENCE_DIR / "qms_qt_002_coupling_observability.json"
out.write_text(json.dumps(evidence, indent=2) + "\n")


print(f"evidence: {out}")
print()
print("COUPLING OBSERVABILITY SWEEP")

for r in results:
    smallest = min(r["singular_values"])
    print(
        f"g={r['coupling']:7.3f} "
        f"rank={r['rank']} "
        f"nullity={r['nullity']} "
        f"cond={r['condition_number_nonzero']:.6g} "
        f"sigma_min={smallest:.6e}"
    )
