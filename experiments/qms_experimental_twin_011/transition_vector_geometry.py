from pathlib import Path
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.glasgow_event_adapter import GlasgowCumulativeArchive


GLASGOW_ZIP = Path(
    os.environ.get(
        "GLASGOW_ZIP",
        str(
            Path.home()
            / "Desktop"
            / "Quantum-Research"
            / "experimental-data"
            / "glasgow-single-photon"
            / "dpi_lab_1"
            / "Heralded Diffraction SM.zip"
        ),
    )
)

GRID = 32
WINDOW = 100

OUT = Path(
    "experiments/qms_experimental_twin_011/"
    "evidence/qms_experimental_twin_011_transition_geometry.json"
)

archive = GlasgowCumulativeArchive(GLASGOW_ZIP)

increments = {}

for record in archive.iter_increments():
    increments[int(record["index"])] = np.asarray(
        record["increment"],
        dtype=float,
    )


def aggregate(start, end):
    return np.sum(
        np.stack([
            increments[i]
            for i in range(start, end + 1)
        ]),
        axis=0,
    )


def distribution(frame):
    h, w = frame.shape

    bh = h // GRID
    bw = w // GRID

    reduced = frame.reshape(
        GRID, bh, GRID, bw
    ).sum(axis=(1, 3))

    p = reduced.ravel().astype(float)

    return p / p.sum()


def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    return float(
        np.dot(a, b) / (na * nb)
    )


# Three consecutive non-overlapping measurement states
A = distribution(
    aggregate(502, 601)
)

B = distribution(
    aggregate(602, 701)
)

C = distribution(
    aggregate(702, 801)
)


# Transition vectors
AB = B - A
BC = C - B

# Direct return vector
AC = C - A


transition_cosine = cosine(
    AB,
    BC
)

reversal_cosine = cosine(
    AB,
    -BC
)

ab_norm = float(
    np.linalg.norm(AB)
)

bc_norm = float(
    np.linalg.norm(BC)
)

ac_norm = float(
    np.linalg.norm(AC)
)


# How much of first transition is undone by second?
projection_of_bc_on_reverse_ab = float(
    np.dot(
        BC,
        -AB
    )
    /
    np.dot(
        AB,
        AB
    )
)


# State similarity A vs C
state_ac_cosine = cosine(
    A,
    C
)


# Fraction of components whose signs reverse
nz = (
    (np.abs(AB) > 1e-15)
    &
    (np.abs(BC) > 1e-15)
)

opposite_sign_fraction = float(
    np.mean(
        np.sign(AB[nz])
        ==
        -np.sign(BC[nz])
    )
)


# Magnitude-weighted sign reversal
weights = (
    np.abs(AB[nz])
    +
    np.abs(BC[nz])
)

weighted_reversal = float(
    np.sum(
        weights[
            np.sign(AB[nz])
            ==
            -np.sign(BC[nz])
        ]
    )
    /
    np.sum(weights)
)


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-011",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "states": {
        "A": [502, 601],
        "B": [602, 701],
        "C": [702, 801],
    },

    "transition_geometry": {
        "cosine_AB_BC":
            transition_cosine,

        "cosine_AB_negative_BC":
            reversal_cosine,

        "norm_AB":
            ab_norm,

        "norm_BC":
            bc_norm,

        "norm_A_to_C":
            ac_norm,

        "reverse_projection_fraction":
            projection_of_bc_on_reverse_ab,

        "opposite_sign_fraction":
            opposite_sign_fraction,

        "magnitude_weighted_reversal_fraction":
            weighted_reversal,

        "state_A_C_cosine":
            state_ac_cosine,
    },

    "scientific_boundary": (
        "This experiment tests whether two consecutive "
        "measurement-distribution transitions are geometrically "
        "opposed in observable detector space. A reversal does "
        "not identify a physical mechanism, detector fault, "
        "quantum-field transition, or causal process."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-011 ===")
print()

print(
    "cos(AB, BC):",
    f"{transition_cosine:.6f}"
)

print(
    "cos(AB, -BC):",
    f"{reversal_cosine:.6f}"
)

print()

print(
    "|AB|:",
    f"{ab_norm:.6f}"
)

print(
    "|BC|:",
    f"{bc_norm:.6f}"
)

print(
    "|AC|:",
    f"{ac_norm:.6f}"
)

print()

print(
    "Reverse projection fraction:",
    f"{projection_of_bc_on_reverse_ab:.6f}"
)

print(
    "Opposite-sign cells:",
    f"{100*opposite_sign_fraction:.2f}%"
)

print(
    "Magnitude-weighted reversal:",
    f"{100*weighted_reversal:.2f}%"
)

print()

print(
    "State cosine A vs C:",
    f"{state_ac_cosine:.6f}"
)

print()
print("Evidence:", OUT)
