from pathlib import Path
import json
import numpy as np


SRC = Path(
    "experiments/qms_experimental_twin_013/"
    "evidence/qms_experimental_twin_013_innovation_geometry.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_016/"
    "evidence/qms_experimental_twin_016_conformal_surprise.json"
)

data = json.loads(SRC.read_text())
rows = data["results"]

windows = np.asarray(
    [int(r["window_end"]) for r in rows],
    dtype=int,
)

x = np.asarray(
    [float(r["transition_l2"]) for r in rows],
    dtype=float,
)

y = np.asarray(
    [float(r["innovation_norm"]) for r in rows],
    dtype=float,
)

N = len(rows)
K = min(10, N - 1)


# ------------------------------------------------------------
# Leave-one-out local predictor
#
# For each transition, predict innovation using nearest
# transition-magnitude neighbours while excluding itself.
# ------------------------------------------------------------

expected = np.zeros(N)
residual = np.zeros(N)
ratio = np.zeros(N)


for i in range(N):

    distance = np.abs(
        x - x[i]
    )

    candidates = [
        j for j in np.argsort(distance)
        if j != i
    ]

    neighbors = candidates[:K]

    expected[i] = np.median(
        y[neighbors]
    )

    residual[i] = (
        y[i] - expected[i]
    )

    if expected[i] > 0:
        ratio[i] = (
            y[i] / expected[i]
        )
    else:
        ratio[i] = np.inf


# ------------------------------------------------------------
# Conformal-style empirical p-values
#
# A large positive residual means the observation was more
# surprising than expected for transitions of similar size.
# ------------------------------------------------------------

p_values = np.zeros(N)

for i in range(N):

    p_values[i] = (
        1
        +
        np.sum(
            residual >= residual[i]
        )
    ) / (N + 1)


# ------------------------------------------------------------
# Exact sequence ranks
# ------------------------------------------------------------

order = np.argsort(
    residual
)[::-1]

rank = np.empty(N, dtype=int)

for r, i in enumerate(
    order,
    start=1
):
    rank[i] = r


results = []

for i in range(N):

    results.append({
        "window":
            int(windows[i]),

        "transition_l2":
            float(x[i]),

        "observed_innovation":
            float(y[i]),

        "loo_expected_innovation":
            float(expected[i]),

        "surprise_residual":
            float(residual[i]),

        "surprise_ratio":
            float(ratio[i]),

        "sequence_rank":
            int(rank[i]),

        "conformal_p":
            float(p_values[i]),
    })


ranked = sorted(
    results,
    key=lambda r:
        r["surprise_residual"],
    reverse=True,
)


targets = {
    str(target): next(
        r for r in results
        if r["window"] == target
    )
    for target in (701, 801)
}


# ------------------------------------------------------------
# Joint top-two test
#
# Compare the two target residuals with the two largest
# residuals in the whole sequence.
# ------------------------------------------------------------

target_sum = (
    targets["701"]["surprise_residual"]
    +
    targets["801"]["surprise_residual"]
)

all_pair_sums = []

for i in range(N):

    for j in range(i + 1, N):

        all_pair_sums.append(
            residual[i]
            +
            residual[j]
        )

all_pair_sums = np.asarray(
    all_pair_sums,
    dtype=float,
)

joint_p = float(
    (
        1
        +
        np.sum(
            all_pair_sums >= target_sum
        )
    )
    /
    (
        len(all_pair_sums)
        + 1
    )
)


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-016",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "method":
        "leave-one-out nearest-transition median predictor with sequence-wide conformal ranking",

    "samples":
        int(N),

    "neighbors":
        int(K),

    "targets":
        targets,

    "joint_701_801": {
        "combined_residual":
            float(target_sum),

        "pairwise_empirical_p":
            joint_p,
    },

    "ranked_results":
        ranked,

    "scientific_boundary": (
        "This experiment measures predictive surprise relative "
        "to transitions of similar observable magnitude using "
        "leave-one-out empirical ranking. Statistical surprise "
        "does not identify physical cause, detector degradation, "
        "hardware failure, or causal mechanism."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-016 ===")
print()

print(
    "Samples:",
    N
)

print()

for target in (701, 801):

    r = targets[str(target)]

    print("TARGET", target)

    print(
        " observed:",
        f"{r['observed_innovation']:.8f}"
    )

    print(
        " expected:",
        f"{r['loo_expected_innovation']:.8f}"
    )

    print(
        " residual:",
        f"{r['surprise_residual']:.8f}"
    )

    print(
        " ratio:",
        f"{r['surprise_ratio']:.2f}x"
    )

    print(
        " sequence rank:",
        f"{r['sequence_rank']}/{N}"
    )

    print(
        " conformal p:",
        f"{r['conformal_p']:.6f}"
    )

    print()


print(
    "Joint 701+801 residual:",
    f"{target_sum:.8f}"
)

print(
    "Pairwise empirical p:",
    f"{joint_p:.6f}"
)

print()

print("Top sequence-wide surprises:")

for r in ranked[:10]:

    print(
        " window=",
        r["window"],
        "residual=",
        f"{r['surprise_residual']:.8f}",
        "ratio=",
        f"{r['surprise_ratio']:.2f}x",
        "rank=",
        r["sequence_rank"],
        "p=",
        f"{r['conformal_p']:.6f}"
    )

print()
print("Evidence:", OUT)
