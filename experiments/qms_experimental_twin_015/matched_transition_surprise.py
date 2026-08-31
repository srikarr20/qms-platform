from pathlib import Path
import json
import numpy as np


SRC = Path(
    "experiments/qms_experimental_twin_013/"
    "evidence/qms_experimental_twin_013_innovation_geometry.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_015/"
    "evidence/qms_experimental_twin_015_matched_surprise.json"
)

data = json.loads(SRC.read_text())
rows = data["results"]

TARGETS = [701, 801]

transition = np.asarray(
    [float(r["transition_l2"]) for r in rows],
    dtype=float,
)

innovation = np.asarray(
    [float(r["innovation_norm"]) for r in rows],
    dtype=float,
)

windows = np.asarray(
    [int(r["window_end"]) for r in rows],
    dtype=int,
)


def target_analysis(target):

    idx = int(
        np.where(windows == target)[0][0]
    )

    t = transition[idx]
    y = innovation[idx]

    # Rank all other transitions by similarity
    # in transition magnitude.
    distance = np.abs(
        transition - t
    )

    candidates = [
        i for i in np.argsort(distance)
        if i != idx
    ]

    # Use nearest 10 comparable transitions.
    K = min(10, len(candidates))

    matched_idx = candidates[:K]

    matched_innov = innovation[
        matched_idx
    ]

    matched_transition = transition[
        matched_idx
    ]

    # Empirical exceedance with finite-sample
    # correction.
    exceedances = int(
        np.sum(
            matched_innov >= y
        )
    )

    p_empirical = (
        exceedances + 1
    ) / (K + 1)

    median = float(
        np.median(
            matched_innov
        )
    )

    mad = float(
        np.median(
            np.abs(
                matched_innov - median
            )
        )
    )

    if mad > 0:
        robust_z = float(
            (y - median)
            /
            (1.4826 * mad)
        )
    else:
        robust_z = None

    return {
        "window":
            target,

        "target_transition_l2":
            float(t),

        "target_innovation":
            float(y),

        "matched_count":
            K,

        "matched_innovation_mean":
            float(
                matched_innov.mean()
            ),

        "matched_innovation_median":
            median,

        "target_to_matched_median_ratio":
            float(
                y / median
            ),

        "matched_empirical_p":
            float(
                p_empirical
            ),

        "matched_robust_z":
            robust_z,

        "matched_transitions": [
            {
                "window":
                    int(windows[i]),

                "transition_l2":
                    float(transition[i]),

                "innovation":
                    float(innovation[i]),

                "transition_distance":
                    float(distance[i]),
            }
            for i in matched_idx
        ],
    }


targets = {
    str(t): target_analysis(t)
    for t in TARGETS
}


# ------------------------------------------------------------
# Sequence-wide normalized surprise
#
# For each state, compare innovation with its nearest
# transition-magnitude neighbours.
# ------------------------------------------------------------

sequence = []

for idx in range(len(rows)):

    distance = np.abs(
        transition - transition[idx]
    )

    candidates = [
        i for i in np.argsort(distance)
        if i != idx
    ]

    K = min(
        10,
        len(candidates)
    )

    neighbours = candidates[:K]

    neighbour_median = float(
        np.median(
            innovation[neighbours]
        )
    )

    ratio = float(
        innovation[idx]
        /
        neighbour_median
    )

    sequence.append({
        "window":
            int(windows[idx]),

        "transition_l2":
            float(transition[idx]),

        "innovation":
            float(innovation[idx]),

        "matched_median_innovation":
            neighbour_median,

        "surprise_ratio":
            ratio,
    })


ranked = sorted(
    sequence,
    key=lambda r:
        r["surprise_ratio"],
    reverse=True,
)


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-015",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "method":
        "nearest-transition-magnitude matched comparison",

    "neighbors":
        10,

    "targets":
        targets,

    "top_matched_surprises":
        ranked[:10],

    "scientific_boundary": (
        "This analysis asks whether predictive innovation "
        "is unusual relative to measurement transitions "
        "of similar observable magnitude. It avoids assuming "
        "a particular linear relationship. It does not identify "
        "physical cause, detector health, degradation, or causality."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-015 ===")
print()

for target in TARGETS:

    r = targets[str(target)]

    print("TARGET", target)

    print(
        " transition:",
        f"{r['target_transition_l2']:.6f}"
    )

    print(
        " innovation:",
        f"{r['target_innovation']:.8f}"
    )

    print(
        " matched median:",
        f"{r['matched_innovation_median']:.8f}"
    )

    print(
        " surprise ratio:",
        f"{r['target_to_matched_median_ratio']:.3f}x"
    )

    print(
        " empirical p:",
        f"{r['matched_empirical_p']:.6f}"
    )

    print(
        " robust-z:",
        "NA"
        if r["matched_robust_z"] is None
        else f"{r['matched_robust_z']:.3f}"
    )

    print()


print("Top matched-transition surprises:")
print()

for r in ranked[:10]:

    print(
        " window=",
        r["window"],
        "innovation=",
        f"{r['innovation']:.8f}",
        "transition=",
        f"{r['transition_l2']:.6f}",
        "matched=",
        f"{r['matched_median_innovation']:.8f}",
        "ratio=",
        f"{r['surprise_ratio']:.2f}x"
    )

print()
print("Evidence:", OUT)
