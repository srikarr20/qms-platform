from pathlib import Path
import json
import numpy as np


SRC = Path(
    "experiments/qms_experimental_twin_004/"
    "evidence/qms_experimental_twin_004_innovation_gate.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_005/"
    "evidence/qms_experimental_twin_005_event_clustering.json"
)

data = json.loads(SRC.read_text())
rows = data["results"]


# Treat P99+ events as high-innovation events.
high = [
    int(r["window_end_increment"])
    for r in rows
    if r["innovation_state"] in (
        "LARGE_DEVIATION",
        "EXTREME_DEVIATION",
    )
]

high = np.array(sorted(high), dtype=int)

if len(high) == 0:
    raise SystemExit("No high-innovation events found.")


# ------------------------------------------------------------
# Adjacent-event clustering
# ------------------------------------------------------------

gaps = np.diff(high)

adjacent_pairs = int(
    np.sum(gaps == 1)
)

close_pairs_5 = int(
    np.sum(gaps <= 5)
)

close_pairs_10 = int(
    np.sum(gaps <= 10)
)


# ------------------------------------------------------------
# Build contiguous clusters
# ------------------------------------------------------------

clusters = []

current = [int(high[0])]

for previous, current_value in zip(
    high[:-1],
    high[1:]
):
    if current_value - previous <= 1:
        current.append(
            int(current_value)
        )
    else:
        clusters.append(current)
        current = [
            int(current_value)
        ]

clusters.append(current)

multi_event_clusters = [
    c for c in clusters
    if len(c) > 1
]


# ------------------------------------------------------------
# Monte Carlo null:
# randomly distribute the same number of events across the
# available window range and ask how many adjacent pairs occur.
# ------------------------------------------------------------

all_windows = np.array([
    int(r["window_end_increment"])
    for r in rows
], dtype=int)

rng = np.random.default_rng(20260829)

TRIALS = 10000

null_adjacent = np.zeros(
    TRIALS,
    dtype=int,
)

null_close5 = np.zeros(
    TRIALS,
    dtype=int,
)

for i in range(TRIALS):

    sample = np.sort(
        rng.choice(
            all_windows,
            size=len(high),
            replace=False,
        )
    )

    dg = np.diff(sample)

    null_adjacent[i] = np.sum(
        dg == 1
    )

    null_close5[i] = np.sum(
        dg <= 5
    )


p_adjacent = float(
    np.mean(
        null_adjacent >= adjacent_pairs
    )
)

p_close5 = float(
    np.mean(
        null_close5 >= close_pairs_5
    )
)


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-005",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "event_definition":
        "P99-or-higher one-step innovation",

    "high_event_count":
        int(len(high)),

    "adjacent_pairs":
        adjacent_pairs,

    "pairs_within_5_windows":
        close_pairs_5,

    "pairs_within_10_windows":
        close_pairs_10,

    "contiguous_clusters":
        clusters,

    "multi_event_clusters":
        multi_event_clusters,

    "monte_carlo_trials":
        TRIALS,

    "null_mean_adjacent_pairs":
        float(
            null_adjacent.mean()
        ),

    "null_mean_pairs_within_5":
        float(
            null_close5.mean()
        ),

    "p_adjacent":
        p_adjacent,

    "p_within_5":
        p_close5,

    "scientific_boundary": (
        "This test evaluates temporal clustering of unusually "
        "large short-horizon measurement-distribution innovations "
        "in one real Glasgow detector sequence. Clustering does "
        "not establish detector failure, degradation mechanism, "
        "or causality."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-005 ===")
print()

print(
    "High-innovation events:",
    len(high)
)

print(
    "Adjacent pairs:",
    adjacent_pairs
)

print(
    "Pairs within 5 windows:",
    close_pairs_5
)

print(
    "Pairs within 10 windows:",
    close_pairs_10
)

print()

print(
    "Multi-event contiguous clusters:"
)

for c in multi_event_clusters:
    print(" ", c)

print()

print(
    "Null mean adjacent pairs:",
    f"{null_adjacent.mean():.4f}"
)

print(
    "Observed adjacent pairs:",
    adjacent_pairs
)

print(
    "Monte Carlo p(adjacent):",
    f"{p_adjacent:.6f}"
)

print()

print(
    "Null mean pairs within 5:",
    f"{null_close5.mean():.4f}"
)

print(
    "Observed within 5:",
    close_pairs_5
)

print(
    "Monte Carlo p(within 5):",
    f"{p_close5:.6f}"
)

print()
print("Evidence:", OUT)
