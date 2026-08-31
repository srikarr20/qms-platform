from pathlib import Path
import json
import numpy as np


SRC = Path(
    "experiments/qms_experimental_twin_003/"
    "evidence/qms_experimental_twin_003_predictive.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_004/"
    "evidence/qms_experimental_twin_004_innovation_gate.json"
)

data = json.loads(SRC.read_text())
rows = data["results"]

innovation = np.asarray(
    [float(r["innovation_norm"]) for r in rows],
    dtype=float,
)

# Empirical thresholds derived from this real dataset.
p95 = float(np.percentile(innovation, 95))
p99 = float(np.percentile(innovation, 99))
p999 = float(np.percentile(innovation, 99.9))


results = []

counts = {
    "EXPECTED": 0,
    "ELEVATED": 0,
    "LARGE_DEVIATION": 0,
    "EXTREME_DEVIATION": 0,
}


for r in rows:

    x = float(r["innovation_norm"])

    if x <= p95:
        state = "EXPECTED"

    elif x <= p99:
        state = "ELEVATED"

    elif x <= p999:
        state = "LARGE_DEVIATION"

    else:
        state = "EXTREME_DEVIATION"

    counts[state] += 1

    results.append({
        "window_end_increment":
            r["window_end_increment"],

        "innovation_norm":
            x,

        "cosine_innovation":
            r["cosine_innovation"],

        "js_innovation":
            r["js_innovation"],

        "measurement_state":
            r["smoothed_measurement_state"],

        "innovation_state":
            state,
    })


# Show largest deviations
largest = sorted(
    results,
    key=lambda r: r["innovation_norm"],
    reverse=True,
)[:20]


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-004",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "threshold_basis":
        "empirical percentiles of one-step-ahead innovation norm",

    "thresholds": {
        "p95": p95,
        "p99": p99,
        "p99_9": p999,
    },

    "counts":
        counts,

    "largest_deviations":
        largest,

    "results":
        results,

    "scientific_boundary": (
        "Innovation classes are relative to the empirical "
        "distribution of short-horizon prediction errors in "
        "this Glasgow detector sequence. They indicate unusual "
        "measurement-distribution changes, not detector failure, "
        "hardware health, CTI, readout noise, or causal degradation."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-004 ===")
print()

print("P95 threshold :", f"{p95:.8f}")
print("P99 threshold :", f"{p99:.8f}")
print("P99.9 threshold:", f"{p999:.8f}")

print()
print("Innovation-state counts:")
for k, v in counts.items():
    print(" ", k, ":", v)

print()
print("Top 10 measurement deviations:")
print()

for r in largest[:10]:

    print(
        "window=",
        r["window_end_increment"],
        "innovation=",
        f"{r['innovation_norm']:.8f}",
        "state=",
        r["innovation_state"],
        "measurement=",
        r["measurement_state"],
    )

print()
print("Evidence:", OUT)
