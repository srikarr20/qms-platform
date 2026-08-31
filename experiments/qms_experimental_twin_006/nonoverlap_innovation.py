from pathlib import Path
import json
import numpy as np


SRC = Path(
    "experiments/qms_experimental_twin_003/"
    "evidence/qms_experimental_twin_003_predictive.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_006/"
    "evidence/qms_experimental_twin_006_nonoverlap.json"
)

WINDOW = 100

data = json.loads(SRC.read_text())
rows = data["results"]


# ------------------------------------------------------------
# Select approximately non-overlapping windows.
# Each original measurement window has length 100.
# We retain states separated by >=100 acquisition increments.
# ------------------------------------------------------------

selected = []

last_window = None

for r in rows:

    w = int(r["window_end_increment"])

    if last_window is None or (w - last_window) >= WINDOW:
        selected.append(r)
        last_window = w


innov = np.asarray(
    [float(r["innovation_norm"]) for r in selected],
    dtype=float,
)

if len(innov) < 10:
    raise RuntimeError(
        f"Too few non-overlapping samples: {len(innov)}"
    )


p95 = float(np.percentile(innov, 95))
p99 = float(np.percentile(innov, 99))

high = [
    {
        "window_end_increment":
            int(r["window_end_increment"]),
        "innovation_norm":
            float(r["innovation_norm"]),
        "measurement_state":
            r["smoothed_measurement_state"],
    }
    for r in selected
    if float(r["innovation_norm"]) > p95
]


# ------------------------------------------------------------
# Serial dependence check on non-overlapping innovations
# ------------------------------------------------------------

if len(innov) > 2:

    x = innov[:-1]
    y = innov[1:]

    lag1 = float(
        np.corrcoef(x, y)[0, 1]
    )

else:
    lag1 = None


# ------------------------------------------------------------
# Compare original innovation distribution with non-overlap
# ------------------------------------------------------------

all_innov = np.asarray(
    [float(r["innovation_norm"]) for r in rows],
    dtype=float,
)


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-006",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "selection":
        "prediction states separated by at least 100 acquisition increments",

    "original_predictions":
        len(rows),

    "nonoverlapping_predictions":
        len(selected),

    "original_mean_innovation":
        float(all_innov.mean()),

    "nonoverlap_mean_innovation":
        float(innov.mean()),

    "nonoverlap_p95":
        p95,

    "nonoverlap_p99":
        p99,

    "nonoverlap_max":
        float(innov.max()),

    "lag1_innovation_correlation":
        lag1,

    "p95_exceedances":
        high,

    "scientific_boundary": (
        "This analysis reduces dependence caused by the "
        "100-increment sliding measurement window by retaining "
        "states separated by at least one full window. "
        "It tests whether unusually large measurement-space "
        "innovations remain visible without heavy window overlap. "
        "No physical failure or degradation mechanism is inferred."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-006 ===")
print()

print(
    "Original predictions:",
    len(rows)
)

print(
    "Non-overlapping predictions:",
    len(selected)
)

print()

print(
    "Original mean innovation:",
    f"{all_innov.mean():.8f}"
)

print(
    "Non-overlap mean innovation:",
    f"{innov.mean():.8f}"
)

print(
    "Non-overlap P95:",
    f"{p95:.8f}"
)

print(
    "Non-overlap P99:",
    f"{p99:.8f}"
)

print(
    "Non-overlap max:",
    f"{innov.max():.8f}"
)

print()

print(
    "Lag-1 innovation correlation:",
    "NA" if lag1 is None else f"{lag1:.6f}"
)

print()

print("P95 exceedances:")

for r in high:
    print(
        " window=",
        r["window_end_increment"],
        "innovation=",
        f"{r['innovation_norm']:.8f}",
        "state=",
        r["measurement_state"],
    )

print()
print("Evidence:", OUT)
