from pathlib import Path
import json
import numpy as np


SRC = Path(
    "experiments/qms_experimental_twin_001/"
    "evidence/qms_experimental_twin_001_glasgow.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_002/"
    "evidence/qms_experimental_twin_002_persistent.json"
)

SMOOTH = 3


data = json.loads(SRC.read_text())
history = data["history"]


delta_cos = np.array([
    np.nan if r["delta_cosine"] is None
    else float(r["delta_cosine"])
    for r in history
])

delta_js = np.array([
    np.nan if r["delta_js"] is None
    else float(r["delta_js"])
    for r in history
])


def rolling_mean(x, n):
    out = np.full(len(x), np.nan)

    for i in range(n - 1, len(x)):
        chunk = x[i - n + 1:i + 1]

        if np.all(np.isfinite(chunk)):
            out[i] = float(np.mean(chunk))

    return out


smooth_cos = rolling_mean(
    delta_cos,
    SMOOTH
)

smooth_js = rolling_mean(
    delta_js,
    SMOOTH
)


results = []

previous_state = None
transitions = 0


for i, source in enumerate(history):

    dc = smooth_cos[i]
    dj = smooth_js[i]

    if not np.isfinite(dc) or not np.isfinite(dj):
        state = "INITIALIZE"

    elif dc > 0 and dj < 0:
        state = "CONVERGING"

    elif dc < 0 and dj > 0:
        state = "DIVERGING"

    else:
        state = "MIXED"

    if (
        previous_state is not None
        and state != previous_state
        and state != "INITIALIZE"
        and previous_state != "INITIALIZE"
    ):
        transitions += 1

    results.append({
        "window_end_increment":
            source["window_end_increment"],

        "cosine_similarity":
            source[
                "cosine_similarity_to_final_distribution"
            ],

        "js_divergence":
            source[
                "js_divergence_to_final_distribution"
            ],

        "raw_delta_cosine":
            source["delta_cosine"],

        "raw_delta_js":
            source["delta_js"],

        "smoothed_delta_cosine":
            None if not np.isfinite(dc)
            else float(dc),

        "smoothed_delta_js":
            None if not np.isfinite(dj)
            else float(dj),

        "state":
            state,
    })

    previous_state = state


counts = {}

for r in results:
    counts[r["state"]] = (
        counts.get(r["state"], 0) + 1
    )


raw_states = [
    r["measurement_state"]
    for r in history
]

raw_transitions = 0

for a, b in zip(
    raw_states[:-1],
    raw_states[1:]
):
    if (
        a != b
        and a != "INITIALIZE"
        and b != "INITIALIZE"
    ):
        raw_transitions += 1


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-002",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "source_type":
        "real experimental detector measurements",

    "smoothing_windows":
        SMOOTH,

    "raw_state_transitions":
        raw_transitions,

    "smoothed_state_transitions":
        transitions,

    "state_counts":
        counts,

    "results":
        results,

    "scientific_boundary": (
        "Temporal smoothing reduces sensitivity to "
        "single-window fluctuations in the real Glasgow "
        "measurement stream. States describe evolution "
        "of detector-distribution similarity to the final "
        "accumulated reference distribution. They are not "
        "hardware-health, CTI, noise, failure, or causal "
        "degradation labels."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-002 ===")
print()

print(
    "Windows:",
    len(results)
)

print(
    "Raw state transitions:",
    raw_transitions
)

print(
    "Smoothed state transitions:",
    transitions
)

if raw_transitions:
    reduction = (
        1.0
        - transitions / raw_transitions
    )

    print(
        "Transition reduction:",
        f"{100*reduction:.3f}%"
    )

print(
    "State counts:",
    counts
)

print()
print("Evidence:", OUT)
