from pathlib import Path
import json


SRC = Path(
    "experiments/qms_experimental_twin_018/"
    "evidence/qms_experimental_twin_018_causal_frozen_reference.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_019/"
    "evidence/qms_experimental_twin_019_alert_episodes.json"
)

REFRACTORY = 100

data = json.loads(SRC.read_text())

future = data["future_results"]

flags = [
    r for r in future
    if r["flagged"]
]

flags = sorted(
    flags,
    key=lambda r: r["window_end"]
)


episodes = []
current = None


for event in flags:

    w = int(event["window_end"])

    if current is None:
        current = {
            "start_window": w,
            "end_window": w,
            "events": [event],
        }
        continue

    if w - current["end_window"] <= REFRACTORY:
        current["events"].append(event)
        current["end_window"] = w
    else:
        episodes.append(current)

        current = {
            "start_window": w,
            "end_window": w,
            "events": [event],
        }


if current is not None:
    episodes.append(current)


summaries = []

for i, ep in enumerate(episodes, start=1):

    strongest = max(
        ep["events"],
        key=lambda r: r["innovation_norm"]
    )

    summaries.append({
        "episode": i,

        "start_window":
            ep["start_window"],

        "end_window":
            ep["end_window"],

        "duration_increments":
            ep["end_window"]
            - ep["start_window"],

        "raw_flag_count":
            len(ep["events"]),

        "peak_window":
            int(strongest["window_end"]),

        "peak_innovation":
            float(strongest["innovation_norm"]),

        "peak_frozen_z":
            float(strongest["robust_z_frozen"]),
    })


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-019",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "design":
        "causal prospective alert episode consolidation",

    "refractory_increments":
        REFRACTORY,

    "raw_prospective_flags":
        len(flags),

    "independent_alert_episodes":
        len(summaries),

    "episodes":
        summaries,

    "scientific_boundary": (
        "Alerts are consolidated to reduce repeated detections "
        "caused by overlapping 100-increment measurement windows. "
        "Episodes represent temporally localized unexpected "
        "measurement-distribution evolution, not detector faults, "
        "degradation mechanisms, or physical causal events."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-019 ===")
print()

print(
    "Raw prospective flags:",
    len(flags)
)

print(
    "Consolidated alert episodes:",
    len(summaries)
)

print()
print("Episodes:")

for r in summaries:

    print(
        " episode=",
        r["episode"],
        "range=",
        f"{r['start_window']}-{r['end_window']}",
        "raw flags=",
        r["raw_flag_count"],
        "peak=",
        r["peak_window"],
        "innovation=",
        f"{r['peak_innovation']:.8f}",
        "z=",
        f"{r['peak_frozen_z']:.3f}",
    )

print()
print("Evidence:", OUT)
