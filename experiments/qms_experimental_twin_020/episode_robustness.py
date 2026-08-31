from pathlib import Path
import json


SRC = Path(
    "experiments/qms_experimental_twin_018/"
    "evidence/qms_experimental_twin_018_causal_frozen_reference.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_020/"
    "evidence/qms_experimental_twin_020_episode_robustness.json"
)

REFRACTORY_VALUES = [
    10,
    25,
    50,
    75,
    100,
    150,
]

data = json.loads(SRC.read_text())

flags = sorted(
    [
        r
        for r in data["future_results"]
        if r["flagged"]
    ],
    key=lambda r: r["window_end"],
)


def build_episodes(refractory):

    episodes = []
    current = None

    for event in flags:

        w = int(
            event["window_end"]
        )

        if current is None:

            current = {
                "start":
                    w,

                "end":
                    w,

                "events":
                    [event],
            }

            continue

        if (
            w - current["end"]
            <= refractory
        ):

            current["events"].append(
                event
            )

            current["end"] = w

        else:

            episodes.append(
                current
            )

            current = {
                "start":
                    w,

                "end":
                    w,

                "events":
                    [event],
            }

    if current is not None:
        episodes.append(current)

    summaries = []

    for i, ep in enumerate(
        episodes,
        start=1,
    ):

        peak = max(
            ep["events"],
            key=lambda r:
                r["innovation_norm"],
        )

        summaries.append({
            "episode":
                i,

            "start":
                ep["start"],

            "end":
                ep["end"],

            "flag_count":
                len(ep["events"]),

            "peak_window":
                int(
                    peak["window_end"]
                ),

            "peak_innovation":
                float(
                    peak[
                        "innovation_norm"
                    ]
                ),
        })

    return summaries


results = {}

for refractory in REFRACTORY_VALUES:

    episodes = build_episodes(
        refractory
    )

    results[str(refractory)] = {
        "episode_count":
            len(episodes),

        "episodes":
            episodes,
    }


# Track the strongest event locations themselves.
# These do not depend on episode grouping.
top_flags = sorted(
    flags,
    key=lambda r:
        r["innovation_norm"],
    reverse=True,
)[:10]


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-020",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "raw_flag_count":
        len(flags),

    "refractory_values":
        REFRACTORY_VALUES,

    "results":
        results,

    "strongest_raw_flags": [
        {
            "window":
                int(r["window_end"]),

            "innovation":
                float(
                    r["innovation_norm"]
                ),

            "frozen_z":
                float(
                    r["robust_z_frozen"]
                ),
        }
        for r in top_flags
    ],

    "scientific_boundary": (
        "This analysis tests robustness of event-level alert "
        "grouping to the chosen refractory interval. Episode "
        "boundaries are algorithmic summaries of overlapping "
        "prospective flags and must not be interpreted as "
        "physical event durations or independent physical events."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-020 ===")
print()

print(
    "Raw prospective flags:",
    len(flags)
)

print()

for refractory in REFRACTORY_VALUES:

    r = results[
        str(refractory)
    ]

    print(
        "Refractory:",
        refractory,
        "-> episodes:",
        r["episode_count"]
    )

    print(
        " peaks:",
        [
            e["peak_window"]
            for e in r["episodes"]
        ]
    )

print()

print("Strongest raw prospective flags:")

for r in top_flags:

    print(
        " window=",
        r["window_end"],
        "innovation=",
        f"{r['innovation_norm']:.8f}",
        "z=",
        f"{r['robust_z_frozen']:.3f}",
    )

print()
print("Evidence:", OUT)
