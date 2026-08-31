from pathlib import Path
import json
import numpy as np


ROOT = Path(__file__).resolve().parent

src = (
    ROOT
    / "qms_pyxel_twin_005_predictive.json"
)

data = json.loads(src.read_text())
rows = data["results"]


# Provisional computational thresholds.
# These are NOT universal detector thresholds.
CTI_REL_THRESHOLD = 0.20
NOISE_REL_THRESHOLD = 0.10


results = []

print()
print("=== QMS-PYXEL-TWIN-006 PREDICTIVE DIVERGENCE ===")
print()


for r in rows:

    cti_err = abs(
        r["cti_prediction_relative_error"]
    )

    noise_err = abs(
        r["noise_prediction_relative_error"]
    )

    cti_alert = (
        cti_err > CTI_REL_THRESHOLD
    )

    noise_alert = (
        noise_err > NOISE_REL_THRESHOLD
    )


    if cti_alert and noise_alert:
        status = "MULTI_PARAMETER_CHANGE"

    elif cti_alert:
        status = "CTI_CHANGE"

    elif noise_alert:
        status = "NOISE_CHANGE"

    else:
        status = "EXPECTED_EVOLUTION"


    trigger_reestimate = (
        cti_alert or noise_alert
    )


    row = {
        "time":
            r["time"],

        "cti_prediction_relative_error":
            cti_err,

        "noise_prediction_relative_error":
            noise_err,

        "cti_alert":
            cti_alert,

        "noise_alert":
            noise_alert,

        "status":
            status,

        "trigger_reestimate":
            trigger_reestimate,
    }

    results.append(row)


    print(r["time"])
    print(
        " CTI innovation error:",
        f"{100*cti_err:.3f}%"
    )
    print(
        " Noise innovation error:",
        f"{100*noise_err:.3f}%"
    )
    print(
        " status:",
        status
    )
    print(
        " trigger re-estimation:",
        trigger_reestimate
    )
    print()


summary = {
    "experiment":
        "QMS-PYXEL-TWIN-006",

    "title":
        "Predictive innovation and divergence-triggered adaptation",

    "thresholds": {
        "cti_relative_prediction_error":
            CTI_REL_THRESHOLD,

        "noise_relative_prediction_error":
            NOISE_REL_THRESHOLD,
    },

    "results":
        results,

    "scientific_boundary": (
        "Thresholds are provisional and selected for this "
        "synthetic Pyxel trajectory. The experiment demonstrates "
        "predictive divergence logic and adaptation triggering, "
        "not validated hardware anomaly thresholds."
    ),
}


out = (
    ROOT
    / "qms_pyxel_twin_006_divergence.json"
)

out.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print("=" * 60)
print("SUMMARY")
print("=" * 60)

for r in results:
    print(
        r["time"],
        "->",
        r["status"],
        "| re-estimate:",
        r["trigger_reestimate"]
    )

print()
print("Evidence:", out)
