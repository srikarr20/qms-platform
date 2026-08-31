from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

trajectory = json.loads(
    (
        ROOT
        / "qms_pyxel_twin_003"
        / "qms_pyxel_twin_003b_calibrated_noise.json"
    ).read_text()
)["results"]

trajectory = sorted(
    trajectory,
    key=lambda r: r["time"]
)

CTI_THRESHOLD = 0.20
NOISE_THRESHOLD = 0.10


def state(row):
    return np.array([
        float(row["estimated_cti_density"]),
        float(row["calibrated_estimated_output_noise"]),
    ])


def predict(a, b):
    # constant-velocity model
    return b + (b - a)


history = []
accepted_states = []


# Bootstrap first two states from measurements
for i in range(2):
    s = state(trajectory[i])

    accepted_states.append(s)

    history.append({
        "time": trajectory[i]["time"],
        "mode": "INITIALIZED_FROM_MEASUREMENT",
        "cti": float(s[0]),
        "noise": float(s[1]),
    })


print()
print("=== QMS-PYXEL-TWIN-007 AUTONOMOUS ADAPTIVE TWIN ===")
print()


for i in range(2, len(trajectory)):

    name = trajectory[i]["time"]

    previous_2 = accepted_states[-2]
    previous_1 = accepted_states[-1]

    predicted = predict(
        previous_2,
        previous_1
    )

    measured = state(
        trajectory[i]
    )

    cti_error = abs(
        predicted[0] - measured[0]
    ) / measured[0]

    noise_error = abs(
        predicted[1] - measured[1]
    ) / measured[1]

    cti_alert = cti_error > CTI_THRESHOLD
    noise_alert = noise_error > NOISE_THRESHOLD

    trigger = cti_alert or noise_alert


    if trigger:
        # In the real architecture this branch invokes
        # the CTI + noise estimators.
        #
        # Here trajectory[i] contains those validated
        # estimator outputs, so we use them as the
        # re-estimated twin state.
        updated = measured.copy()

        mode = "REESTIMATED_AND_ADAPTED"

    else:
        # Prediction is considered compatible with
        # expected evolution.
        #
        # Assimilate the measured state without calling
        # expensive mechanism re-identification.
        updated = measured.copy()

        mode = "MEASUREMENT_ASSIMILATED"


    accepted_states.append(updated)


    history.append({
        "time":
            name,

        "predicted_cti":
            float(predicted[0]),

        "predicted_noise":
            float(predicted[1]),

        "measured_cti":
            float(measured[0]),

        "measured_noise":
            float(measured[1]),

        "cti_innovation_relative":
            float(cti_error),

        "noise_innovation_relative":
            float(noise_error),

        "cti_alert":
            bool(cti_alert),

        "noise_alert":
            bool(noise_alert),

        "trigger_reestimate":
            bool(trigger),

        "mode":
            mode,

        "updated_cti":
            float(updated[0]),

        "updated_noise":
            float(updated[1]),
    })


    print(name)

    print(
        " predicted:",
        f"CTI={predicted[0]:.6e}",
        f"noise={predicted[1]:.6e}"
    )

    print(
        " observed :",
        f"CTI={measured[0]:.6e}",
        f"noise={measured[1]:.6e}"
    )

    print(
        " innovation:",
        f"CTI={100*cti_error:.3f}%",
        f"noise={100*noise_error:.3f}%"
    )

    print(
        " action:",
        mode
    )

    print()


reestimated = sum(
    r.get("trigger_reestimate", False)
    for r in history
)

assimilated = sum(
    r.get("mode") == "MEASUREMENT_ASSIMILATED"
    for r in history
)


summary = {
    "experiment":
        "QMS-PYXEL-TWIN-007",

    "title":
        "Autonomous predictive adaptive detector twin",

    "controller": {
        "prediction":
            "constant-velocity parameter extrapolation",

        "cti_innovation_threshold":
            CTI_THRESHOLD,

        "noise_innovation_threshold":
            NOISE_THRESHOLD,
    },

    "states":
        len(history),

    "reestimation_events":
        reestimated,

    "measurement_assimilation_events":
        assimilated,

    "history":
        history,

    "scientific_boundary": (
        "Controller operates on parameter estimates obtained from "
        "controlled Pyxel simulations. Re-estimation outputs are "
        "previously validated estimators within known mechanism "
        "families. This demonstrates closed-loop computational twin "
        "logic, not autonomous real-hardware operation."
    ),
}


out = (
    ROOT
    / "qms_pyxel_twin_007"
    / "qms_pyxel_twin_007_closed_loop.json"
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

print()
print(
    "States processed:",
    summary["states"]
)

print(
    "Re-estimation events:",
    summary["reestimation_events"]
)

print(
    "Measurement-only assimilation events:",
    summary["measurement_assimilation_events"]
)

print()
print("Evidence:", out)
