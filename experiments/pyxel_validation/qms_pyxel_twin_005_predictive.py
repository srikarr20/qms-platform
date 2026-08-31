from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent

src = (
    ROOT
    / "qms_pyxel_twin_003"
    / "qms_pyxel_twin_003b_calibrated_noise.json"
)

data = json.loads(src.read_text())
rows = data["results"]

# Sort by time label
rows = sorted(rows, key=lambda r: r["time"])

# Use inferred states, not the hidden truth, as the twin's internal state
density = np.array([
    r["estimated_cti_density"]
    for r in rows
], dtype=float)

noise = np.array([
    r["calibrated_estimated_output_noise"]
    for r in rows
], dtype=float)

true_density = np.array([
    r["true_cti_density"]
    for r in rows
], dtype=float)

true_noise = np.array([
    r["true_output_noise"]
    for r in rows
], dtype=float)

times = [r["time"] for r in rows]

results = []

print()
print("=== QMS-PYXEL-TWIN-005 PREDICTIVE TWIN ===")
print()

for i in range(2, len(rows)):

    # Simple constant-velocity predictor in parameter space
    pred_density = (
        density[i - 1]
        + (density[i - 1] - density[i - 2])
    )

    pred_noise = (
        noise[i - 1]
        + (noise[i - 1] - noise[i - 2])
    )

    actual_density = density[i]
    actual_noise = noise[i]

    density_prediction_error = (
        pred_density - actual_density
    )

    noise_prediction_error = (
        pred_noise - actual_noise
    )

    density_relative_error = (
        density_prediction_error
        / actual_density
    )

    noise_relative_error = (
        noise_prediction_error
        / actual_noise
    )

    # Innovation: how far the observed twin state moved away from prediction
    density_innovation = (
        actual_density - pred_density
    )

    noise_innovation = (
        actual_noise - pred_noise
    )

    row = {
        "time": times[i],

        "predicted_cti_density":
            float(pred_density),

        "observed_cti_density":
            float(actual_density),

        "true_cti_density":
            float(true_density[i]),

        "cti_prediction_relative_error":
            float(density_relative_error),

        "cti_innovation":
            float(density_innovation),

        "predicted_output_noise":
            float(pred_noise),

        "observed_output_noise":
            float(actual_noise),

        "true_output_noise":
            float(true_noise[i]),

        "noise_prediction_relative_error":
            float(noise_relative_error),

        "noise_innovation":
            float(noise_innovation),
    }

    results.append(row)

    print(times[i])

    print(
        " CTI predicted:",
        f"{pred_density:.6e}"
    )
    print(
        " CTI observed :",
        f"{actual_density:.6e}"
    )
    print(
        " CTI pred err :",
        f"{100*abs(density_relative_error):.3f}%"
    )

    print(
        " Noise predicted:",
        f"{pred_noise:.6e}"
    )
    print(
        " Noise observed :",
        f"{actual_noise:.6e}"
    )
    print(
        " Noise pred err :",
        f"{100*abs(noise_relative_error):.3f}%"
    )

    print()


cti_err = np.array([
    abs(r["cti_prediction_relative_error"])
    for r in results
])

noise_err = np.array([
    abs(r["noise_prediction_relative_error"])
    for r in results
])


summary = {
    "experiment":
        "QMS-PYXEL-TWIN-005",

    "title":
        "One-step-ahead predictive detector twin",

    "predictor":
        "constant-velocity extrapolation in inferred parameter space",

    "predictions":
        len(results),

    "mean_absolute_cti_prediction_relative_error":
        float(cti_err.mean()),

    "max_absolute_cti_prediction_relative_error":
        float(cti_err.max()),

    "mean_absolute_noise_prediction_relative_error":
        float(noise_err.mean()),

    "max_absolute_noise_prediction_relative_error":
        float(noise_err.max()),

    "results":
        results,

    "scientific_boundary": (
        "Prediction is performed on a short synthetic time-indexed "
        "Pyxel trajectory using a simple deterministic extrapolator. "
        "This demonstrates predictive-twin mechanics, not validated "
        "forecasting of real detector aging or failure."
    ),
}

out = (
    ROOT
    / "qms_pyxel_twin_005_predictive.json"
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
    "Mean CTI prediction error:",
    f"{100*summary['mean_absolute_cti_prediction_relative_error']:.3f}%"
)

print(
    "Max CTI prediction error:",
    f"{100*summary['max_absolute_cti_prediction_relative_error']:.3f}%"
)

print()

print(
    "Mean noise prediction error:",
    f"{100*summary['mean_absolute_noise_prediction_relative_error']:.3f}%"
)

print(
    "Max noise prediction error:",
    f"{100*summary['max_absolute_noise_prediction_relative_error']:.3f}%"
)

print()
print("Evidence:", out)
