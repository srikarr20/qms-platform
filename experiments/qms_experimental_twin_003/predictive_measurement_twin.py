from pathlib import Path
import json
import numpy as np


SRC = Path(
    "experiments/qms_experimental_twin_002/"
    "evidence/qms_experimental_twin_002_persistent.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_003/"
    "evidence/qms_experimental_twin_003_predictive.json"
)

data = json.loads(SRC.read_text())
rows = data["results"]


cos = np.asarray(
    [float(r["cosine_similarity"]) for r in rows],
    dtype=float,
)

js = np.asarray(
    [float(r["js_divergence"]) for r in rows],
    dtype=float,
)


results = []

for i in range(2, len(rows)):

    # Constant-velocity predictor in measurement-state space
    pred_cos = cos[i - 1] + (cos[i - 1] - cos[i - 2])
    pred_js = js[i - 1] + (js[i - 1] - js[i - 2])

    obs_cos = cos[i]
    obs_js = js[i]

    innovation_cos = obs_cos - pred_cos
    innovation_js = obs_js - pred_js

    results.append({
        "window_end_increment":
            rows[i]["window_end_increment"],

        "predicted_cosine":
            float(pred_cos),

        "observed_cosine":
            float(obs_cos),

        "cosine_innovation":
            float(innovation_cos),

        "predicted_js":
            float(pred_js),

        "observed_js":
            float(obs_js),

        "js_innovation":
            float(innovation_js),

        "innovation_norm":
            float(
                np.sqrt(
                    innovation_cos ** 2
                    + innovation_js ** 2
                )
            ),

        "smoothed_measurement_state":
            rows[i]["state"],
    })


innovation_norm = np.asarray(
    [r["innovation_norm"] for r in results],
    dtype=float,
)

cos_abs = np.asarray(
    [abs(r["cosine_innovation"]) for r in results],
    dtype=float,
)

js_abs = np.asarray(
    [abs(r["js_innovation"]) for r in results],
    dtype=float,
)


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-003",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "predictor":
        "constant-velocity extrapolation in cosine/JS measurement-state space",

    "predictions":
        len(results),

    "mean_absolute_cosine_innovation":
        float(cos_abs.mean()),

    "p95_absolute_cosine_innovation":
        float(np.percentile(cos_abs, 95)),

    "mean_absolute_js_innovation":
        float(js_abs.mean()),

    "p95_absolute_js_innovation":
        float(np.percentile(js_abs, 95)),

    "mean_innovation_norm":
        float(innovation_norm.mean()),

    "p95_innovation_norm":
        float(np.percentile(innovation_norm, 95)),

    "max_innovation_norm":
        float(innovation_norm.max()),

    "results":
        results,

    "scientific_boundary": (
        "Prediction is performed only in observable "
        "measurement-distribution space using real Glasgow "
        "detector data. Innovation quantifies unexpected "
        "short-horizon distribution change relative to this "
        "simple predictor. It is not a detector-health, failure, "
        "CTI, noise, or causal degradation score."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-003 ===")
print()

print(
    "Predictions:",
    summary["predictions"]
)

print(
    "Mean |cos innovation|:",
    f"{summary['mean_absolute_cosine_innovation']:.8f}"
)

print(
    "P95 |cos innovation|:",
    f"{summary['p95_absolute_cosine_innovation']:.8f}"
)

print(
    "Mean |JS innovation|:",
    f"{summary['mean_absolute_js_innovation']:.8f}"
)

print(
    "P95 |JS innovation|:",
    f"{summary['p95_absolute_js_innovation']:.8f}"
)

print(
    "Mean innovation norm:",
    f"{summary['mean_innovation_norm']:.8f}"
)

print(
    "P95 innovation norm:",
    f"{summary['p95_innovation_norm']:.8f}"
)

print(
    "Max innovation norm:",
    f"{summary['max_innovation_norm']:.8f}"
)

print()
print("Evidence:", OUT)
