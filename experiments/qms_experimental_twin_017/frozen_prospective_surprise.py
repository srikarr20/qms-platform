from pathlib import Path
import json
import numpy as np


SRC = Path(
    "experiments/qms_experimental_twin_003/"
    "evidence/qms_experimental_twin_003_predictive.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_017/"
    "evidence/qms_experimental_twin_017_frozen_prospective.json"
)

TRAIN_END = 699

data = json.loads(SRC.read_text())
rows = data["results"]


windows = np.asarray(
    [int(r["window_end_increment"]) for r in rows],
    dtype=int,
)

innovation = np.asarray(
    [float(r["innovation_norm"]) for r in rows],
    dtype=float,
)


# ------------------------------------------------------------
# Freeze calibration using ONLY pre-701 observations.
# ------------------------------------------------------------

train_mask = windows <= TRAIN_END
test_mask = windows > TRAIN_END

train = innovation[train_mask]

if len(train) < 20:
    raise RuntimeError(
        f"Too few training observations: {len(train)}"
    )


median = float(
    np.median(train)
)

mad = float(
    np.median(
        np.abs(train - median)
    )
)

p95 = float(
    np.percentile(train, 95)
)

p99 = float(
    np.percentile(train, 99)
)


# Robust threshold:
# median + 6 MAD-scaled deviations.
robust_threshold = float(
    median + 6.0 * 1.4826 * mad
)

# Use the more conservative of empirical P99
# and robust six-MAD threshold.
threshold = max(
    p99,
    robust_threshold,
)


# ------------------------------------------------------------
# Evaluate future observations WITHOUT recalibration.
# ------------------------------------------------------------

results = []

for r in rows:

    w = int(
        r["window_end_increment"]
    )

    if w <= TRAIN_END:
        continue

    x = float(
        r["innovation_norm"]
    )

    robust_z = (
        (x - median)
        /
        (1.4826 * mad)
        if mad > 0
        else None
    )

    flagged = bool(
        x > threshold
    )

    results.append({
        "window":
            w,

        "innovation":
            x,

        "robust_z_against_frozen_baseline":
            None
            if robust_z is None
            else float(robust_z),

        "flagged":
            flagged,
    })


flagged = [
    r for r in results
    if r["flagged"]
]

target_results = {
    str(target): next(
        (
            r for r in results
            if r["window"] == target
        ),
        None,
    )
    for target in (701, 801)
}


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-017",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "design":
        "prospective frozen-baseline evaluation",

    "training": {
        "last_window":
            TRAIN_END,

        "samples":
            int(len(train)),

        "median_innovation":
            median,

        "mad":
            mad,

        "p95":
            p95,

        "p99":
            p99,

        "robust_6mad_threshold":
            robust_threshold,

        "frozen_threshold":
            threshold,
    },

    "test": {
        "samples":
            len(results),

        "flagged_count":
            len(flagged),

        "flagged_fraction":
            (
                len(flagged)
                / len(results)
                if results
                else 0.0
            ),
    },

    "targets":
        target_results,

    "flagged_events":
        flagged,

    "scientific_boundary": (
        "The surprise threshold is calibrated exclusively on "
        "measurements preceding window 701 and then frozen. "
        "Subsequent flags therefore represent prospective "
        "measurement-space deviations relative to that baseline. "
        "They do not establish detector failure, degradation, "
        "hardware health, quantum-field dynamics, or physical cause."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-017 ===")
print()

print("Frozen training:")
print(" samples:", len(train))
print(" last window:", TRAIN_END)
print(" median:", f"{median:.8f}")
print(" MAD:", f"{mad:.8f}")
print(" P95:", f"{p95:.8f}")
print(" P99:", f"{p99:.8f}")
print(
    " 6-MAD threshold:",
    f"{robust_threshold:.8f}"
)
print(
    " frozen threshold:",
    f"{threshold:.8f}"
)

print()
print("Prospective test:")
print(" samples:", len(results))
print(" flags:", len(flagged))

print()

for target in (701, 801):

    r = target_results[str(target)]

    print("TARGET", target)

    if r is None:
        print(" not found")
    else:
        print(
            " innovation:",
            f"{r['innovation']:.8f}"
        )
        print(
            " frozen robust-z:",
            f"{r['robust_z_against_frozen_baseline']:.3f}"
        )
        print(
            " FLAGGED:",
            r["flagged"]
        )

    print()


print("First prospective flags:")

for r in flagged[:20]:
    print(
        " window=",
        r["window"],
        "innovation=",
        f"{r['innovation']:.8f}",
        "z=",
        f"{r['robust_z_against_frozen_baseline']:.3f}",
    )

print()
print("Evidence:", OUT)
