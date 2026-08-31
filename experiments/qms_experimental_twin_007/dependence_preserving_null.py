from pathlib import Path
import json
import numpy as np


SRC = Path(
    "experiments/qms_experimental_twin_006/"
    "evidence/qms_experimental_twin_006_nonoverlap.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_007/"
    "evidence/qms_experimental_twin_007_dependence_null.json"
)

data = json.loads(SRC.read_text())

# Recover the retained non-overlapping innovation sequence
src3 = Path(
    "experiments/qms_experimental_twin_003/"
    "evidence/qms_experimental_twin_003_predictive.json"
)

rows = json.loads(src3.read_text())["results"]

WINDOW = 100

selected = []
last = None

for r in rows:
    w = int(r["window_end_increment"])

    if last is None or (w - last) >= WINDOW:
        selected.append(r)
        last = w


innov = np.asarray(
    [float(r["innovation_norm"]) for r in selected],
    dtype=float,
)

windows = np.asarray(
    [int(r["window_end_increment"]) for r in selected],
    dtype=int,
)


# ------------------------------------------------------------
# Observed statistics
# ------------------------------------------------------------

median = float(np.median(innov))
mad = float(
    np.median(
        np.abs(innov - median)
    )
)

if mad == 0:
    raise RuntimeError("MAD is zero; robust normalization unavailable.")

robust_z = (
    innov - median
) / (1.4826 * mad)

observed_max_z = float(
    np.max(robust_z)
)

observed_top2_sum = float(
    np.sort(robust_z)[-2:].sum()
)


# ------------------------------------------------------------
# Dependence-preserving null
#
# Circularly shift the sequence relative to a slowly varying
# local baseline. This retains the ordering and autocorrelation
# structure of the innovations.
# ------------------------------------------------------------

N = len(innov)
rng = np.random.default_rng(20260829)

TRIALS = 10000


def local_baseline(x, radius=3):
    out = np.empty_like(x)

    for i in range(len(x)):
        vals = []
        for j in range(i - radius, i + radius + 1):
            vals.append(
                x[j % len(x)]
            )

        out[i] = np.median(vals)

    return out


baseline = local_baseline(
    innov,
    radius=3
)

residual = innov - baseline


null_max = np.zeros(TRIALS)
null_top2 = np.zeros(TRIALS)


for t in range(TRIALS):

    shift = int(
        rng.integers(
            1,
            N
        )
    )

    shifted = np.roll(
        residual,
        shift
    )

    synthetic = (
        baseline + shifted
    )

    med = np.median(synthetic)

    md = np.median(
        np.abs(
            synthetic - med
        )
    )

    if md == 0:
        continue

    z = (
        synthetic - med
    ) / (1.4826 * md)

    null_max[t] = np.max(z)

    null_top2[t] = (
        np.sort(z)[-2:].sum()
    )


p_max = (
    np.sum(
        null_max >= observed_max_z
    ) + 1
) / (TRIALS + 1)

p_top2 = (
    np.sum(
        null_top2 >= observed_top2_sum
    ) + 1
) / (TRIALS + 1)


top_idx = np.argsort(
    innov
)[::-1][:10]


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-007",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "samples":
        int(N),

    "spacing":
        ">=100 detector increments",

    "lag1_correlation":
        float(
            np.corrcoef(
                innov[:-1],
                innov[1:]
            )[0, 1]
        ),

    "robust_statistics": {
        "median":
            median,

        "mad":
            mad,

        "observed_max_robust_z":
            observed_max_z,

        "observed_top2_robust_z_sum":
            observed_top2_sum,
    },

    "null": {
        "type":
            "circular-shift residual null preserving temporal ordering",

        "trials":
            TRIALS,

        "p_max":
            float(p_max),

        "p_top2":
            float(p_top2),
    },

    "largest_nonoverlap_innovations": [
        {
            "window":
                int(windows[i]),

            "innovation":
                float(innov[i]),

            "robust_z":
                float(robust_z[i]),
        }
        for i in top_idx
    ],

    "scientific_boundary": (
        "This analysis tests whether the largest innovations "
        "remain unusual under a dependence-preserving null. "
        "It does not identify their physical cause and does not "
        "establish detector degradation, failure, or causality."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-007 ===")
print()

print("Non-overlap samples:", N)
print(
    "Lag-1 correlation:",
    f"{summary['lag1_correlation']:.6f}"
)

print()

print(
    "Observed max robust-z:",
    f"{observed_max_z:.4f}"
)

print(
    "Observed top-2 z sum:",
    f"{observed_top2_sum:.4f}"
)

print()

print(
    "Dependence-null p(max):",
    f"{p_max:.6f}"
)

print(
    "Dependence-null p(top2):",
    f"{p_top2:.6f}"
)

print()
print("Largest independent-spacing innovations:")

for i in top_idx:
    print(
        " window=",
        windows[i],
        "innovation=",
        f"{innov[i]:.8f}",
        "robust_z=",
        f"{robust_z[i]:.3f}",
    )

print()
print("Evidence:", OUT)
