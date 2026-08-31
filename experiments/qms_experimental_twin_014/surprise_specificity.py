from pathlib import Path
import json
import numpy as np


SRC = Path(
    "experiments/qms_experimental_twin_013/"
    "evidence/qms_experimental_twin_013_innovation_geometry.json"
)

OUT = Path(
    "experiments/qms_experimental_twin_014/"
    "evidence/qms_experimental_twin_014_surprise_specificity.json"
)

data = json.loads(SRC.read_text())
rows = data["results"]

innovation = np.asarray(
    [float(r["innovation_norm"]) for r in rows],
    dtype=float,
)

transition = np.asarray(
    [float(r["transition_l2"]) for r in rows],
    dtype=float,
)

windows = np.asarray(
    [int(r["window_end"]) for r in rows],
    dtype=int,
)


# ------------------------------------------------------------
# Fit simple linear relationship:
#
# expected innovation given transition magnitude
# ------------------------------------------------------------

X = np.column_stack([
    np.ones(len(transition)),
    transition,
])

beta, *_ = np.linalg.lstsq(
    X,
    innovation,
    rcond=None,
)

predicted = X @ beta

residual = (
    innovation - predicted
)


# ------------------------------------------------------------
# Robust residual surprise
# ------------------------------------------------------------

median = float(
    np.median(residual)
)

mad = float(
    np.median(
        np.abs(residual - median)
    )
)

if mad == 0:
    raise RuntimeError(
        "Residual MAD is zero."
    )

robust_surprise = (
    residual - median
) / (1.4826 * mad)


# ------------------------------------------------------------
# Rank measurement states by surprise beyond transition size
# ------------------------------------------------------------

order = np.argsort(
    robust_surprise
)[::-1]

ranked = []

for rank, i in enumerate(order, start=1):

    percentile = float(
        100.0
        * np.mean(
            robust_surprise
            <= robust_surprise[i]
        )
    )

    ranked.append({
        "rank":
            rank,

        "window_end":
            int(windows[i]),

        "innovation_norm":
            float(innovation[i]),

        "transition_l2":
            float(transition[i]),

        "expected_innovation_from_transition":
            float(predicted[i]),

        "innovation_residual":
            float(residual[i]),

        "robust_surprise":
            float(robust_surprise[i]),

        "percentile":
            percentile,
    })


# ------------------------------------------------------------
# Leave-one-out check:
#
# Prevent 701/801 themselves from determining their expected
# innovation relationship.
# ------------------------------------------------------------

loo = []

for target_index in range(len(rows)):

    mask = np.ones(
        len(rows),
        dtype=bool,
    )

    mask[target_index] = False

    X_train = X[mask]
    y_train = innovation[mask]

    b, *_ = np.linalg.lstsq(
        X_train,
        y_train,
        rcond=None,
    )

    expected = float(
        X[target_index] @ b
    )

    error = float(
        innovation[target_index]
        - expected
    )

    train_residual = (
        y_train
        - X_train @ b
    )

    med = np.median(
        train_residual
    )

    md = np.median(
        np.abs(
            train_residual - med
        )
    )

    if md > 0:
        z = float(
            (error - med)
            /
            (1.4826 * md)
        )
    else:
        z = None

    loo.append({
        "window_end":
            int(windows[target_index]),

        "observed_innovation":
            float(innovation[target_index]),

        "loo_expected_innovation":
            expected,

        "loo_residual":
            error,

        "loo_robust_surprise":
            z,
    })


target_results = {
    str(target): next(
        r for r in loo
        if r["window_end"] == target
    )
    for target in (701, 801)
}


summary = {
    "experiment":
        "QMS-EXPERIMENTAL-TWIN-014",

    "dataset":
        "Glasgow Heralded Diffraction SM",

    "samples":
        len(rows),

    "conditioning_metric":
        "transition_l2",

    "linear_model": {
        "intercept":
            float(beta[0]),

        "slope":
            float(beta[1]),
    },

    "target_leave_one_out":
        target_results,

    "top_surprises":
        ranked[:10],

    "scientific_boundary": (
        "This analysis tests whether predictive innovation "
        "remains unusual after accounting for ordinary "
        "measurement-transition magnitude. Surprise means "
        "unexpected relative to this simple empirical model; "
        "it does not identify physical cause, detector failure, "
        "degradation, or causality."
    ),
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2
    ) + "\n"
)


print()
print("=== QMS-EXPERIMENTAL-TWIN-014 ===")
print()

print(
    "Linear model:"
)

print(
    " innovation ≈",
    f"{beta[0]:.8f}",
    "+",
    f"{beta[1]:.6f}",
    "* transition_L2"
)

print()

print("701/801 leave-one-out surprise:")

for target in (701, 801):

    r = target_results[str(target)]

    print(
        " window=",
        target,
        "observed=",
        f"{r['observed_innovation']:.8f}",
        "expected=",
        f"{r['loo_expected_innovation']:.8f}",
        "residual=",
        f"{r['loo_residual']:.8f}",
        "robust-z=",
        "NA"
        if r["loo_robust_surprise"] is None
        else f"{r['loo_robust_surprise']:.3f}",
    )


print()
print("Top surprise states:")

for r in ranked[:10]:

    print(
        " window=",
        r["window_end"],
        "innovation=",
        f"{r['innovation_norm']:.8f}",
        "transition=",
        f"{r['transition_l2']:.6f}",
        "expected=",
        f"{r['expected_innovation_from_transition']:.8f}",
        "surprise=",
        f"{r['robust_surprise']:.3f}",
        "percentile=",
        f"{r['percentile']:.2f}%"
    )

print()
print("Evidence:", OUT)
