import json
from pathlib import Path

import numpy as np

from qms_core.representation import (
    effective_dimension,
    explained_variance_fraction,
    observable_sensitivity,
)


def make_state_samples(n=200, noise=0.0, seed=42):
    rng = np.random.default_rng(seed)

    controls = np.linspace(0.0, 1.0, n)

    # Controlled synthetic measurement representation.
    # Signal lives primarily in two dimensions, with optional noise
    # added across all four measurement channels.
    measurements = np.column_stack([
        controls,
        1.0 - controls,
        controls ** 2,
        np.sin(np.pi * controls),
    ])

    if noise > 0:
        measurements += rng.normal(
            0.0,
            noise,
            size=measurements.shape,
        )

    return controls, measurements


def main():
    results = []

    for noise in [0.0, 0.01, 0.05, 0.10, 0.20]:
        control, measurements = make_state_samples(
            noise=noise
        )

        eff_dim = effective_dimension(measurements)
        evf2 = explained_variance_fraction(
            measurements,
            2,
        )

        # Candidate scalar observables
        energy = np.mean(measurements ** 2, axis=1)
        variance = np.var(measurements, axis=1)

        energy_sensitivity = observable_sensitivity(
            energy,
            control,
        )

        variance_sensitivity = observable_sensitivity(
            variance,
            control,
        )

        row = {
            "noise": noise,
            "effective_dimension": eff_dim,
            "variance_explained_first_2": evf2,
            "energy_correlation": energy_sensitivity[
                "correlation"
            ],
            "energy_slope": energy_sensitivity[
                "slope"
            ],
            "variance_correlation": variance_sensitivity[
                "correlation"
            ],
            "variance_slope": variance_sensitivity[
                "slope"
            ],
        }

        results.append(row)

    outdir = Path(
        "experiments/qms_rep_001/evidence"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / "qms_rep_001_results.json"

    outfile.write_text(
        json.dumps(results, indent=2)
    )

    for row in results:
        print(row)

    print()
    print("Saved:", outfile)


if __name__ == "__main__":
    main()
