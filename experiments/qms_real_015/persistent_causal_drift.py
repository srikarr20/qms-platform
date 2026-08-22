from __future__ import annotations

import argparse
import json
from pathlib import Path


WARNING_Z = 3.0
ALERT_Z = 5.0

WARNING_PERSISTENCE = 2
ALERT_PERSISTENCE = 3


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    raw = json.loads(
        args.input_json.read_text()
    )

    rows = raw["windows"]

    warning_run = 0
    alert_run = 0

    results = []

    first_warning = None
    first_alert = None

    for row in rows:

        result = dict(row)

        if row.get("phase") != "locked_monitoring":
            result["persistent_state"] = row.get(
                "state",
                "prelock",
            )

            results.append(result)
            continue

        z = float(
            row.get(
                "robust_js_z",
                0.0,
            )
        )

        if z >= WARNING_Z:
            warning_run += 1
        else:
            warning_run = 0

        if z >= ALERT_Z:
            alert_run += 1
        else:
            alert_run = 0

        if alert_run >= ALERT_PERSISTENCE:
            state = "drift_alert"

        elif warning_run >= WARNING_PERSISTENCE:
            state = "warning"

        else:
            state = "nominal"

        result[
            "warning_run"
        ] = warning_run

        result[
            "alert_run"
        ] = alert_run

        result[
            "persistent_state"
        ] = state

        if (
            state == "warning"
            and first_warning is None
        ):
            first_warning = {
                "window_index":
                    row["window_index"],
                "frame_start":
                    row["frame_start"],
                "frame_end":
                    row["frame_end"],
                "robust_js_z":
                    z,
            }

        if (
            state == "drift_alert"
            and first_alert is None
        ):
            first_alert = {
                "window_index":
                    row["window_index"],
                "frame_start":
                    row["frame_start"],
                "frame_end":
                    row["frame_end"],
                "robust_js_z":
                    z,
            }

        results.append(result)

    warning_windows = [
        r for r in results
        if r.get("persistent_state")
        == "warning"
    ]

    alert_windows = [
        r for r in results
        if r.get("persistent_state")
        == "drift_alert"
    ]

    summary = {
        "warning_z_threshold":
            WARNING_Z,

        "alert_z_threshold":
            ALERT_Z,

        "warning_persistence_windows":
            WARNING_PERSISTENCE,

        "alert_persistence_windows":
            ALERT_PERSISTENCE,

        "baseline_lock_frame":
            raw["summary"][
                "baseline_lock_frame"
            ],

        "first_persistent_warning":
            first_warning,

        "first_persistent_drift_alert":
            first_alert,

        "n_persistent_warning_windows":
            len(warning_windows),

        "n_persistent_alert_windows":
            len(alert_windows),

        "comparison_reference": {
            "retrospective_robust_drift_frame":
                3034,
            "note":
                (
                    "3034 is used only for post-hoc comparison. "
                    "It is not used by the causal detector."
                ),
        },

        "interpretation": (
            "This experiment tests whether requiring sustained "
            "distribution departure suppresses transient or "
            "premature alerts while preserving causal detection."
        ),
    }

    output = {
        "experiment":
            "QMS-REAL-015",
        "dataset":
            raw["dataset"],
        "method":
            (
                "persistent causal drift detection "
                "using QMS-REAL-014 locked baseline"
            ),
        "windows":
            results,
        "summary":
            summary,
        "caution":
            (
                "Persistence parameters are provisional. "
                "Agreement with the retrospective drift region "
                "would support, but not prove, prospective "
                "drift-detection validity."
            ),
    }

    outdir = Path(
        "experiments/qms_real_015/evidence"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = (
        outdir /
        "qms_real_015_persistent_drift.json"
    )

    outfile.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print("=== SUMMARY ===")
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("Saved:", outfile)


if __name__ == "__main__":
    main()
