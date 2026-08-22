from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("experiments")

OUTDIR = Path(
    "experiments/qms_real_025/evidence"
)

OUTFILE = (
    OUTDIR
    / "qms_real_025_reproducibility_manifest.json"
)


EXPERIMENTS = [
    {
        "id": "QMS-REAL-001",
        "directory": "qms_real_001",
        "script": "ista_iq_representation.py",
        "evidence": "evidence/qms_real_001_results.json",
        "dataset":
            "ISTA AllOpticalSCQreadout_data / Fig_4a IQ blobs",
        "source_type":
            "real superconducting-qubit IQ measurements",
        "evidence_class":
            "validated_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-002",
        "directory": "qms_real_002",
        "script": "bootstrap_iq_diagnostics.py",
        "evidence": "evidence/qms_real_002_bootstrap.json",
        "dataset":
            "ISTA AllOpticalSCQreadout_data / Fig_4a IQ blobs",
        "source_type":
            "real superconducting-qubit IQ measurements",
        "evidence_class":
            "validated_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-003",
        "directory": "qms_real_003",
        "script": "unsupervised_iq_health.py",
        "evidence": "evidence/qms_real_003_results.json",
        "dataset":
            "ISTA AllOpticalSCQreadout_data / Fig_4a IQ blobs",
        "source_type":
            "real superconducting-qubit IQ measurements",
        "evidence_class":
            "validated_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-004",
        "directory": "qms_real_004",
        "script": "bootstrap_unsupervised_health.py",
        "evidence": "evidence/qms_real_004_bootstrap.json",
        "dataset":
            "ISTA AllOpticalSCQreadout_data / Fig_4a IQ blobs",
        "source_type":
            "real superconducting-qubit IQ measurements",
        "evidence_class":
            "validated_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-005",
        "directory": "qms_real_005",
        "script": "measurement_health_score.py",
        "evidence": None,
        "dataset":
            "ISTA AllOpticalSCQreadout_data / Fig_4a IQ blobs",
        "source_type":
            "real superconducting-qubit IQ measurements",
        "evidence_class":
            "provisional_heuristic",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-006",
        "directory": "qms_real_006",
        "script": "glasgow_raw_event_geometry.py",
        "evidence": "evidence/qms_real_006_glasgow_raw.json",
        "dataset":
            "Glasgow Heralded Diffraction SM.zip",
        "source_type":
            "raw 512x512 sparse single-photon ASC frames",
        "evidence_class":
            "validated_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-007",
        "directory": "qms_real_007",
        "script": "glasgow_convergence_vs_drift.py",
        "evidence": "evidence/qms_real_007_convergence.json",
        "dataset":
            "Glasgow Heralded Diffraction SM.zip",
        "source_type":
            "raw 512x512 sparse single-photon ASC frames",
        "evidence_class":
            "validated_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-008",
        "directory": "qms_real_008",
        "script": "glasgow_measurement_state.py",
        "evidence":
            "evidence/qms_real_008_state_classification.json",
        "dataset":
            "Glasgow Heralded Diffraction SM.zip",
        "source_type":
            "derived from real raw-detector convergence metrics",
        "evidence_class":
            "provisional_heuristic",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-009",
        "directory": "qms_real_009",
        "script": "state_classification_robustness.py",
        "evidence":
            "evidence/qms_real_009_state_robustness.json",
        "dataset":
            "Glasgow Heralded Diffraction SM.zip",
        "source_type":
            "derived from real raw-detector acquisition",
        "evidence_class":
            "transfer_or_robustness",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-010",
        "directory": "qms_real_010",
        "script": "glasgow_drift_early_warning.py",
        "evidence":
            "evidence/qms_real_010_early_warning.json",
        "dataset":
            "Glasgow Heralded Diffraction SM.zip",
        "source_type":
            "derived from real raw-detector acquisition",
        "evidence_class":
            "exploratory",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-011",
        "directory": "qms_real_011",
        "script": "change_point_lead_time.py",
        "evidence":
            "evidence/qms_real_011_change_points.json",
        "dataset":
            "Glasgow Heralded Diffraction SM.zip",
        "source_type":
            "derived from real raw-detector acquisition",
        "evidence_class":
            "negative_result",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-012",
        "directory": "qms_real_012",
        "script": "glasgow_causal_drift.py",
        "evidence":
            "evidence/qms_real_012_causal_drift.json",
        "dataset":
            "Glasgow Heralded Diffraction SM.zip",
        "source_type":
            "raw detector sequence",
        "evidence_class":
            "negative_result",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-013",
        "directory": "qms_real_013",
        "script": "glasgow_frozen_reference_drift.py",
        "evidence":
            "evidence/qms_real_013_frozen_reference.json",
        "dataset":
            "Glasgow Heralded Diffraction SM.zip",
        "source_type":
            "raw detector sequence",
        "evidence_class":
            "negative_result",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-014",
        "directory": "qms_real_014",
        "script": "glasgow_auto_baseline_lock.py",
        "evidence":
            "evidence/qms_real_014_auto_baseline_lock.json",
        "dataset":
            "Glasgow Heralded Diffraction SM.zip",
        "source_type":
            "raw detector sequence",
        "evidence_class":
            "negative_result",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-015",
        "directory": "qms_real_015",
        "script": "persistent_causal_drift.py",
        "evidence":
            "evidence/qms_real_015_persistent_drift.json",
        "dataset":
            "derived from QMS-REAL-014",
        "source_type":
            "causal post-processing of real detector sequence",
        "evidence_class":
            "negative_result",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-016",
        "directory": "qms_real_016",
        "script": "causal_drift_acceleration.py",
        "evidence":
            "evidence/qms_real_016_drift_acceleration.json",
        "dataset":
            "derived from QMS-REAL-014",
        "source_type":
            "causal post-processing of real detector sequence",
        "evidence_class":
            "negative_result",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-017",
        "directory": "qms_real_017",
        "script": "glasgow_cross_condition_transfer.py",
        "evidence":
            "evidence/qms_real_017_ghost_diffraction_sm.json",
        "dataset":
            "Glasgow Ghost Diffraction SM.zip",
        "source_type":
            "raw 512x512 sparse single-photon ASC frames",
        "evidence_class":
            "transfer_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-018",
        "directory": "qms_real_018",
        "script":
            "cross_condition_trajectory_consistency.py",
        "evidence":
            "evidence/qms_real_018_cross_condition_consistency.json",
        "dataset":
            "Heralded Diffraction SM + Ghost Diffraction SM",
        "source_type":
            "cross-condition comparison",
        "evidence_class":
            "transfer_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-019",
        "directory": "qms_real_019",
        "script": "state_classifier_transfer.py",
        "evidence":
            "evidence/qms_real_019_state_transfer.json",
        "dataset":
            "Glasgow Ghost Diffraction SM.zip",
        "source_type":
            "unchanged classifier transfer",
        "evidence_class":
            "transfer_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-020",
        "directory": "qms_real_020",
        "script":
            "heralded_imaging_state_transfer.py",
        "evidence":
            "evidence/qms_real_020_heralded_imaging_mm.json",
        "dataset":
            "Glasgow Heralded Imaging MM set 1.zip",
        "source_type":
            "raw 512x512 sparse single-photon ASC frames",
        "evidence_class":
            "transfer_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-021",
        "directory": "qms_real_021",
        "script":
            "multidataset_transfer_summary.py",
        "evidence":
            "evidence/qms_real_021_multidataset_summary.json",
        "dataset":
            "three Glasgow acquisitions",
        "source_type":
            "aggregate transfer analysis",
        "evidence_class":
            "transfer_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-022",
        "directory": "qms_real_022",
        "script":
            "ghost_imaging_state_transfer.py",
        "evidence":
            "evidence/qms_real_022_ghost_imaging_mm.json",
        "dataset":
            "Glasgow Ghost Imaging MM set 1.zip",
        "source_type":
            "raw 512x512 sparse single-photon ASC frames",
        "evidence_class":
            "transfer_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-023",
        "directory": "qms_real_023",
        "script":
            "four_dataset_transfer_summary.py",
        "evidence":
            "evidence/qms_real_023_four_dataset_summary.json",
        "dataset":
            "four Glasgow acquisitions",
        "source_type":
            "aggregate transfer analysis",
        "evidence_class":
            "transfer_finding",
        "status":
            "completed",
    },

    {
        "id": "QMS-REAL-024",
        "directory": "qms_real_024",
        "script":
            "build_evidence_registry.py",
        "evidence":
            "evidence/qms_real_024_evidence_registry.json",
        "dataset":
            "QMS real-measurement evidence corpus",
        "source_type":
            "evidence consolidation",
        "evidence_class":
            "evidence_registry",
        "status":
            "completed",
    },
]


def check_experiment(item):
    base = ROOT / item["directory"]

    script_path = (
        base / item["script"]
        if item["script"]
        else None
    )

    evidence_path = (
        base / item["evidence"]
        if item["evidence"]
        else None
    )

    return {
        **item,

        "script_exists":
            (
                script_path.exists()
                if script_path
                else None
            ),

        "evidence_exists":
            (
                evidence_path.exists()
                if evidence_path
                else None
            ),

        "script_path":
            (
                str(script_path)
                if script_path
                else None
            ),

        "evidence_path":
            (
                str(evidence_path)
                if evidence_path
                else None
            ),
    }


def main():
    checked = [
        check_experiment(item)
        for item in EXPERIMENTS
    ]

    missing_scripts = [
        item["id"]
        for item in checked
        if item["script_exists"] is False
    ]

    missing_evidence = [
        item["id"]
        for item in checked
        if item["evidence_exists"] is False
    ]

    by_class = {}

    for item in checked:
        key = item["evidence_class"]

        by_class[key] = (
            by_class.get(key, 0) + 1
        )

    manifest = {
        "manifest":
            "QMS-REAL-025",

        "purpose":
            (
                "Reproducibility and provenance manifest "
                "for QMS real-measurement experiments."
            ),

        "experiment_count":
            len(checked),

        "evidence_class_counts":
            by_class,

        "missing_scripts":
            missing_scripts,

        "missing_evidence":
            missing_evidence,

        "all_declared_scripts_present":
            len(missing_scripts) == 0,

        "all_declared_evidence_present":
            len(missing_evidence) == 0,

        "experiments":
            checked,

        "scientific_scope": {
            "real_modalities": [
                "superconducting-qubit IQ",
                "spatial single-photon detection",
            ],

            "glasgow_families": [
                "diffraction",
                "imaging",
            ],

            "glasgow_conditions": [
                "heralded diffraction SM",
                "ghost diffraction SM",
                "heralded imaging MM set 1",
                "ghost imaging MM set 1",
            ],

            "important_boundary":
                (
                    "Current Glasgow transfer evidence is "
                    "cross-condition within one experimental "
                    "platform, not cross-laboratory validation."
                ),

            "causal_early_warning_status":
                "not established",
        },

        "reproduction_note":
            (
                "Raw external datasets are not duplicated into "
                "this manifest. Each experiment records its "
                "dataset dependency and generated evidence "
                "artifact. Reproduction requires access to the "
                "corresponding source datasets."
            ),
    }

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTFILE.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    print(
        "=== QMS-REAL-025 REPRODUCIBILITY MANIFEST ==="
    )

    print(
        "Experiments:",
        manifest["experiment_count"],
    )

    print(
        "Evidence classes:",
        json.dumps(
            by_class,
            indent=2,
        ),
    )

    print(
        "Missing scripts:",
        missing_scripts,
    )

    print(
        "Missing evidence:",
        missing_evidence,
    )

    print(
        "All declared scripts present:",
        manifest[
            "all_declared_scripts_present"
        ],
    )

    print(
        "All declared evidence present:",
        manifest[
            "all_declared_evidence_present"
        ],
    )

    print()
    print("Saved:", OUTFILE)


if __name__ == "__main__":
    main()
