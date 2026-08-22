from __future__ import annotations

import json
from pathlib import Path


OUTDIR = Path(
    "experiments/qms_real_024/evidence"
)

OUTFILE = (
    OUTDIR
    / "qms_real_024_evidence_registry.json"
)


def main():
    registry = {
        "experiment":
            "QMS-REAL-024",

        "purpose":
            (
                "Consolidate QMS real-measurement evidence "
                "and separate validated findings, transfer "
                "findings, negative results, provisional "
                "heuristics, and unsupported claims."
            ),

        "evidence_classes": {

            "validated_findings": [

                {
                    "id":
                        "REAL-001",

                    "finding":
                        (
                            "On six ISTA all-optical "
                            "superconducting-qubit readout "
                            "conditions, raw-IQ representation "
                            "diagnostics tracked reported QND "
                            "fidelity degradation monotonically."
                        ),

                    "scope":
                        (
                            "Single experimental system; "
                            "six operating conditions."
                        ),

                    "strength":
                        "real experimental convergent validation",
                },

                {
                    "id":
                        "REAL-002",

                    "finding":
                        (
                            "Bootstrap analysis showed that "
                            "the principal labeled IQ diagnostics "
                            "were stable to shot-level resampling."
                        ),

                    "scope":
                        "ISTA IQ dataset",

                    "strength":
                        "bootstrap robustness",
                },

                {
                    "id":
                        "REAL-003",

                    "finding":
                        (
                            "Several label-free IQ representation "
                            "metrics tracked reported QND fidelity, "
                            "while other candidate statistics such "
                            "as covariance trace and entropy were "
                            "weak."
                        ),

                    "scope":
                        "ISTA IQ dataset",

                    "strength":
                        "real-data metric screening",
                },

                {
                    "id":
                        "REAL-004",

                    "finding":
                        (
                            "Selected unsupervised representation "
                            "metrics remained stable under bootstrap "
                            "resampling."
                        ),

                    "scope":
                        "ISTA IQ dataset",

                    "strength":
                        "bootstrap robustness",
                },

                {
                    "id":
                        "REAL-006",

                    "finding":
                        (
                            "The same label-free representation "
                            "diagnostic architecture operated on "
                            "raw Glasgow sparse single-photon "
                            "detector frames."
                        ),

                    "scope":
                        (
                            "Glasgow Heralded Diffraction SM; "
                            "4070 raw ASC frames."
                        ),

                    "strength":
                        "cross-hardware applicability",
                },

                {
                    "id":
                        "REAL-007",

                    "finding":
                        (
                            "The Glasgow acquisition showed a "
                            "structured convergence trajectory "
                            "toward its mature detector "
                            "distribution followed by a small "
                            "late-stage redistribution."
                        ),

                    "scope":
                        "retrospective distribution reference",

                    "strength":
                        "real-data temporal characterization",
                },
            ],

            "transfer_findings": [

                {
                    "id":
                        "REAL-017",

                    "finding":
                        (
                            "Fixed QMS representation and "
                            "convergence diagnostics transferred "
                            "without metric changes from Heralded "
                            "Diffraction SM to Ghost Diffraction SM."
                        ),

                    "scope":
                        (
                            "Two related Glasgow diffraction "
                            "acquisitions."
                        ),
                },

                {
                    "id":
                        "REAL-018",

                    "finding":
                        (
                            "Distribution-level convergence "
                            "trajectories were highly reproducible "
                            "between Heralded and Ghost Diffraction "
                            "SM, with cosine and JS trajectory "
                            "correlations approximately 0.997."
                        ),

                    "scope":
                        "two Glasgow diffraction acquisitions",

                    "important_boundary":
                        (
                            "Detailed representation-geometry "
                            "trajectories were much less "
                            "correlated and appear condition "
                            "dependent."
                        ),
                },

                {
                    "id":
                        "REAL-019",

                    "finding":
                        (
                            "An unchanged convergence-state "
                            "classifier reproduced the same "
                            "converging -> stable/mixed -> "
                            "drifting sequence in Ghost "
                            "Diffraction SM."
                        ),

                    "scope":
                        (
                            "Classifier transferred without "
                            "retuning from Heralded Diffraction SM."
                        ),
                },

                {
                    "id":
                        "REAL-020",

                    "finding":
                        (
                            "The same fixed convergence and state "
                            "logic remained meaningful after "
                            "transfer from diffraction to Heralded "
                            "Imaging MM."
                        ),

                    "scope":
                        (
                            "1000 raw frames; only 10 fixed "
                            "100-frame windows."
                        ),
                },

                {
                    "id":
                        "REAL-022",

                    "finding":
                        (
                            "The same unchanged classifier also "
                            "transferred to Ghost Imaging MM."
                        ),

                    "scope":
                        (
                            "1000 raw frames; 10 windows."
                        ),
                },

                {
                    "id":
                        "REAL-023",

                    "finding":
                        (
                            "Across four Glasgow acquisitions "
                            "spanning heralded/ghost and "
                            "diffraction/imaging conditions, "
                            "the unchanged state logic "
                            "consistently identified convergence "
                            "followed by late-acquisition "
                            "departure."
                        ),

                    "quantitative_summary": {
                        "n_datasets":
                            4,

                        "normalized_drift_position_mean":
                            0.812289312039312,

                        "normalized_drift_position_std":
                            0.041974989316435465,

                        "normalized_drift_position_min":
                            0.7495085995085995,

                        "normalized_drift_position_max":
                            0.8505,

                        "diffraction_mean":
                            0.774078624078624,

                        "imaging_mean":
                            0.8505,
                    },

                    "scope":
                        (
                            "Cross-condition transfer within one "
                            "Glasgow experimental platform."
                        ),
                },
            ],

            "negative_results": [

                {
                    "id":
                        "REAL-012",

                    "finding":
                        (
                            "A causal rolling-reference detector "
                            "produced no warning or drift alerts."
                        ),

                    "interpretation":
                        (
                            "Adaptive rolling references can "
                            "absorb gradual redistribution."
                        ),
                },

                {
                    "id":
                        "REAL-013",

                    "finding":
                        (
                            "A frozen early reference generated "
                            "many premature alerts."
                        ),

                    "interpretation":
                        (
                            "The first acquisition windows were "
                            "not a valid stationary baseline."
                        ),
                },

                {
                    "id":
                        "REAL-014",

                    "finding":
                        (
                            "Automatic stabilization-based "
                            "baseline locking improved baseline "
                            "selection but still generated an "
                            "early drift alert shortly after "
                            "locking."
                        ),

                    "interpretation":
                        (
                            "Baseline departure is not equivalent "
                            "to a regime-change drift event."
                        ),
                },

                {
                    "id":
                        "REAL-015",

                    "finding":
                        (
                            "Persistence requirements delayed "
                            "the alert but still detected sustained "
                            "baseline departure earlier than the "
                            "retrospective transition region."
                        ),
                },

                {
                    "id":
                        "REAL-016",

                    "finding":
                        (
                            "Causal JS-slope acceleration logic "
                            "produced no warning or drift alert."
                        ),

                    "interpretation":
                        (
                            "The late redistribution was not "
                            "characterized by a sufficiently sharp "
                            "causal acceleration under the tested "
                            "formulation."
                        ),
                },

                {
                    "id":
                        "REAL-011",

                    "finding":
                        (
                            "Global retrospective change-point "
                            "analysis primarily detected early "
                            "convergence/saturation structure "
                            "rather than a specific precursor to "
                            "late drift."
                        ),

                    "interpretation":
                        (
                            "This does not validate predictive "
                            "early warning."
                        ),
                },
            ],

            "provisional_heuristics": [

                {
                    "id":
                        "REAL-005",

                    "name":
                        "Measurement Degradation Index",

                    "status":
                        "provisional",

                    "reason":
                        (
                            "Reference-specific composite score "
                            "derived from ISTA metrics."
                        ),

                    "boundary":
                        (
                            "Not established as a universal "
                            "measurement-health score."
                        ),
                },

                {
                    "id":
                        "REAL-008",

                    "name":
                        "Measurement-state classifier",

                    "rule": {
                        "converging":
                            (
                                "delta smoothed cosine > 0 "
                                "and delta smoothed JS < 0"
                            ),

                        "drifting":
                            (
                                "delta smoothed cosine < 0 "
                                "and delta smoothed JS > 0"
                            ),

                        "stable_or_mixed":
                            "otherwise",
                    },

                    "status":
                        (
                            "empirically transferable within "
                            "tested Glasgow conditions"
                        ),

                    "boundary":
                        (
                            "Operational state descriptor, "
                            "not calibrated hardware-failure "
                            "classification."
                        ),
                },
            ],

            "unsupported_or_not_yet_established": [

                {
                    "claim":
                        (
                            "QMS provides universal detector "
                            "health monitoring."
                        ),

                    "status":
                        "not established",
                },

                {
                    "claim":
                        (
                            "A normalized acquisition position "
                            "near 0.8 is a universal drift "
                            "threshold."
                        ),

                    "status":
                        "not established",
                },

                {
                    "claim":
                        (
                            "The Glasgow late-stage state is "
                            "confirmed hardware degradation."
                        ),

                    "status":
                        "not established",
                },

                {
                    "claim":
                        (
                            "QMS has validated causal early "
                            "warning of detector drift."
                        ),

                    "status":
                        "not established",
                },

                {
                    "claim":
                        (
                            "Fine representation geometry is "
                            "universal across measurement "
                            "conditions."
                        ),

                    "status":
                        "contradicted by current evidence",
                },

                {
                    "claim":
                        (
                            "Cross-laboratory transfer of the "
                            "Glasgow state classifier has been "
                            "validated."
                        ),

                    "status":
                        "not yet tested",
                },
            ],
        },

        "current_defensible_claims": [

            (
                "QMS supports label-free representation "
                "diagnostics across at least two distinct "
                "real quantum measurement modalities: "
                "superconducting-qubit IQ and spatial "
                "single-photon detection."
            ),

            (
                "Distribution-level convergence diagnostics "
                "show stronger cross-condition transfer than "
                "fine representation-geometry trajectories."
            ),

            (
                "Across four real Glasgow acquisitions spanning "
                "diffraction and imaging, unchanged QMS state "
                "logic consistently identifies convergence "
                "followed by late-acquisition departure."
            ),

            (
                "Current Glasgow evidence supports "
                "cross-condition transfer on one experimental "
                "platform, not universal detector behavior."
            ),

            (
                "Current causal experiments do not establish "
                "prospective early-warning capability."
            ),
        ],

        "architecture_implication": [
            "representation diagnostics",
            "convergence/reference diagnostics",
            "operational state classification",
            "observability diagnostics",
            "reconstruction assurance",
            "error decomposition",
            "validated health interpretation",
        ],

        "recommended_next_validation":
            (
                "Apply the frozen diagnostic and state logic "
                "to an additional independent acquisition family "
                "or, preferably, a different experimental "
                "platform/laboratory without retuning."
            ),
    }

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTFILE.write_text(
        json.dumps(
            registry,
            indent=2,
        )
    )

    print("=== QMS-REAL-024 EVIDENCE REGISTRY ===")
    print()
    print(
        "Validated findings:",
        len(
            registry[
                "evidence_classes"
            ][
                "validated_findings"
            ]
        ),
    )

    print(
        "Transfer findings:",
        len(
            registry[
                "evidence_classes"
            ][
                "transfer_findings"
            ]
        ),
    )

    print(
        "Negative results:",
        len(
            registry[
                "evidence_classes"
            ][
                "negative_results"
            ]
        ),
    )

    print(
        "Provisional heuristics:",
        len(
            registry[
                "evidence_classes"
            ][
                "provisional_heuristics"
            ]
        ),
    )

    print(
        "Unsupported/not established:",
        len(
            registry[
                "evidence_classes"
            ][
                "unsupported_or_not_yet_established"
            ]
        ),
    )

    print()
    print("Current defensible claims:")

    for i, claim in enumerate(
        registry["current_defensible_claims"],
        start=1,
    ):
        print(f"{i}. {claim}")

    print()
    print("Saved:", OUTFILE)


if __name__ == "__main__":
    main()
