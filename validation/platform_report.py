import json
from pathlib import Path

from validation.evidence import (
    PlatformEvidenceReport,
)

from validation.glasgow import (
    GlasgowValidationSuite,
)


class QMSPlatformValidationReport:
    """
    Consolidated scientific evidence report.

    Separates:
        architecture validation
        synthetic inverse-reconstruction validation
        real experimental validation
    """

    def __init__(
        self,
        glasgow_zip=None,
        artifacts_dir="artifacts",
    ):
        self.glasgow_zip = glasgow_zip
        self.artifacts_dir = artifacts_dir


    def build(self):
        report = PlatformEvidenceReport()


        # ====================================================
        # ARCHITECTURE
        # ====================================================

        report.add(
            name="Unified measurement-twin runtime",

            status="DEMONSTRATED",

            evidence_type="architecture",

            basis=(
                "MeasurementTwinPlatform successfully executes "
                "measurement ingestion, reconstruction, "
                "observability, dynamics, and persistence."
            ),

            claim=(
                "A common runtime can support multiple "
                "measurement modalities through adapters."
            ),

            limitation=(
                "Architecture correctness does not establish "
                "physical validity for every modality."
            ),
        )


        report.add(
            name="Optical raw-ingest runtime",

            status="DEMONSTRATED",

            evidence_type="synthetic",

            basis=(
                "Raw four-quadrature detector arrays were "
                "ingested through the public platform API."
            ),

            metrics={
                "recovered_depth_m":
                    0.073,

                "dynamics_ready":
                    True,
            },

            claim=(
                "Optical quadrature measurements can flow "
                "through DPI reconstruction and shared "
                "dynamics automatically."
            ),

            limitation=(
                "Validated with controlled matched synthetic "
                "measurements."
            ),
        )


        report.add(
            name="MRI raw k-space runtime",

            status="DEMONSTRATED",

            evidence_type="synthetic",

            basis=(
                "Single-coil Cartesian k-space was ingested, "
                "inverse reconstructed, and passed through the "
                "shared observability/AURORA stack."
            ),

            metrics={
                "normalized_complex_error":
                    2.830444456425e-16,

                "manifold_shape":
                    [11, 3],

                "dynamics_ready":
                    True,
            },

            claim=(
                "Raw Cartesian MRI k-space can enter the same "
                "measurement-twin architecture."
            ),

            limitation=(
                "Current MRI reconstruction is a matched "
                "single-coil Cartesian inverse FFT, not a "
                "general scanner reconstruction stack."
            ),
        )


        # ====================================================
        # DPI SYNTHETIC INVERSE RECONSTRUCTION
        # ====================================================

        report.add(
            name="Matched DPI upstream reconstruction",

            status="DEMONSTRATED",

            evidence_type="synthetic",

            basis=(
                "Four-quadrature synthetic measurements were "
                "inverted through matched propagation physics."
            ),

            metrics={
                "dpi_4q_mean_complex_error":
                    0.04578757,

                "dpi_4q_phase_error_rad":
                    0.00443363,

                "centroid_error_microns":
                    0.110701,

                "separation_error_microns":
                    3.142758,
            },

            claim=(
                "DPI reconstructs upstream source structure "
                "accurately when the measurement model is "
                "matched and phase information is available."
            ),

            limitation=(
                "This is synthetic matched-model evidence, "
                "not yet real experimental upstream recovery."
            ),
        )


        report.add(
            name="Blind source depth inference",

            status="DEMONSTRATED",

            evidence_type="synthetic",

            basis=(
                "Virtual propagation search inferred unknown "
                "source depth without being given the true z."
            ),

            metrics={
                "noiseless_depth_mae_mm":
                    0.000069,

                "depth_mae_at_4pct_noise_mm":
                    1.056474,

                "within_1mm_at_4pct_noise_percent":
                    60.0,
            },

            claim=(
                "Source depth can be inferred from reconstructed "
                "field structure under the tested model."
            ),

            limitation=(
                "Depth is degenerate with wavelength/model "
                "parameters unless additional measurement "
                "information is supplied."
            ),
        )


        report.add(
            name="Active observability",

            status="DEMONSTRATED",

            evidence_type="synthetic",

            basis=(
                "The twin evaluated candidate wavelengths and "
                "selected a second measurement that improved "
                "depth inference."
            ),

            metrics={
                "selected_wavelength_nm":
                    850.0,

                "single_measurement_mae_mm":
                    0.093470,

                "active_mae_mm":
                    0.060774,

                "mae_improvement_percent":
                    34.98,
            },

            claim=(
                "The twin can use observability analysis to "
                "select a useful additional measurement."
            ),

            limitation=(
                "Selection currently uses a heuristic curvature "
                "criterion rather than calibrated expected "
                "information gain."
            ),
        )


        # ====================================================
        # REAL GLASGOW DATA
        # ====================================================

        if self.glasgow_zip is not None:

            suite = GlasgowValidationSuite(
                self.glasgow_zip,
                artifacts_dir=
                    self.artifacts_dir,
            )

            glasgow = suite.run_all()

            dataset = glasgow["dataset"]

            report.add(
                name="Glasgow real detector ingestion",

                status="DEMONSTRATED",

                evidence_type="real_data",

                basis=(
                    "Full experimental cumulative detector "
                    "archive was converted into acquisition "
                    "increments."
                ),

                metrics={
                    "detector_shape":
                        dataset[
                            "detector_shape"
                        ],

                    "acquisition_increments":
                        dataset[
                            "acquisition_increments"
                        ],

                    "total_event_weight":
                        dataset[
                            "total_event_weight"
                        ],
                },

                claim=(
                    "The platform can process the full Glasgow "
                    "experimental detector sequence."
                ),

                limitation=(
                    "The dataset contains sparse detector-count "
                    "information rather than quadrature complex "
                    "field measurements."
                ),
            )


            heldout = glasgow[
                "heldout_prediction"
            ]

            report.add(
                name="Real held-out spatial prediction",

                status="SUPPORTED",

                evidence_type="real_data",

                basis=(
                    "Spatial model trained on earlier events "
                    "was evaluated on untouched future events."
                ),

                metrics={
                    "information_gain_bits_per_event":
                        heldout[
                            "information_gain_bits_per_event"
                        ],

                    "training_events":
                        heldout[
                            "training_pool"
                        ],

                    "heldout_events":
                        heldout[
                            "heldout_events"
                        ],
                },

                claim=(
                    "The experimental detector sequence contains "
                    "predictive spatial structure."
                ),

                limitation=(
                    "Predictability is spatial; it does not "
                    "establish upstream complex-field recovery."
                ),
            )


            joint = glasgow[
                "joint_spatial_information"
            ]

            report.add(
                name="Real joint X-Y information",

                status="SUPPORTED",

                evidence_type="real_data",

                basis=(
                    "Joint spatial modeling outperformed "
                    "factorized X/Y modeling on untouched data."
                ),

                metrics={
                    "joint_vs_factorized_bits_per_event":
                        joint[
                            "joint_vs_factorized_bits_per_event"
                        ],

                    "permutation_p":
                        joint[
                            "permutation_p"
                        ],

                    "blocks":
                        joint[
                            "blocks_joint_beats_factorized"
                        ],
                },

                claim=(
                    "The detector distribution contains genuine "
                    "two-dimensional spatial dependence."
                ),

                limitation=(
                    "The measured dependence is detector-plane "
                    "information, not direct proof of a unique "
                    "upstream physical source state."
                ),
            )


            temporal = glasgow[
                "temporal_order_null"
            ]

            report.add(
                name="Glasgow acquisition-order dynamics",

                status="NOT_SUPPORTED",

                evidence_type="real_data",

                basis=(
                    "Observed AURORA/CKE temporal metrics were "
                    "compared with shuffled acquisition-order "
                    "null distributions across 63 windows."
                ),

                metrics={
                    "bonferroni_threshold":
                        temporal[
                            "bonferroni_threshold"
                        ],

                    "drift_significant_windows":
                        temporal["drift"][
                            "bonferroni_significant_windows"
                        ],

                    "distortion_significant_windows":
                        temporal[
                            "attractor_distortion"
                        ][
                            "bonferroni_significant_windows"
                        ],

                    "instability_significant_windows":
                        temporal[
                            "mean_local_instability"
                        ][
                            "bonferroni_significant_windows"
                        ],
                },

                claim=(
                    "Strong acquisition-order-sensitive dynamics "
                    "were not demonstrated by the current "
                    "observable representation."
                ),

                limitation=(
                    "This does not prove the physical system has "
                    "no dynamics; it constrains what these data "
                    "and metrics support."
                ),
            )


            stationarity = glasgow[
                "stationarity_null"
            ]

            report.add(
                name="Glasgow slow drift",

                status="NOT_SUPPORTED",

                evidence_type="real_data",

                basis=(
                    "Centroid, width, and JS variation were "
                    "compared with stationarity nulls."
                ),

                metrics={
                    "x_range_p":
                        stationarity[
                            "x_range_p"
                        ],

                    "y_range_p":
                        stationarity[
                            "y_range_p"
                        ],

                    "width_range_p":
                        stationarity[
                            "width_range_p"
                        ],

                    "max_js_adj_p":
                        stationarity[
                            "max_js_adj_p"
                        ],
                },

                claim=(
                    "Slow detector-plane drift beyond sampling "
                    "variation was not demonstrated."
                ),

                limitation=(
                    "The test is specific to the spatial "
                    "statistics and null model used."
                ),
            )


            report.add(
                name="Glasgow upstream complex-field reconstruction",

                status="NOT_AVAILABLE",

                evidence_type="real_data",

                basis=(
                    "The Glasgow archive supplies cumulative "
                    "detector counts/event locations but does "
                    "not provide the phase/quadrature information "
                    "used by the current DPI complex inversion."
                ),

                claim=(
                    "No real-data DPI upstream reconstruction "
                    "claim is made from this dataset."
                ),

                limitation=(
                    "A phase-sensitive experimental dataset or "
                    "a different validated inverse measurement "
                    "model is required."
                ),
            )


        return report


    def save(
        self,
        path,
    ):
        report = self.build()

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "summary":
                report.summary(),

            **report.to_dict(),
        }

        output.write_text(
            json.dumps(
                payload,
                indent=2,
            )
        )

        return output
