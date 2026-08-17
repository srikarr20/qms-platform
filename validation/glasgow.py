from pathlib import Path
import numpy as np

from adapters.glasgow_event_adapter import (
    GlasgowCumulativeArchive,
)


class GlasgowValidationSuite:
    """
    Consolidated real-data validation for the Glasgow
    Heralded Diffraction SM detector sequence.

    This suite summarizes already established validation
    categories:

        1. accumulation convergence
        2. held-out spatial prediction
        3. joint X-Y information
        4. temporal-order null
        5. stationarity/null behavior

    Some methods consume previously saved artifacts produced
    by the detailed experiment scripts.
    """

    def __init__(
        self,
        zip_path,
        artifacts_dir="artifacts",
    ):
        self.zip_path = Path(
            zip_path
        ).expanduser()

        self.artifacts_dir = Path(
            artifacts_dir
        )

        self.archive = (
            GlasgowCumulativeArchive(
                self.zip_path
            )
        )


    def dataset_summary(
        self,
    ):
        increments = 0
        total_weight = 0.0
        changed_pixels = 0

        shape = None

        for record in self.archive.iter_increments():
            increment = record[
                "increment"
            ]

            if shape is None:
                shape = increment.shape

            increments += 1

            total_weight += float(
                record[
                    "added_count"
                ]
            )

            changed_pixels += int(
                record[
                    "changed_pixels"
                ]
            )

        return {
            "detector_shape":
                shape,

            "acquisition_increments":
                increments,

            "total_event_weight":
                total_weight,

            "changed_pixel_records":
                changed_pixels,
        }


    def convergence(
        self,
    ):
        path = (
            self.artifacts_dir
            / "glasgow_real_event_convergence.npz"
        )

        if not path.exists():
            return {
                "available": False,
                "artifact": str(path),
            }

        with np.load(path) as data:

            checkpoints = data[
                "checkpoints"
            ]

            correlation = data[
                "correlation"
            ]

            nrmse = data[
                "nrmse"
            ]

            js = data[
                "js"
            ]

        def first(
            values,
            condition,
        ):
            for n, value in zip(
                checkpoints,
                values,
            ):
                if condition(value):
                    return int(n)

            return None

        return {
            "available": True,

            "events_to_corr_090":
                first(
                    correlation,
                    lambda v: v >= 0.90,
                ),

            "events_to_corr_095":
                first(
                    correlation,
                    lambda v: v >= 0.95,
                ),

            "events_to_corr_099":
                first(
                    correlation,
                    lambda v: v >= 0.99,
                ),

            "events_to_js_005":
                first(
                    js,
                    lambda v: v <= 0.05,
                ),

            "events_to_nrmse_025":
                first(
                    nrmse,
                    lambda v: v <= 0.25,
                ),

            "final_correlation":
                float(
                    correlation[-1]
                ),

            "final_js":
                float(
                    js[-1]
                ),

            "final_nrmse":
                float(
                    nrmse[-1]
                ),
        }


    def heldout_prediction(
        self,
    ):
        """
        Consolidated previously established real-data result.

        These values come from DPI-LAB-15C.
        """

        return {
            "available": True,

            "training_pool":
                32088,

            "heldout_events":
                8022,

            "uniform_mean_log_likelihood":
                -12.47664899,

            "learned_mean_log_likelihood":
                -11.73898141,

            "information_gain_bits_per_event":
                1.06422936,

            "first_positive_gain_training_events":
                19252,

            "interpretation":
                (
                    "Learned spatial structure predicts "
                    "held-out future events better than "
                    "a uniform detector model."
                ),
        }


    def joint_spatial_information(
        self,
    ):
        """
        Consolidated DPI-LAB-15E result.
        """

        return {
            "available": True,

            "train_events":
                24066,

            "validation_events":
                8022,

            "test_events":
                8022,

            "factorized_vs_uniform_bits_per_event":
                1.48662209,

            "joint_vs_uniform_bits_per_event":
                1.56309501,

            "joint_vs_factorized_bits_per_event":
                0.07647292,

            "blocks_joint_beats_factorized":
                "8/8",

            "permutation_p":
                0.009901,

            "interpretation":
                (
                    "Joint X-Y spatial structure contains "
                    "predictive information beyond independent "
                    "X and Y marginals."
                ),
        }


    def temporal_order_null(
        self,
    ):
        path = (
            self.artifacts_dir
            / "glasgow_multiwindow_null_results.npz"
        )

        if not path.exists():
            return {
                "available": False,
                "artifact": str(path),
            }

        with np.load(path) as data:
            pvalues = data[
                "pvalues"
            ]

            threshold = float(
                data[
                    "bonf_threshold"
                ]
            )

        names = [
            "drift",
            "attractor_distortion",
            "mean_local_instability",
        ]

        result = {
            "available": True,
            "bonferroni_threshold":
                threshold,
        }

        for i, name in enumerate(names):

            result[name] = {
                "minimum_p":
                    float(
                        np.min(
                            pvalues[:, i]
                        )
                    ),

                "raw_p_lt_005_windows":
                    int(
                        np.sum(
                            pvalues[:, i]
                            < 0.05
                        )
                    ),

                "bonferroni_significant_windows":
                    int(
                        np.sum(
                            pvalues[:, i]
                            < threshold
                        )
                    ),
            }

        result[
            "interpretation"
        ] = (
            "Current CKE/AURORA summary metrics do not "
            "show strong acquisition-order-sensitive "
            "structure after multiple-testing correction."
        )

        return result


    def stationarity_null(
        self,
    ):
        """
        Consolidated DPI-LAB-15H stationarity-null result.
        """

        return {
            "available": True,

            "x_range_p":
                0.638723,

            "y_range_p":
                0.598802,

            "width_range_p":
                0.133733,

            "mean_centroid_step_p":
                0.504990,

            "max_js_first_p":
                0.165669,

            "mean_js_adj_p":
                0.239521,

            "max_js_adj_p":
                0.091816,

            "interpretation":
                (
                    "Observed slow spatial variation is "
                    "consistent with stationarity/sampling "
                    "fluctuation under the tested null."
                ),
        }


    def run_all(
        self,
    ):
        return {
            "dataset":
                self.dataset_summary(),

            "convergence":
                self.convergence(),

            "heldout_prediction":
                self.heldout_prediction(),

            "joint_spatial_information":
                self.joint_spatial_information(),

            "temporal_order_null":
                self.temporal_order_null(),

            "stationarity_null":
                self.stationarity_null(),
        }
