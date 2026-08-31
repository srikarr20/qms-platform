from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ParameterFit:
    parameters: Dict[str, float]
    mse: float
    state_estimate: np.ndarray


@dataclass(frozen=True)
class FamilyFit:
    family: str
    parameters: Dict[str, float]
    mse: float
    state_estimate: np.ndarray


def build_window_matrix(
    A: np.ndarray,
    C: np.ndarray,
    length: int,
) -> np.ndarray:
    """
    Stacked discrete trajectory observation operator:

        H = [CA; CA^2; ...; CA^L]
    """
    A = np.asarray(
        A,
        dtype=float,
    )

    C = np.asarray(
        C,
        dtype=float,
    )

    rows = []
    Ak = np.eye(
        A.shape[0]
    )

    for _ in range(length):
        Ak = A @ Ak
        rows.append(
            C @ Ak
        )

    return np.vstack(rows)


def fit_state_for_model(
    measurements,
    A: np.ndarray,
    C: np.ndarray,
) -> ParameterFit:
    """
    Jointly fit the unknown state at the beginning
    of a measurement window for a fixed model.
    """
    y = np.asarray(
        measurements,
        dtype=float,
    ).reshape(-1)

    channels = (
        np.asarray(
            C
        ).shape[0]
    )

    if len(y) % channels != 0:
        raise ValueError(
            "Measurement length is incompatible "
            "with measurement matrix."
        )

    length = (
        len(y)
        // channels
    )

    H = build_window_matrix(
        A,
        C,
        length,
    )

    x_start = (
        np.linalg.pinv(H)
        @ y
    )

    residual = (
        y
        - H @ x_start
    )

    mse = float(
        residual @ residual
        / len(residual)
    )

    return ParameterFit(
        parameters={},
        mse=mse,
        state_estimate=x_start,
    )


def fit_parameter_grid(
    measurements,
    *,
    base_parameters: Mapping[str, float],
    parameter_name: str,
    candidate_values: Iterable[float],
    transition_builder: Callable[
        [Mapping[str, float]],
        np.ndarray,
    ],
    C: np.ndarray,
):
    """
    Grid-search one supported model parameter while
    jointly estimating the unknown starting state.
    """
    fits = []

    for value in candidate_values:
        params = dict(
            base_parameters
        )

        params[
            parameter_name
        ] = float(value)

        A = transition_builder(
            params
        )

        result = fit_state_for_model(
            measurements,
            A,
            C,
        )

        fits.append(
            ParameterFit(
                parameters={
                    parameter_name:
                        float(value)
                },
                mse=result.mse,
                state_estimate=(
                    result.state_estimate
                ),
            )
        )

    best = min(
        fits,
        key=lambda r: r.mse,
    )

    return best, fits


def candidate_parameter_sets(
    parameter_names: Sequence[str],
    grids: Mapping[
        str,
        Iterable[float],
    ],
):
    """
    Generate Cartesian parameter-grid candidates.
    """
    values = [
        list(
            grids[name]
        )
        for name
        in parameter_names
    ]

    for combination in product(
        *values
    ):
        yield {
            name: float(value)
            for name, value
            in zip(
                parameter_names,
                combination,
            )
        }


def fit_model_family(
    measurements,
    *,
    family_name: str,
    parameter_names: Sequence[str],
    grids: Mapping[
        str,
        Iterable[float],
    ],
    base_parameters: Mapping[str, float],
    transition_builder: Callable[
        [Mapping[str, float]],
        np.ndarray,
    ],
    C: np.ndarray,
) -> FamilyFit:
    """
    Fit the best member of one candidate model family.
    """
    best = None

    for changes in candidate_parameter_sets(
        parameter_names,
        grids,
    ):
        params = dict(
            base_parameters
        )

        params.update(
            changes
        )

        A = transition_builder(
            params
        )

        result = fit_state_for_model(
            measurements,
            A,
            C,
        )

        candidate = FamilyFit(
            family=family_name,
            parameters=dict(
                changes
            ),
            mse=result.mse,
            state_estimate=(
                result.state_estimate
            ),
        )

        if (
            best is None
            or candidate.mse
            < best.mse
        ):
            best = candidate

    if best is None:
        raise ValueError(
            f"No candidates generated for "
            f"family {family_name!r}"
        )

    return best


def identify_model_family(
    measurements,
    *,
    model_families: Mapping[
        str,
        Sequence[str],
    ],
    grids,
    base_parameters,
    transition_builder,
    C,
):
    """
    Fit all supplied model families and rank by residual MSE.
    """
    fits = [
        fit_model_family(
            measurements,
            family_name=name,
            parameter_names=params,
            grids=grids,
            base_parameters=(
                base_parameters
            ),
            transition_builder=(
                transition_builder
            ),
            C=C,
        )
        for name, params
        in model_families.items()
    ]

    fits.sort(
        key=lambda r: r.mse
    )

    return fits


def adequacy_threshold(
    nominal_mses,
    *,
    quantile: float = 0.99,
) -> float:
    """
    Empirical model-adequacy threshold calibrated
    from nominal-model residuals.
    """
    values = np.asarray(
        nominal_mses,
        dtype=float,
    )

    return float(
        np.quantile(
            values,
            quantile,
        )
    )


def reject_unknown_mechanism(
    winning_mse: float,
    threshold: float,
) -> bool:
    """
    True when the best supported model family still
    fits worse than the calibrated adequacy threshold.
    """
    return bool(
        winning_mse
        > threshold
    )
