from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.constants import k, m_e
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class PyxelCTIConfig:
    beta: float = 0.3
    electron_volume: float = 1.62e-10
    transfer_period: float = 9.4722e-4
    full_well_capacity: float = 100000.0
    temperature: float = 300.0
    effective_mass_fraction: float = 0.5
    trap_release_time: float = 3e-2
    capture_cross_section: float = 1e-10


@dataclass(frozen=True)
class CTIEstimate:
    trap_density: float
    residual_rmse: float
    optimization_success: bool
    evaluations: int


def _rmse(a, b):
    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(
                        a,
                        dtype=float,
                    )
                    - np.asarray(
                        b,
                        dtype=float,
                    )
                ) ** 2
            )
        )
    )


def simulate_parallel_cti(
    baseline_pixel,
    log10_density: float,
    *,
    config: Optional[PyxelCTIConfig] = None,
):
    """
    Run the validated Pyxel CDM parallel-transfer model.

    Pyxel is imported lazily so the generic QMS detector-twin
    package can still be imported without Pyxel installed.
    """
    from pyxel.models.charge_transfer.utils_cdm import (
        run_cdm_parallel,
    )

    cfg = (
        config
        if config is not None
        else PyxelCTIConfig()
    )

    baseline = np.asarray(
        baseline_pixel,
        dtype=np.float64,
    )

    density = (
        10.0
        ** float(
            log10_density
        )
    )

    effective_mass = (
        cfg.effective_mass_fraction
        * m_e
    )

    thermal_velocity = (
        100.0
        * np.sqrt(
            3.0
            * k
            * cfg.temperature
            / effective_mass
        )
    )

    return run_cdm_parallel(
        array=baseline.copy(),
        beta=cfg.beta,
        vg=cfg.electron_volume,
        t=cfg.transfer_period,
        fwc=cfg.full_well_capacity,
        vth=thermal_velocity,
        tr=np.array(
            [
                cfg.trap_release_time
            ],
            dtype=float,
        ),
        nt=np.array(
            [
                density
            ],
            dtype=float,
        ),
        sigma=np.array(
            [
                cfg.capture_cross_section
            ],
            dtype=float,
        ),
        charge_injection=False,
        chg_inj_parallel_transfers=(
            baseline.shape[0]
        ),
    )


def estimate_parallel_cti_density(
    baseline_pixel,
    observed_pixel,
    *,
    log10_bounds=(7.0, 11.5),
    xatol: float = 1e-8,
    config: Optional[PyxelCTIConfig] = None,
) -> CTIEstimate:
    """
    Infer trap density by minimizing Pixel-stage RMSE
    against the known Pyxel CDM family.
    """

    def objective(
        log_density,
    ):
        predicted = (
            simulate_parallel_cti(
                baseline_pixel,
                log_density,
                config=config,
            )
        )

        return _rmse(
            predicted,
            observed_pixel,
        )

    result = minimize_scalar(
        objective,
        bounds=log10_bounds,
        method="bounded",
        options={
            "xatol":
                xatol
        },
    )

    return CTIEstimate(
        trap_density=float(
            10.0
            ** result.x
        ),

        residual_rmse=float(
            result.fun
        ),

        optimization_success=bool(
            result.success
        ),

        evaluations=int(
            result.nfev
        ),
    )
