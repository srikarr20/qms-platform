import os
import sys
from pathlib import Path
import numpy as np

from qms_core import DynamicState


DEFAULT_AURORA_ROOT = (
    Path.home()
    / "Desktop"
    / "aurora-github"
)


def resolve_aurora_root(explicit_root=None):
    """
    Resolve external AURORA repository path.

    Priority:
        1. explicit path
        2. AURORA_RUNTIME_ROOT environment variable
        3. known local fallback
    """

    if explicit_root is not None:
        root = Path(
            explicit_root
        ).expanduser()

        if root.exists():
            return root

        raise FileNotFoundError(
            f"Explicit AURORA root does not exist: {root}"
        )


    env_root = os.environ.get(
        "AURORA_RUNTIME_ROOT"
    )

    if env_root:
        root = Path(
            env_root
        ).expanduser()

        if root.exists():
            return root

        raise FileNotFoundError(
            "AURORA_RUNTIME_ROOT is set but does not exist: "
            f"{root}"
        )


    if DEFAULT_AURORA_ROOT.exists():
        return DEFAULT_AURORA_ROOT


    raise FileNotFoundError(
        "Could not locate AURORA repository. "
        "Pass aurora_root explicitly or set "
        "AURORA_RUNTIME_ROOT."
    )


def load_aurora_engines(
    aurora_root=None,
):
    root = resolve_aurora_root(
        aurora_root
    )

    package_root = (
        root
        / "src_instrument"
    )

    package_root_str = str(
        package_root
    )

    if package_root_str not in sys.path:
        sys.path.insert(
            0,
            package_root_str,
        )

    from aurora_project.core.trajectory_engine import (
        compute_trajectory,
    )

    from aurora_project.core.phase_engine import (
        build_phase_trajectory,
    )

    from aurora_project.core.attractor_engine import (
        fit_reference_circle,
        compute_attractor_distortion,
        compute_local_instability,
    )

    return {
        "compute_trajectory":
            compute_trajectory,

        "build_phase_trajectory":
            build_phase_trajectory,

        "fit_reference_circle":
            fit_reference_circle,

        "compute_attractor_distortion":
            compute_attractor_distortion,

        "compute_local_instability":
            compute_local_instability,

        "root":
            root,
    }


def enrich_with_aurora_dynamics(
    platform_state,
    aurora_root=None,
):
    """
    Add AURORA dynamics to PlatformTwinState.

    Requires:
        platform_state.manifold.state

    The AURORA repository is resolved dynamically.
    """

    if platform_state.manifold is None:
        raise ValueError(
            "PlatformTwinState has no ObservableManifold."
        )

    X = np.asarray(
        platform_state.manifold.state
    )

    if X.ndim != 2:
        raise ValueError(
            "Observable manifold must be a 2D trajectory."
        )

    if X.shape[0] < 3:
        raise ValueError(
            "Need at least 3 manifold states."
        )

    engines = load_aurora_engines(
        aurora_root
    )

    compute_trajectory = (
        engines[
            "compute_trajectory"
        ]
    )

    build_phase_trajectory = (
        engines[
            "build_phase_trajectory"
        ]
    )

    fit_reference_circle = (
        engines[
            "fit_reference_circle"
        ]
    )

    compute_attractor_distortion = (
        engines[
            "compute_attractor_distortion"
        ]
    )

    compute_local_instability = (
        engines[
            "compute_local_instability"
        ]
    )


    # --------------------------------------------------------
    # Trajectory
    # --------------------------------------------------------

    velocities, speed, drift = (
        compute_trajectory(
            X
        )
    )


    # --------------------------------------------------------
    # Phase
    #
    # Use E coordinate when available.
    # --------------------------------------------------------

    if X.shape[1] >= 3:
        phase_signal = X[:, 2]
    else:
        phase_signal = X[:, 0]

    (
        phase_trajectory,
        phase,
    ) = build_phase_trajectory(
        phase_signal
    )


    # --------------------------------------------------------
    # Attractor geometry
    # --------------------------------------------------------

    attractor_input = (
        X[:, :2]
    )

    center, radius = (
        fit_reference_circle(
            attractor_input
        )
    )

    (
        distortion_score,
        distortion,
    ) = compute_attractor_distortion(
        attractor_input,
        center,
        radius,
    )

    local_instability = (
        compute_local_instability(
            attractor_input
        )
    )


    # --------------------------------------------------------
    # Unified DynamicState
    # --------------------------------------------------------

    dynamics = DynamicState(
        trajectory={
            "state":
                X,

            "velocities":
                velocities,

            "speed":
                speed,

            "drift":
                float(
                    drift
                ),
        },

        phase={
            "signal":
                phase_signal,

            "trajectory":
                phase_trajectory,

            "phase":
                phase,
        },

        attractor={
            "center":
                center,

            "radius":
                float(
                    radius
                ),

            "distortion_score":
                float(
                    distortion_score
                ),

            "distortion":
                distortion,
        },

        instability=
            local_instability,

        metadata={
            "engine":
                "AURORA",

            "aurora_repository":
                str(
                    engines["root"]
                ),

            "trajectory_engine":
                "compute_trajectory",

            "phase_engine":
                "build_phase_trajectory",

            "attractor_engine":
                (
                    "fit_reference_circle + "
                    "compute_attractor_distortion + "
                    "compute_local_instability"
                ),

            "source_manifold":
                platform_state.manifold.names,
        },
    )

    platform_state.dynamics = dynamics

    return platform_state
