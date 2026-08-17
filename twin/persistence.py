from pathlib import Path
import json
import numpy as np


def _to_jsonable(value):
    """
    Convert common NumPy/Python objects into JSON-safe values.
    Large arrays should be stored separately in NPZ files.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (list, tuple)):
        return [
            _to_jsonable(v)
            for v in value
        ]

    if isinstance(value, dict):
        return {
            str(k): _to_jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    return str(value)


class TwinRecorder:
    """
    Persist a MeasurementTwinPlatform timeline.

    Layout:

        run_dir/
            manifest.json
            states/
                state_000001.json
                state_000001.npz
                ...
    """

    def __init__(
        self,
        run_dir,
    ):
        self.run_dir = Path(run_dir)
        self.states_dir = (
            self.run_dir / "states"
        )

        self.states_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.records = []


    def record(
        self,
        state,
    ):
        version = int(
            state.version
        )

        stem = (
            f"state_{version:06d}"
        )

        json_path = (
            self.states_dir
            / f"{stem}.json"
        )

        npz_path = (
            self.states_dir
            / f"{stem}.npz"
        )

        arrays = {}

        # ----------------------------------------------------
        # Reconstructed field
        # ----------------------------------------------------

        if (
            state.reconstructed_field
            is not None
            and
            isinstance(
                state.reconstructed_field.data,
                np.ndarray,
            )
        ):
            arrays[
                "reconstructed_field"
            ] = (
                state.reconstructed_field.data
            )

        # ----------------------------------------------------
        # Detector state
        # ----------------------------------------------------

        if (
            state.detector_state
            is not None
            and
            isinstance(
                state.detector_state.data,
                np.ndarray,
            )
        ):
            arrays[
                "detector_state"
            ] = (
                state.detector_state.data
            )

        # ----------------------------------------------------
        # Observable manifold
        # ----------------------------------------------------

        if (
            state.manifold
            is not None
            and
            isinstance(
                state.manifold.state,
                np.ndarray,
            )
        ):
            arrays[
                "manifold"
            ] = (
                state.manifold.state
            )

        # ----------------------------------------------------
        # Dynamics arrays
        # ----------------------------------------------------

        if state.dynamics is not None:

            traj = (
                state.dynamics.trajectory
            )

            if isinstance(
                traj,
                dict,
            ):
                for key in [
                    "state",
                    "velocities",
                    "speed",
                ]:
                    value = traj.get(key)

                    if isinstance(
                        value,
                        np.ndarray,
                    ):
                        arrays[
                            f"dynamics_trajectory_{key}"
                        ] = value

            phase = (
                state.dynamics.phase
            )

            if isinstance(
                phase,
                dict,
            ):
                for key in [
                    "signal",
                    "trajectory",
                    "phase",
                ]:
                    value = phase.get(key)

                    if isinstance(
                        value,
                        np.ndarray,
                    ):
                        arrays[
                            f"dynamics_phase_{key}"
                        ] = value

            instability = (
                state.dynamics.instability
            )

            if isinstance(
                instability,
                np.ndarray,
            ):
                arrays[
                    "dynamics_instability"
                ] = instability

        if arrays:
            np.savez_compressed(
                npz_path,
                **arrays,
            )

        # ----------------------------------------------------
        # Metadata summary
        # ----------------------------------------------------

        record = {
            "version":
                version,

            "measurement": {
                "modality":
                    getattr(
                        state.measurement,
                        "modality",
                        None,
                    ),

                "detector_id":
                    getattr(
                        state.measurement,
                        "detector_id",
                        None,
                    ),

                "timestamp":
                    getattr(
                        state.measurement,
                        "timestamp",
                        None,
                    ),
            },

            "detector_diagnostics":
                _to_jsonable(
                    None
                    if state.detector_diagnostics is None
                    else {
                        "visibility":
                            state.detector_diagnostics.visibility,

                        "quality_status":
                            state.detector_diagnostics.quality_status,

                        "metadata":
                            state.detector_diagnostics.metadata,
                    }
                ),

            "upstream":
                _to_jsonable(
                    None
                    if state.upstream is None
                    else {
                        "x":
                            state.upstream.x,

                        "y":
                            state.upstream.y,

                        "z":
                            state.upstream.z,

                        "confidence":
                            state.upstream.confidence,

                        "metadata":
                            state.upstream.metadata,
                    }
                ),

            "reconstructed_field":
                _to_jsonable(
                    None
                    if state.reconstructed_field is None
                    else {
                        "domain":
                            state.reconstructed_field.domain,

                        "metadata":
                            state.reconstructed_field.metadata,
                    }
                ),

            "detector_state":
                _to_jsonable(
                    None
                    if state.detector_state is None
                    else {
                        "detector_type":
                            state.detector_state.detector_type,

                        "metadata":
                            state.detector_state.metadata,
                    }
                ),

            "manifold":
                _to_jsonable(
                    None
                    if state.manifold is None
                    else {
                        "names":
                            state.manifold.names,

                        "metadata":
                            state.manifold.metadata,
                    }
                ),

            "dynamics":
                _to_jsonable(
                    None
                    if state.dynamics is None
                    else {
                        "trajectory":
                            state.dynamics.trajectory,

                        "phase":
                            state.dynamics.phase,

                        "attractor":
                            state.dynamics.attractor,

                        "regime":
                            state.dynamics.regime,

                        "prediction":
                            state.dynamics.prediction,

                        "instability":
                            state.dynamics.instability,

                        "metadata":
                            state.dynamics.metadata,
                    }
                ),

            "observability":
                _to_jsonable(
                    None
                    if state.observability is None
                    else {
                        "depth_score":
                            state.observability.depth_score,

                        "parameter_map":
                            state.observability.parameter_map,

                        "uncertainty":
                            state.observability.uncertainty,

                        "degeneracy":
                            state.observability.degeneracy,

                        "metadata":
                            state.observability.metadata,
                    }
                ),

            "metadata":
                _to_jsonable(
                    state.metadata
                ),

            "arrays_file":
                (
                    npz_path.name
                    if arrays
                    else None
                ),
        }

        json_path.write_text(
            json.dumps(
                record,
                indent=2,
            )
        )

        self.records.append(
            record
        )

        return record


    def finalize(self):
        manifest = {
            "state_count":
                len(self.records),

            "versions":
                [
                    r["version"]
                    for r in self.records
                ],

            "states":
                [
                    f"states/state_{r['version']:06d}.json"
                    for r in self.records
                ],
        }

        manifest_path = (
            self.run_dir
            / "manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            )
        )

        return manifest_path


class TwinReplay:
    """
    Lightweight replay reader for a recorded run.
    """

    def __init__(
        self,
        run_dir,
    ):
        self.run_dir = Path(run_dir)

        manifest_path = (
            self.run_dir
            / "manifest.json"
        )

        self.manifest = json.loads(
            manifest_path.read_text()
        )


    def __len__(self):
        return int(
            self.manifest[
                "state_count"
            ]
        )


    def state_metadata(
        self,
        version,
    ):
        path = (
            self.run_dir
            / "states"
            / f"state_{version:06d}.json"
        )

        return json.loads(
            path.read_text()
        )


    def state_arrays(
        self,
        version,
    ):
        path = (
            self.run_dir
            / "states"
            / f"state_{version:06d}.npz"
        )

        if not path.exists():
            return {}

        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            return {
                key: data[key]
                for key in data.files
            }
