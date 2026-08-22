# QMS Platform

QMS Platform is a modality-aware measurement-twin runtime for connecting downstream measurements to reconstruction, observability, temporal dynamics, and replay.

## Architecture

The current architecture supports two validated branches.
Optical / DPI:
raw 4-quadrature detector data -> MeasurementAdapter -> DPI inverse reconstruction -> ReconstructedField -> DetectorState -> ObservableManifold -> AURORA dynamics -> PlatformTwinState
MRI:
raw single-coil Cartesian k-space -> MeasurementAdapter -> MRI inverse FFT reconstruction -> ReconstructedField -> DetectorState -> ObservableManifold -> AURORA dynamics -> PlatformTwinState

## Core Idea

The platform separates measurement physics from the shared twin layer.
Each modality provides a MeasurementAdapter and a ReconstructionAdapter. The common runtime handles reconstructed-field history, detector-conditioned temporal state, observable manifolds, AURORA dynamics, persistence, and replay.
Public runtime pattern: state = platform.ingest(raw_measurement)

## Public API

import qms_platform as qmp
Key entry points: qmp.MeasurementTwinPlatform, qmp.create_optical_platform, qmp.create_mri_platform, qmp.DPIReconstructionAdapter, qmp.MRIKSpaceReconstructionAdapter, qmp.TwinRecorder, qmp.TwinReplay.

## Optical Branch

The optical path currently uses four phase-shifted detector measurements: I0, I90, I180, I270. These are used to recover a complex detector field and perform virtual upstream propagation/source inference through DPI.
Synthetic matched-model validation: DPI 4Q mean normalized complex error 0.04578757; weighted phase error 0.00443363 rad; centroid error 0.110701 microns; separation error 3.142758 microns.
Blind source-depth inference has also been demonstrated in simulation. At 4% noise: depth MAE 1.056474 mm; within 1 mm: 60%.
Depth/wavelength degeneracy has been explicitly tested and is treated as an observability limitation rather than ignored.

## MRI Branch

The current MRI path supports single-coil Cartesian k-space -> centered inverse FFT -> complex 4D MRI field.
In the matched synthetic reconstruction test: normalized complex reconstruction error approximately 2.8e-16; reconstructed field shape (T, H, W, Z); shared observable manifold X(t) = [C, K, E]; AURORA dynamics active.
This is currently a minimal MRI reconstruction path. It does not yet include multi-coil sensitivity reconstruction, SENSE/GRAPPA, non-Cartesian trajectories, compressed sensing, scanner-specific corrections, or raw vendor-format ingestion.

## Shared Observability Layer

A reconstructed temporal field sequence is transformed as: field evolution -> temporal detector state -> C(t), K(t), E(t) -> normalized observable manifold X(t).
AURORA then operates on this trajectory to estimate trajectory speed, drift, phase evolution, attractor geometry, attractor distortion, and local instability.

## Real-Data Validation

The Glasgow Heralded Diffraction SM dataset has been processed end to end on the detector/observability side.
Dataset: detector 512 x 512; acquisition increments 4069; total detector-event weight 40110.
Supported: reproducible detector-plane spatial structure; held-out future-event spatial prediction; genuine joint X-Y information.
Held-out prediction: training events 32088; held-out events 8022; information gain 1.06422936 bits/event.
Joint spatial information: joint vs factorized gain 0.07647292 bits/event; joint model beat factorized model in 8/8 blocks; permutation p-value 0.009901.
Not supported by current tests: strong acquisition-order-sensitive dynamics and slow detector-plane drift beyond stationarity/sampling fluctuations. Across 63 temporal windows, no current AURORA/CKE metric survived multiple-testing correction.
Not yet available: real phase-aware DPI upstream reconstruction has not been demonstrated with the Glasgow dataset because it contains sparse detector counts/event locations rather than the phase-sensitive quadrature information required by the current DPI inversion.

## Evidence Categories

The platform explicitly separates DEMONSTRATED, SUPPORTED, NOT_SUPPORTED, and NOT_AVAILABLE.
A machine-readable evidence report is generated locally at artifacts/qms_platform_evidence.json.

## Persistence and Replay

A live twin run can be recorded with qmp.TwinRecorder("artifacts/run") and replayed with qmp.TwinReplay("artifacts/run").

## Installation

Editable local install: python3 -m pip install -e .
Then: import qms_platform as qmp

## External Integrations

QMS diagnostics and AURORA dynamics are loaded from external repositories. They can be resolved through QMS_RUNTIME_ROOT and AURORA_RUNTIME_ROOT.

## Scientific Boundary

QMS Platform distinguishes carefully between measurement evidence, reconstructed state, observable state, and dynamical interpretation.
A reconstructed upstream field is a measurement-conditioned estimate, not a claim that a physically measured hidden plane directly exists in the dataset. Likewise, detector-plane predictability does not by itself establish a unique upstream physical state.

## Current Status

Architecture: multi-modality runtime demonstrated; optical raw-ingest path demonstrated; MRI raw-k-space path demonstrated; persistence/replay demonstrated.
Synthetic physics: matched DPI upstream reconstruction demonstrated; blind depth inference demonstrated; active observability demonstrated; depth/wavelength degeneracy characterized.
Real data: detector ingestion demonstrated; spatial predictability supported; joint 2D structure supported; strong acquisition-order dynamics not supported; real phase-aware upstream reconstruction not yet available.

## License

License not yet specified.

## Paper

The QMS Platform framework and initial validation are described in:

**Srikar R.**

*QMS Platform: A Quantum Measurement Observability and Reconstruction-Assurance Framework with Initial Validation.*

Zenodo, 2026.

**DOI:** https://doi.org/10.5281/zenodo.22057334

[![Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22057334.svg)](https://doi.org/10.5281/zenodo.22057334)

### Related research release

The integrated Quantum-Research snapshot corresponding to this work is archived at:

**DOI:** https://doi.org/10.5281/zenodo.22057237
