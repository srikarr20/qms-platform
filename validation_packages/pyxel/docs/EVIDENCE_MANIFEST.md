# QMS + Pyxel Adaptive Measurement Twin — Evidence Manifest

This manifest maps the consolidated report to the supporting experiment outputs.

## Master Report

- `docs/pyxel_twin/QMS_Pyxel_Adaptive_Measurement_Twin_Report.docx`

## Pyxel Computational Twin Evidence

### Controlled degradation observability

- `experiments/pyxel_validation/qms_pyxel_001/`
- output-node-noise sweep
- CTI sensitivity sweep
- stage observability analysis
- matched-severity CTI vs output-noise comparison

### Snapshot and inverse twin

- `experiments/pyxel_validation/qms_pyxel_twin_001/`
- blind CTI condition
- continuous CTI inverse fitting
- mixed CTI + output-node-noise decomposition
- adaptive Pyxel prediction

### Monte Carlo validation

- `experiments/pyxel_validation/qms_pyxel_twin_002/`
- 20 independent mixed-mechanism trials
- CTI stage-isolation validation
- empirical noise-estimation stability

### Time-indexed twin

- `experiments/pyxel_validation/qms_pyxel_twin_003/`
- six-state CTI + output-noise trajectory
- calibrated state reconstruction
- `qms_pyxel_twin_003b_calibrated_noise.json`

### Nonlinear noise calibration

- `experiments/pyxel_validation/qms_pyxel_twin_004/`
- 11-level output-node-noise calibration
- `qms_pyxel_twin_004_noise_calibration.json`

### Predictive twin

- `experiments/pyxel_validation/qms_pyxel_twin_005_predictive.json`

Key result:
- one-step prediction introduced explicit innovation/state-surprise logic

### Predictive divergence gating

- `experiments/pyxel_validation/qms_pyxel_twin_006_divergence.json`

Key result:
- expected evolution separated from parameter changes requiring adaptation

### Closed-loop adaptive twin

- `experiments/pyxel_validation/qms_pyxel_twin_007/`
- `qms_pyxel_twin_007_closed_loop.json`

Key result:
- predict → observe → innovation → assimilate/re-estimate → update

### Streaming twin

- `experiments/pyxel_validation/qms_pyxel_twin_008/`

Key result:
- sequential measurement consumption with persistent twin state

### External measurement interface

- `experiments/pyxel_validation/qms_pyxel_twin_009/`

Key result:
- observation source separated from Pyxel model side
- externalized measurement recovered CTI and output-noise parameters

---

## Real Glasgow Experimental Twin Evidence

### Experimental Twin 001

- `experiments/qms_experimental_twin_001/`
- real Glasgow acquisition increments
- 100-increment observable-state stream

### Experimental Twin 002

- `experiments/qms_experimental_twin_002/evidence/qms_experimental_twin_002_persistent.json`

Key result:
- raw state transitions: 2550
- smoothed state transitions: 1603
- transition reduction: 37.137%

### Experimental Twin 003

- `experiments/qms_experimental_twin_003/evidence/qms_experimental_twin_003_predictive.json`

Key result:
- 3968 one-step predictions
- mean innovation norm ≈ 0.00592392
- P95 ≈ 0.01432935
- max ≈ 0.15447367

### Experimental Twin 004

- `experiments/qms_experimental_twin_004/evidence/qms_experimental_twin_004_innovation_gate.json`

Key result:
- empirical innovation-state gating

### Experimental Twin 005

- `experiments/qms_experimental_twin_005/evidence/qms_experimental_twin_005_event_clustering.json`

Important limitation:
- adjacent-window clustering is inflated by heavy sliding-window overlap
- not used as final physical-event evidence

### Experimental Twin 006

- `experiments/qms_experimental_twin_006/evidence/qms_experimental_twin_006_nonoverlap.json`

Key result:
- large innovations survive one-window spacing
- lag-1 innovation correlation ≈ 0.456596

### Experimental Twin 007

- `experiments/qms_experimental_twin_007/evidence/qms_experimental_twin_007_dependence_null.json`

Key result:
- dependence-aware analysis
- no physical mechanism assigned

### Experimental Twin 008

- `experiments/qms_experimental_twin_008/evidence/qms_experimental_twin_008_excursions.json`

Key result:
- raw detector characterization around major innovation regions

### Experimental Twin 009

- `experiments/qms_experimental_twin_009/evidence/qms_experimental_twin_009_window_decomposition.json`

Key result:
- large innovations correspond to substantial normalized spatial redistribution at the actual 100-frame twin scale

### Experimental Twin 010

- `experiments/qms_experimental_twin_010/evidence/qms_experimental_twin_010_spatial_localization.json`

Key result:
- redistribution is broad rather than confined to a single detector cell

### Experimental Twin 011

- `experiments/qms_experimental_twin_011/evidence/qms_experimental_twin_011_transition_geometry.json`

Key result:
- partial transition reversal identified

### Experimental Twin 012

- `experiments/qms_experimental_twin_012/evidence/qms_experimental_twin_012_reversal_baseline.json`

Key result:
- 701→801 reversal is not exceptional sequence-wide
- stronger physical reversal interpretation rejected

### Experimental Twin 013

- `experiments/qms_experimental_twin_013/evidence/qms_experimental_twin_013_innovation_geometry.json`

Key result:
- innovation only weakly associated with ordinary transition metrics
- supports distinction between amount of change and prediction surprise

### Experimental Twin 014

- `experiments/qms_experimental_twin_014/evidence/qms_experimental_twin_014_surprise_specificity.json`

Key result:
- 701 and 801 remain unusually surprising after accounting for transition magnitude

### Experimental Twin 015

- `experiments/qms_experimental_twin_015/evidence/qms_experimental_twin_015_matched_surprise.json`

Key result:
- window 701 surprise ratio ≈ 23.31×
- window 801 surprise ratio ≈ 15.19×

### Experimental Twin 016

- `experiments/qms_experimental_twin_016/evidence/qms_experimental_twin_016_conformal_surprise.json`

Key result:
- window 701 rank 1/34
- window 801 rank 2/34

Important limitation:
- retrospective ranking; not treated as prospective validation

### Experimental Twin 017

- `experiments/qms_experimental_twin_017/evidence/qms_experimental_twin_017_frozen_prospective.json`

Important limitation:
- threshold was frozen prospectively, but observable reference inherited future-data leakage
- superseded scientifically by TWIN-018

### Experimental Twin 018

- `experiments/qms_experimental_twin_018/evidence/qms_experimental_twin_018_causal_frozen_reference.json`

Key result:
- causal reference uses increments 0–399 only
- calibration frozen through window 699
- future data used: false
- window 701 flagged
- window 801 flagged
- 3369 future samples
- 64 raw prospective flags

This is the primary real-data prospective evidence.

### Experimental Twin 019

- `experiments/qms_experimental_twin_019/evidence/qms_experimental_twin_019_alert_episodes.json`

Key result:
- 64 raw prospective flags consolidated into 6 macro alert episodes

### Experimental Twin 020

- `experiments/qms_experimental_twin_020/evidence/qms_experimental_twin_020_episode_robustness.json`

Key result:
- at refractory intervals 100 and 150, the same six dominant peaks are recovered:
  - 951
  - 1598
  - 2316
  - 2761
  - 3458
  - 3875

### Experimental Twin 021

- `experiments/qms_experimental_twin_021/evidence/qms_experimental_twin_021_baseline_robustness.json`

Key result:
- the same six dominant peaks recur in 6/6 tested causal reference/calibration configurations:
  - 951
  - 1598
  - 2316
  - 2761
  - 3458
  - 3875

Secondary peak:
- 2993 appears in 5/6 configurations

---

# Supported Integrated Claim

The current evidence supports the following bounded statement:

> QMS can operate as an observability, inference-assurance, prediction, and adaptive-twin layer around Pyxel detector models. In controlled Pyxel simulations it can localize detector-model divergence by pipeline stage, infer supported parameters, decompose stage-separated mechanisms, predict detector state, quantify innovation, and selectively adapt the model. On real Glasgow detector measurements it can construct a strictly causal observable-state twin that prospectively flags unexpected measurement-distribution evolution and identifies robust alert regions without assigning unsupported physical degradation labels.

---

# Scientific Boundaries

This evidence does not establish:

- calibrated real-detector CTI inference;
- calibrated real-detector read-noise inference;
- detector-health scoring;
- detector failure prediction;
- radiation-damage diagnosis;
- ESA hardware validation;
- hardware-in-the-loop operation;
- universal anomaly thresholds;
- arbitrary unknown mechanism identification;
- causal physical degradation;
- quantum-field reconstruction or dynamics.
